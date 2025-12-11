"""Q-ILS: Iterated Local Search guided by Q-Learning for TSP."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from src.tsp.constructive import CONSTRUCTIVES
from src.tsp.instance import TSPInstance
from src.tsp.local_search import (
    LOCAL_SEARCHES,
    two_opt,
    lin_kernighan,
    _build_neighbor_lists,
    _resolve_k,
    _THRESHOLD_NN,
)
from src.tsp.perturbation import PERTURBATIONS
from src.tsp.solution import Solution
from src.rl.q_table import QTable


@dataclass
class RunStats:
    """Statistics from a Q-ILS run."""

    # Timing (in seconds)
    total_time: float = 0.0
    init_time: float = 0.0  # Time for initial solution

    # Iterations
    total_iterations: int = 0
    best_iteration: int = 0  # Iteration where best solution was found
    improvements: int = 0  # Number of improvements found
    early_stopped: bool = False  # True if stopped early by reaching target

    # Solution quality
    initial_cost: float = 0.0
    initial_gap: float = 0.0  # Gap % of initial solution
    final_cost: float = 0.0
    final_gap: float = 0.0

    # Action/state distribution
    action_counts: dict[str, int] = field(default_factory=dict)
    state_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to flat dictionary for CSV export."""
        d = {
            "total_time": self.total_time,
            "init_time": self.init_time,
            "total_iterations": self.total_iterations,
            "best_iteration": self.best_iteration,
            "improvements": self.improvements,
            "early_stopped": self.early_stopped,
            "initial_cost": self.initial_cost,
            "initial_gap": self.initial_gap,
            "final_cost": self.final_cost,
            "final_gap": self.final_gap,
        }
        # Flatten action counts
        for action in Action:
            d[f"action_{action.name}"] = self.action_counts.get(action.name, 0)
        # Flatten state counts
        for state in State:
            d[f"state_{state.name}"] = self.state_counts.get(state.name, 0)
        return d


class State(IntEnum):
    """
    MDP states based on gap percentage from optimal.

    Gap = ((cost - opt_cost) / opt_cost) * 100
    """

    EXCELLENT = 0  # gap in [0, 2%]: reward 75
    GOOD = 1  # gap in (2%, 5%]: reward 50
    REGULAR = 2  # gap in (5%, 10%]: reward 25
    POOR = 3  # gap > 10%: reward 0
    BETTER = 4  # gap < 0 (better than known optimal): reward 100


# State rewards mapping
STATE_REWARDS: dict[State, int] = {
    State.EXCELLENT: 75,
    State.GOOD: 50,
    State.REGULAR: 25,
    State.POOR: 0,
    State.BETTER: 100,
}


class Action(IntEnum):
    """
    MDP actions: combinations of (perturbation, local_search).

    Light perturbations: two_swap, segment_reverse
    Destructive perturbations: random, nearest, cheapest, grasp (rebuild from scratch)
    """

    TWO_SWAP_2OPT = 0
    TWO_SWAP_LK = 1
    SEGMENT_REVERSE_2OPT = 2
    SEGMENT_REVERSE_LK = 3
    RANDOM_2OPT = 4
    NEAREST_2OPT = 5
    CHEAPEST_2OPT = 6
    NEAREST_LK = 7
    GRASP_2OPT = 8


# Action to (perturbation, local_search) mapping
ACTION_DECODE: dict[Action, tuple[str, str]] = {
    Action.TWO_SWAP_2OPT: ("two_swap", "two_opt"),
    Action.TWO_SWAP_LK: ("two_swap", "lin_kernighan"),
    Action.SEGMENT_REVERSE_2OPT: ("segment_reverse", "two_opt"),
    Action.SEGMENT_REVERSE_LK: ("segment_reverse", "lin_kernighan"),
    Action.RANDOM_2OPT: ("random", "two_opt"),
    Action.NEAREST_2OPT: ("nearest", "two_opt"),
    Action.CHEAPEST_2OPT: ("cheapest", "two_opt"),
    Action.NEAREST_LK: ("nearest", "lin_kernighan"),
    Action.GRASP_2OPT: ("grasp", "two_opt"),
}

# Reverse mapping
ACTION_ENCODE: dict[tuple[str, str], Action] = {v: k for k, v in ACTION_DECODE.items()}

N_STATES = len(State)
N_ACTIONS = len(Action)


class QILS:
    """
    Q-ILS: Iterated Local Search where a Q-Learning agent decides which
    (perturbation, local_search) pair to apply at each iteration.

    The agent observes the current state (based on gap percentage) and
    selects an action according to the learned Q-table.
    """

    def __init__(self, problem: TSPInstance, k: int | float = 0.5) -> None:
        self.problem = problem
        # Reuse precomputed distance matrix from instance (already 0-based)
        self.dist_matrix = problem.dist_matrix

        self.n_states = N_STATES
        self.n_actions = N_ACTIONS

        self.q_table: Optional[QTable] = None
        self.last_action: Optional[Action] = None
        self.last_state: Optional[State] = None
        self.last_stats: Optional[RunStats] = None

        # Pre-compute neighbor lists for instances large enough to benefit
        n = problem.dimension
        if n >= _THRESHOLD_NN:
            k_resolved = _resolve_k(k, n)
            self._neighbors = _build_neighbor_lists(self.dist_matrix, k_resolved)
        else:
            self._neighbors = None

    def load_q_table(self, path: Union[str, Path]) -> None:
        """Load Q-table from file."""
        self.q_table = QTable.from_txt(path)

    def get_state(self, cost: float, opt_cost: float) -> tuple[State, float]:
        """
        Map cost to discrete state and continuous reward based on gap %.

        State is discretized for Q-table indexing, but reward is continuous
        for more granular feedback: reward = max(0, 100 - gap).

        Args:
            cost: Current solution cost.
            opt_cost: Optimal (or best known) cost.

        Returns:
            (state, reward): Discrete state and continuous reward.
        """
        gap = ((cost - opt_cost) / opt_cost) * 100
        gap = round(gap, 7)

        if gap < 0:
            state = State.BETTER
        elif gap <= 2:
            state = State.EXCELLENT
        elif gap <= 5:
            state = State.GOOD
        elif gap <= 10:
            state = State.REGULAR
        else:
            state = State.POOR

        # Continuous reward in [0, 1]: 1.0 at gap=0%, 0.0 at gap>=100%
        reward = max(0.0, 1.0 - gap / 100.0)

        return state, reward

    def choose_action(self, state: State, epsilon: float = 0.0) -> Action:
        """
        Select action using epsilon-greedy policy from Q-table.

        Args:
            state: Current state.
            epsilon: Exploration rate.

        Returns:
            Selected action.

        Raises:
            ValueError: If Q-table not loaded.
        """
        if self.q_table is None:
            raise ValueError("Q-table not loaded. Call load_q_table() first.")

        # Exploration
        if random.random() < epsilon:
            return Action(random.randrange(self.n_actions))

        # Exploitation (greedy)
        q_row = self.q_table.table[state.value]
        return Action(int(q_row.argmax().item()))

    def _apply_perturbation(self, solution: Solution, pert_type: str) -> Solution:
        """
        Apply perturbation to solution.

        Light perturbations modify the existing solution.
        Destructive perturbations rebuild from scratch (ignore current solution).

        Args:
            solution: Current solution.
            pert_type: Perturbation type name.

        Returns:
            Perturbed solution.
        """
        # Light perturbations (modify existing solution)
        if pert_type in PERTURBATIONS:
            return PERTURBATIONS[pert_type](solution)

        # Destructive perturbations (rebuild from scratch using constructive)
        if pert_type in CONSTRUCTIVES:
            tour, _ = CONSTRUCTIVES[pert_type](self.problem)
            return Solution(tour, self.dist_matrix, is_closed=True)

        raise ValueError(f"Unknown perturbation type: {pert_type}")

    def _apply_local_search(self, solution: Solution, ls_type: str) -> Solution:
        """
        Apply local search to solution.

        Uses pre-computed neighbor lists when available for 2-opt variants.

        Args:
            solution: Input solution.
            ls_type: Local search type name.

        Returns:
            Improved solution.
        """
        if ls_type == "two_opt":
            # Use cached neighbors if available
            return two_opt(solution, neighbors=self._neighbors)
        elif ls_type == "lin_kernighan":
            return lin_kernighan(solution)
        elif ls_type in LOCAL_SEARCHES:
            return LOCAL_SEARCHES[ls_type](solution)
        else:
            raise ValueError(f"Unknown local search type: {ls_type}")

    def _get_initial_solution(self) -> Solution:
        """Generate initial solution via random constructive + 2-opt."""
        constructive_choice = random.choice(list(CONSTRUCTIVES.keys()))
        tour, _ = CONSTRUCTIVES[constructive_choice](self.problem)
        initial_solution = Solution(tour, self.dist_matrix, is_closed=True)
        return two_opt(initial_solution, neighbors=self._neighbors)

    def generate_transitions(
        self,
        max_iter: int = 50,
        opt_cost: float = 0.0,
        out_path: Union[str, Path] = "transitions.txt",
        beta: float = 0.0,
    ) -> Solution:
        """
        Generate transition data for MDP training.

        Runs ILS with random action choices, recording (s, a, r, s') tuples.

        Args:
            max_iter: Maximum iterations without improvement.
            opt_cost: Optimal cost for state calculation.
            out_path: Output file path.
            beta: Time discount weight. Penalizes slower operators by subtracting
                  beta * normalized_time from the reward. Time is normalized by
                  O(n²) so that beta has consistent meaning across instance sizes.
                  Reference: 1 second at n=100. Default 0.0 (no penalty).

        Returns:
            Best solution found.
        """
        ls_solution = self._get_initial_solution()
        best_solution = ls_solution.copy()

        iter_without_improvement = 0
        output_lines: list[str] = []

        action_list = list(ACTION_DECODE.keys())

        # Time normalization: t_ref = 1s for n=100 (scales as O(n²))
        n = self.problem.dimension
        t_ref = (n / 100) ** 2

        while iter_without_improvement < max_iter:
            # Current state
            i_state, _ = self.get_state(ls_solution.cost, opt_cost)

            # Random action selection
            action = random.choice(action_list)
            pert_type, ls_type = ACTION_DECODE[action]

            # Apply perturbation and local search (time in seconds)
            t_start = time.perf_counter()
            perturbed = self._apply_perturbation(ls_solution, pert_type)
            new_solution = self._apply_local_search(perturbed, ls_type)
            operator_time = time.perf_counter() - t_start

            # Acceptance criterion
            if new_solution.cost < best_solution.cost:
                best_solution = new_solution.copy()
                iter_without_improvement = 0
            else:
                iter_without_improvement += 1

            # Update current solution
            ls_solution = new_solution

            # Record transition with time-discounted reward
            f_state, reward = self.get_state(new_solution.cost, opt_cost)
            if beta > 0:
                reward -= beta * (operator_time / t_ref)

            output_lines.append(f"{i_state.value} {action.value} {reward:.2f} {f_state.value}")

        with open(out_path, "w") as f:
            f.write("\n".join(output_lines))

        return best_solution

    def run(
        self,
        max_iter: int = 50,
        opt_cost: float = 0.0,
        epsilon: float = 0.0,
        verbose: bool = True,
        early_stop: bool = True,
        early_stop_target: Optional[float] = None,
    ) -> Solution:
        """
        Run Q-ILS using the learned Q-table.

        Args:
            max_iter: Maximum iterations without improvement.
            opt_cost: Optimal cost for state calculation and gap reporting.
            epsilon: Exploration rate for action selection.
            verbose: Print progress information.
            early_stop: Stop early when reaching target cost.
            early_stop_target: Target cost for early stop (default: opt_cost).
                Use lower_bound when mip_gap > 0, or None to disable.

        Returns:
            Best solution found. Access self.last_stats for detailed metrics.

        Raises:
            ValueError: If opt_cost is invalid or Q-table not loaded.
        """
        if opt_cost <= 0:
            raise ValueError("opt_cost must be positive.")

        if self.q_table is None:
            raise ValueError("Q-table not loaded. Call load_q_table() first.")

        # Initialize stats
        stats = RunStats()
        stats.action_counts = {a.name: 0 for a in Action}
        stats.state_counts = {s.name: 0 for s in State}

        # Track total time
        t_start = time.perf_counter()

        # Generate initial solution
        t_init = time.perf_counter()
        ls_solution = self._get_initial_solution()
        best_solution = ls_solution.copy()
        stats.init_time = time.perf_counter() - t_init

        # Record initial solution quality
        stats.initial_cost = ls_solution.cost
        stats.initial_gap = ((ls_solution.cost - opt_cost) / opt_cost) * 100

        iter_without_improvement = 0
        iteration = 0
        best_iteration = 0

        while iter_without_improvement < max_iter:
            iteration += 1

            # Observe current state
            i_state, _ = self.get_state(ls_solution.cost, opt_cost)
            stats.state_counts[i_state.name] += 1

            # Select action via Q-table
            action = self.choose_action(i_state, epsilon=epsilon)
            stats.action_counts[action.name] += 1
            self.last_action = action
            self.last_state = i_state

            # Decode and apply action
            pert_type, ls_type = ACTION_DECODE[action]
            perturbed = self._apply_perturbation(ls_solution, pert_type)
            new_solution = self._apply_local_search(perturbed, ls_type)

            # Acceptance criterion
            if new_solution.cost < best_solution.cost:
                best_solution = new_solution.copy()
                best_iteration = iteration
                stats.improvements += 1
                iter_without_improvement = 0

                # Early stop: check if we reached the target
                # Use early_stop_target if provided, else opt_cost
                target = early_stop_target if early_stop_target is not None else opt_cost
                if early_stop and early_stop_target is not None and best_solution.cost <= target:
                    stats.early_stopped = True
                    if verbose:
                        print(f"[Q-ILS] Early stop: reached target cost {target:.4f}")
                    break
            else:
                iter_without_improvement += 1

            # Update current solution
            ls_solution = new_solution

            if verbose:
                print(
                    f"[Q-ILS] state={i_state.name}, action={action.name} "
                    f"({pert_type} + {ls_type}) "
                    f"best_cost={best_solution.cost:.4f}"
                )

        # Finalize stats
        stats.total_time = time.perf_counter() - t_start
        stats.total_iterations = iteration
        stats.best_iteration = best_iteration
        stats.final_cost = best_solution.cost
        stats.final_gap = ((best_solution.cost - opt_cost) / opt_cost) * 100

        self.last_stats = stats
        return best_solution
