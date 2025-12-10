"""Q-ILS: Iterated Local Search guided by Q-Learning for TSP."""

from __future__ import annotations

import random
from enum import IntEnum
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from src.tsp.constructive import CONSTRUCTIVES
from src.tsp.instance import TSPInstance
from src.tsp.local_search import LOCAL_SEARCHES, two_opt
from src.tsp.perturbation import PERTURBATIONS
from src.tsp.solution import Solution
from src.rl.q_table import QTable


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
    Destructive perturbations: random, nearest, cheapest (rebuild from scratch)
    """

    TWO_SWAP_2OPT = 0
    TWO_SWAP_LK = 1
    SEGMENT_REVERSE_2OPT = 2
    SEGMENT_REVERSE_LK = 3
    RANDOM_2OPT = 4
    NEAREST_2OPT = 5
    CHEAPEST_2OPT = 6
    NEAREST_LK = 7


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

    def __init__(self, problem: TSPInstance) -> None:
        """
        Initialize Q-ILS solver.

        Args:
            problem: TSP instance to solve.
        """
        self.problem = problem
        # Reuse precomputed distance matrix from instance (already 0-based)
        self.dist_matrix = problem.dist_matrix

        self.n_states = N_STATES
        self.n_actions = N_ACTIONS

        self.q_table: Optional[QTable] = None
        self.last_action: Optional[Action] = None
        self.last_state: Optional[State] = None

    def load_q_table(self, path: Union[str, Path]) -> None:
        """
        Load Q-table from file.

        Args:
            path: Path to Q-table text file.
        """
        self.q_table = QTable.from_txt(path)

    def get_state(self, cost: float, opt_cost: float) -> tuple[State, int]:
        """
        Map current cost to discrete state based on gap percentage.

        Args:
            cost: Current solution cost.
            opt_cost: Optimal (or best known) cost.

        Returns:
            Tuple of (state, reward).
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

        return state, STATE_REWARDS[state]

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

        Args:
            solution: Input solution.
            ls_type: Local search type name.

        Returns:
            Improved solution.
        """
        if ls_type not in LOCAL_SEARCHES:
            raise ValueError(f"Unknown local search type: {ls_type}")
        return LOCAL_SEARCHES[ls_type](solution)

    def _get_initial_solution(self) -> Solution:
        """
        Generate initial solution using random constructive heuristic.

        Returns:
            Initial solution improved by 2-opt.
        """
        constructive_choice = random.choice(list(CONSTRUCTIVES.keys()))
        tour, _ = CONSTRUCTIVES[constructive_choice](self.problem)
        initial_solution = Solution(tour, self.dist_matrix, is_closed=True)
        return two_opt(initial_solution)

    def generate_transitions(
        self,
        max_iter: int = 50,
        opt_cost: float = 0.0,
        out_path: Union[str, Path] = "transitions.txt",
    ) -> Solution:
        """
        Generate transition data for MDP training.

        Runs ILS with random action choices, recording (s, a, r, s') tuples.

        Args:
            max_iter: Maximum iterations without improvement.
            opt_cost: Optimal cost for state calculation.
            out_path: Output file path.

        Returns:
            Best solution found.
        """
        ls_solution = self._get_initial_solution()
        best_solution = ls_solution.copy()

        iter_without_improvement = 0
        output_lines: list[str] = []

        action_list = list(ACTION_DECODE.keys())

        while iter_without_improvement < max_iter:
            # Current state
            i_state, _ = self.get_state(ls_solution.cost, opt_cost)

            # Random action selection
            action = random.choice(action_list)
            pert_type, ls_type = ACTION_DECODE[action]

            # Apply perturbation and local search
            perturbed = self._apply_perturbation(ls_solution, pert_type)
            new_solution = self._apply_local_search(perturbed, ls_type)

            # Acceptance criterion
            if new_solution.cost < best_solution.cost:
                best_solution = new_solution.copy()
                iter_without_improvement = 0
            else:
                iter_without_improvement += 1

            # Update current solution
            ls_solution = new_solution

            # Record transition
            f_state, reward = self.get_state(new_solution.cost, opt_cost)
            output_lines.append(f"{i_state.value} {action.value} {reward} {f_state.value}")

        with open(out_path, "w") as f:
            f.write("\n".join(output_lines))

        return best_solution

    def run(
        self,
        max_iter: int = 50,
        opt_cost: float = 0.0,
        epsilon: float = 0.0,
        verbose: bool = True,
    ) -> Solution:
        """
        Run Q-ILS using the learned Q-table.

        Args:
            max_iter: Maximum iterations without improvement.
            opt_cost: Optimal cost for state calculation.
            epsilon: Exploration rate for action selection.
            verbose: Print progress information.

        Returns:
            Best solution found.

        Raises:
            ValueError: If opt_cost is invalid or Q-table not loaded.
        """
        if opt_cost <= 0:
            raise ValueError("opt_cost must be positive.")

        if self.q_table is None:
            raise ValueError("Q-table not loaded. Call load_q_table() first.")

        ls_solution = self._get_initial_solution()
        best_solution = ls_solution.copy()

        iter_without_improvement = 0

        while iter_without_improvement < max_iter:
            # Observe current state
            i_state, _ = self.get_state(ls_solution.cost, opt_cost)

            # Select action via Q-table
            action = self.choose_action(i_state, epsilon=epsilon)
            self.last_action = action
            self.last_state = i_state

            # Decode and apply action
            pert_type, ls_type = ACTION_DECODE[action]
            perturbed = self._apply_perturbation(ls_solution, pert_type)
            new_solution = self._apply_local_search(perturbed, ls_type)

            # Acceptance criterion
            if new_solution.cost < best_solution.cost:
                best_solution = new_solution.copy()
                iter_without_improvement = 0
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

        return best_solution
