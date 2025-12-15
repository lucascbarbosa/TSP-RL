"""DQN environment wrapper for Q-ILS operators."""

import random
import time
from typing import Optional

import numpy as np

from src.rl.dqn.state import DQNState, normalize_gap, compute_delta_reward


# Action decoding (local copy to avoid circular import with q_ils)
# Maps action index to (perturbation, local_search) pair
ACTION_DECODE: dict[int, tuple[str, str]] = {
    0: ("two_swap", "two_opt"),
    1: ("two_swap", "lin_kernighan"),
    2: ("segment_reverse", "two_opt"),
    3: ("segment_reverse", "lin_kernighan"),
    4: ("random", "two_opt"),
    5: ("nearest", "two_opt"),
    6: ("cheapest", "two_opt"),
    7: ("nearest", "lin_kernighan"),
    8: ("grasp", "two_opt"),
}
N_ACTIONS = len(ACTION_DECODE)
from src.tsp.constructive import CONSTRUCTIVES
from src.tsp.instance import TSPInstance
from src.tsp.local_search import (
    two_opt,
    lin_kernighan,
    _build_neighbor_lists,
    _resolve_k,
    _THRESHOLD_NN,
)
from src.tsp.perturbation import PERTURBATIONS
from src.tsp.solution import Solution


class DQNEnv:
    """
    Gym-like environment for DQN training on TSP.

    Wraps the existing TSP operators (constructives, perturbations, local searches)
    into a standard RL environment interface.

    Episode structure:
    1. reset(): Sample instance, generate initial solution, start timer
    2. step(action): Apply (perturbation, local_search), return (state, reward, done)
    3. Episode ends when time budget is exhausted
    """

    def __init__(
        self,
        instance: TSPInstance,
        time_budget: float,
        history_len: int = 3,
        k: int | float = 0.5,
    ) -> None:
        """
        Initialize environment.

        Args:
            instance: TSP instance to solve.
            time_budget: Maximum episode duration in seconds.
            history_len: Number of past actions to track in state.
            k: Neighbor list parameter for 2-opt.
        """
        self.instance = instance
        self.time_budget = time_budget
        self.history_len = history_len

        # Reuse precomputed distance matrix
        self.dist_matrix = instance.dist_matrix

        # Compute optimal cost for gap calculation
        opt_tour = instance.opt_tour
        if opt_tour:
            self.opt_cost = sum(
                instance.get_weight(opt_tour[i], opt_tour[(i + 1) % len(opt_tour)]) for i in range(len(opt_tour))
            )
        else:
            self.opt_cost = 1.0

        # Pre-compute neighbor lists for large instances
        n = instance.dimension
        if n >= _THRESHOLD_NN:
            k_resolved = _resolve_k(k, n)
            self._neighbors = _build_neighbor_lists(self.dist_matrix, k_resolved)
        else:
            self._neighbors = None

        # Episode state (set by reset())
        self.solution: Optional[Solution] = None
        self.best_gap: float = float("inf")
        self.history: list[int] = []
        self.t_start: float = 0.0

    def reset(self) -> DQNState:
        """
        Reset environment for new episode.

        Returns:
            Initial state.
        """
        # Generate initial solution: random constructive + 2-opt
        constructive = random.choice(list(CONSTRUCTIVES.keys()))
        tour, _ = CONSTRUCTIVES[constructive](self.instance)
        self.solution = Solution(tour, self.dist_matrix, is_closed=True)
        self.solution = two_opt(self.solution, neighbors=self._neighbors)

        # Initialize tracking
        gap = ((self.solution.cost - self.opt_cost) / self.opt_cost) * 100
        self.best_gap = gap
        self.history = [-1] * self.history_len
        self.t_start = time.perf_counter()

        return self._get_state()

    def step(self, action: int) -> tuple[DQNState, float, bool]:
        """
        Execute action and return (next_state, reward, done).

        Args:
            action: Action index (0 to N_ACTIONS-1).

        Returns:
            Tuple of (next_state, reward, done).
        """
        if self.solution is None:
            raise RuntimeError("Call reset() before step()")

        # Decode and apply action
        pert_type, ls_type = ACTION_DECODE[action]
        perturbed = self._apply_perturbation(self.solution, pert_type)
        new_solution = self._apply_local_search(perturbed, ls_type)

        # Update solution
        self.solution = new_solution

        # Compute reward (improvement in best gap)
        gap = ((new_solution.cost - self.opt_cost) / self.opt_cost) * 100
        old_best = self.best_gap
        new_best = min(self.best_gap, gap)
        reward = compute_delta_reward(old_best, new_best)
        self.best_gap = new_best

        # Update history
        self.history = self.history[1:] + [action]

        # Check if done (time budget exhausted)
        elapsed = time.perf_counter() - self.t_start
        done = elapsed >= self.time_budget

        return self._get_state(), reward, done

    def _get_state(self) -> DQNState:
        """Get current state."""
        assert self.solution is not None

        gap = ((self.solution.cost - self.opt_cost) / self.opt_cost) * 100
        elapsed = time.perf_counter() - self.t_start
        t_ratio = max(0.0, 1.0 - elapsed / self.time_budget)

        return DQNState(
            g=normalize_gap(gap),
            g_best=normalize_gap(self.best_gap),
            t_ratio=t_ratio,
            history=tuple(self.history),
        )

    def _apply_perturbation(self, solution: Solution, pert_type: str) -> Solution:
        """Apply perturbation to solution."""
        if pert_type in PERTURBATIONS:
            return PERTURBATIONS[pert_type](solution)
        if pert_type in CONSTRUCTIVES:
            tour, _ = CONSTRUCTIVES[pert_type](self.instance)
            return Solution(tour, self.dist_matrix, is_closed=True)
        raise ValueError(f"Unknown perturbation: {pert_type}")

    def _apply_local_search(self, solution: Solution, ls_type: str) -> Solution:
        """Apply local search to solution."""
        if ls_type == "two_opt":
            return two_opt(solution, neighbors=self._neighbors)
        elif ls_type == "lin_kernighan":
            return lin_kernighan(solution)
        raise ValueError(f"Unknown local search: {ls_type}")

    @property
    def state_dim(self) -> int:
        """State vector dimension."""
        return DQNState.dim(self.history_len)

    @property
    def n_actions(self) -> int:
        """Number of available actions."""
        return N_ACTIONS


__all__ = ["DQNEnv"]
