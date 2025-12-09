"""Local search algorithms for TSP."""

from __future__ import annotations

from typing import Callable, Dict, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from src.tsp.solution import Solution


def two_opt(solution: Solution) -> Solution:
    """
    Apply 2-opt local search to improve the solution.

    Uses best-improvement strategy with multiple passes until no
    improvement is found.

    Args:
        solution: Input solution (closed tour).

    Returns:
        Improved solution.
    """
    best_tour = solution.tour[:]
    best_cost = solution.cost
    dist_matrix = solution.dist_matrix

    n = len(best_tour) - 1  # number of actual cities

    improved = True
    while improved:
        improved = False
        # Fix first node to avoid equivalent rotations
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue  # skip adjacent edges

                new_tour = best_tour[:]
                # 2-opt: reverse segment [i, j)
                new_tour[i:j] = reversed(best_tour[i:j])

                new_cost = Solution.compute_cost_static(new_tour, dist_matrix, is_closed=True)

                if new_cost < best_cost:
                    best_tour = new_tour
                    best_cost = new_cost
                    improved = True

    return Solution(best_tour, dist_matrix, is_closed=True)


def lin_kernighan(solution: Solution, max_depth: int = 2) -> Solution:
    """
    Apply Lin-Kernighan heuristic (simplified variable-depth 2-opt chains).

    At each iteration, explores chains of 2-opt moves starting from
    various positions. Each step in the chain must maintain positive
    cumulative gain.

    Args:
        solution: Input solution (closed tour).
        max_depth: Maximum depth of 2-opt chains.

    Returns:
        Improved solution.
    """
    dist_matrix = solution.dist_matrix
    current_tour = solution.tour[:]
    current_cost = solution.cost
    n = len(current_tour) - 1

    improved = True
    while improved:
        improved = False
        best_global_gain = 0.0
        best_global_tour = current_tour

        # Try starting the chain at each internal position
        for start_idx in range(1, n):
            used_positions: Set[int] = {start_idx}
            best_tour_local, best_gain_local = _lk_variable_depth(
                current_tour,
                start_idx,
                used_positions,
                depth=0,
                max_depth=max_depth,
                dist_matrix=dist_matrix,
                current_gain=0.0,
            )

            if best_gain_local > best_global_gain + 1e-12:
                best_global_gain = best_gain_local
                best_global_tour = best_tour_local

        # Apply best chain found
        if best_global_gain > 1e-12:
            current_tour = best_global_tour
            current_cost = Solution.compute_cost_static(current_tour, dist_matrix, is_closed=True)
            improved = True

    return Solution(current_tour, dist_matrix, is_closed=True)


def _lk_variable_depth(
    tour: List[int],
    last_pos: int,
    used_positions: Set[int],
    depth: int,
    max_depth: int,
    dist_matrix: NDArray[np.float64],
    current_gain: float,
) -> Tuple[List[int], float]:
    """
    Recursively explore variable-depth 2-opt chains.

    Args:
        tour: Current closed tour.
        last_pos: Last position used in the chain (1..n-1).
        used_positions: Positions already used (to avoid repetition).
        depth: Current recursion depth.
        max_depth: Maximum allowed depth.
        dist_matrix: Distance matrix.
        current_gain: Cumulative gain so far.

    Returns:
        Tuple of (best_tour, best_gain) from this subtree.
    """
    n = len(tour) - 1
    best_gain = current_gain
    best_tour = tour

    if depth >= max_depth:
        return best_tour, best_gain

    for j in range(1, n):
        if j == last_pos or abs(j - last_pos) == 1 or j in used_positions:
            continue

        move_gain = _two_opt_gain(tour, last_pos, j, dist_matrix)
        new_total_gain = current_gain + move_gain

        # Only continue if cumulative gain is positive
        if new_total_gain <= 0:
            continue

        new_tour = _apply_two_opt(tour, last_pos, j)

        if new_total_gain > best_gain + 1e-12:
            best_gain = new_total_gain
            best_tour = new_tour

        # Try deeper chains
        new_used = set(used_positions)
        new_used.add(j)

        deeper_tour, deeper_gain = _lk_variable_depth(
            new_tour,
            j,
            new_used,
            depth + 1,
            max_depth,
            dist_matrix,
            new_total_gain,
        )

        if deeper_gain > best_gain + 1e-12:
            best_gain = deeper_gain
            best_tour = deeper_tour

    return best_tour, best_gain


def _two_opt_gain(
    tour: List[int],
    i: int,
    j: int,
    dist_matrix: NDArray[np.float64],
) -> float:
    """
    Calculate gain (cost reduction) from a 2-opt move.

    The move reverses tour[i:j].

    Args:
        tour: Closed tour [c0, ..., cn-1, c0].
        i, j: Positions in 1..n-1 (i != j).
        dist_matrix: Distance matrix.

    Returns:
        Gain (positive = improvement).
    """
    if i > j:
        i, j = j, i

    a, b = tour[i - 1], tour[i]
    c, d = tour[j - 1], tour[j]

    a_, b_, c_, d_ = a - 1, b - 1, c - 1, d - 1

    removed = dist_matrix[a_, b_] + dist_matrix[c_, d_]
    added = dist_matrix[a_, c_] + dist_matrix[b_, d_]

    return removed - added


def _apply_two_opt(tour: List[int], i: int, j: int) -> List[int]:
    """
    Apply 2-opt move to tour.

    Args:
        tour: Input tour.
        i, j: Positions to swap.

    Returns:
        New tour with segment reversed.
    """
    if i > j:
        i, j = j, i

    new_tour = tour[:]
    new_tour[i:j] = reversed(tour[i:j])

    # Ensure tour remains closed
    if new_tour[0] != new_tour[-1]:
        new_tour[-1] = new_tour[0]

    return new_tour


# Registry: local search name -> function
LOCAL_SEARCHES: Dict[str, Callable[[Solution], Solution]] = {
    "two_opt": two_opt,
    "lin_kernighan": lin_kernighan,
}
