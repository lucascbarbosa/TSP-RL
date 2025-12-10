"""Local search algorithms for TSP."""

from __future__ import annotations

from typing import Callable, Dict, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from src.tsp.solution import Solution


def _two_opt_delta(
    tour: List[int],
    i: int,
    j: int,
    dist_matrix: NDArray[np.float64],
) -> float:
    """
    Calculate cost change (delta) from a 2-opt move in O(1).

    A 2-opt move reverses the segment tour[i:j], which replaces edges
    (i-1, i) and (j-1, j) with edges (i-1, j-1) and (i, j).

    Args:
        tour: Closed tour [c0, ..., cn-1, c0] with 1-based city indices.
        i: Start of segment to reverse (1 <= i < j).
        j: End of segment to reverse (i < j <= n-1).
        dist_matrix: Distance matrix (0-based indexing).

    Returns:
        Cost delta (negative means improvement).
    """
    # Cities involved (convert to 0-based for matrix access)
    a = tour[i - 1] - 1  # before segment
    b = tour[i] - 1  # start of segment
    c = tour[j - 1] - 1  # end of segment
    d = tour[j] - 1  # after segment

    # Edges removed: (a, b) and (c, d)
    # Edges added: (a, c) and (b, d)
    removed = dist_matrix[a, b] + dist_matrix[c, d]
    added = dist_matrix[a, c] + dist_matrix[b, d]

    return added - removed


def two_opt(solution: Solution) -> Solution:
    """
    Apply 2-opt local search to improve the solution.

    Uses best-improvement strategy with incremental delta calculation.
    Complexity: O(n²) per pass instead of O(n³).

    Args:
        solution: Input solution (closed tour).

    Returns:
        Improved solution.
    """
    tour = solution.tour[:]
    cost = solution.cost
    dist_matrix = solution.dist_matrix

    n = len(tour) - 1  # number of actual cities

    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_i, best_j = -1, -1

        # Find best improving move
        for i in range(1, n - 1):
            for j in range(i + 2, n):  # j > i+1 to skip adjacent
                delta = _two_opt_delta(tour, i, j, dist_matrix)
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_i, best_j = i, j

        # Apply best move if improvement found
        if best_delta < -1e-10:
            tour[best_i:best_j] = reversed(tour[best_i:best_j])
            cost += best_delta
            improved = True

    return Solution(tour, dist_matrix, is_closed=True, cost=cost)


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
            current_cost -= best_global_gain  # Update cost incrementally
            improved = True

    return Solution(current_tour, dist_matrix, is_closed=True, cost=current_cost)


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
    # Gain is negative of delta (delta < 0 means improvement)
    return -_two_opt_delta(tour, i, j, dist_matrix)


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
