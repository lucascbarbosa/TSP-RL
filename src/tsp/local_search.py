"""Local search algorithms for TSP."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from src.tsp.solution import Solution


# =============================================================================
# Neighbor Lists (for large instances)
# =============================================================================


def _build_neighbor_lists(
    dist_matrix: NDArray[np.float64],
    k: int,
) -> NDArray[np.int32]:
    """
    Build k-nearest neighbor lists for each city.

    Args:
        dist_matrix: Distance matrix (n x n), 0-based indexing.
        k: Number of nearest neighbors to keep.

    Returns:
        Array of shape (n, k) where neighbors[i] contains indices of
        k nearest neighbors of city i, sorted by distance.
    """
    n = dist_matrix.shape[0]
    k = min(k, n - 1)  # can't have more neighbors than n-1

    # For each city, get indices sorted by distance (excluding self)
    # argsort gives indices; we skip index 0 (self, distance=0) and take next k
    neighbors = np.zeros((n, k), dtype=np.int32)

    for i in range(n):
        # Sort by distance, exclude self
        sorted_indices = np.argsort(dist_matrix[i])
        # Skip self (first element after sort, which has distance 0)
        neighbor_indices = sorted_indices[sorted_indices != i][:k]
        neighbors[i, : len(neighbor_indices)] = neighbor_indices

    return neighbors


def two_opt_nn(solution: Solution, k: int = 20) -> Solution:
    """
    2-opt with neighbor lists for O(n·k) complexity per pass.

    Only considers moves where the new edge (a,c) connects city a to one of
    its k nearest neighbors. This dramatically speeds up large instances
    while typically finding solutions within 1% of full 2-opt.

    Args:
        solution: Input solution (closed tour).
        k: Number of nearest neighbors to consider (default 20).

    Returns:
        Improved solution.
    """
    tour = solution.tour[:]
    cost = solution.cost
    dist_matrix = solution.dist_matrix
    n = len(tour) - 1

    # Build neighbor lists (0-based city indices)
    neighbors = _build_neighbor_lists(dist_matrix, k)

    # Position array: pos[city] = position in tour (1-based city to 0-based position)
    # tour uses 1-based cities, so pos[city-1] = position
    pos = np.zeros(n, dtype=np.int32)
    for idx in range(n):
        pos[tour[idx] - 1] = idx

    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_i, best_j = -1, -1

        # For each position i (where segment starts)
        for i in range(1, n - 1):
            a = tour[i - 1] - 1  # city before segment (0-based)

            # Check only k nearest neighbors of a as potential new connection
            for c in neighbors[a]:
                # c is a potential new neighbor of a after the 2-opt move
                # In 2-opt reversing [i:j], new edge is (a, c) where c = tour[j-1]
                # So we need j-1 position where tour[j-1]-1 == c
                j_minus_1 = pos[c]  # position of city c+1 (1-based)
                j = j_minus_1 + 1

                # Validate: j must be > i+1 and < n (valid segment)
                if j <= i + 1 or j >= n:
                    continue

                delta = _two_opt_delta(tour, i, j, dist_matrix)
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_i, best_j = i, j

        # Apply best move if found
        if best_delta < -1e-10:
            # Update tour
            tour[best_i:best_j] = reversed(tour[best_i:best_j])
            cost += best_delta
            improved = True

            # Update position array for affected cities
            for idx in range(best_i, best_j):
                pos[tour[idx] - 1] = idx

    return Solution(tour, dist_matrix, is_closed=True, cost=cost)


def two_opt_dlb(solution: Solution, k: int = 20) -> Solution:
    """
    2-opt with Neighbor Lists + Don't Look Bits for maximum speed.

    DLB skips cities that haven't led to improvements recently, reducing
    redundant checks. Combined with neighbor lists, achieves near-optimal
    results with sub-quadratic runtime.

    Args:
        solution: Input solution (closed tour).
        k: Number of nearest neighbors to consider (default 20).

    Returns:
        Improved solution.
    """
    tour = solution.tour[:]
    cost = solution.cost
    dist_matrix = solution.dist_matrix
    n = len(tour) - 1

    # Build neighbor lists
    neighbors = _build_neighbor_lists(dist_matrix, k)

    # Position array
    pos = np.zeros(n, dtype=np.int32)
    for idx in range(n):
        pos[tour[idx] - 1] = idx

    # Don't Look Bits: dlb[city] = True means skip this city
    dlb = np.zeros(n, dtype=bool)

    improved = True
    while improved:
        improved = False

        for i in range(1, n - 1):
            city_at_i = tour[i] - 1  # 0-based city index

            # Skip if this city is marked as "don't look"
            if dlb[city_at_i]:
                continue

            a = tour[i - 1] - 1  # city before position i

            found_improvement = False
            best_delta = 0.0
            best_j = -1

            # Check neighbors of city a
            for c in neighbors[a]:
                j_minus_1 = pos[c]
                j = j_minus_1 + 1

                if j <= i + 1 or j >= n:
                    continue

                delta = _two_opt_delta(tour, i, j, dist_matrix)
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_j = j
                    found_improvement = True

            if found_improvement:
                # Apply move
                tour[i:best_j] = reversed(tour[i:best_j])
                cost += best_delta
                improved = True

                # Update position array
                for idx in range(i, best_j):
                    pos[tour[idx] - 1] = idx

                # Clear DLB for affected cities (they may now have good moves)
                for idx in range(max(0, i - 1), min(n, best_j + 1)):
                    dlb[tour[idx] - 1] = False
            else:
                # No improvement found for this city, mark as "don't look"
                dlb[city_at_i] = True

    return Solution(tour, dist_matrix, is_closed=True, cost=cost)


# =============================================================================
# Adaptive 2-opt (auto-selects best variant based on instance size)
# =============================================================================

# Thresholds for adaptive selection (based on benchmarks)
_THRESHOLD_NN = 50  # Use neighbor lists above this size
_THRESHOLD_DLB = 80  # Use DLB above this size


def two_opt_adaptive(solution: Solution, k: int = 20) -> Solution:
    """
    Adaptive 2-opt that selects the best variant based on instance size.

    Selection rules (based on empirical benchmarks):
    - n < 50:  two_opt (full O(n²), overhead of neighbor lists not worth it)
    - 50 ≤ n ≤ 80: two_opt_nn (neighbor lists, good quality/speed balance)
    - n > 80: two_opt_dlb (DLB + neighbor lists, max speed, ~1-4% quality loss)

    Args:
        solution: Input solution (closed tour).
        k: Number of nearest neighbors for nn/dlb variants (default 20).

    Returns:
        Improved solution.
    """
    n = len(solution.tour) - 1  # exclude closing city

    if n < _THRESHOLD_NN:
        return two_opt_full(solution)
    elif n <= _THRESHOLD_DLB:
        return two_opt_nn(solution, k=k)
    else:
        return two_opt_dlb(solution, k=k)


def _two_opt_delta(
    tour: list[int],
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


def two_opt_full(solution: Solution) -> Solution:
    """
    Full 2-opt local search with O(n²) complexity per pass.

    Uses best-improvement strategy with incremental delta calculation.
    For large instances (n > 50), consider two_opt_nn or two_opt_dlb instead.

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
        best_global_tour = None

        # Try starting the chain at each internal position
        for start_idx in range(1, n):
            best_tour_local, best_gain_local = _lk_chain(current_tour, start_idx, max_depth, dist_matrix)

            if best_gain_local > best_global_gain + 1e-12:
                best_global_gain = best_gain_local
                best_global_tour = best_tour_local

        # Apply best chain found
        if best_global_gain > 1e-12:
            current_tour = best_global_tour
            current_cost -= best_global_gain
            improved = True

    return Solution(current_tour, dist_matrix, is_closed=True, cost=current_cost)


def _lk_chain(
    tour: list[int],
    start_pos: int,
    max_depth: int,
    dist: NDArray[np.float64],
) -> tuple[list[int], float]:
    """Explore LK chain from start_pos. Returns (best_tour, best_gain)."""
    n = len(tour) - 1
    best_tour = tour
    best_gain = 0.0

    # Stack: (current_tour, last_pos, used_set, depth, cumulative_gain)
    stack = [(tour[:], start_pos, {start_pos}, 0, 0.0)]

    while stack:
        cur_tour, last_pos, used, depth, gain = stack.pop()

        if depth >= max_depth:
            continue

        for j in range(1, n):
            if j == last_pos or abs(j - last_pos) == 1 or j in used:
                continue

            # Calculate gain
            i, jj = (last_pos, j) if last_pos < j else (j, last_pos)
            move_gain = -_two_opt_delta(cur_tour, i, jj, dist)
            new_gain = gain + move_gain

            if new_gain <= 1e-12:
                continue

            # Apply move
            new_tour = cur_tour[:]
            new_tour[i:jj] = reversed(cur_tour[i:jj])

            # Update best if improved
            if new_gain > best_gain + 1e-12:
                best_gain = new_gain
                best_tour = new_tour

            # Continue exploring (only if depth allows)
            if depth + 1 < max_depth:
                new_used = used | {j}
                stack.append((new_tour, j, new_used, depth + 1, new_gain))

    return best_tour, best_gain


def _two_opt_gain(
    tour: list[int],
    i: int,
    j: int,
    dist_matrix: NDArray[np.float64],
) -> float:
    """Return gain (positive = improvement) from reversing tour[i:j]."""
    if i > j:
        i, j = j, i
    # Gain is negative of delta (delta < 0 means improvement)
    return -_two_opt_delta(tour, i, j, dist_matrix)


def _apply_two_opt(tour: list[int], i: int, j: int) -> list[int]:
    """Return new tour with segment [i:j] reversed."""
    if i > j:
        i, j = j, i

    new_tour = tour[:]
    new_tour[i:j] = reversed(tour[i:j])

    # Ensure tour remains closed
    if new_tour[0] != new_tour[-1]:
        new_tour[-1] = new_tour[0]

    return new_tour


# Alias: two_opt points to adaptive by default
two_opt = two_opt_adaptive

# Registry: local search name -> function
# "two_opt" uses adaptive selection by default (auto-selects best variant by instance size)
# Individual variants (two_opt_full, two_opt_nn, two_opt_dlb) available via direct import
LOCAL_SEARCHES: dict[str, Callable[[Solution], Solution]] = {
    "two_opt": two_opt,
    "lin_kernighan": lin_kernighan,
}
