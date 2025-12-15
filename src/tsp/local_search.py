"""Local search algorithms for TSP."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.tsp.solution import Solution


# =============================================================================
# Numba-compiled kernels (10-100x faster than pure Python)
# =============================================================================


@njit(cache=True)
def _two_opt_delta_nb(
    tour: np.ndarray,
    i: int,
    j: int,
    dist: np.ndarray,
) -> float:
    """
    Numba-compiled 2-opt delta calculation.

    ~50x faster than pure Python version.
    """
    a = tour[i - 1] - 1
    b = tour[i] - 1
    c = tour[j - 1] - 1
    d = tour[j] - 1
    return dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]


@njit(cache=True)
def _two_opt_full_core(
    tour: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Numba-compiled 2-opt full search core loop.

    Returns (improved_tour, total_delta).
    """
    n = len(tour) - 1
    total_delta = 0.0

    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_i, best_j = -1, -1

        for i in range(1, n):
            for j in range(i + 2, n + 1):
                delta = _two_opt_delta_nb(tour, i, j, dist)
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_i, best_j = i, j

        if best_delta < -1e-10:
            # Reverse segment in-place
            left, right = best_i, best_j - 1
            while left < right:
                tour[left], tour[right] = tour[right], tour[left]
                left += 1
                right -= 1
            total_delta += best_delta
            improved = True

    return tour, total_delta


@njit(cache=True)
def _two_opt_nn_core(
    tour: np.ndarray,
    dist: np.ndarray,
    neighbors: np.ndarray,
    pos: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Numba-compiled 2-opt with neighbor lists.

    Returns (improved_tour, total_delta, updated_pos).
    """
    n = len(tour) - 1
    total_delta = 0.0
    k = neighbors.shape[1]

    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_i, best_j = -1, -1

        for i in range(1, n):
            a = tour[i - 1] - 1  # 0-based city index

            for ki in range(k):
                c = neighbors[a, ki]
                j_minus_1 = pos[c]
                j = j_minus_1 + 1

                if j <= i + 1 or j > n:
                    continue

                delta = _two_opt_delta_nb(tour, i, j, dist)
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_i, best_j = i, j

        if best_delta < -1e-10:
            # Reverse segment
            left, right = best_i, best_j - 1
            while left < right:
                tour[left], tour[right] = tour[right], tour[left]
                left += 1
                right -= 1
            total_delta += best_delta
            improved = True

            # Update position array
            for idx in range(best_i, best_j):
                pos[tour[idx] - 1] = idx

    return tour, total_delta, pos


@njit(cache=True)
def _two_opt_dlb_core(
    tour: np.ndarray,
    dist: np.ndarray,
    neighbors: np.ndarray,
    pos: np.ndarray,
    dlb: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Numba-compiled 2-opt with neighbor lists + don't look bits.

    Returns (improved_tour, total_delta, updated_pos, updated_dlb).
    """
    n = len(tour) - 1
    total_delta = 0.0
    k = neighbors.shape[1]

    improved = True
    while improved:
        improved = False

        for i in range(1, n):
            city_at_i = tour[i] - 1

            if dlb[city_at_i]:
                continue

            a = tour[i - 1] - 1

            found_improvement = False
            best_delta = 0.0
            best_j = -1

            for ki in range(k):
                c = neighbors[a, ki]
                j_minus_1 = pos[c]
                j = j_minus_1 + 1

                if j <= i + 1 or j > n:
                    continue

                delta = _two_opt_delta_nb(tour, i, j, dist)
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_j = j
                    found_improvement = True

            if found_improvement:
                # Reverse segment
                left, right = i, best_j - 1
                while left < right:
                    tour[left], tour[right] = tour[right], tour[left]
                    left += 1
                    right -= 1
                total_delta += best_delta
                improved = True

                # Update position array
                for idx in range(i, best_j):
                    pos[tour[idx] - 1] = idx

                # Clear DLB for affected cities
                for idx in range(max(0, i - 1), min(n, best_j + 1)):
                    dlb[tour[idx] - 1] = False
            else:
                dlb[city_at_i] = True

    return tour, total_delta, pos, dlb


# =============================================================================
# Neighbor Lists (for large instances)
# =============================================================================


def _build_neighbor_lists(
    dist_matrix: NDArray[np.float64],
    k: int,
) -> NDArray[np.int32]:
    """
    Build k-nearest neighbor lists for each city.

    Uses argpartition for O(n + k log k) per city instead of O(n log n).

    Args:
        dist_matrix: Distance matrix (n x n), 0-based indexing.
        k: Number of nearest neighbors to keep.

    Returns:
        Array of shape (n, k) where neighbors[i] contains indices of
        k nearest neighbors of city i, sorted by distance.
    """
    n = dist_matrix.shape[0]
    k = min(k, n - 1)  # can't have more neighbors than n-1

    neighbors = np.zeros((n, k), dtype=np.int32)

    for i in range(n):
        dists = dist_matrix[i]
        # argpartition: O(n) to get k+1 smallest (including self)
        # We need k+1 because self (distance 0) will be among the smallest
        kp1 = min(k + 1, n)
        candidate_indices = np.argpartition(dists, kp1 - 1)[:kp1]
        # Remove self from candidates
        candidate_indices = candidate_indices[candidate_indices != i]
        # Sort the k candidates by distance: O(k log k)
        sorted_candidates = candidate_indices[np.argsort(dists[candidate_indices])]
        neighbors[i, : len(sorted_candidates)] = sorted_candidates[:k]

    return neighbors


def two_opt_nn(
    solution: Solution,
    k: int | float = 0.5,
    neighbors: NDArray[np.int32] | None = None,
) -> Solution:
    """
    2-opt with neighbor lists for O(n·k) complexity per pass.

    Uses Numba-compiled kernel for ~50x speedup over pure Python.
    Only considers moves where the new edge (a,c) connects city a to one of
    its k nearest neighbors.

    Args:
        solution: Input solution (closed tour).
        k: Number of nearest neighbors. If int, used as-is. If float in (0,1),
           interpreted as proportion of n (default 0.5 = 50% of cities).
        neighbors: Pre-computed neighbor lists (optional). If provided, k is
           ignored and these lists are used directly. Shape: (n, k_actual).

    Returns:
        Improved solution.
    """
    dist_matrix = solution.dist_matrix
    cost = solution.cost
    n = len(solution.tour) - 1

    # Use pre-computed neighbors or build new ones
    if neighbors is None:
        k_resolved = _resolve_k(k, n)
        neighbors = _build_neighbor_lists(dist_matrix, k_resolved)

    # Convert to numpy arrays for Numba kernel
    tour_arr = np.array(solution.tour, dtype=np.int32)

    # Position array: pos[city] = position in tour
    pos = np.zeros(n, dtype=np.int32)
    for idx in range(n):
        pos[tour_arr[idx] - 1] = idx

    # Run compiled kernel
    tour_arr, total_delta, _ = _two_opt_nn_core(tour_arr, dist_matrix, neighbors, pos)

    return Solution(tour_arr.tolist(), dist_matrix, is_closed=True, cost=cost + total_delta)


def two_opt_dlb(
    solution: Solution,
    k: int | float = 0.5,
    neighbors: NDArray[np.int32] | None = None,
) -> Solution:
    """
    2-opt with Neighbor Lists + Don't Look Bits for maximum speed.

    Uses Numba-compiled kernel for ~50x speedup over pure Python.
    DLB skips cities that haven't led to improvements recently.

    Args:
        solution: Input solution (closed tour).
        k: Number of nearest neighbors. If int, used as-is. If float in (0,1),
           interpreted as proportion of n (default 0.5 = 50% of cities).
        neighbors: Pre-computed neighbor lists (optional). If provided, k is
           ignored and these lists are used directly. Shape: (n, k_actual).

    Returns:
        Improved solution.
    """
    dist_matrix = solution.dist_matrix
    cost = solution.cost
    n = len(solution.tour) - 1

    # Use pre-computed neighbors or build new ones
    if neighbors is None:
        k_resolved = _resolve_k(k, n)
        neighbors = _build_neighbor_lists(dist_matrix, k_resolved)

    # Convert to numpy arrays for Numba kernel
    tour_arr = np.array(solution.tour, dtype=np.int32)

    # Position array
    pos = np.zeros(n, dtype=np.int32)
    for idx in range(n):
        pos[tour_arr[idx] - 1] = idx

    # Don't Look Bits
    dlb = np.zeros(n, dtype=np.bool_)

    # Run compiled kernel
    tour_arr, total_delta, _, _ = _two_opt_dlb_core(tour_arr, dist_matrix, neighbors, pos, dlb)

    return Solution(tour_arr.tolist(), dist_matrix, is_closed=True, cost=cost + total_delta)


# =============================================================================
# Adaptive 2-opt (auto-selects best variant based on instance size)
# =============================================================================

# Thresholds for adaptive selection (based on benchmarks)
_THRESHOLD_NN = 40  # Use neighbor lists above this size
_THRESHOLD_DLB = 80  # Use DLB at or above this size


def _resolve_k(k: int | float, n: int) -> int:
    """Resolve k parameter: int as-is, float (0,1) as proportion of n."""
    if isinstance(k, float) and 0 < k < 1:
        return max(1, int(k * n))
    return int(k)


def two_opt_adaptive(
    solution: Solution,
    k: int | float = 0.5,
    neighbors: NDArray[np.int32] | None = None,
) -> Solution:
    """
    Adaptive 2-opt that selects the best variant based on instance size.

    Selection rules (based on empirical benchmarks):
    - n < 40:  two_opt_full (full O(n²), overhead of neighbor lists not worth it)
    - 40 ≤ n < 80: two_opt_nn (neighbor lists, good quality/speed balance)
    - n ≥ 80: two_opt_dlb (DLB + neighbor lists, max speed, ~1-4% quality loss)

    Args:
        solution: Input solution (closed tour).
        k: Number of nearest neighbors. If int, used as-is. If float in (0,1),
           interpreted as proportion of n (default 0.5 = 50% of cities).
        neighbors: Pre-computed neighbor lists (optional). If provided, k is
           ignored and these lists are used directly. Shape: (n, k_actual).

    Returns:
        Improved solution.
    """
    n = len(solution.tour) - 1  # exclude closing city

    if n < _THRESHOLD_NN:
        return two_opt_full(solution)
    elif n < _THRESHOLD_DLB:
        return two_opt_nn(solution, k=k, neighbors=neighbors)
    else:
        return two_opt_dlb(solution, k=k, neighbors=neighbors)


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

    Uses Numba-compiled kernel for ~50x speedup over pure Python.
    For large instances (n > 50), consider two_opt_nn or two_opt_dlb instead.

    Args:
        solution: Input solution (closed tour).

    Returns:
        Improved solution.
    """
    # Convert to numpy array for Numba kernel
    tour_arr = np.array(solution.tour, dtype=np.int32)
    dist_matrix = solution.dist_matrix
    cost = solution.cost

    # Run compiled kernel
    tour_arr, total_delta = _two_opt_full_core(tour_arr, dist_matrix)

    return Solution(tour_arr.tolist(), dist_matrix, is_closed=True, cost=cost + total_delta)


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

        for j in range(1, n + 1):
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
