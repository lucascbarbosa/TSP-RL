"""Constructive heuristics for TSP."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from src.tsp.instance import TSPInstance


def random_tour(problem: TSPInstance) -> tuple[list[int], float]:
    """Generate a random tour. Returns (closed_tour, cost)."""
    nodes = list(problem.get_nodes())
    n = len(nodes)

    random.shuffle(nodes)

    # Compute cost
    tour_cost = 0.0
    for i in range(n):
        curr = nodes[i]
        nxt = nodes[(i + 1) % n]
        tour_cost += problem.get_weight(curr, nxt)

    # Close the tour
    closed_tour = nodes + [nodes[0]]

    return closed_tour, tour_cost


def nearest_neighbor(
    problem: TSPInstance,
    start_node: int | None = None,
) -> tuple[list[int], float]:
    """Nearest neighbor heuristic. Returns (closed_tour, cost)."""
    n = problem.dimension
    if start_node is None:
        start_node = random.choice(list(problem.get_nodes()))

    unvisited = set(problem.get_nodes())
    unvisited.remove(start_node)

    tour = [start_node]
    current_node = start_node
    tour_cost = 0.0

    while unvisited:
        next_node = min(unvisited, key=lambda node: problem.get_weight(current_node, node))
        tour_cost += problem.get_weight(current_node, next_node)
        tour.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node

    # Return to start
    tour_cost += problem.get_weight(current_node, start_node)
    tour.append(start_node)

    return tour, tour_cost


def cheapest_insertion(
    problem: TSPInstance,
    start_node: int | None = None,
) -> tuple[list[int], float]:
    """
    Cheapest insertion heuristic with vectorized delta computation.

    Uses numpy broadcasting to compute all insertion deltas simultaneously,
    reducing complexity from O(n³) with high constants to O(n³) with low constants
    (actual speedup ~10-50x for typical instances).

    Returns (closed_tour, cost).
    """
    n = problem.dimension
    dist = problem.dist_matrix  # 0-based indexing

    if n <= 2:
        return random_tour(problem)

    # Work with 0-based indices internally
    if start_node is None:
        start_idx = random.randrange(n)
    else:
        start_idx = start_node - 1  # convert 1-based to 0-based

    # Find nearest neighbor to start (0-based)
    dists_from_start = dist[start_idx].copy()
    dists_from_start[start_idx] = np.inf  # exclude self
    nearest_idx = int(np.argmin(dists_from_start))

    # Initial tour: [start, nearest] (open, will close at end)
    # tour_indices stores 0-based indices
    tour_indices = [start_idx, nearest_idx]
    tour_cost = dist[start_idx, nearest_idx] + dist[nearest_idx, start_idx]

    # Track visited cities
    visited = np.zeros(n, dtype=bool)
    visited[start_idx] = True
    visited[nearest_idx] = True
    n_visited = 2

    # Insert remaining nodes
    while n_visited < n:
        # Get unvisited indices
        unvisited_mask = ~visited
        unvisited_indices = np.where(unvisited_mask)[0]

        # Current tour edges: (tour[i], tour[i+1]) for i in range(len-1), plus closing edge
        # For closed tour: edges are (0,1), (1,2), ..., (k-1, 0) where k = len(tour_indices)
        k = len(tour_indices)
        tour_arr = np.array(tour_indices, dtype=np.int32)

        # Edge endpoints: a[i] -> b[i]
        a_indices = tour_arr  # [t0, t1, ..., tk-1]
        b_indices = np.roll(tour_arr, -1)  # [t1, t2, ..., t0]

        # Compute deltas for all (unvisited_city, edge) pairs using broadcasting
        # delta[c, e] = dist[a[e], c] + dist[c, b[e]] - dist[a[e], b[e]]
        # Shapes: a_indices (k,), b_indices (k,), unvisited_indices (m,)

        # dist[a_indices, unvisited_indices] -> need (k, m) matrix
        # dist[a_indices][:, unvisited_indices] -> (k, m)
        dist_a_to_c = dist[a_indices][:, unvisited_indices]  # (k, m)
        dist_c_to_b = dist[unvisited_indices][:, b_indices].T  # (m, k).T = (k, m)
        dist_a_to_b = dist[a_indices, b_indices]  # (k,)

        # delta[e, c] = dist_a_to_c[e, c] + dist_c_to_b[e, c] - dist_a_to_b[e]
        deltas = dist_a_to_c + dist_c_to_b - dist_a_to_b[:, np.newaxis]  # (k, m)

        # Find minimum delta
        min_flat_idx = np.argmin(deltas)
        best_edge_idx, best_unvisited_idx = np.unravel_index(min_flat_idx, deltas.shape)
        best_delta = deltas[best_edge_idx, best_unvisited_idx]
        best_city_idx = unvisited_indices[best_unvisited_idx]

        # Insert: after position best_edge_idx in tour_indices
        tour_indices.insert(best_edge_idx + 1, int(best_city_idx))
        tour_cost += best_delta
        visited[best_city_idx] = True
        n_visited += 1

    # Convert to 1-based closed tour
    closed_tour = [idx + 1 for idx in tour_indices] + [tour_indices[0] + 1]

    return closed_tour, float(tour_cost)


# Registry: constructive heuristic name -> function
CONSTRUCTIVES: dict[str, Callable[[TSPInstance], tuple[list[int], float]]] = {
    "random": random_tour,
    "nearest": nearest_neighbor,
    "cheapest": cheapest_insertion,
}
