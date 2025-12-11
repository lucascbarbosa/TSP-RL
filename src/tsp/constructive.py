"""Constructive heuristics for TSP."""

from __future__ import annotations

import random
from typing import Callable

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
    """Cheapest insertion heuristic. Returns (closed_tour, cost)."""
    nodes = list(problem.get_nodes())
    n = len(nodes)

    if n <= 2:
        return random_tour(problem)

    if start_node is None:
        start_node = random.choice(nodes)

    unvisited = set(nodes)
    unvisited.remove(start_node)

    # Find nearest neighbor to start initial cycle
    nearest = min(unvisited, key=lambda node: problem.get_weight(start_node, node))
    unvisited.remove(nearest)

    # Initial closed tour: start -> nearest -> start
    tour = [start_node, nearest, start_node]
    tour_cost = problem.get_weight(start_node, nearest) + problem.get_weight(nearest, start_node)

    # Insert remaining nodes by minimum cost increase
    while unvisited:
        best_delta = float("inf")
        best_city = None
        best_pos = None

        for city in unvisited:
            for i in range(len(tour) - 1):
                a, b = tour[i], tour[i + 1]
                delta = problem.get_weight(a, city) + problem.get_weight(city, b) - problem.get_weight(a, b)

                if delta < best_delta:
                    best_delta = delta
                    best_city = city
                    best_pos = i

        tour.insert(best_pos + 1, best_city)
        tour_cost += best_delta
        unvisited.remove(best_city)

    return tour, tour_cost


# Registry: constructive heuristic name -> function
CONSTRUCTIVES: dict[str, Callable[[TSPInstance], tuple[list[int], float]]] = {
    "random": random_tour,
    "nearest": nearest_neighbor,
    "cheapest": cheapest_insertion,
}
