"""Perturbation operators for ILS."""

from __future__ import annotations

import random
from typing import Callable, Dict

from src.tsp.solution import Solution


def two_swap(solution: Solution) -> Solution:
    """
    Swap two random internal vertices.

    Light perturbation that maintains most of the tour structure.

    Args:
        solution: Input solution (closed tour).

    Returns:
        Perturbed solution.
    """
    tour = solution.tour[:]
    n = len(tour) - 1  # number of actual cities

    if n <= 3:
        return solution.copy()

    # Swap two random internal positions (not first/last)
    i, j = random.sample(range(1, n), 2)
    tour[i], tour[j] = tour[j], tour[i]

    # Maintain closed tour
    tour[-1] = tour[0]

    return Solution(tour, solution.dist_matrix, is_closed=True)


def segment_reverse(solution: Solution) -> Solution:
    """
    Reverse a random segment of the tour.

    Stronger perturbation than two_swap.

    Args:
        solution: Input solution (closed tour).

    Returns:
        Perturbed solution.
    """
    tour = solution.tour[:]
    n = len(tour) - 1

    if n <= 4:
        return solution.copy()

    # Select random segment to reverse
    i, j = sorted(random.sample(range(1, n), 2))
    tour[i:j] = reversed(tour[i:j])
    tour[-1] = tour[0]

    return Solution(tour, solution.dist_matrix, is_closed=True)


# Registry: perturbation name -> function
PERTURBATIONS: Dict[str, Callable[[Solution], Solution]] = {
    "two_swap": two_swap,
    "segment_reverse": segment_reverse,
}
