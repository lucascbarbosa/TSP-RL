"""Shared fixtures for TSP-RL tests."""

import numpy as np
import pytest


@pytest.fixture
def simple_dist_matrix():
    """
    Simple 4-city distance matrix (symmetric).

    Cities arranged in a square:
        1 --- 2
        |     |
        4 --- 3

    Distances: adjacent = 1.0, diagonal = sqrt(2) ~ 1.414
    Optimal tour: 1-2-3-4-1 with cost 4.0
    """
    sqrt2 = np.sqrt(2)
    return np.array(
        [
            [0.0, 1.0, sqrt2, 1.0],
            [1.0, 0.0, 1.0, sqrt2],
            [sqrt2, 1.0, 0.0, 1.0],
            [1.0, sqrt2, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def optimal_tour():
    """Optimal tour for simple_dist_matrix (1-based, closed)."""
    return [1, 2, 3, 4, 1]


@pytest.fixture
def suboptimal_tour():
    """Suboptimal tour for simple_dist_matrix (crosses diagonal)."""
    return [1, 3, 2, 4, 1]  # cost = sqrt(2) + 1 + sqrt(2) + 1 ~ 4.83


@pytest.fixture
def larger_dist_matrix():
    """
    10-city random distance matrix for performance tests.
    """
    np.random.seed(42)
    n = 10
    coords = np.random.rand(n, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    return dist
