"""Integration tests for TSP-RL components."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.tsp.solution import Solution
from src.tsp.instance import TSPInstance
from src.tsp.local_search import two_opt, lin_kernighan
from src.tsp.perturbation import two_swap, segment_reverse
from src.tsp.constructive import random_tour, nearest_neighbor, cheapest_insertion
from src.ils.q_ils import QILS, State


@pytest.fixture
def temp_instance_file():
    """Create a temporary JSON file with a small TSP instance."""
    data = [
        {
            "coords": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "tour": [0, 1, 2, 3],  # 0-based optimal tour
        }
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        return Path(f.name)


class TestTSPInstanceIntegration:
    """Tests for TSPInstance integration."""

    def test_load_instance(self, temp_instance_file):
        """Should load instance and compute distance matrix."""
        instance = TSPInstance(temp_instance_file, instance_id=0)
        assert instance.dimension == 4
        assert instance.dist_matrix.shape == (4, 4)
        # Check a known distance (0,0) to (1,0) = 1.0
        assert instance.get_weight(1, 2) == pytest.approx(1.0)

    def test_opt_tour_loaded(self, temp_instance_file):
        """Should load optimal tour (converted to 1-based)."""
        instance = TSPInstance(temp_instance_file, instance_id=0)
        assert instance.opt_tour == [1, 2, 3, 4]


class TestQILSIntegration:
    """Tests for QILS integration."""

    def test_qils_reuses_dist_matrix(self, temp_instance_file):
        """QILS should reuse instance's precomputed distance matrix."""
        instance = TSPInstance(temp_instance_file, instance_id=0)
        qils = QILS(instance)
        # Should be the same object (not a copy)
        assert qils.dist_matrix is instance.dist_matrix

    def test_qils_get_state(self, temp_instance_file):
        """QILS state calculation should work correctly."""
        instance = TSPInstance(temp_instance_file, instance_id=0)
        qils = QILS(instance)

        # Optimal cost for unit square = 4.0
        opt_cost = 4.0

        # Exact optimal -> EXCELLENT
        state, reward = qils.get_state(4.0, opt_cost)
        assert state == State.EXCELLENT
        assert reward == 75

        # 1% gap -> EXCELLENT
        state, _ = qils.get_state(4.04, opt_cost)
        assert state == State.EXCELLENT

        # 3% gap -> GOOD
        state, _ = qils.get_state(4.12, opt_cost)
        assert state == State.GOOD

        # 7% gap -> REGULAR
        state, _ = qils.get_state(4.28, opt_cost)
        assert state == State.REGULAR

        # 15% gap -> POOR
        state, _ = qils.get_state(4.60, opt_cost)
        assert state == State.POOR


class TestConstructiveToLocalSearch:
    """Tests for constructive -> local search pipeline."""

    def test_nearest_neighbor_then_two_opt(self, temp_instance_file):
        """Nearest neighbor + 2-opt should find good solution."""
        instance = TSPInstance(temp_instance_file, instance_id=0)
        tour, _ = nearest_neighbor(instance)
        sol = Solution(tour, instance.dist_matrix)
        improved = two_opt(sol)
        # Should find optimal (4.0) for this simple instance
        assert improved.cost == pytest.approx(4.0)

    def test_random_then_two_opt(self, temp_instance_file):
        """Random tour + 2-opt should improve or maintain quality."""
        np.random.seed(42)
        instance = TSPInstance(temp_instance_file, instance_id=0)
        tour, _ = random_tour(instance)
        sol = Solution(tour, instance.dist_matrix)
        improved = two_opt(sol)
        # 2-opt should not make it worse
        assert improved.cost <= sol.cost + 1e-10
        # For this small instance, should reach optimal or near-optimal
        assert improved.cost <= 5.0  # Reasonable upper bound


class TestPerturbationIntegration:
    """Tests for perturbation operators."""

    def test_two_swap_then_two_opt(self, temp_instance_file):
        """Two-swap + 2-opt should produce valid solution."""
        np.random.seed(42)
        instance = TSPInstance(temp_instance_file, instance_id=0)
        tour, _ = nearest_neighbor(instance)
        sol = Solution(tour, instance.dist_matrix)
        optimal = two_opt(sol)

        # Perturb
        perturbed = two_swap(optimal)
        # Recover
        recovered = two_opt(perturbed)
        # Should be a valid tour with reasonable cost
        assert recovered.cost <= 5.0  # Upper bound for 4-city square

    def test_segment_reverse_then_two_opt(self, temp_instance_file):
        """Segment reverse + 2-opt should recover optimal."""
        instance = TSPInstance(temp_instance_file, instance_id=0)
        tour, _ = nearest_neighbor(instance)
        sol = Solution(tour, instance.dist_matrix)
        optimal = two_opt(sol)

        # Perturb
        perturbed = segment_reverse(optimal)
        # Recover
        recovered = two_opt(perturbed)
        assert recovered.cost == pytest.approx(4.0)
