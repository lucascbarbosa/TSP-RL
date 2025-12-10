"""Tests for Solution class."""

import numpy as np
import pytest

from src.tsp.solution import Solution


class TestSolutionCost:
    """Tests for Solution cost computation."""

    def test_cost_computed_lazily(self, simple_dist_matrix, optimal_tour):
        """Cost should not be computed until accessed."""
        sol = Solution(optimal_tour, simple_dist_matrix)
        # Access internal _cost directly to check it's None before first access
        assert sol._cost is None
        # Now access cost property
        cost = sol.cost
        assert cost == pytest.approx(4.0)
        assert sol._cost == pytest.approx(4.0)

    def test_cost_precomputed(self, simple_dist_matrix, optimal_tour):
        """Precomputed cost should be used directly without recalculation."""
        precomputed = 4.0
        sol = Solution(optimal_tour, simple_dist_matrix, cost=precomputed)
        assert sol._cost == precomputed
        assert sol.cost == precomputed

    def test_cost_setter(self, simple_dist_matrix, optimal_tour):
        """Cost setter should work for delta updates."""
        sol = Solution(optimal_tour, simple_dist_matrix)
        sol.cost = 123.45
        assert sol.cost == 123.45

    def test_copy_preserves_cost(self, simple_dist_matrix, optimal_tour):
        """Copy should preserve precomputed cost."""
        sol = Solution(optimal_tour, simple_dist_matrix, cost=4.0)
        copy = sol.copy()
        # Cost should be copied, not None
        assert copy._cost == 4.0
        assert copy.cost == 4.0

    def test_copy_with_lazy_cost(self, simple_dist_matrix, optimal_tour):
        """Copy should preserve None cost if not yet computed."""
        sol = Solution(optimal_tour, simple_dist_matrix)
        copy = sol.copy()
        # Both should have None initially
        assert copy._cost is None
        # Computing one shouldn't affect the other
        _ = sol.cost
        assert copy._cost is None


class TestSolutionCorrectness:
    """Tests for Solution correctness."""

    def test_optimal_tour_cost(self, simple_dist_matrix, optimal_tour):
        """Optimal tour (square perimeter) should have cost 4.0."""
        sol = Solution(optimal_tour, simple_dist_matrix)
        assert sol.cost == pytest.approx(4.0)

    def test_suboptimal_tour_cost(self, simple_dist_matrix, suboptimal_tour):
        """Suboptimal tour should have higher cost."""
        sol = Solution(suboptimal_tour, simple_dist_matrix)
        expected = 2 * np.sqrt(2) + 2  # ~4.83
        assert sol.cost == pytest.approx(expected)
        assert sol.cost > 4.0

    def test_tour_is_copied(self, simple_dist_matrix, optimal_tour):
        """Tour should be defensively copied."""
        original = optimal_tour[:]
        sol = Solution(optimal_tour, simple_dist_matrix)
        optimal_tour[1] = 999  # Mutate original
        assert sol.tour == original  # Solution unchanged
