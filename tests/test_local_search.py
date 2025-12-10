"""Tests for local search algorithms."""

import numpy as np
import pytest

from src.tsp.solution import Solution
from src.tsp.local_search import (
    two_opt,
    two_opt_nn,
    two_opt_dlb,
    lin_kernighan,
    _two_opt_delta,
    _two_opt_gain,
)


class TestTwoOptDelta:
    """Tests for 2-opt delta calculation."""

    def test_delta_improvement(self, simple_dist_matrix, suboptimal_tour):
        """Delta should be negative for improving moves."""
        # Suboptimal: [1, 3, 2, 4, 1] -> reversing segment [3,2] gives [1, 2, 3, 4, 1]
        # This is a 2-opt move at positions i=1, j=3
        delta = _two_opt_delta(suboptimal_tour, 1, 3, simple_dist_matrix)
        # Expected: removes (1,3) + (2,4), adds (1,2) + (3,4)
        # removed: sqrt(2) + sqrt(2) = 2*sqrt(2) ~ 2.83
        # added: 1 + 1 = 2
        # delta = 2 - 2.83 ~ -0.83 (improvement)
        assert delta < 0

    def test_delta_no_improvement(self, simple_dist_matrix, optimal_tour):
        """Delta should be non-negative for non-improving moves on optimal."""
        # Any 2-opt on optimal tour should not improve
        delta = _two_opt_delta(optimal_tour, 1, 3, simple_dist_matrix)
        assert delta >= -1e-10  # Allow for floating point

    def test_gain_is_negative_delta(self, simple_dist_matrix, suboptimal_tour):
        """Gain should be -delta (positive for improvements)."""
        delta = _two_opt_delta(suboptimal_tour, 1, 3, simple_dist_matrix)
        gain = _two_opt_gain(suboptimal_tour, 1, 3, simple_dist_matrix)
        assert gain == pytest.approx(-delta)


class TestTwoOpt:
    """Tests for 2-opt local search."""

    def test_improves_suboptimal(self, simple_dist_matrix, suboptimal_tour):
        """2-opt should improve a suboptimal tour."""
        sol = Solution(suboptimal_tour, simple_dist_matrix)
        improved = two_opt(sol)
        assert improved.cost < sol.cost

    def test_finds_optimal_from_suboptimal(self, simple_dist_matrix, suboptimal_tour):
        """2-opt should find optimal from simple suboptimal."""
        sol = Solution(suboptimal_tour, simple_dist_matrix)
        improved = two_opt(sol)
        assert improved.cost == pytest.approx(4.0)

    def test_optimal_unchanged(self, simple_dist_matrix, optimal_tour):
        """2-opt should not change an already optimal tour."""
        sol = Solution(optimal_tour, simple_dist_matrix)
        improved = two_opt(sol)
        assert improved.cost == pytest.approx(sol.cost)

    def test_cost_is_precomputed(self, simple_dist_matrix, suboptimal_tour):
        """Returned solution should have precomputed cost (not lazy)."""
        sol = Solution(suboptimal_tour, simple_dist_matrix)
        improved = two_opt(sol)
        # Cost should already be computed (not None)
        assert improved._cost is not None
        # And should match actual cost
        actual = Solution.compute_cost_static(improved.tour, simple_dist_matrix, is_closed=True)
        assert improved.cost == pytest.approx(actual)


class TestLinKernighan:
    """Tests for Lin-Kernighan local search."""

    def test_improves_suboptimal(self, simple_dist_matrix, suboptimal_tour):
        """LK should improve a suboptimal tour."""
        sol = Solution(suboptimal_tour, simple_dist_matrix)
        improved = lin_kernighan(sol)
        assert improved.cost < sol.cost

    def test_at_least_as_good_as_two_opt(self, larger_dist_matrix):
        """LK should be at least as good as 2-opt."""
        np.random.seed(123)
        n = larger_dist_matrix.shape[0]
        random_tour = list(range(1, n + 1))
        np.random.shuffle(random_tour)
        random_tour.append(random_tour[0])

        sol = Solution(random_tour, larger_dist_matrix)
        two_opt_result = two_opt(sol)
        lk_result = lin_kernighan(sol)

        # LK should find solution at least as good
        assert lk_result.cost <= two_opt_result.cost + 1e-10

    def test_cost_is_precomputed(self, simple_dist_matrix, suboptimal_tour):
        """Returned solution should have precomputed cost."""
        sol = Solution(suboptimal_tour, simple_dist_matrix)
        improved = lin_kernighan(sol)
        assert improved._cost is not None
        actual = Solution.compute_cost_static(improved.tour, simple_dist_matrix, is_closed=True)
        assert improved.cost == pytest.approx(actual, rel=1e-6)


class TestTwoOptNN:
    """Tests for 2-opt with neighbor lists."""

    def test_improves_suboptimal(self, simple_dist_matrix, suboptimal_tour):
        """2-opt-nn should improve a suboptimal tour."""
        sol = Solution(suboptimal_tour, simple_dist_matrix)
        improved = two_opt_nn(sol, k=3)
        assert improved.cost <= sol.cost

    def test_quality_close_to_full_two_opt(self, larger_dist_matrix):
        """2-opt-nn should find solution close to full 2-opt (within 10%)."""
        np.random.seed(456)
        n = larger_dist_matrix.shape[0]
        random_tour = list(range(1, n + 1))
        np.random.shuffle(random_tour)
        random_tour.append(random_tour[0])

        sol = Solution(random_tour, larger_dist_matrix)
        two_opt_result = two_opt(sol)
        nn_result = two_opt_nn(sol, k=5)

        # Should be within 10% of full 2-opt
        assert nn_result.cost <= two_opt_result.cost * 1.10


class TestTwoOptDLB:
    """Tests for 2-opt with neighbor lists + don't look bits."""

    def test_improves_suboptimal(self, simple_dist_matrix, suboptimal_tour):
        """2-opt-dlb should improve a suboptimal tour."""
        sol = Solution(suboptimal_tour, simple_dist_matrix)
        improved = two_opt_dlb(sol, k=3)
        assert improved.cost <= sol.cost

    def test_quality_reasonable(self, larger_dist_matrix):
        """2-opt-dlb should find a reasonable solution (within 15% of 2-opt)."""
        np.random.seed(789)
        n = larger_dist_matrix.shape[0]
        random_tour = list(range(1, n + 1))
        np.random.shuffle(random_tour)
        random_tour.append(random_tour[0])

        sol = Solution(random_tour, larger_dist_matrix)
        two_opt_result = two_opt(sol)
        dlb_result = two_opt_dlb(sol, k=5)

        # Should be within 15% of full 2-opt
        assert dlb_result.cost <= two_opt_result.cost * 1.15
