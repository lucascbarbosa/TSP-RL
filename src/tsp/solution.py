"""TSP solution representation."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray


class Solution:
    """
    TSP solution representation.

    Attributes:
        tour: List of city indices (1-based). Closed tour: [c1, ..., cn, c1].
        dist_matrix: Distance matrix (n x n), indexed 0..n-1.
        cost: Total tour cost (lazy-computed if not provided).
        is_closed: Whether the tour is closed (last == first).
    """

    __slots__ = ("tour", "dist_matrix", "is_closed", "_cost")

    def __init__(
        self,
        tour: List[int],
        dist_matrix: NDArray[np.float64],
        is_closed: bool = True,
        cost: Optional[float] = None,
    ) -> None:
        self.dist_matrix = dist_matrix
        self.is_closed = is_closed
        self.tour = tour[:]  # defensive copy
        self._cost: Optional[float] = cost  # lazy: None means not computed yet

    @property
    def cost(self) -> float:
        """Return tour cost, computing lazily if needed."""
        if self._cost is None:
            self._cost = self._compute_cost()
        return self._cost

    @cost.setter
    def cost(self, value: float) -> None:
        """Allow setting cost directly (for delta updates)."""
        self._cost = value

    def _compute_cost(self) -> float:
        return self.compute_cost_static(self.tour, self.dist_matrix, self.is_closed)

    @staticmethod
    def compute_cost_static(
        tour: List[int],
        dist_matrix: NDArray[np.float64],
        is_closed: bool = True,
    ) -> float:
        """
        Compute tour cost using the distance matrix.

        Args:
            tour: City indices (1-based).
            dist_matrix: Distance matrix (0-based indexing).
            is_closed: Whether tour is closed.

        Returns:
            Total tour cost.
        """
        if not tour:
            return 0.0

        cost = 0.0
        working_tour = tour

        if is_closed and tour[0] != tour[-1]:
            working_tour = tour[:] + [tour[0]]

        for i in range(len(working_tour) - 1):
            a = working_tour[i] - 1
            b = working_tour[i + 1] - 1
            cost += dist_matrix[a, b]

        return cost

    def copy(self) -> Solution:
        """Create a deep copy of this solution, preserving computed cost."""
        return Solution(self.tour, self.dist_matrix, self.is_closed, self._cost)

    def __repr__(self) -> str:
        return f"Solution(cost={self.cost:.2f}, tour_len={len(self.tour)})"
