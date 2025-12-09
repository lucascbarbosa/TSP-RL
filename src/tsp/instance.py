"""TSP instance loading and dataset management."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
from numpy.typing import NDArray


class TSPInstance:
    """
    TSP instance loader for randomly generated instances in JSON format.

    Compatible with tsplib95 API conventions.

    Attributes:
        coords: List of (x, y) coordinates.
        n: Number of cities.
        dimension: Alias for n.
        name: Instance identifier.
        dist_matrix: Precomputed distance matrix.
        opt_tour: Optimal tour if available.
    """

    def __init__(
        self,
        path: Union[str, Path],
        instance_id: int = 0,
        num_cities: Optional[int] = None,
        preloaded_data: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize TSP instance.

        Args:
            path: Path to JSON file containing instances.
            instance_id: Index of instance within the JSON file.
            num_cities: Limit number of cities (None = use all).
            preloaded_data: Pre-loaded JSON data (avoids re-reading file).
        """
        if preloaded_data is not None:
            data = preloaded_data
        else:
            with open(path, "r") as f:
                data = json.load(f)

        entry = data[instance_id]
        coords = entry["coords"]

        if num_cities is not None:
            coords = coords[:num_cities]

        self.coords = coords
        self.n = len(self.coords)
        self.dimension = self.n
        self.name = f"random_instance_{instance_id}_nodes_{self.n}"

        # Compute distance matrix
        self.dist_matrix: NDArray[np.float64] = np.zeros((self.n, self.n))
        for i in range(self.n):
            xi, yi = self.coords[i]
            for j in range(self.n):
                xj, yj = self.coords[j]
                self.dist_matrix[i, j] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

        # Load optimal tour if available
        if "tour" in entry and entry["tour"] is not None:
            full_tour = [c + 1 for c in entry["tour"]]
            self.opt_tour: Optional[List[int]] = [c for c in full_tour if c <= self.n]
        else:
            self.opt_tour = None

    def get_nodes(self) -> range:
        """Return node indices {1, 2, ..., n}."""
        return range(1, self.n + 1)

    def get_weight(self, i: int, j: int) -> float:
        """
        Get distance between nodes i and j (1-based indexing).

        Args:
            i: First node (1-based).
            j: Second node (1-based).

        Returns:
            Distance between nodes.
        """
        return float(self.dist_matrix[i - 1, j - 1])


class TSPDataset:
    """
    Dataset wrapper for batch processing of TSP instances.

    Loads all instances into memory for efficient parallel processing.
    """

    def __init__(
        self,
        json_file_path: Union[str, Path],
        active_indices: List[int],
    ) -> None:
        """
        Initialize dataset.

        Args:
            json_file_path: Path to JSON file with instances.
            active_indices: List of instance IDs to include (e.g., train IDs).
        """
        self.path = str(json_file_path)
        self.indices = active_indices

        print(f"Loading {json_file_path} into memory...")
        with open(json_file_path, "r") as f:
            self.data_in_memory = json.load(f)
        print(f"Loaded {len(self.data_in_memory)} raw instances. Active subset: {len(self.indices)}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> TSPInstance:
        """Get instance by index within active subset."""
        real_instance_id = self.indices[idx]
        return TSPInstance(
            path=self.path,
            instance_id=real_instance_id,
            preloaded_data=self.data_in_memory,
        )

    def __iter__(self) -> Iterator[TSPInstance]:
        """Iterate over all active instances."""
        for i in range(len(self)):
            yield self[i]
