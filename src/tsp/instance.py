"""TSP instance loading and dataset management."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Distance Metrics (TSPLIB-compatible)
# =============================================================================


def _euc_2d(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Euclidean distance (double precision, no rounding).

    Used for generated instances with coords in [0,1]².
    """
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist[i, j] = dist[j, i] = math.sqrt(dx * dx + dy * dy)
    return dist


def _att(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Pseudo-Euclidean distance (ATT).

    Special scaling by 1/sqrt(10) with conditional rounding.
    Used for att48, att532 etc. from TSPLIB.
    """
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            xd = coords[i, 0] - coords[j, 0]
            yd = coords[i, 1] - coords[j, 1]
            rij = math.sqrt((xd * xd + yd * yd) / 10.0)
            tij = int(rij + 0.5)
            if tij < rij:
                dij = tij + 1
            else:
                dij = tij
            dist[i, j] = dist[j, i] = dij
    return dist


def _geo(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Geographic distance (great-circle on Earth).

    Coordinates are in TSPLIB degree format (DDD.MM where MM is minutes).
    Used for ulysses16, ulysses22 etc. from TSPLIB.
    """
    n = len(coords)
    RRR = 6378.388  # Earth radius in km
    PI = 3.141592

    # Convert TSPLIB degree format to radians
    lats = np.zeros(n)
    lons = np.zeros(n)
    for i in range(n):
        deg_lat = int(coords[i, 0])
        min_lat = coords[i, 0] - deg_lat
        lats[i] = PI * (deg_lat + 5.0 * min_lat / 3.0) / 180.0

        deg_lon = int(coords[i, 1])
        min_lon = coords[i, 1] - deg_lon
        lons[i] = PI * (deg_lon + 5.0 * min_lon / 3.0) / 180.0

    # Compute great-circle distances
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            q1 = math.cos(lons[i] - lons[j])
            q2 = math.cos(lats[i] - lats[j])
            q3 = math.cos(lats[i] + lats[j])
            dij = int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
            dist[i, j] = dist[j, i] = dij
    return dist


# Registry: edge_weight_type -> distance function
DISTANCE_METRICS: Dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
    "EUC_2D": _euc_2d,
    "ATT": _att,
    "GEO": _geo,
}


def _infer_edge_weight_type(path: Union[str, Path]) -> str:
    """Infer edge weight type from file path (e.g., 'data/GEO.json' -> 'GEO')."""
    stem = Path(path).stem.upper()
    if stem in DISTANCE_METRICS:
        return stem
    return "EUC_2D"  # Default fallback


class TSPInstance:
    """
    TSP instance loader for randomly generated instances in JSON format.

    Compatible with tsplib95 API conventions.

    Attributes:
        coords: List of (x, y) coordinates.
        n: Number of cities.
        dimension: Alias for n.
        name: Instance identifier.
        edge_weight_type: Distance metric type (EUC_2D, ATT, GEO).
        dist_matrix: Precomputed distance matrix.
        opt_tour: Optimal tour if available.
    """

    def __init__(
        self,
        path: Union[str, Path],
        instance_id: int = 0,
        num_cities: Optional[int] = None,
        preloaded_data: Optional[List[Dict[str, Any]]] = None,
        edge_weight_type: Optional[str] = None,
    ) -> None:
        """
        Initialize TSP instance.

        Args:
            path: Path to JSON file containing instances.
            instance_id: Index of instance within the JSON file.
            num_cities: Limit number of cities (None = use all).
            preloaded_data: Pre-loaded JSON data (avoids re-reading file).
            edge_weight_type: Distance metric (EUC_2D, ATT, GEO). Inferred from path if None.
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

        # Determine edge weight type
        self.edge_weight_type = edge_weight_type or _infer_edge_weight_type(path)

        # Compute distance matrix using appropriate metric
        coords_array = np.array(self.coords, dtype=np.float64)
        distance_func = DISTANCE_METRICS.get(self.edge_weight_type, _euc_2d)
        self.dist_matrix: NDArray[np.float64] = distance_func(coords_array)

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
        edge_weight_type: Optional[str] = None,
    ) -> None:
        """
        Initialize dataset.

        Args:
            json_file_path: Path to JSON file with instances.
            active_indices: List of instance IDs to include (e.g., train IDs).
            edge_weight_type: Distance metric (inferred from path if None).
        """
        self.path = str(json_file_path)
        self.indices = active_indices
        self.edge_weight_type = edge_weight_type or _infer_edge_weight_type(json_file_path)

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
            edge_weight_type=self.edge_weight_type,
        )

    def __iter__(self) -> Iterator[TSPInstance]:
        """Iterate over all active instances."""
        for i in range(len(self)):
            yield self[i]
