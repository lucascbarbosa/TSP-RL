"""TSP instance loading and dataset management."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# JSON Loading (supports .json and .json.zip)
# =============================================================================


def _resolve_json_path(path: Union[str, Path]) -> Path:
    """
    Resolve JSON path, preferring .json over .json.zip.

    Args:
        path: Path to JSON file (with or without .zip extension).

    Returns:
        Resolved path (may have .zip added if .json doesn't exist).

    Raises:
        FileNotFoundError: If neither .json nor .json.zip exists.
    """
    p = Path(path)

    # If path exists as-is, use it
    if p.exists():
        return p

    # If path ends with .json, try .json.zip
    if p.suffix == ".json":
        zip_path = p.with_suffix(".json.zip")
        if zip_path.exists():
            return zip_path

    # If path ends with .json.zip, try .json
    if str(p).endswith(".json.zip"):
        json_path = Path(str(p)[:-4])  # Remove .zip
        if json_path.exists():
            return json_path

    raise FileNotFoundError(f"Neither {p} nor {p}.zip exists")


def _load_json(path: Union[str, Path]) -> list[dict[str, Any]]:
    """
    Load JSON data from .json or .json.zip file.

    Args:
        path: Path to JSON file (resolved via _resolve_json_path).

    Returns:
        Parsed JSON data.
    """
    resolved = _resolve_json_path(path)

    if resolved.suffix == ".zip":
        # Extract JSON from zip (assumes single file inside with .json name)
        with zipfile.ZipFile(resolved, "r") as zf:
            # Get the JSON filename (same as zip but without .zip)
            json_name = resolved.stem  # e.g., "EUC_2D.json" from "EUC_2D.json.zip"
            with zf.open(json_name) as f:
                return json.load(f)
    else:
        with open(resolved, "r") as f:
            return json.load(f)


# =============================================================================
# Distance Metrics (TSPLIB-compatible)
# =============================================================================


def _euc_2d(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Euclidean distance (double precision, no rounding).

    Used for generated instances with coords in [0,1]².
    Vectorized implementation: O(n²) memory, but ~10-20x faster than loops.
    """
    # Compute pairwise differences: diff[i,j,k] = coords[i,k] - coords[j,k]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # Sum of squared differences along coordinate axis, then sqrt
    return np.sqrt(np.sum(diff * diff, axis=2))


def _att(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Pseudo-Euclidean distance (ATT).

    Special scaling by 1/sqrt(10) with conditional rounding.
    Used for att48, att532 etc. from TSPLIB.
    Vectorized implementation.
    """
    # Pairwise squared differences
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    sum_sq = np.sum(diff * diff, axis=2)

    # ATT formula: rij = sqrt(sum_sq / 10), tij = round(rij)
    rij = np.sqrt(sum_sq / 10.0)
    tij = np.floor(rij + 0.5)  # int(x + 0.5) = floor(x + 0.5)

    # dij = tij + 1 if tij < rij else tij
    return np.where(tij < rij, tij + 1, tij)


def _geo(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Geographic distance (great-circle on Earth).

    Coordinates are in TSPLIB degree format (DDD.MM where MM is minutes).
    Example: 38.24 means 38° 24' (38 degrees, 24 minutes).
    Used for ulysses16, ulysses22 etc. from TSPLIB.
    Vectorized implementation.
    """
    RRR = 6378.388  # Earth radius in km
    PI = 3.141592

    # Convert TSPLIB degree format to radians (vectorized)
    # Use trunc() to match TSPLIB int() behavior (truncate towards zero)
    # floor() would give wrong results for negative coords (-38.24 -> -39 vs -38)
    deg = np.trunc(coords)
    minutes = coords - deg
    radians = PI * (deg + 5.0 * minutes / 3.0) / 180.0

    lats = radians[:, 0]  # shape (n,)
    lons = radians[:, 1]  # shape (n,)

    # Compute pairwise differences for great-circle formula
    # q1 = cos(lon_i - lon_j), q2 = cos(lat_i - lat_j), q3 = cos(lat_i + lat_j)
    lon_diff = lons[:, np.newaxis] - lons[np.newaxis, :]  # (n, n)
    lat_diff = lats[:, np.newaxis] - lats[np.newaxis, :]  # (n, n)
    lat_sum = lats[:, np.newaxis] + lats[np.newaxis, :]  # (n, n)

    q1 = np.cos(lon_diff)
    q2 = np.cos(lat_diff)
    q3 = np.cos(lat_sum)

    # Great-circle distance formula
    # Clamp argument to [-1, 1] to avoid numerical issues with arccos
    arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
    arg = np.clip(arg, -1.0, 1.0)

    return np.floor(RRR * np.arccos(arg) + 1.0)


# Registry: edge_weight_type -> distance function
DISTANCE_METRICS: dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
    "EUC_2D": _euc_2d,
    "ATT": _att,
    "GEO": _geo,
}


def _infer_edge_weight_type(path: Union[str, Path]) -> str:
    """Infer edge weight type from filename (e.g., 'GEO.json' or 'GEO.json.zip' -> 'GEO')."""
    name = Path(path).name.upper()
    # Handle .json.zip -> remove both suffixes
    if name.endswith(".JSON.ZIP"):
        stem = name[:-9]
    elif name.endswith(".JSON"):
        stem = name[:-5]
    else:
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
        preloaded_data: Optional[list[dict[str, Any]]] = None,
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
            data = _load_json(path)

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
            self.opt_tour: Optional[list[int]] = [c for c in full_tour if c <= self.n]
        else:
            self.opt_tour = None

        # Load MIP gap (duality gap from solver, as fraction: 0.03 = 3%)
        # gap=0 means tour is provably optimal
        # gap>0 means solver didn't close gap (lower_bound = primal / (1 + gap))
        # gap=None means no gap info available
        self.mip_gap: Optional[float] = entry.get("gap", None)

        # Compute optimal tour cost from distance matrix
        if self.opt_tour is not None:
            # opt_tour is 1-based closed tour, compute cost
            self.opt_cost: Optional[float] = sum(
                self.dist_matrix[self.opt_tour[i] - 1, self.opt_tour[(i + 1) % len(self.opt_tour)] - 1]
                for i in range(len(self.opt_tour))
            )
            # Validate against JSON cost if provided
            json_cost = entry.get("cost", None)
            if json_cost is not None:
                if abs(self.opt_cost - json_cost) > 1e-6:
                    print(
                        f"[WARNING] {self.name}: computed cost {self.opt_cost:.6f} "
                        f"differs from JSON cost {json_cost:.6f}"
                    )
        else:
            self.opt_cost = None

    def get_nodes(self) -> range:
        """Node indices {1, ..., n}."""
        return range(1, self.n + 1)

    def get_weight(self, i: int, j: int) -> float:
        """Distance between nodes i and j (1-based)."""
        return float(self.dist_matrix[i - 1, j - 1])


class TSPDataset:
    """
    Dataset wrapper for batch processing of TSP instances.

    Loads all instances into memory for efficient parallel processing.
    """

    def __init__(
        self,
        json_file_path: Union[str, Path],
        active_indices: list[int],
        edge_weight_type: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize dataset.

        Args:
            json_file_path: Path to JSON file with instances.
            active_indices: List of instance IDs to include (e.g., train IDs).
            edge_weight_type: Distance metric (inferred from path if None).
            verbose: Print loading messages (default True, set False for workers).
        """
        self.path = str(json_file_path)
        self.indices = active_indices
        self.edge_weight_type = edge_weight_type or _infer_edge_weight_type(json_file_path)

        if verbose:
            print(f"Loading {json_file_path} into memory...")
        self.data_in_memory = _load_json(json_file_path)
        if verbose:
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
