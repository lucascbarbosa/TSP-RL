"""TSP instance management and utilities."""
import numpy as np
from dataclasses import dataclass


@dataclass
class TSPInstance:
    """Represents a TSP problem instance."""
    name: str
    n_cities: int
    distances: np.ndarray  # n x n distance matrix

    @classmethod
    def from_coordinates(cls, name: str, coords: np.ndarray) -> 'TSPInstance':
        """Create TSP instance from city coordinates."""
        n = len(coords)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = np.linalg.norm(coords[i] - coords[j])
        return cls(name, n, dist)

    @classmethod
    def from_tsplib(cls, filepath: str) -> 'TSPInstance':
        """Load TSP instance from TSPLIB format file."""
        coords = []
        reading_coords = False
        name = filepath.split('/')[-1].split('.')[0]

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('NODE_COORD_SECTION'):
                    reading_coords = True
                    continue
                if line == 'EOF' or not line:
                    break
                if reading_coords:
                    parts = line.split()
                    coords.append([float(parts[1]), float(parts[2])])

        if len(coords) > 0:
            return cls.from_coordinates(name, np.array(coords))
        else:
            return None
