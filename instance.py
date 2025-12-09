import json
import numpy as np
import math


class RandomTSPInstance:
    """
    Leitor de instâncias TSP geradas aleatoriamente no formato JSON,
    compatível com a API tsplib95.

    Parâmetros:
        path        = caminho para o arquivo JSON
        instance_id = qual instância carregar dentro do JSON
        num_cities  = número de cidades a usar (None = todas)
    """

    def __init__(self, path, instance_id=0, num_cities=None, preloaded_data=None):

        # MODIFICATION: If data is already in memory, use it.
        # Otherwise, read from file (old behavior).
        if preloaded_data is not None:
            data = preloaded_data
        else:
            with open(path, "r") as f:
                data = json.load(f)

        entry = data[instance_id]

        # --- The rest of your original logic remains exactly the same ---
        coords = entry["coords"]

        if num_cities is not None:
            coords = coords[:num_cities]

        self.coords = coords
        self.n = len(self.coords)
        self.dimension = self.n
        self.name = f"random_instance_{instance_id}_nodes_{self.n}"

        self.dist_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            xi, yi = self.coords[i]
            for j in range(self.n):
                xj, yj = self.coords[j]
                self.dist_matrix[i, j] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

        if "tour" in entry and entry["tour"] is not None:
            full_tour = [c + 1 for c in entry["tour"]]
            self.opt_tour = [c for c in full_tour if c <= self.n]
        else:
            self.opt_tour = None

    def get_nodes(self):
        """Retorna {1, 2, ..., n}."""
        return range(1, self.n + 1)

    def get_weight(self, i, j):
        """
        Distância entre nós i e j (1-based).
        """
        return self.dist_matrix[i - 1, j - 1]


class TSPDataset:
    def __init__(self, json_file_path, active_indices):
        """
        Args:
            json_file_path (str): Path to the .json file containing the instances.
            active_indices (list): List of integers (IDs) to use (e.g., train_ids).
        """
        self.path = json_file_path
        self.indices = active_indices

        print(f"Loading {json_file_path} into memory...")
        with open(json_file_path, "r") as f:
            # We load the WHOLE list of dicts into memory once
            self.data_in_memory = json.load(f)
        print(f"Loaded {len(self.data_in_memory)} raw instances. Active subset: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Allows usage like: dataset[0] to get the first active instance.
        """
        # 1. Get the real ID from your subset of indices
        real_instance_id = self.indices[idx]

        # 2. Create the object using the preloaded data
        # We pass 'self.path' just to satisfy the signature, but 'preloaded_data' does the work.
        instance = RandomTSPInstance(path=self.path, instance_id=real_instance_id, preloaded_data=self.data_in_memory)
        return instance

    def __iter__(self):
        """
        Allows usage like: for instance in dataset: ...
        """
        for i in range(len(self)):
            yield self[i]
