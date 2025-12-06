# solution.py
import numpy as np


class Solution:
    """
    Representa uma solução do TSP.

    tour: lista de cidades em 1..n
          aqui assumimos tour FECHADO: [c1, ..., ck, c1]
    dist_matrix: matriz de distâncias (n x n), indexada em 0..n-1
    """

    def __init__(self, tour, dist_matrix, is_closed=True):
        self.dist_matrix = dist_matrix
        self.is_closed = is_closed
        self.tour = tour[:]  # cópia defensiva
        self.cost = self._compute_cost()

    def _compute_cost(self) -> float:
        return self.compute_cost_static(self.tour, self.dist_matrix, self.is_closed)

    @staticmethod
    def compute_cost_static(tour, dist_matrix, is_closed=True) -> float:
        """
        Calcula o custo de um tour usando dist_matrix.
        tour usa nós em 1..n, dist_matrix é 0..n-1.
        """
        cost = 0.0

        if not tour:
            return 0.0

        # Se for fechado, assumimos que o último já é igual ao primeiro.
        # Se não for, podemos fechar implicitamente.
        if is_closed:
            # Se não estiver fechado, fecha na marra:
            if tour[0] != tour[-1]:
                tour = tour[:] + [tour[0]]

            for i in range(len(tour) - 1):
                a = tour[i] - 1
                b = tour[i + 1] - 1
                cost += dist_matrix[a, b]
        else:
            # Rota aberta: só soma arestas consecutivas
            for i in range(len(tour) - 1):
                a = tour[i] - 1
                b = tour[i + 1] - 1
                cost += dist_matrix[a, b]

        return cost

    def copy(self):
        return Solution(self.tour, self.dist_matrix, self.is_closed)

    def __repr__(self):
        return f"Solution(cost={self.cost}, tour_len={len(self.tour)})"
