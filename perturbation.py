# perturbation.py
import copy
import random
from solution import Solution


class Perturbation:
    @staticmethod
    def random_two_swap(solution: Solution) -> Solution:
        """
        Perturba a solução trocando dois vértices aleatórios na parte interna da rota.
        Mantém a rota fechada [c0, ..., c_{n-1}, c0].
        """
        # copia rasa do tour
        tour = solution.tour[:]

        # assumindo rota fechada: primeiro == último
        # vamos mexer apenas nos índices 1 .. len(tour) - 2
        n = len(tour) - 1  # número de cidades reais
        if n <= 3:
            # muito pequena, devolve cópia
            return solution.copy()

        i, j = random.sample(range(1, n), 2)
        tour[i], tour[j] = tour[j], tour[i]

        # garante que continua fechada
        tour[-1] = tour[0]

        return Solution(tour, solution.dist_matrix, is_closed=True)

    @staticmethod
    def random_segment_reverse(solution: Solution) -> Solution:
        """
        Perturbação um pouco mais forte: reverte um segmento aleatório do tour interno.
        """
        tour = solution.tour[:]
        n = len(tour) - 1
        if n <= 4:
            return solution.copy()

        i, j = sorted(random.sample(range(1, n), 2))
        tour[i:j] = reversed(tour[i:j])
        tour[-1] = tour[0]

        return Solution(tour, solution.dist_matrix, is_closed=True)
