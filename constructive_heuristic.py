import random


class ConstructiveHeuristic:
    def random_tour(self, problem):
        # Nós do problema (em geral 1..n)
        nodes = list(problem.get_nodes())
        n = len(nodes)

        # Embaralha a ordem dos nós
        random.shuffle(nodes)

        # Calcula o custo percorrendo todos os nós + volta ao início
        tour_cost = 0.0
        for i in range(n):
            curr = nodes[i]
            nxt = nodes[(i + 1) % n]  # último volta para o primeiro
            tour_cost += problem.get_weight(curr, nxt)

        # Fecha explicitamente a rota na representação
        closed_tour = nodes + [nodes[0]]

        return closed_tour, tour_cost

    def nearest_neighbor_tour(self, problem, start_node=None):
        n = problem.dimension
        if start_node is None:
            start_node = random.choice(list(problem.get_nodes()))

        unvisited = set(problem.get_nodes())
        unvisited.remove(start_node)

        tour = [start_node]
        current_node = start_node
        tour_cost = 0.0

        while unvisited:
            next_node = min(
                unvisited, key=lambda node: problem.get_weight(current_node, node)
            )
            tour_cost += problem.get_weight(current_node, next_node)
            tour.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node

        # Return to starting node
        tour_cost += problem.get_weight(current_node, start_node)
        tour.append(start_node)

        # print(f"Nearest neighbor tour: {tour}")
        # print(f"Nearest neighbor tour length: {tour_cost}")

        return tour, tour_cost

    def cheapest_insertion_tour(self, problem, start_node=None):
        """
        Construtivo do TSP pelo método de Cheapest Insertion.

        - Começa com um ciclo de 2 nós: start_node e o vizinho mais próximo.
        - Em cada passo, insere o nó ainda não visitado na posição (entre duas
            cidades consecutivas do tour) que causar o menor aumento de custo.
        - Retorna tour fechado [c0, ..., ck, c0] e o custo total.
        """
        nodes = list(problem.get_nodes())
        n = len(nodes)

        # Casos muito pequenos: usa random_tour como fallback
        if n <= 2:
            return self.random_tour(problem)

        # Escolhe nó inicial
        if start_node is None:
            start_node = random.choice(nodes)

        unvisited = set(nodes)
        unvisited.remove(start_node)

        # Escolhe o vizinho mais próximo do nó inicial para começar o ciclo
        nearest = min(
            unvisited,
            key=lambda node: problem.get_weight(start_node, node),
        )
        unvisited.remove(nearest)

        # Tour inicial fechado: start -> nearest -> start
        tour = [start_node, nearest, start_node]
        tour_cost = problem.get_weight(start_node, nearest) + problem.get_weight(
            nearest, start_node
        )

        # Enquanto existirem nós não visitados, insere sempre pelo menor aumento de custo
        while unvisited:
            best_delta = float("inf")
            best_city = None
            best_pos = None  # posição i tal que inserimos entre tour[i] e tour[i+1]

            for city in unvisited:
                # testa inserir city em cada aresta (tour[i], tour[i+1])
                for i in range(len(tour) - 1):
                    a = tour[i]
                    b = tour[i + 1]
                    # aumento de custo ao inserir city entre a e b
                    delta = (
                        problem.get_weight(a, city)
                        + problem.get_weight(city, b)
                        - problem.get_weight(a, b)
                    )

                    if delta < best_delta:
                        best_delta = delta
                        best_city = city
                        best_pos = i

            # Faz a melhor inserção encontrada
            tour.insert(best_pos + 1, best_city)
            tour_cost += best_delta
            unvisited.remove(best_city)

        # Por segurança, garante que está fechado
        if tour[0] != tour[-1]:
            tour.append(tour[0])
            tour_cost += problem.get_weight(tour[-2], tour[-1])

        return tour, tour_cost
