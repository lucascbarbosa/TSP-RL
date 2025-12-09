"""Q-ILS: Iterated Local Search guiado por Q-Learning para o TSP."""

import constructive_heuristic
import local_search
import perturbation
from solution import Solution
import numpy as np
import random


class Q_ILS:
    """
    Framework Q-ILS: ILS onde um agente RL decide qual par (perturbação, busca local)
    aplicar a cada iteração, em função do estado atual (gap percentual).

    Perturbações:
    - Leves: two_swap, segment_reverse
    - Destrutivas (construtivos): random, nearest, cheapest

    Buscas locais: 2-opt, Lin-Kernighan (simplificado, depth=2)
    """

    def __init__(self, problem):
        n = problem.dimension
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i][j] = problem.get_weight(i + 1, j + 1)
        self.problem = problem
        self.dist_matrix = dist_matrix
        self.constructive = constructive_heuristic.ConstructiveHeuristic()
        self.local_search = local_search.LocalSearch()

        # Mapeamento de ações: (perturbação, busca local)
        # 8 ações = 5 perturbações × 2 buscas locais (com algumas omissões)
        self.action_map = {
            # Perturbações leves
            ("two_swap", "two_opt"): 0,
            ("two_swap", "lin_kernighan"): 1,
            ("segment_reverse", "two_opt"): 2,
            ("segment_reverse", "lin_kernighan"): 3,
            # Perturbações destrutivas (construtivos)
            ("random", "two_opt"): 4,
            ("nearest", "two_opt"): 5,
            ("cheapest", "two_opt"): 6,
            ("nearest", "lin_kernighan"): 7,
        }

        # Mapeamento inverso para decodificar ações
        self.action_decode = {v: k for k, v in self.action_map.items()}

        self.n_actions = len(self.action_map)
        self.n_states = 5  # estados baseados no gap

        self.action = None
        self.state = None
        self.qtable = None

    def load_qtable(self, path: str = "qtable_output.txt"):
        """Carrega Q-table de um arquivo texto."""
        with open(path, "r") as f:
            header = f.readline().strip().split()
            if len(header) != 2:
                raise ValueError("Cabeçalho da Q-table inválido (esperado: 'n_states n_actions').")

            n_states, n_actions = map(int, header)

            data = []
            for i in range(n_states):
                line = f.readline()
                if not line:
                    raise ValueError("Número de linhas da Q-table menor que n_states.")
                row_vals = list(map(float, line.strip().split()))
                if len(row_vals) != n_actions:
                    raise ValueError(f"Linha {i+2} da Q-table tem {len(row_vals)} colunas, esperado {n_actions}.")
                data.append(row_vals)

        self.qtable = np.array(data, dtype=float)

    def choose_action_from_q(self, state: int, epsilon: float = 0.0) -> int:
        """
        Dado um estado, seleciona uma ação usando a Q-table (epsilon-greedy).
        """
        if self.qtable is None:
            raise ValueError("Q-table ainda não carregada. Use load_qtable() antes.")

        n_states, n_actions = self.qtable.shape

        if state < 0 or state >= n_states:
            raise ValueError(f"Estado {state} fora do intervalo [0, {n_states-1}].")

        # Exploração
        if random.random() < epsilon:
            return random.randrange(n_actions)

        # Greedy
        q_row = self.qtable[state]
        return int(np.argmax(q_row))

    def get_action(self, perturbation_choice: str, local_search_choice: str) -> int:
        """Retorna o id da ação para a combinação (perturbação, busca local)."""
        key = (perturbation_choice, local_search_choice)
        if key not in self.action_map:
            raise ValueError(f"Combinação desconhecida: {key}")
        return self.action_map[key]

    def get_state(self, cost: float, opt_cost: float):
        """
        Mapeia o custo atual para um estado discreto baseado no gap percentual.

        Retorna: (estado, recompensa)
        """
        gap = ((cost - opt_cost) / opt_cost) * 100

        cost = round(cost, 7)
        opt_cost = round(opt_cost, 7)
        gap = round(gap, 7)

        if gap >= 0 and gap <= 2:
            return 0, 75  # Excelente
        elif gap > 2 and gap <= 5:
            return 1, 50  # Bom
        elif gap > 5 and gap <= 10:
            return 2, 25  # Regular
        elif gap > 10:
            return 3, 0  # Ruim
        elif gap < 0:
            return 4, 100  # Melhor que ótimo conhecido

    def _apply_perturbation(self, solution: Solution, pert_type: str) -> Solution:
        """
        Aplica a perturbação especificada à solução.

        Perturbações leves mantêm estrutura da solução.
        Perturbações destrutivas (construtivos) ignoram a solução atual.
        """
        if pert_type == "two_swap":
            return perturbation.Perturbation.random_two_swap(solution)
        elif pert_type == "segment_reverse":
            return perturbation.Perturbation.random_segment_reverse(solution)
        elif pert_type == "random":
            tour, _ = self.constructive.random_tour(self.problem)
            return Solution(tour, self.dist_matrix, is_closed=True)
        elif pert_type == "nearest":
            tour, _ = self.constructive.nearest_neighbor_tour(self.problem)
            return Solution(tour, self.dist_matrix, is_closed=True)
        elif pert_type == "cheapest":
            tour, _ = self.constructive.cheapest_insertion_tour(self.problem)
            return Solution(tour, self.dist_matrix, is_closed=True)
        else:
            raise ValueError(f"Tipo de perturbação desconhecido: {pert_type}")

    def _apply_local_search(self, solution: Solution, ls_type: str) -> Solution:
        """Aplica a busca local especificada à solução."""
        if ls_type == "two_opt":
            return self.local_search.two_opt(solution)
        elif ls_type == "lin_kernighan":
            return self.local_search.lin_kernighan(solution)
        else:
            raise ValueError(f"Tipo de busca local desconhecido: {ls_type}")

    def generate_transition(self, max_iter: int = 50, opt_cost: float = None, out_path: str = "transicao.txt"):
        """
        Gera dados de transição para treinamento do MDP.

        Executa ILS com escolhas aleatórias de (perturbação, busca local),
        registrando tuplas (s, a, r, s') para cada transição.
        """
        # Solução inicial via construtivo aleatório
        constructive_choice = random.choice(["random", "nearest", "cheapest"])
        tour, _ = getattr(
            self.constructive,
            {"random": "random_tour", "nearest": "nearest_neighbor_tour", "cheapest": "cheapest_insertion_tour"}[
                constructive_choice
            ],
        )(self.problem)

        initial_solution = Solution(tour, self.dist_matrix, is_closed=True)

        # Busca local inicial
        ls_solution = self.local_search.two_opt(initial_solution)
        best_solution = ls_solution.copy()

        iter_wto_impr = 0
        output = ""

        # Amostragem proporcional às ações disponíveis
        action_list = list(self.action_map.keys())

        while iter_wto_impr < max_iter:
            # Estado antes da ação
            i_state, _ = self.get_state(ls_solution.cost, opt_cost)

            # Escolha aleatória de uma ação válida
            pert_choice, ls_choice = random.choice(action_list)

            # Aplica perturbação
            perturbed_solution = self._apply_perturbation(ls_solution, pert_choice)

            # Aplica busca local
            new_solution = self._apply_local_search(perturbed_solution, ls_choice)

            # Registra a ação
            action = self.get_action(pert_choice, ls_choice)

            # Critério de aceitação
            if new_solution.cost < best_solution.cost:
                best_solution = new_solution.copy()
                iter_wto_impr = 0
            else:
                iter_wto_impr += 1

            # Atualiza solução corrente (ILS aceita qualquer solução do ótimo local)
            ls_solution = new_solution

            # Estado após a ação
            f_state, reward = self.get_state(new_solution.cost, opt_cost)

            output += f"{i_state} {action} {reward} {f_state}\n"

        with open(out_path, "w") as f:
            f.write(output)

        return best_solution

    def exec_q_table(
        self,
        max_iter: int = 50,
        opt_cost: float = None,
        epsilon: float = 0.0,
    ):
        """
        Executa o ILS guiado pela Q-table treinada.

        A cada iteração, o agente observa o estado (gap) e escolhe
        a melhor ação (perturbação, busca local) segundo a Q-table.
        """
        if opt_cost is None:
            raise ValueError("opt_cost não pode ser None.")

        if self.qtable is None:
            raise ValueError("Q-table ainda não carregada. Use load_qtable() antes.")

        # Solução inicial
        constructive_choice = random.choice(["random", "nearest", "cheapest"])
        if constructive_choice == "random":
            tour, _ = self.constructive.random_tour(self.problem)
        elif constructive_choice == "nearest":
            tour, _ = self.constructive.nearest_neighbor_tour(self.problem)
        else:
            tour, _ = self.constructive.cheapest_insertion_tour(self.problem)

        initial_solution = Solution(tour, self.dist_matrix, is_closed=True)

        # Busca local inicial
        ls_solution = self.local_search.two_opt(initial_solution)
        best_solution = ls_solution.copy()

        iter_wto_impr = 0

        while iter_wto_impr < max_iter:
            # Observa o estado atual
            i_state, _ = self.get_state(ls_solution.cost, opt_cost)

            # Escolhe ação via Q-table (epsilon-greedy)
            action = self.choose_action_from_q(i_state, epsilon=epsilon)
            self.action = action

            # Decodifica a ação
            pert_type, ls_type = self.action_decode[action]

            # Aplica perturbação
            perturbed_solution = self._apply_perturbation(ls_solution, pert_type)

            # Aplica busca local
            new_solution = self._apply_local_search(perturbed_solution, ls_type)

            # Critério de aceitação
            if new_solution.cost < best_solution.cost:
                best_solution = new_solution.copy()
                iter_wto_impr = 0
            else:
                iter_wto_impr += 1

            # Atualiza solução corrente
            ls_solution = new_solution

            print(
                f"[Q-ILS] state={i_state}, action={action} "
                f"({pert_type} + {ls_type}) "
                f"best_cost={best_solution.cost:.4f}"
            )

        return best_solution
