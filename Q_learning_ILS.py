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

    O espaço de ações inclui:
    - Perturbações leves: two_swap, segment_reverse
    - Perturbações destrutivas (construtivos): random, nearest, cheapest

    Combinadas com buscas locais: 2-opt, 3-opt, Lin-Kernighan
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

        # Novo mapeamento de ações: (perturbação, busca local)
        # Perturbações:
        #   - "two_swap": perturbação leve (troca 2 vértices)
        #   - "segment_reverse": perturbação média (reverte segmento)
        #   - "random": perturbação destrutiva (ignora solução atual)
        #   - "nearest": perturbação destrutiva (ignora solução atual)
        #   - "cheapest": perturbação destrutiva (ignora solução atual)
        # Buscas locais: "two_opt", "three_opt", "lin_kernighan"

        self.action_map = {
            # Perturbações leves + buscas locais
            ("two_swap", "two_opt"): 0,
            ("two_swap", "three_opt"): 1,
            ("two_swap", "lin_kernighan"): 2,
            ("segment_reverse", "two_opt"): 3,
            ("segment_reverse", "three_opt"): 4,
            ("segment_reverse", "lin_kernighan"): 5,
            # Perturbações destrutivas (construtivos) + buscas locais
            ("random", "two_opt"): 6,
            ("random", "three_opt"): 7,
            ("random", "lin_kernighan"): 8,
            ("nearest", "two_opt"): 9,
            ("nearest", "three_opt"): 10,
            ("nearest", "lin_kernighan"): 11,
            ("cheapest", "two_opt"): 12,
            ("cheapest", "three_opt"): 13,
            ("cheapest", "lin_kernighan"): 14,
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
                raise ValueError(
                    "Cabeçalho da Q-table inválido (esperado: 'n_states n_actions')."
                )

            n_states, n_actions = map(int, header)

            data = []
            for i in range(n_states):
                line = f.readline()
                if not line:
                    raise ValueError("Número de linhas da Q-table menor que n_states.")
                row_vals = list(map(float, line.strip().split()))
                if len(row_vals) != n_actions:
                    raise ValueError(
                        f"Linha {i+2} da Q-table tem {len(row_vals)} colunas, esperado {n_actions}."
                    )
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
            return 3, 0   # Ruim
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
        elif ls_type == "three_opt":
            return self.local_search.three_opt(solution)
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
        tour, _ = getattr(self.constructive,
                         {"random": "random_tour",
                          "nearest": "nearest_neighbor_tour",
                          "cheapest": "cheapest_insertion_tour"}[constructive_choice])(self.problem)

        initial_solution = Solution(tour, self.dist_matrix, is_closed=True)

        # Busca local inicial
        ls_solution = self.local_search.three_opt(initial_solution)
        best_solution = ls_solution.copy()

        iter_wto_impr = 0
        output = ""

        perturbation_choices = ["two_swap", "segment_reverse", "random", "nearest", "cheapest"]
        local_search_choices = ["two_opt", "three_opt", "lin_kernighan"]

        while iter_wto_impr < max_iter:
            # Estado antes da ação
            i_state, _ = self.get_state(ls_solution.cost, opt_cost)

            # Escolha aleatória de perturbação e busca local
            pert_choice = random.choice(perturbation_choices)
            ls_choice = random.choice(local_search_choices)

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
        ls_solution = self.local_search.three_opt(initial_solution)
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


# Compatibilidade: manter mapeamento antigo para Q-tables existentes
class Q_ILS_Legacy(Q_ILS):
    """Versão compatível com Q-tables treinadas no formato antigo (9 ações)."""

    def __init__(self, problem):
        super().__init__(problem)

        # Mapeamento antigo: (construtivo, busca local)
        self.action_map = {
            ("random", "two_opt"): 0,
            ("random", "three_opt"): 1,
            ("nearest", "two_opt"): 2,
            ("nearest", "three_opt"): 3,
            ("cheapest", "two_opt"): 4,
            ("cheapest", "three_opt"): 5,
            ("random", "lin_kernighan"): 6,
            ("nearest", "lin_kernighan"): 7,
            ("cheapest", "lin_kernighan"): 8,
        }
        self.action_decode = {v: k for k, v in self.action_map.items()}
        self.n_actions = 9

    def _apply_perturbation(self, solution: Solution, pert_type: str) -> Solution:
        """No modo legado, perturbação = perturbação padrão, construtivo após."""
        return perturbation.Perturbation.random_two_swap(solution)

    def exec_q_table(self, max_iter: int = 50, opt_cost: float = None, epsilon: float = 0.0):
        """Execução no modo legado (construtivo + busca local como no código original)."""
        if opt_cost is None:
            raise ValueError("opt_cost não pode ser None.")
        if self.qtable is None:
            raise ValueError("Q-table ainda não carregada.")

        # Solução inicial
        constructive_choice = random.choice(["random", "nearest", "cheapest"])
        if constructive_choice == "random":
            tour, _ = self.constructive.random_tour(self.problem)
        elif constructive_choice == "nearest":
            tour, _ = self.constructive.nearest_neighbor_tour(self.problem)
        else:
            tour, _ = self.constructive.cheapest_insertion_tour(self.problem)

        initial_solution = Solution(tour, self.dist_matrix, is_closed=True)
        ls_solution = self.local_search.three_opt(initial_solution)
        best_solution = ls_solution.copy()

        iter_wto_impr = 0

        while iter_wto_impr < max_iter:
            perturbed_solution = perturbation.Perturbation.random_two_swap(ls_solution)
            i_state, _ = self.get_state(perturbed_solution.cost, opt_cost)

            action = self.choose_action_from_q(i_state, epsilon=epsilon)
            self.action = action

            # Decodifica ação no formato antigo
            if action in (0, 1, 6):
                action_tag = "random"
            elif action in (2, 3, 7):
                action_tag = "nearest"
            else:
                action_tag = "cheapest"

            if action % 3 == 0:
                ls_solution = self.local_search.two_opt(perturbed_solution)
            elif action % 3 == 1:
                ls_solution = self.local_search.three_opt(perturbed_solution)
            else:
                ls_solution = self.local_search.lin_kernighan(perturbed_solution)

            if ls_solution.cost < best_solution.cost:
                best_solution = ls_solution.copy()
                iter_wto_impr = 0
            else:
                iter_wto_impr += 1

            print(
                f"[Q-ILS-Legacy] state={i_state}, action={action} "
                f"best_cost={best_solution.cost:.4f}"
            )

        return best_solution
