import constructive_heuristic
import local_search
import perturbation
from solution import Solution
import numpy as np
import random


# ILS  escolhendo estratégias de busca local e construção
# fazer o exemplo simples, retornando uma tabela de transição


class Q_ILS:
    def __init__(self, problem):
        # distance matrix (0..n-1)
        n = problem.dimension
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i][j] = problem.get_weight(i + 1, j + 1)
        self.problem = problem
        self.dist_matrix = dist_matrix
        self.constructive = constructive_heuristic.ConstructiveHeuristic()
        self.local_search = local_search.LocalSearch()

        # q-learning action mapping
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

        self.action = None
        self.state = None
        self.qtable = None

    def load_qtable(self, path: str = "qtable_output.txt"):
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

    # ------------------------------------------------------------------
    # DADO UM ESTADO, QUAL MELHOR AÇÃO? (ε-greedy)
    # ------------------------------------------------------------------
    def choose_action_from_q(self, state: int, epsilon: float = 0.0) -> int:
        """
        Dado um estado (índice inteiro), seleciona uma ação usando a Q-table:
        - com probabilidade epsilon: ação aleatória (exploração)
        - caso contrário: argmax_a Q[state, a] (aproveitamento)
        """
        if self.qtable is None:
            raise ValueError("Q-table ainda não carregada. Use load_qtable() antes.")

        n_states, n_actions = self.qtable.shape

        if state < 0 or state >= n_states:
            raise ValueError(f"Estado {state} fora do intervalo [0, {n_states-1}].")

        # exploração
        if random.random() < epsilon:
            return random.randrange(n_actions)

        # greedy: pega o índice da maior Q[state, a]
        q_row = self.qtable[state]
        return int(np.argmax(q_row))

    def get_action(self, constructive_choice, local_search_choice):
        """Retorna o id do estado (0..5) para a combinação escolhida."""
        key = (constructive_choice, local_search_choice)
        if key not in self.action_map:
            raise ValueError(f"Combinação desconhecida: {key}")
        return self.action_map[key]

    def get_state(self, cost, opt_cost):

        gap = ((cost - opt_cost) / opt_cost) * 100

        # aproxime o valor do custo, do ótimo e do gap para evitar problemas de precisão
        cost = round(cost, 7)
        opt_cost = round(opt_cost, 7)
        gap = round(gap, 7)

        if gap >= 0 and gap <= 2:
            return 0, 75  # state 0, reward 75
        elif gap > 2 and gap <= 5:
            return 1, 50  # state 1, reward 50
        elif gap > 5 and gap <= 10:
            return 2, 25  # state 2, reward 25
        elif gap > 10:
            return 3, 0  # state 3, reward 0
        elif gap < 0:
            print("Found solution better than optimal!")
            print(f"Cost: {cost}, Opt Cost: {opt_cost}, Gap: {gap}%")
            return 4, 100  # state 4 (better than opt), reward 100
        else:
            print("ERROR: Gap value out of expected range")

    def generate_transition(self, max_iter=50, opt_cost=None, out_path="transicao.txt"):

        # choose randomly one of the constructive heuristics
        constructive_choice = random.choice(["random", "nearest", "cheapest"])
        if constructive_choice == "random":
            tour, tour_cost = self.constructive.random_tour(self.problem)
        elif constructive_choice == "nearest":
            tour, tour_cost = self.constructive.nearest_neighbor_tour(self.problem)
        else:
            tour, tour_cost = self.constructive.cheapest_insertion_tour(self.problem)

        initial_solution = Solution(tour, self.dist_matrix, is_closed=True)

        ls_solution = self.local_search.three_opt(initial_solution)
        best_solution = ls_solution.copy()

        iter_wto_impr = 0
        output = ""
        while iter_wto_impr < max_iter:
            # perturbation
            perturbed_solution = perturbation.Perturbation.random_two_swap(ls_solution)
            i_state, _ = self.get_state(
                perturbed_solution.cost, opt_cost
            )  # map solution to initial state just before action

            local_search_choice = random.choice(
                ["two_opt", "three_opt", "lin_kernighan"]
            )
            if local_search_choice == "two_opt":
                ls_solution = self.local_search.two_opt(perturbed_solution)
            elif local_search_choice == "three_opt":
                ls_solution = self.local_search.three_opt(perturbed_solution)
            else:
                ls_solution = self.local_search.lin_kernighan(perturbed_solution)

            # acceptance criterion
            # registra o estado escolhido (0..5)
            self.action = self.get_action(constructive_choice, local_search_choice)
            #print(
            #    f"action: {self.action} ({constructive_choice} + {local_search_choice})"
            #)

            if ls_solution.cost < best_solution.cost:
                best_solution = ls_solution.copy()
                iter_wto_impr = 0
            else:
                iter_wto_impr += 1

            f_state, reward = self.get_state(
                ls_solution.cost, opt_cost
            )  # map solution after action (construtivo + busca local) to final state just after action

            #print(
            #    f"state before action: {i_state}, action: {self.action}, reward: {reward}, state after action: {f_state}"
            #)
            output += f"{i_state} {self.action} {reward} {f_state}\n"

        with open(out_path, "w") as f:
            f.write(output)

        # print("Best Solution Cost after ILS:", best_solution.cost)
        return best_solution

        # ----------------------------------------------------

    def exec_q_table(
        self,
        max_iter: int = 50,
        opt_cost: float = None,
        epsilon: float = 0.0,
    ):
        """
        Executa um ILS onde, dado o estado da solução, escolhemos a MELHOR
        ação via Q-table, no formato do qtable_output.txt.

        - Estado (s) é dado por get_state(cost, opt_cost)
        - Ação (a) é um inteiro 0..5 (mesmo mapeamento usado no treino)
        """

        if opt_cost is None:
            raise ValueError(
                "opt_cost não pode ser None (preciso do ótimo para calcular o estado)."
            )

        if self.qtable is None:
            raise ValueError("Q-table ainda não carregada. Use load_qtable() antes.")

        # 1) Gera uma solução inicial com QUALQUER construtivo
        constructive_choice = random.choice(["random", "nearest", "cheapest"])
        if constructive_choice == "random":
            tour, tour_cost = self.constructive.random_tour(self.problem)
        elif constructive_choice == "nearest":
            tour, tour_cost = self.constructive.nearest_neighbor_tour(self.problem)
        else:  # "cheapest"
            tour, tour_cost = self.constructive.cheapest_insertion_tour(self.problem)

        initial_solution = Solution(tour, self.dist_matrix, is_closed=True)

        # 2) Aplica uma busca local inicial (como você já fazia)
        ls_solution = self.local_search.three_opt(initial_solution)
        best_solution = ls_solution.copy()

        iter_wto_impr = 0

        # 3) Loop principal do ILS
        while iter_wto_impr < max_iter:
            # 3.1) Perturba a solução atual
            perturbed_solution = perturbation.Perturbation.random_two_swap(ls_solution)

            # 3.2) Calcula o estado da solução perturbada
            i_state, _ = self.get_state(perturbed_solution.cost, opt_cost)

            # 3.3) Escolhe ação CONSULTANDO A Q-TABLE
            action = self.choose_action_from_q(i_state, epsilon=epsilon)
            self.action = action  # guarda se quiser inspecionar depois

            # # (Opcional: apenas para log, decodificar "tipo" de ação)
            if action in (0, 1):
                action_tag = "random"
            elif action in (2, 3):
                action_tag = "nearest"
            else:
                action_tag = "cheapest"

            # 3.4) A partir da ação, escolhemos o TIPO de busca local
            # qual das tres buscas locais usar?
            if action % 3 == 0:
                action_tag += " + two_opt"
                ls_solution = self.local_search.two_opt(perturbed_solution)
            elif action % 3 == 1:
                action_tag += " + three_opt"
                ls_solution = self.local_search.three_opt(perturbed_solution)
            else:
                action_tag += " + lin_kernighan"
                ls_solution = self.local_search.lin_kernighan(perturbed_solution)

            # 3.5) Critério de aceitação (mesmo do ILS original)
            if ls_solution.cost < best_solution.cost:
                best_solution = ls_solution.copy()
                iter_wto_impr = 0
            else:
                iter_wto_impr += 1

            # # 3.6) (Opcional) Estado após ação, para logging
            # f_state, reward = self.get_state(ls_solution.cost, opt_cost)

            print(
                f"[Q-ILS] state={i_state}, action={action} "
                f"action={action_tag} "
                f"best_cost={best_solution.cost:.4f}"
            )

        return best_solution
