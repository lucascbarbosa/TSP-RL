# local_search.py
from solution import Solution


class LocalSearch:
    def __init__(self):
        pass

    def two_opt(self, solution: Solution) -> Solution:
        """
        Recebe uma Solution (rota fechada) e devolve uma nova Solution melhorada
        pelo vizinhança 2-opt.
        """
        best_tour = solution.tour[:]  # [c0, ..., c_{n-1}, c0]
        best_cost = solution.cost
        dist_matrix = solution.dist_matrix

        # como a rota é fechada, o último == primeiro
        # número de cidades "reais":
        n = len(best_tour) - 1

        improved = True
        while improved:
            improved = False
            # fixamos o primeiro nó para evitar rota equivalente rodada
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue  # pula adjacentes

                    new_tour = best_tour[:]
                    # 2-opt: reversão do segmento [i, j-1]
                    new_tour[i:j] = reversed(best_tour[i:j])

                    new_cost = Solution.compute_cost_static(
                        new_tour, dist_matrix, is_closed=True
                    )

                    if new_cost < best_cost:
                        best_tour = new_tour
                        best_cost = new_cost
                        improved = True

        return Solution(best_tour, dist_matrix, is_closed=True)

    def three_opt(self, solution: Solution) -> Solution:
        """
        Busca local 3-opt sobre uma Solution (rota fechada).

        Estratégia:
        - Tour fechado: [c0, ..., c_{n-1}, c0]
        - Escolhemos índices 1 <= i < j < k <= n-1 como pontos de corte.
        - Quebramos o tour em quatro partes: A | B | C | D, onde:
              A = tour[0:i]
              B = tour[i:j]
              C = tour[j:k]
              D = tour[k:]
          (note que D termina em c0)
        - Geramos diversas recombinações 3-opt, avaliamos todas e aceitamos
          a primeira melhora (first-improvement) em cada varredura.
        """
        best_tour = solution.tour[:]  # [c0, ..., c_{n-1}, c0]
        best_cost = solution.cost
        dist_matrix = solution.dist_matrix

        # número de cidades reais (sem contar a repetição final)
        n = len(best_tour) - 1

        improved = True
        while improved:
            improved = False

            # i, j, k são índices de corte na parte "interna" do tour
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        # segmentação
                        A = best_tour[0:i]
                        B = best_tour[i:j]
                        C = best_tour[j:k]
                        D = best_tour[k:]  # inclui o c0 final

                        # Gera candidatos 3-opt (alguns são 2-opt, mas ok)
                        candidates = []

                        # 1) A + B^R + C + D
                        candidates.append(A + B[::-1] + C + D)

                        # 2) A + B + C^R + D
                        candidates.append(A + B + C[::-1] + D)

                        # 3) A + B^R + C^R + D
                        candidates.append(A + B[::-1] + C[::-1] + D)

                        # 4) A + C + B + D
                        candidates.append(A + C + B + D)

                        # 5) A + C^R + B + D
                        candidates.append(A + C[::-1] + B + D)

                        # 6) A + C + B^R + D
                        candidates.append(A + C + B[::-1] + D)

                        # 7) A + C^R + B^R + D
                        candidates.append(A + C[::-1] + B[::-1] + D)

                        improved_here = False

                        for new_tour in candidates:
                            # garante que está fechado (por segurança)
                            if new_tour[0] != new_tour[-1]:
                                new_tour[-1] = new_tour[0]

                            new_cost = Solution.compute_cost_static(
                                new_tour, dist_matrix, is_closed=True
                            )

                            if new_cost < best_cost - 1e-12:
                                best_cost = new_cost
                                best_tour = new_tour
                                improved = True
                                improved_here = True
                                break  # sai do loop de candidatos

                        if improved_here:
                            break  # sai do loop de k
                    if improved:
                        break  # sai do loop de j
                if improved:
                    break  # sai do loop de i

        return Solution(best_tour, dist_matrix, is_closed=True)

    def lin_kernighan(self, solution: Solution, max_depth: int = 4) -> Solution:
        """
        Heurística de Lin–Kernighan (versão simplificada) usando cadeias
        de movimentos 2-opt de profundidade variável.

        - A solução é uma rota FECHADA: [c1, ..., cn, c1].
        - A cada iteração, tentamos construir cadeias de 2-opt começando
          em vários pontos da rota.
        - Cada passo da cadeia deve manter o ganho acumulado POSITIVO
          (ideia central do LK).
        - Escolhemos a melhor cadeia encontrada e repetimos o processo
          até não haver mais melhoria.
        """
        dist_matrix = solution.dist_matrix
        current_tour = solution.tour[:]
        current_cost = solution.cost
        n = len(current_tour) - 1  # número de cidades "reais"

        improved = True
        while improved:
            improved = False
            best_global_gain = 0.0
            best_global_tour = current_tour

            # tenta iniciar a cadeia em cada posição interna da rota
            for start_idx in range(1, n):
                used_positions = {start_idx}
                best_tour_local, best_gain_local = self._lk_variable_depth(
                    current_tour,
                    start_idx,
                    used_positions,
                    depth=0,
                    max_depth=max_depth,
                    dist_matrix=dist_matrix,
                    current_gain=0.0,
                )

                # ganho total relativo à solução corrente
                total_gain = best_gain_local  # current_gain inicial é 0

                if total_gain > best_global_gain + 1e-12:
                    best_global_gain = total_gain
                    best_global_tour = best_tour_local

            # aplica a melhor cadeia encontrada, se houver
            if best_global_gain > 1e-12:
                current_tour = best_global_tour
                current_cost = Solution.compute_cost_static(
                    current_tour, dist_matrix, is_closed=True
                )
                improved = True

        return Solution(current_tour, dist_matrix, is_closed=True)

    # ---------------------------------------------------------
    # Função recursiva: profundidade variável de cadeias 2-opt
    # ---------------------------------------------------------
    def _lk_variable_depth(
        self,
        tour,
        last_pos,
        used_positions,
        depth,
        max_depth,
        dist_matrix,
        current_gain,
    ):
        """
        Explora recursivamente cadeias de movimentos 2-opt.

        tour          : tour fechado atual
        last_pos      : índice da última posição usada na cadeia (1..n-1)
        used_positions: posições já usadas na cadeia (para não repetir)
        depth         : profundidade atual da recursão
        max_depth     : profundidade máxima permitida
        current_gain  : ganho acumulado até o estado atual
        """
        n = len(tour) - 1  # número de cidades reais

        # melhor ganho total (desde o início da cadeia) a partir deste estado
        best_gain = current_gain
        best_tour = tour

        if depth >= max_depth:
            return best_tour, best_gain

        for j in range(1, n):
            if j == last_pos:
                continue
            if abs(j - last_pos) == 1:
                continue  # não corta arestas adjacentes
            if j in used_positions:
                continue  # evita usar a mesma posição duas vezes na cadeia

            move_gain = self._two_opt_gain(tour, last_pos, j, dist_matrix)
            new_total_gain = current_gain + move_gain

            # mantemos apenas cadeias com ganho acumulado positivo
            if new_total_gain <= 0:
                continue

            # aplica o 2-opt correspondente
            new_tour = self._apply_two_opt(tour, last_pos, j)

            # atualiza melhor solução desta subárvore
            if new_total_gain > best_gain + 1e-12:
                best_gain = new_total_gain
                best_tour = new_tour

            # tenta aprofundar mais a cadeia
            new_used = set(used_positions)
            new_used.add(j)

            deeper_tour, deeper_gain = self._lk_variable_depth(
                new_tour,
                j,
                new_used,
                depth + 1,
                max_depth,
                dist_matrix,
                new_total_gain,
            )

            if deeper_gain > best_gain + 1e-12:
                best_gain = deeper_gain
                best_tour = deeper_tour

        return best_tour, best_gain

    # ---------------------------------------------------------
    # Ganho de um movimento 2-opt (compatível com seu two_opt)
    # ---------------------------------------------------------
    def _two_opt_gain(self, tour, i, j, dist_matrix) -> float:
        """
        Calcula o ganho (redução de custo) ao aplicar um 2-opt que
        reverte o segmento tour[i:j].

        Tour é FECHADO: [c0, ..., cn-1, c0]
        i, j são índices em 1..n-1, com i != j.
        """
        if i > j:
            i, j = j, i

        # arestas antes:
        # ... - a (=tour[i-1]) - b (=tour[i]) - ... - c (=tour[j-1]) - d (=tour[j]) - ...
        a = tour[i - 1]
        b = tour[i]
        c = tour[j - 1]
        d = tour[j]

        a_, b_, c_, d_ = a - 1, b - 1, c - 1, d - 1

        # removemos (a,b) e (c,d); adicionamos (a,c) e (b,d)
        removed = dist_matrix[a_, b_] + dist_matrix[c_, d_]
        added = dist_matrix[a_, c_] + dist_matrix[b_, d_]

        return removed - added  # > 0 => melhora

    # ---------------------------------------------------------
    # Aplica o movimento 2-opt na rota
    # ---------------------------------------------------------
    def _apply_two_opt(self, tour, i, j):
        """
        Aplica em 'tour' a operação 2-opt entre as posições i e j,
        revertendo o segmento tour[i:j]. Retorna uma NOVA lista.
        """
        if i > j:
            i, j = j, i

        new_tour = tour[:]
        new_tour[i:j] = reversed(tour[i:j])

        # garante que a rota permanece fechada
        if new_tour[0] != new_tour[-1]:
            new_tour[-1] = new_tour[0]

        return new_tour
