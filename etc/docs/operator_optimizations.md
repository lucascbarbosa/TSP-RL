# Otimizações dos Operadores TSP

Descrição das otimizações implementadas nos operadores do TSP-RL.

**Período coberto:**
- Parte I (commits `0a30d30` a `de50e65`): Estruturas de dados e algoritmos
- Parte II (commits após `de50e65`): Numba JIT, GRASP e redesenho de ações

## 1. Buscas Locais (2-opt)

### 1.1. Estado Inicial

Antes das otimizações, existia apenas uma implementação de 2-opt (`two_opt_full`) com complexidade O(n²) por iteração, usando estratégia best-improvement. Para instâncias pequenas (n < 50), o desempenho era aceitável, mas para instâncias maiores tornava-se um gargalo significativo.

### 1.2. Variantes Introduzidas

#### `two_opt_nn` — Neighbor Lists

Restringe a busca a um subconjunto de k vizinhos mais próximos de cada cidade, reduzindo a complexidade de O(n²) para O(n·k) por iteração.

**Funcionamento:**
- Para cada cidade `a`, considera apenas trocas onde a nova aresta `(a, c)` conecta `a` a uma das suas k cidades mais próximas
- Mantém array de posições `pos[city]` para localizar cidades no tour em O(1)
- Estratégia: best-improvement (avalia todas as trocas válidas e aplica a melhor)

**Estrutura de dados auxiliar — Neighbor Lists:**
```
neighbors[i] = [c1, c2, ..., ck]  # k vizinhos mais próximos de i, ordenados por distância
```

Speedup reportado: ~5.7x para n=200.

#### `two_opt_dlb` — Don't Look Bits

Combina neighbor lists com Don't Look Bits (DLB), uma técnica que marca cidades como "não olhar" quando não produziram melhorias recentes, evitando verificações redundantes.

**Funcionamento:**
- `dlb[city] = True` indica que a cidade deve ser ignorada
- Quando uma melhoria é encontrada, limpa-se o DLB das cidades afetadas
- Estratégia: first-improvement (aplica a melhoria assim que encontrada)

Speedup reportado: ~360x para n=300, com tradeoff de qualidade de ~1-4%.

### 1.3. Seleção Adaptativa

A função `two_opt_adaptive` (alias padrão de `two_opt`) seleciona automaticamente a melhor variante baseada no tamanho da instância:

| n | Variante | Justificativa |
|---|----------|---------------|
| < 40 | `two_opt_full` | Overhead de neighbor lists não compensa |
| 40 ≤ n < 80 | `two_opt_nn` | Bom equilíbrio qualidade/velocidade |
| ≥ 80 | `two_opt_dlb` | Máxima velocidade para instâncias grandes |

Os thresholds foram definidos empiricamente via benchmarks.

### 1.4. Otimização na Construção de Neighbor Lists

A função `_build_neighbor_lists` foi otimizada usando `np.argpartition` em vez de `np.argsort`:

- **Antes:** O(n log n) por cidade para ordenar todas as distâncias
- **Depois:** O(n + k log k) por cidade — `argpartition` obtém os k menores em O(n), depois ordena apenas esses k elementos

### 1.5. Parâmetro k Flexível

O parâmetro `k` aceita dois formatos:
- **Inteiro:** valor absoluto (ex: `k=20` → 20 vizinhos)
- **Float em (0,1):** proporção de n (ex: `k=0.5` → 50% das cidades)

Default: `k=0.5`.

### 1.6. Cache de Neighbor Lists

As variantes `two_opt_nn`, `two_opt_dlb` e `two_opt_adaptive` aceitam um parâmetro opcional `neighbors` para reutilizar listas pré-computadas. Isso permite que chamadas repetidas sobre a mesma instância (comum em ILS/metaheurísticas) evitem recalcular as neighbor lists a cada invocação.

### 1.7. Numba JIT Kernels (commit `08d14cf`)

Os loops críticos de busca local foram reescritos como kernels Numba JIT-compiled, eliminando o overhead do interpretador Python.

**Kernels implementados:**

| Kernel | Função |
|--------|--------|
| `_two_opt_delta_nb` | Cálculo de delta para uma troca 2-opt |
| `_two_opt_full_core` | Loop principal do 2-opt full (best-improvement) |
| `_two_opt_nn_core` | 2-opt com neighbor lists |
| `_two_opt_dlb_core` | 2-opt com neighbor lists + don't look bits |

**Características:**
- Decorados com `@njit(cache=True)` para compilação persistente
- Evitam recompilação entre execuções (cache em disco)
- Trabalham diretamente com arrays NumPy (sem overhead de objetos Python)

**Speedup medido:** ~20-100x nos loops críticos, dependendo do tamanho da instância.

**Exemplo de ganho no cálculo de delta:**
```python
# Antes (Python puro)
def _two_opt_delta(tour, i, j, dist):
    a, b = tour[i - 1], tour[i]
    c, d = tour[j - 1], tour[j]
    return dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]

# Depois (Numba JIT) - ~50x mais rápido
@njit(cache=True)
def _two_opt_delta_nb(tour, i, j, dist):
    a = tour[i - 1] - 1  # conversão para 0-based
    b = tour[i] - 1
    c = tour[j - 1] - 1
    d = tour[j] - 1
    return dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
```

## 2. Construtivos

### 2.1. `cheapest_insertion` Vetorizado

O construtivo cheapest insertion foi completamente reescrito usando operações vetorizadas do NumPy.

**Antes (versão escalar):**
```python
for city in unvisited:
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        delta = dist(a, city) + dist(city, b) - dist(a, b)
        # atualiza melhor se delta < best_delta
```
- Dois loops aninhados com chamadas individuais a `get_weight`
- Constantes altas devido ao overhead de Python

**Depois (versão vetorizada):**
```python
# Para m cidades não visitadas e k arestas no tour parcial:
dist_a_to_c = dist[a_indices][:, unvisited_indices]  # (k, m)
dist_c_to_b = dist[unvisited_indices][:, b_indices].T  # (k, m)
dist_a_to_b = dist[a_indices, b_indices]  # (k,)
deltas = dist_a_to_c + dist_c_to_b - dist_a_to_b[:, np.newaxis]  # (k, m)
```
- Broadcasting numpy calcula todos os deltas simultaneamente
- Uma única chamada a `np.argmin` encontra a melhor inserção

**Speedups medidos:**

| n | Tempo antes | Tempo depois | Speedup |
|---|-------------|--------------|---------|
| 100 | 309ms | 10ms | 31x |
| 200 | 3.7s | 47ms | 78x |
| 500 | 70s | 437ms | 159x |

A complexidade assintótica permanece O(n³), mas as constantes são drasticamente reduzidas pelo uso de operações SIMD do NumPy.

### 2.2. GRASP (commit `b594560`)

Adicionado o construtivo GRASP (Greedy Randomized Adaptive Search Procedure), que constrói soluções incrementalmente usando uma **Restricted Candidate List (RCL)**.

**Funcionamento:**
1. A cada passo, calcula distâncias para todas as cidades não visitadas
2. Define threshold: `min_dist + α × (upper_bound - min_dist)`
3. Seleciona aleatoriamente da RCL (cidades com distância ≤ threshold)

**Parâmetro α (greediness):**

| α | Comportamento |
|---|--------------|
| 0.0 | Guloso puro (equivalente a nearest neighbor) |
| 0.03 | Quase-guloso, mínima diversificação |
| 0.1 | Balanço entre qualidade e diversificação |
| 0.3 | Mais aleatório, boa diversificação |
| 1.0 | Completamente aleatório |

**Uso no DQN-ILS:**
O GRASP é usado como operador de restart/perturbação "construtiva" — gera uma nova solução do zero quando o agente decide diversificar. Três variantes com α fixo (0.03, 0.1, 0.3) são disponibilizadas como ações distintas.

## 3. Perturbações

As perturbações (`two_swap`, `segment_reverse`) não sofreram otimizações de performance significativas neste período, pois já eram O(1) ou O(n) e não representavam gargalos. Houve apenas correções de corretude para garantir comportamento consistente.

## 4. Espaço de Ações (commit `d603e67`)

O espaço de ações do MDP foi redesenhado de 21 para 16 ações, baseado em análise empírica de modelos treinados.

### 4.1. Remoções

**`two_opt_nn` (busca local):**
- Uso observado: ~4% das ações
- Justificativa: `two_opt_dlb` é mais rápido com qualidade similar; `two_opt_full` é melhor para intensificação

**`random` (construtivo):**
- Substituído por `grasp_0.3` que oferece diversificação controlada

### 4.2. Adições

Três variantes de GRASP com α fixo:

| Ação | α | Descrição |
|------|---|-----------|
| `grasp_0.03` | 0.03 | Quase-guloso, diversificação mínima |
| `grasp_0.1` | 0.1 | Balanço qualidade/diversificação |
| `grasp_0.3` | 0.3 | Mais aleatório (substitui `random`) |

### 4.3. Estrutura Final (16 ações)

| Faixa | Perturbação | Buscas Locais |
|-------|-------------|---------------|
| 0-2 | two_swap | dlb, full, LK |
| 3-4 | segment_reverse | dlb, full |
| 5-7 | nearest | dlb, full, LK |
| 8-9 | cheapest | dlb, full |
| 10-11 | grasp_0.03 | dlb, full |
| 12-13 | grasp_0.1 | dlb, full |
| 14-15 | grasp_0.3 | dlb, full |

### 4.4. Impacto na Rede Neural

- **Dimensão do estado:** 45 → 35 (com history_len=2) ou 19 (history_len=1)
- **Rede mais compacta:** Menos parâmetros, convergência mais rápida
- **Remoção de ruído:** Ações pouco úteis não competem por probabilidade

## Resumo de Impacto

### Parte I (estruturas de dados e algoritmos)

| Componente | Melhoria Principal |
|------------|-------------------|
| 2-opt (instâncias grandes) | Até ~360x mais rápido com DLB |
| Neighbor lists build | O(n log n) → O(n + k log k) via argpartition |
| Cheapest insertion | 30-160x mais rápido via vetorização numpy |
| Metaheurísticas (ILS) | Cache de neighbor lists economiza ~8s por instância (n=1000) |

### Parte II (Numba JIT, GRASP, espaço de ações)

| Componente | Melhoria Principal |
|------------|-------------------|
| 2-opt (Numba JIT) | 20-100x adicional nos loops críticos |
| GRASP construtivo | Diversificação controlada com parâmetro α |
| Espaço de ações | 21 → 16 ações, rede mais compacta |
| Estado do MDP | 45 → 19 dimensões (history_len=1) |
