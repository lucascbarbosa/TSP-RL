# Q-ILS: Q-Learning para Iterated Local Search no TSP

Framework que integra **Q-Learning** ao **Iterated Local Search (ILS)** para resolver o Problema do Caixeiro Viajante (TSP).

## Ideia Central

No ILS tradicional, a cada iteração aplicamos uma perturbação seguida de uma busca local. O Q-ILS permite que um **agente de RL aprenda qual par (perturbação, busca local)** usar em função do estado atual da solução.

```
┌─────────────────────────────────────────────────────────────┐
│                        Loop do Q-ILS                        │
├─────────────────────────────────────────────────────────────┤
│  1. Observar estado s (gap percentual)                      │
│  2. Consultar Q-table: a = argmax Q(s, a)                   │
│  3. Decodificar ação: a → (perturbação, busca local)        │
│  4. Aplicar perturbação à solução atual                     │
│  5. Aplicar busca local                                     │
│  6. Atualizar melhor solução se houve melhora               │
└─────────────────────────────────────────────────────────────┘
```

## Modelagem MDP

### Estados (5)

O estado é definido pelo **gap percentual** entre a solução atual e o ótimo conhecido:

| Estado    | Gap (%)     | Recompensa | Interpretação          |
|-----------|-------------|------------|------------------------|
| EXCELLENT | 0 – 2       | 75         | Excelente              |
| GOOD      | 2 – 5       | 50         | Bom                    |
| REGULAR   | 5 – 10      | 25         | Regular                |
| POOR      | > 10        | 0          | Ruim                   |
| BETTER    | < 0         | 100        | Melhor que o ótimo     |

### Ações (8)

Cada ação é um par **(perturbação, busca local)**:

| Ação                 | Perturbação      | Busca Local    | Uso                      |
|----------------------|------------------|----------------|--------------------------|
| TWO_SWAP_2OPT        | two_swap         | 2-opt          | Refinamento leve         |
| TWO_SWAP_LK          | two_swap         | Lin-Kernighan  | Refinamento moderado     |
| SEGMENT_REVERSE_2OPT | segment_reverse  | 2-opt          | Perturbação média        |
| SEGMENT_REVERSE_LK   | segment_reverse  | Lin-Kernighan  | Perturbação + intensif.  |
| RANDOM_2OPT          | random           | 2-opt          | Restart rápido           |
| NEAREST_2OPT         | nearest          | 2-opt          | Restart de qualidade     |
| CHEAPEST_2OPT        | cheapest         | 2-opt          | Restart alta qualidade   |
| NEAREST_LK           | nearest          | Lin-Kernighan  | Restart + intensificação |

**Tipos de perturbação:**
- **Leves** (`two_swap`, `segment_reverse`): modificam levemente a solução atual
- **Destrutivas** (`random`, `nearest`, `cheapest`): ignoram a solução atual e constroem uma nova do zero

**Buscas locais:**
- **2-opt**: O(n²), rápido e eficiente
- **Lin-Kernighan**: Cadeias de 2-opt (depth=2), mais intensivo

## Estrutura do Projeto

```
TSP-RL/
├── src/                         # Código fonte principal
│   ├── tsp/                     # Componentes core do TSP
│   │   ├── solution.py          # Representação de soluções
│   │   ├── instance.py          # TSPInstance, TSPDataset
│   │   ├── local_search.py      # two_opt(), lin_kernighan() + LOCAL_SEARCHES
│   │   ├── perturbation.py      # two_swap(), segment_reverse() + PERTURBATIONS
│   │   └── constructive.py      # random_tour(), nearest_neighbor(), cheapest_insertion() + CONSTRUCTIVES
│   ├── ils/                     # Framework ILS
│   │   └── q_ils.py             # QILS, State, Action, N_STATES, N_ACTIONS
│   └── rl/                      # Reinforcement Learning
│       ├── q_table.py           # Classe QTable
│       ├── q_learning.py        # Q-Learning (value iteration)
│       ├── mdp.py               # Construção do MDP
│       └── transition.py        # Carga de transições
├── scripts/
│   ├── generate_splits.py       # Gera splits train/test
│   ├── train_transitions.py     # Geração de transições (paralelo)
│   ├── train_qtable.py          # Treinamento de Q-tables
│   ├── evaluate.py              # Avaliação com Q-table treinada
│   ├── clear.sh                 # Remove arquivos gerados
│   └── exemplos.sh              # Pipeline de exemplo (subconjunto pequeno)
├── utils/
│   └── plot.py                  # Visualizações (convergência, heatmap, violin plots)
├── data/
│   ├── EUC_2D.json              # Instâncias euclidianas
│   ├── ATT.json                 # Instâncias ATT
│   ├── GEO.json                 # Instâncias geográficas
│   ├── splits.json              # Divisão treino/teste (gerado)
│   ├── train/                   # Dados de transição (gerado)
│   ├── q_tables/                # Q-tables treinadas (gerado)
│   ├── plots/                   # Gráficos de treinamento (gerado)
│   └── results/                 # Resultados de avaliação (gerado)
└── etc/
    ├── plantuml/                # Diagramas do framework
    └── slides/                  # Apresentação
```

## Quickstart

Para rodar o pipeline completo em um subconjunto pequeno (tipos EUC_2D/ATT/GEO, tamanhos 10/20, 100 instâncias por par):

```bash
./scripts/exemplos.sh
```

Para limpar todos os arquivos gerados:

```bash
./scripts/clear.sh          # dry-run (mostra o que seria deletado)
./scripts/clear.sh --force  # deleta de fato
```

### Paralelismo

Os scripts `train_transitions.py` e `evaluate.py` usam por padrão `cpu_count - 2` workers. Para sobrescrever:

```bash
python scripts/train_transitions.py ... --workers 8
python scripts/evaluate.py ... --workers 8
```

## Fluxo de Execução

### 1. Geração de Transições (offline)

Executa ILS com escolhas aleatórias de (perturbação, busca local), registrando tuplas `(s, a, r, s')`:

```bash
python scripts/train_transitions.py --help
python scripts/train_transitions.py --split_path data/splits.json --dataset_path data/EUC_2D.json --output_dir data/train/EUC_2D --limit 3333
python scripts/train_transitions.py --split_path data/splits.json --dataset_path data/ATT.json --output_dir data/train/ATT --limit 3333
python scripts/train_transitions.py --split_path data/splits.json --dataset_path data/GEO.json --output_dir data/train/GEO --limit 3333
```

### 2. Treinamento da Q-table

Constrói o MDP a partir das transições e treina via Q-Learning:

```bash
python scripts/train_qtable.py --help
python scripts/train_qtable.py --types EUC_2D ATT GEO --sizes 10 20 30
```

### 3. Avaliação

Executa o ILS guiado pela Q-table em instâncias de teste:

```bash
python scripts/evaluate.py --help
python scripts/evaluate.py --types EUC_2D ATT GEO --sizes 10 20 30 --workers 16
```

Ao final da avaliação, são gerados automaticamente **violin plots** mostrando a distribuição dos gaps por tamanho de instância (um gráfico por tipo). Use `--no-plots` para desabilitar.

## Componentes

Os operadores são funções standalone com registros `Dict[str, Callable]` para acesso dinâmico:

### Heurísticas Construtivas (`CONSTRUCTIVES`)

- **random**: tour aleatório
- **nearest**: adiciona sempre a cidade mais próxima
- **cheapest**: insere na posição de menor aumento de custo

### Buscas Locais (`LOCAL_SEARCHES`)

- **two_opt**: troca 2 arestas; usa seleção adaptativa por padrão
- **lin_kernighan**: cadeias de 2-opt com profundidade limitada (depth=2)

#### Seleção Adaptativa do 2-opt

O `two_opt` no registry usa seleção adaptativa que escolhe automaticamente a melhor variante baseado no tamanho da instância:

| Tamanho (n) | Variante interna | Complexidade | Motivo |
|-------------|------------------|--------------|--------|
| n < 40      | `two_opt_full`   | O(n²)        | Overhead de neighbor lists não compensa |
| 40 ≤ n < 80 | `two_opt_nn`     | O(n·k)       | Bom equilíbrio qualidade/velocidade |
| n ≥ 80      | `two_opt_dlb`    | O(n·k) + DLB | Máxima velocidade, ~1-4% de perda de qualidade |

O parâmetro `k` (número de vizinhos) aceita:
- **int**: usado diretamente (ex: `k=20` = 20 vizinhos)
- **float em (0,1)**: interpretado como proporção de n (ex: `k=0.5` = 50% das cidades)

Padrão: `k=0.5` (50% das cidades como vizinhos).

As variantes individuais (`two_opt_full`, `two_opt_nn`, `two_opt_dlb`) estão disponíveis via import direto se necessário.

### Perturbações (`PERTURBATIONS`)

- **two_swap**: troca dois vértices aleatórios
- **segment_reverse**: reverte um segmento aleatório do tour

## Uso Básico

```python
from src import QILS, TSPInstance

# Carregar instância
problem = TSPInstance("data/EUC_2D.json", instance_id=123)

# Calcular custo ótimo
opt_tour = problem.opt_tour
opt_cost = sum(
    problem.get_weight(opt_tour[i], opt_tour[(i + 1) % len(opt_tour)])
    for i in range(len(opt_tour))
)

# Criar solver e carregar Q-table
solver = QILS(problem)
solver.load_q_table("data/q_tables/EUC_2D/instance_size_50.txt")

# Executar
best_solution = solver.run(max_iter=50, opt_cost=opt_cost, epsilon=0.1)

print(f"Custo: {best_solution.cost}")
print(f"Gap: {((best_solution.cost - opt_cost) / opt_cost) * 100:.2f}%")
```

### Usando os Registros Diretamente

```python
from src import LOCAL_SEARCHES, PERTURBATIONS, CONSTRUCTIVES, N_STATES, N_ACTIONS

# Número de operadores disponíveis
print(f"Buscas locais: {len(LOCAL_SEARCHES)}")      # 2
print(f"Perturbações: {len(PERTURBATIONS)}")        # 2
print(f"Construtivas: {len(CONSTRUCTIVES)}")        # 3
print(f"Estados: {N_STATES}, Ações: {N_ACTIONS}")   # 5, 8

# Chamar operadores pelo nome
from src import Solution
tour, cost = CONSTRUCTIVES["nearest"](problem)
solution = Solution(tour, dist_matrix, is_closed=True)
improved = LOCAL_SEARCHES["two_opt"](solution)  # usa seleção adaptativa por padrão
```

## Split de Dados

O split train/test usa seed=42 para reprodutibilidade e compatibilidade com outros grupos:

```bash
python scripts/generate_splits.py --seed 42 --train_ratio 0.9
```

Resultado: 90% treino, 10% teste (1111 instâncias de teste por tipo).

## Referências

- Lourenço, H. R., Martin, O. C., & Stützle, T. (2003). *Iterated Local Search*. Handbook of Metaheuristics.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Bengio, Y., Lodi, A., & Prouvost, A. (2020). *Machine Learning for Combinatorial Optimization*. arXiv:1811.06128.
