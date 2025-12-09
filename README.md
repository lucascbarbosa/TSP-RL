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

| Estado | Gap (%)     | Recompensa | Interpretação          |
|--------|-------------|------------|------------------------|
| 0      | 0 – 2       | 75         | Excelente              |
| 1      | 2 – 5       | 50         | Bom                    |
| 2      | 5 – 10      | 25         | Regular                |
| 3      | > 10        | 0          | Ruim                   |
| 4      | < 0         | 100        | Melhor que o ótimo     |

### Ações (8)

Cada ação é um par **(perturbação, busca local)**:

| Ação | Perturbação      | Busca Local    | Uso                      |
|------|------------------|----------------|--------------------------|
| 0    | two_swap         | 2-opt          | Refinamento leve         |
| 1    | two_swap         | Lin-Kernighan  | Refinamento moderado     |
| 2    | segment_reverse  | 2-opt          | Perturbação média        |
| 3    | segment_reverse  | Lin-Kernighan  | Perturbação + intensif.  |
| 4    | random           | 2-opt          | Restart rápido           |
| 5    | nearest          | 2-opt          | Restart de qualidade     |
| 6    | cheapest         | 2-opt          | Restart alta qualidade   |
| 7    | nearest          | Lin-Kernighan  | Restart + intensificação |

**Tipos de perturbação:**
- **Leves** (`two_swap`, `segment_reverse`): modificam levemente a solução atual
- **Destrutivas** (`random`, `nearest`, `cheapest`): ignoram a solução atual e constroem uma nova do zero

**Buscas locais:**
- **2-opt**: O(n²), rápido e eficiente
- **Lin-Kernighan**: Cadeias de 2-opt (depth=2), mais intensivo

## Estrutura do Projeto

```
TSP-RL/
├── Q_learning_ILS.py         # Classe Q_ILS principal
├── main.py                   # Avaliação com Q-table treinada
├── experimentRun.py          # Geração de transições (paralelo)
├── instance.py               # Carga de instâncias TSP
├── solution.py               # Representação de soluções
├── constructive_heuristic.py # Random, Nearest Neighbor, Cheapest Insertion
├── local_search.py           # 2-opt, Lin-Kernighan
├── perturbation.py           # Perturbações para ILS
├── scripts/
│   └── generate_splits.py    # Gera splits train/test reprodutíveis
├── utils/
│   ├── mdp.py                # Construção do MDP a partir de transições
│   ├── transition.py         # Carga de arquivos de transição
│   ├── q_learning/
│   │   ├── single.py         # Single Q-Learning (iteração de valor)
│   │   └── table.py          # Classe QTable
│   └── plot.py               # Visualizações
├── data/
│   ├── EUC_2D.json           # Instâncias euclidianas
│   ├── ATT.json              # Instâncias ATT
│   ├── GEO.json              # Instâncias geográficas
│   ├── splits.json           # Divisão treino/teste (90/10, seed=42)
│   ├── q_tables/             # Q-tables treinadas
│   └── train/                # Dados de transição (gitignored)
└── etc/
    ├── plantuml/             # Diagramas do framework
    └── slides/               # Apresentação
```

## Fluxo de Execução

### 1. Geração de Transições (offline)

Executa ILS com escolhas aleatórias de (perturbação, busca local), registrando tuplas `(s, a, r, s')`:

```bash
python experimentRun.py \
    --split_path data/splits.json \
    --dataset_path data/EUC_2D.json \
    --output_dir data/train/EUC_2D
```

### 2. Treinamento da Q-table

Constrói o MDP a partir das transições e treina via Q-Learning:

```bash
python -m utils.q_learning.single --types EUC_2D --sizes 10 20 30
```

### 3. Avaliação

Executa o ILS guiado pela Q-table em instâncias de teste:

```bash
python main.py
```

## Componentes

### Heurísticas Construtivas

- **Random**: tour aleatório
- **Nearest Neighbor**: adiciona sempre a cidade mais próxima
- **Cheapest Insertion**: insere na posição de menor aumento de custo

### Buscas Locais

- **2-opt**: troca 2 arestas; O(n²) por iteração
- **Lin-Kernighan**: cadeias de 2-opt com profundidade limitada (depth=2)

### Perturbações

- **two_swap**: troca dois vértices aleatórios
- **segment_reverse**: reverte um segmento aleatório do tour

## Uso Básico

```python
from Q_learning_ILS import Q_ILS
from instance import RandomTSPInstance

# Carregar instância
problem = RandomTSPInstance("data/EUC_2D.json", instance_id=123)

# Calcular custo ótimo
opt_tour = problem.opt_tour
opt_cost = sum(
    problem.get_weight(opt_tour[i], opt_tour[(i + 1) % len(opt_tour)])
    for i in range(len(opt_tour))
)

# Criar solver e carregar Q-table
q_ils = Q_ILS(problem)
q_ils.load_qtable("data/q_tables/EUC_2D/instance_size_50.txt")

# Executar
best_solution = q_ils.exec_q_table(max_iter=50, opt_cost=opt_cost, epsilon=0.1)

print(f"Custo: {best_solution.cost}")
print(f"Gap: {((best_solution.cost - opt_cost) / opt_cost) * 100:.2f}%")
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
