# DQN-ILS: Deep Q-Network para Iterated Local Search no TSP

Framework que integra **Deep Q-Network (DQN)** ao **Iterated Local Search (ILS)** para resolver o Problema do Caixeiro Viajante (TSP).

## Ideia Central

No ILS tradicional, a cada iteração aplicamos uma perturbação seguida de uma busca local. O DQN-ILS permite que um **agente de RL aprenda qual par (perturbação, busca local)** usar em função do estado contínuo da busca.

```
┌─────────────────────────────────────────────────────────────┐
│                       Loop do DQN-ILS                       │
├─────────────────────────────────────────────────────────────┤
│  1. Observar estado s = (g, g_best, t_ratio, history)       │
│  2. Consultar rede Q: a = argmax Q(s, a; θ)                 │
│  3. Decodificar ação: a -> (perturbação, busca local)       │
│  4. Aplicar perturbação + busca local                       │
│  5. Calcular recompensa: r = Δg_best (melhoria no recorde)  │
│  6. Repetir até esgotar time budget T                       │
└─────────────────────────────────────────────────────────────┘
```

## Modelagem MDP

### Estado (contínuo, 45 dimensões)

O estado é um vetor contínuo que captura:

| Componente | Dimensão | Descrição |
|------------|----------|-----------|
| `g` | 1 | Gap atual normalizado (log scale) |
| `g_best` | 1 | Melhor gap do episódio (normalizado) |
| `t_ratio` | 1 | Tempo restante / T ∈ [0, 1] |
| `history` | 42 | Últimas 2 ações (one-hot encoded, 21 cada) |

**Normalização do gap:** `g_norm = log(1 + gap) / log(101)` — comprime gaps grandes, preserva resolução em gaps pequenos.

### Recompensa

A recompensa é a **melhoria no melhor gap** encontrado:

```
reward = g_best_old - g_best_new
```

- Recompensa > 0 apenas quando o agente melhora o recorde
- Alinha o sinal de recompensa com o objetivo real (minimizar gap)
- Sparse mas informativo

### Ações (21)

Cada ação é um par **(perturbação, busca local)**, com variantes de 2-opt expostas individualmente:

**Perturbações leves** (0-7):
| Ação | Perturbação | Busca Local |
|------|-------------|-------------|
| 0-3 | two_swap | 2-opt_full, 2-opt_nn, 2-opt_dlb, LK |
| 4-7 | segment_reverse | 2-opt_full, 2-opt_nn, 2-opt_dlb, LK |

**Perturbações destrutivas** (8-20):
| Ação | Perturbação | Busca Local |
|------|-------------|-------------|
| 8-10 | random | 2-opt_full, 2-opt_nn, 2-opt_dlb |
| 11-14 | nearest | 2-opt_full, 2-opt_nn, 2-opt_dlb, LK |
| 15-17 | cheapest | 2-opt_full, 2-opt_nn, 2-opt_dlb |
| 18-20 | grasp | 2-opt_full, 2-opt_nn, 2-opt_dlb |

O agente aprende qual variante de 2-opt usar em cada situação. Operadores lentos são naturalmente penalizados pelo desconto temporal.

## Estrutura do Projeto

```
TSP-RL/
├── src/
│   ├── tsp/                      # Core TSP
│   │   ├── solution.py           # Representação de soluções
│   │   ├── instance.py           # TSPInstance, TSPDataset
│   │   ├── local_search.py       # two_opt_full, two_opt_nn, two_opt_dlb, lin_kernighan
│   │   ├── perturbation.py       # two_swap, segment_reverse
│   │   └── constructive.py       # random, nearest, cheapest, grasp
│   └── rl/
│       └── dqn/                  # Deep Q-Network
│           ├── state.py          # DQNState, normalize_gap
│           ├── network.py        # QNetwork (MLP)
│           ├── buffer.py         # ReplayBuffer
│           ├── env.py            # DQNEnv, ACTION_DECODE, N_ACTIONS
│           └── trainer.py        # train_dqn, evaluate_dqn, DQNConfig
├── scripts/
│   ├── pipeline.py               # Pipeline Python (recomendado)
│   ├── pipeline.sh               # Pipeline Bash (alternativo)
│   ├── train_dqn.py              # Treinamento DQN (paralelo)
│   ├── evaluate_dqn.py           # Avaliação de modelos (paralela)
│   ├── generate_splits.py        # Gera splits train/test
│   ├── generate_plots.py         # Gera gráficos de resultados
│   └── clear.sh                  # Remove arquivos gerados
├── models/
│   └── dqn/                      # Modelos treinados (.pt)
├── data/
│   ├── {EUC_2D,ATT,GEO}.json     # Instâncias (11110 por tipo)
│   └── splits.json               # Split 90/10 (seed=42)
└── tests/
```

## Quickstart

### Pipeline completo (recomendado)

```bash
# Pipeline Python (recomendado) - executa splits -> treino -> avaliação -> plots
python scripts/pipeline.py --types EUC_2D --sizes 10 20 --workers 16

# Pipeline Bash (alternativo)
./scripts/pipeline.sh --types "EUC_2D" --sizes "10 20" --workers 16

# Ver todas as opções
python scripts/pipeline.py --help
```

### Treinar um modelo DQN

```bash
# Gerar splits (se necessário)
python scripts/generate_splits.py --seed 42

# Treinar para EUC_2D, tamanho 50, 2000 episódios (16 workers paralelos)
python scripts/train_dqn.py --type EUC_2D --sizes 50 --episodes 2000 --workers 16

# Treinar múltiplos tamanhos
python scripts/train_dqn.py --type EUC_2D --sizes 10 20 30 50 --episodes 1000
```

### Avaliar modelo treinado

```bash
# Avaliar um modelo específico (avaliação paralela de instâncias)
python scripts/evaluate_dqn.py --model models/dqn/EUC_2D_n050.pt --workers 16

# Avaliar com comparação de baseline (GRASP+2opt, mesmo time budget)
python scripts/evaluate_dqn.py --model models/dqn/EUC_2D_n050.pt --baseline

# Avaliar todos os modelos de um tipo
python scripts/evaluate_dqn.py --model "models/dqn/EUC_2D_*.pt"
```

### Limpar arquivos gerados

```bash
./scripts/clear.sh          # dry-run
./scripts/clear.sh --force  # deleta
```

## Uso Programático

### Treinamento

```python
from src import TSPDataset, DQNConfig, train_dqn, evaluate_dqn

# Carregar instâncias de treino
dataset = TSPDataset("data/EUC_2D.json", instance_ids=range(1000))
instances = list(dataset)

# Configurar e treinar
config = DQNConfig(
    time_budget=10.0,      # segundos (escala com n²)
    n_episodes=2000,
    gamma=0.99,
    lr=0.001,
    hidden_dim=64,
)

model, stats = train_dqn(instances, config)
print(f"Gap médio final: {np.mean(stats.episode_best_gaps[-100:]):.2f}%")
```

### Avaliação

```python
from src import TSPDataset, load_model, evaluate_dqn, DQNConfig, N_ACTIONS

# Carregar modelo (arquitetura inferida do checkpoint)
model = load_model("models/dqn/EUC_2D_n050.pt")

# Inferir history_len do modelo
history_len = (model.state_dim - 3) // N_ACTIONS

# Avaliar em instâncias de teste
test_instances = list(TSPDataset("data/EUC_2D.json", instance_ids=range(1000, 1111)))
config = DQNConfig(time_budget=10.0, history_len=history_len)

gaps = evaluate_dqn(model, test_instances, config)
print(f"Gap médio: {sum(gaps)/len(gaps):.2f}%")
```

### Ambiente (para experimentos)

```python
from src import TSPInstance, DQNEnv, N_ACTIONS

instance = TSPInstance("data/EUC_2D.json", instance_id=0)
env = DQNEnv(instance, time_budget=5.0, history_len=2)

state = env.reset()
print(f"State dim: {env.state_dim}")  # 45
print(f"N_ACTIONS: {N_ACTIONS}")      # 21

while True:
    action = 0  # ou política aprendida
    next_state, reward, done = env.step(action)
    if done:
        break

print(f"Best gap: {env.best_gap:.2f}%")
```

## Hiperparâmetros

### DQN (`train_dqn.py`)

| Parâmetro | Flag | Default | Descrição |
|-----------|------|---------|-----------|
| Time budget | `--time_budget` | 10.0 | Budget base em segundos (escala com n²) |
| Episódios | `--episodes` | 2000 | Número de episódios de treino |
| γ (gamma) | `--gamma` | 0.99 | Discount factor |
| Learning rate | `--lr` | 0.001 | Taxa de aprendizado |
| Hidden dim | `--hidden_dim` | 64 | Neurônios por camada oculta |
| Workers | `--workers` | 1 | Workers paralelos (batch episodes) |

### Configuração completa (`DQNConfig`)

```python
@dataclass
class DQNConfig:
    # Ambiente
    time_budget: float = 10.0   # T(n) = (n/100)² * time_budget
    history_len: int = 2        # Ações no histórico

    # DQN
    gamma: float = 0.99
    lr: float = 0.001
    batch_size: int = 64
    buffer_size: int = 50000
    target_update_freq: int = 50

    # Exploração
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

    # Treinamento
    n_episodes: int = 2000
    updates_per_episode: int = 5
    n_workers: int = 1          # Workers paralelos (batch episodes)
```

## Arquitetura da Rede

```
QNetwork: state (45) -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Linear(21)
```

- Input: estado contínuo (45 dims com history_len=2 e 21 ações)
- Output: Q-values para cada uma das 21 ações
- ~6K parâmetros

## Operadores TSP

### Buscas Locais (4 variantes)

| Variante | Complexidade | Descrição |
|----------|--------------|-----------|
| `two_opt_full` | O(n²) | Melhor qualidade, mais lento |
| `two_opt_nn` | O(n·k) | Neighbor lists, bom equilíbrio |
| `two_opt_dlb` | O(n·k) | Don't look bits, mais rápido |
| `lin_kernighan` | Variável | Cadeias de 2-opt com depth=2 |

O parâmetro `k` (número de vizinhos) para `two_opt_nn` e `two_opt_dlb`:
- **int**: usado diretamente (ex: `k=20`)
- **float em (0,1)**: proporção de n (ex: `k=0.5` = 50% das cidades)
- **Default**: `k=0.5`

### Perturbações Leves

- **two_swap**: Troca 2 vértices aleatórios
- **segment_reverse**: Reverte segmento aleatório do tour

### Construtivas (perturbações destrutivas)

- **random**: Tour aleatório
- **nearest**: Vizinho mais próximo
- **cheapest**: Inserção mais barata
- **grasp**: GRASP com RCL (α=0.2)

## Dados

- **3 tipos**: EUC_2D, ATT, GEO
- **10 tamanhos**: 10, 20, ..., 100 nós
- **11.110 instâncias por tipo** (1.111 por tamanho)
- **Split**: 90% treino, 10% teste (seed=42)

## Referências

- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- Lourenço et al. (2003). *Iterated Local Search*. Handbook of Metaheuristics.
- Bengio et al. (2020). *Machine Learning for Combinatorial Optimization*. arXiv:1811.06128.
