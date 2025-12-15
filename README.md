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
│  3. Decodificar ação: a → (perturbação, busca local)        │
│  4. Aplicar perturbação + busca local                       │
│  5. Calcular recompensa: r = Δg_best (melhoria no recorde)  │
│  6. Repetir até esgotar time budget T                       │
└─────────────────────────────────────────────────────────────┘
```

## Modelagem MDP

### Estado (contínuo, 30 dimensões)

O estado é um vetor contínuo que captura:

| Componente | Dimensão | Descrição |
|------------|----------|-----------|
| `g` | 1 | Gap atual normalizado (log scale) |
| `g_best` | 1 | Melhor gap do episódio (normalizado) |
| `t_ratio` | 1 | Tempo restante / T ∈ [0, 1] |
| `history` | 27 | Últimas 3 ações (one-hot encoded, 9 cada) |

**Normalização do gap:** `g_norm = log(1 + gap) / log(101)` — comprime gaps grandes, preserva resolução em gaps pequenos.

### Recompensa

A recompensa é a **melhoria no melhor gap** encontrado:

```
reward = g_best_old - g_best_new
```

- Recompensa > 0 apenas quando o agente melhora o recorde
- Alinha o sinal de recompensa com o objetivo real (minimizar gap)
- Sparse mas informativo

### Ações (9)

Cada ação é um par **(perturbação, busca local)**:

| Ação | Perturbação | Busca Local | Uso |
|------|-------------|-------------|-----|
| 0 | two_swap | 2-opt | Refinamento leve |
| 1 | two_swap | Lin-Kernighan | Refinamento intenso |
| 2 | segment_reverse | 2-opt | Perturbação média |
| 3 | segment_reverse | Lin-Kernighan | Perturbação + intensificação |
| 4 | random | 2-opt | Restart rápido |
| 5 | nearest | 2-opt | Restart de qualidade |
| 6 | cheapest | 2-opt | Restart alta qualidade |
| 7 | nearest | Lin-Kernighan | Restart + intensificação |
| 8 | grasp | 2-opt | Restart diversificado |

**Perturbações leves** (`two_swap`, `segment_reverse`): modificam a solução atual.
**Perturbações destrutivas** (`random`, `nearest`, `cheapest`, `grasp`): reconstroem do zero.

## Estrutura do Projeto

```
TSP-RL/
├── src/
│   ├── tsp/                      # Core TSP
│   │   ├── solution.py           # Representação de soluções
│   │   ├── instance.py           # TSPInstance, TSPDataset
│   │   ├── local_search.py       # two_opt (adaptativo), lin_kernighan
│   │   ├── perturbation.py       # two_swap, segment_reverse
│   │   └── constructive.py       # random, nearest, cheapest, grasp
│   ├── ils/
│   │   └── q_ils.py              # QILS, State, Action, RunStats
│   └── rl/
│       ├── dqn/                  # Deep Q-Network
│       │   ├── state.py          # DQNState, normalize_gap
│       │   ├── network.py        # QNetwork (MLP)
│       │   ├── buffer.py         # ReplayBuffer
│       │   ├── env.py            # DQNEnv (ambiente gym-like)
│       │   └── trainer.py        # train_dqn, evaluate_dqn, DQNConfig
│       ├── q_table.py            # QTable (baseline)
│       ├── q_learning.py         # Value iteration (baseline)
│       └── mdp.py                # MDP (baseline)
├── scripts/
│   ├── train_dqn.py              # Treinamento DQN
│   ├── generate_splits.py        # Gera splits train/test
│   ├── evaluate.py               # Avaliação
│   └── clear.sh                  # Remove arquivos gerados
├── models/
│   └── dqn/                      # Modelos treinados (.pt)
├── data/
│   ├── {EUC_2D,ATT,GEO}.json     # Instâncias (11110 por tipo)
│   └── splits.json               # Split 90/10 (seed=42)
└── tests/
```

## Quickstart

### Treinar um modelo DQN

```bash
# Gerar splits (se necessário)
python scripts/generate_splits.py --seed 42

# Treinar para EUC_2D, tamanho 50, 2000 episódios
python scripts/train_dqn.py --type EUC_2D --sizes 50 --episodes 2000

# Treinar múltiplos tamanhos
python scripts/train_dqn.py --type EUC_2D --sizes 10 20 30 50 --episodes 1000
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
print(f"Gap médio final: {stats.episode_best_gaps[-100:]:.2f}%")
```

### Avaliação

```python
from src.rl.dqn import load_model, evaluate_dqn, DQNConfig

# Carregar modelo treinado
model = load_model("models/dqn/EUC_2D_n050.pt", state_dim=30)

# Avaliar em instâncias de teste
test_instances = list(TSPDataset("data/EUC_2D.json", instance_ids=range(1000, 1111)))
config = DQNConfig(time_budget=10.0)

gaps = evaluate_dqn(model, test_instances, config)
print(f"Gap médio: {sum(gaps)/len(gaps):.2f}%")
```

### Ambiente (para experimentos)

```python
from src import TSPInstance, DQNEnv

instance = TSPInstance("data/EUC_2D.json", instance_id=0)
env = DQNEnv(instance, time_budget=5.0, history_len=3)

state = env.reset()
print(f"State dim: {env.state_dim}")  # 30

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

### Configuração completa (`DQNConfig`)

```python
@dataclass
class DQNConfig:
    # Ambiente
    time_budget: float = 10.0   # T(n) = (n/100)² × time_budget
    history_len: int = 3        # Ações no histórico

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
```

## Arquitetura da Rede

```
QNetwork: state (30) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(9)
```

- Input: estado contínuo (30 dims com history_len=3)
- Output: Q-values para cada uma das 9 ações
- ~5K parâmetros

## Operadores TSP

### Buscas Locais

- **2-opt**: Seleção adaptativa baseada em n
- **Lin-Kernighan**: Cadeias de 2-opt com depth=2

#### 2-opt Adaptativo

O `two_opt` seleciona automaticamente a variante ideal baseado no tamanho da instância:

| Tamanho (n) | Variante | Complexidade | Motivo |
|-------------|----------|--------------|--------|
| n < 40 | `two_opt_full` | O(n²) | Overhead de neighbor lists não compensa |
| 40 ≤ n < 80 | `two_opt_nn` | O(n·k) | Bom equilíbrio qualidade/velocidade |
| n ≥ 80 | `two_opt_dlb` | O(n·k) + DLB | Máxima velocidade, ~1-4% de perda de qualidade |

Parâmetro `k` (número de vizinhos):
- **int**: usado diretamente (ex: `k=20`)
- **float em (0,1)**: proporção de n (ex: `k=0.5` = 50% das cidades)
- **Default**: `k=0.5`

As variantes individuais (`two_opt_full`, `two_opt_nn`, `two_opt_dlb`) estão disponíveis via import direto de `src.tsp.local_search`.

### Perturbações

- **two_swap**: Troca 2 vértices aleatórios
- **segment_reverse**: Reverte segmento aleatório do tour

### Construtivas (perturbações destrutivas)

- **random**: Tour aleatório
- **nearest**: Vizinho mais próximo
- **cheapest**: Inserção mais barata
- **grasp**: GRASP com RCL (α=0.2, seleciona de lista restrita de candidatos "bons o suficiente")

## Dados

- **3 tipos**: EUC_2D, ATT, GEO
- **10 tamanhos**: 10, 20, ..., 100 nós
- **11.110 instâncias por tipo** (1.111 por tamanho)
- **Split**: 90% treino, 10% teste (seed=42)

## Referências

- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- Lourenço et al. (2003). *Iterated Local Search*. Handbook of Metaheuristics.
- Bengio et al. (2020). *Machine Learning for Combinatorial Optimization*. arXiv:1811.06128.
