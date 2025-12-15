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

### Estado (contínuo, 3 + 16×history_len dims; padrão 19)

O estado é um vetor contínuo que captura:

| Componente | Dimensão | Descrição |
|------------|----------|-----------|
| `g` | 1 | Gap atual normalizado (log scale) |
| `g_best` | 1 | Melhor gap do episódio (normalizado) |
| `t_ratio` | 1 | Tempo restante / T ∈ [0, 1] |
| `history` | 16×h | Últimas h ações one-hot (h=history_len; padrão 1) |

**Normalização do gap:** `g_norm = log(1 + gap) / log(101)` — comprime gaps grandes, preserva resolução em gaps pequenos.

### Recompensa (`--reward_type`)

Duas opções de recompensa disponíveis via `--reward_type`:

| Tipo | Fórmula | Descrição |
|------|---------|-----------|
| `delta` (padrão) | `g_best_old - g_best_new` | Recompensa por melhoria no melhor gap |
| `sparse` | `-normalize_gap(best_gap)` no final | Recompensa apenas no fim do episódio |

- **delta**: Recompensa > 0 apenas quando o agente melhora o recorde. Alinha o sinal com o objetivo (minimizar gap).
- **sparse**: Zero durante episódio, recompensa final baseada no gap (menor gap → maior reward).

### Referência para o Gap

O gap é calculado como `(custo_atual - referência) / referência × 100%`.

**Referência para o estado**: Tanto em treino quanto em avaliação, usamos `baseline_cost` (GRASP+2opt) como referência. Isso garante:
- Distribuição consistente entre treino e avaliação
- Método funciona sem conhecer o custo ótimo (cenário real de deploy)

**Gap reportado**: Quando disponível, o gap é reportado vs `opt_cost` para comparação com a literatura.

**Cálculo do baseline:** O baseline competidor (GRASP+2opt) roda com o mesmo time budget do DQN, selecionando α aleatoriamente de {0.03, 0.1, 0.3} a cada iteração — os mesmos valores disponíveis ao agente DQN.

### Ações (16)

Cada ação é um par **(perturbação, busca local)**. Baseado em análise empírica de modelos treinados, removemos operadores pouco usados (two_opt_nn ~4%, random) e adicionamos GRASP parametrizado.

**Perturbações leves** (0-4):
| Ação | Perturbação | Busca Local |
|------|-------------|-------------|
| 0-2 | two_swap | 2-opt_dlb, 2-opt_full, LK |
| 3-4 | segment_reverse | 2-opt_dlb, 2-opt_full |

**Restart com construtivas** (5-9):
| Ação | Perturbação | Busca Local |
|------|-------------|-------------|
| 5-7 | nearest | 2-opt_dlb, 2-opt_full, LK |
| 8-9 | cheapest | 2-opt_dlb, 2-opt_full |

**GRASP parametrizado** (10-15):
| Ação | Perturbação | Busca Local |
|------|-------------|-------------|
| 10-11 | grasp α=0.03 | 2-opt_dlb, 2-opt_full |
| 12-13 | grasp α=0.1 | 2-opt_dlb, 2-opt_full |
| 14-15 | grasp α=0.3 | 2-opt_dlb, 2-opt_full |

- **α=0.03**: quase-guloso, mínima diversificação
- **α=0.1**: balanço entre qualidade e diversificação
- **α=0.3**: mais aleatório, substitui "random" com diversificação controlada

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
│   ├── compare_dqn_double.py     # Comparação DQN vs Double DQN
│   ├── generate_splits.py        # Gera splits train/val/test
│   ├── generate_plots.py         # Gera gráficos de resultados
│   └── clear.sh                  # Remove arquivos gerados
├── models/
│   └── dqn/                      # Modelos treinados (.pt)
├── data/
│   ├── {EUC_2D,ATT,GEO}.json     # Instâncias (11110 por tipo)
│   └── splits.json               # Split 80/10/10 (seed=42)
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

# Treinar para EUC_2D, tamanho 50, 128 episódios (16 workers paralelos)
python scripts/train_dqn.py --type EUC_2D --sizes 50 --episodes 128 --workers 16

# Treinar múltiplos tamanhos
python scripts/train_dqn.py --type EUC_2D --sizes 10 20 30 --episodes 128
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
    n_episodes=128,
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
env = DQNEnv(instance, time_budget=5.0)

state = env.reset()
print(f"State dim: {env.state_dim}")  # 19
print(f"N_ACTIONS: {N_ACTIONS}")      # 16

while True:
    action = 0  # ou política aprendida
    next_state, reward, done = env.step(action)
    if done:
        break

print(f"Best gap: {env.best_gap:.2f}%")
```

## Hiperparâmetros

### Pipeline (`pipeline.py`)

| Parâmetro | Flag | Default | Descrição |
|-----------|------|---------|-----------|
| Time budget | `--time_budget` | 5.0 | Budget base em segundos (escala com n²) |
| Episódios | `--episodes` | 200 | Número de episódios de treino |
| γ (gamma) | `--gamma` | 0.99 | Discount factor |
| Learning rate | `--lr` | 0.001 | Taxa de aprendizado |
| Hidden dim | `--hidden_dim` | 64 | Neurônios por camada oculta |
| Reward type | `--reward_type` | delta | Tipo de recompensa (`delta` ou `sparse`) |
| Workers | `--workers` | 16 | Workers paralelos |

### Configuração completa (`DQNConfig`)

```python
@dataclass
class DQNConfig:
    # Ambiente
    time_budget: float = 10.0   # T(n) = (n/100)² * time_budget
    history_len: int = 1        # Ações no histórico
    reward_type: str = "delta"  # "delta" (melhoria) ou "sparse" (só no final)

    # DQN
    gamma: float = 0.99
    lr: float = 0.001
    batch_size: int = 64
    buffer_size: int = 50000
    target_update_freq: int = 16

    # Exploração
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05

    # Treinamento
    n_episodes: int = 2000
    updates_per_episode: int = 5
    n_workers: int = 1          # Workers paralelos

    # Double DQN
    use_double_dqn: bool = True # Reduz superestimação de Q-values
```

### Double DQN

Por padrão, `use_double_dqn=True`. Double DQN separa seleção (rede online) de avaliação (target network), reduzindo o viés de superestimação:

```
# Standard DQN:  target = r + γ * max_a Q_target(s', a)
# Double DQN:    target = r + γ * Q_target(s', argmax_a Q_online(s', a))
```

Para comparar as variantes:

```bash
python scripts/compare_dqn_double.py --type EUC_2D --sizes 30 50 --episodes 128
```

Gera automaticamente em `data/results/comparison/plots/`:
- Curvas de aprendizado comparativas
- Evolução dos Q-values (indicador de superestimação)
- Distribuição de ações de cada variante

## Arquitetura da Rede

```
QNetwork: state (19) -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Linear(16)
```

- Input: estado contínuo (19 dims com history_len=1 e 16 ações)
- Output: Q-values para cada uma das 16 ações
- ~3K parâmetros

## Operadores TSP

### Buscas Locais (usadas nas ações)

| Variante | Complexidade | Descrição |
|----------|--------------|-----------|
| `two_opt_dlb` | O(n·k) | Don't look bits, rápido (exploração) |
| `two_opt_full` | O(n²) | Melhor qualidade (intensificação) |
| `lin_kernighan` | Variável | Cadeias de 2-opt, depth=2 |

Todas as variantes usam kernels Numba JIT (~20-100x mais rápido que Python puro).

### Perturbações Leves

- **two_swap**: Troca 2 vértices aleatórios
- **segment_reverse**: Reverte segmento aleatório do tour

### Construtivas (restart)

- **nearest**: Vizinho mais próximo
- **cheapest**: Inserção mais barata
- **grasp_α**: GRASP parametrizado (α ∈ {0.03, 0.1, 0.3})

## Dados

- **3 tipos**: EUC_2D, ATT, GEO
- **10 tamanhos**: 10, 20, ..., 100 nós
- **11.110 instâncias por tipo** (1.111 por tamanho)
- **Split**: 80% treino, 10% val, 10% teste (seed=42)

## Outputs do Pipeline

O pipeline gera os seguintes arquivos em `data/plots/`:

| Arquivo | Descrição |
|---------|-----------|
| `{type}_n{size}_learning_curve.png` | Curva de aprendizado (gap por episódio) |
| `{type}_n{size}_action_dist.png` | Distribuição de ações (últimos 10% episódios) |
| `{type}_n{size}_q_heatmap.png` | Heatmap Q-values (gap × ação, t=50%) |
| `{type}_n{size}_q_heatmap_time.png` | Heatmap comparativo: t=80% vs t=20% |
| `{type}_gaps_violin.png` | Violin plot de gaps por tamanho |
| `{type}_time_vs_gap.png` | Scatter tempo × gap |

O heatmap comparativo (`*_q_heatmap_time.png`) mostra como a política muda com o tempo restante:
- **Top (80%)**: Início do episódio — espera-se mais diversificação (GRASP, restarts)
- **Bottom (20%)**: Fim do episódio — espera-se mais intensificação (perturbações leves)

## Referências

- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- Lourenço et al. (2003). *Iterated Local Search*. Handbook of Metaheuristics.
- Bengio et al. (2020). *Machine Learning for Combinatorial Optimization*. arXiv:1811.06128.
