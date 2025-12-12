"""TSP-RL: DQN-guided Iterated Local Search for TSP."""

from src.tsp.solution import Solution
from src.tsp.instance import TSPInstance, TSPDataset
from src.tsp.local_search import (
    LOCAL_SEARCHES,
    two_opt,
    two_opt_full,
    two_opt_nn,
    two_opt_dlb,
    lin_kernighan,
)
from src.tsp.perturbation import PERTURBATIONS, two_swap, segment_reverse
from src.tsp.constructive import CONSTRUCTIVES, random_tour, nearest_neighbor, cheapest_insertion
from src.rl.dqn import (
    # Actions
    N_ACTIONS,
    ACTION_DECODE,
    # State
    DQNState,
    normalize_gap,
    compute_delta_reward,
    # Network
    QNetwork,
    # Buffer
    ReplayBuffer,
    # Environment
    DQNEnv,
    # Trainer
    DQNConfig,
    TrainingStats,
    train_dqn,
    evaluate_dqn,
    compute_time_budget,
    save_model,
    load_model,
)

__all__ = [
    # TSP core
    "Solution",
    "TSPInstance",
    "TSPDataset",
    # Local search
    "LOCAL_SEARCHES",
    "two_opt",
    "two_opt_full",
    "two_opt_nn",
    "two_opt_dlb",
    "lin_kernighan",
    # Perturbations
    "PERTURBATIONS",
    "two_swap",
    "segment_reverse",
    # Constructives
    "CONSTRUCTIVES",
    "random_tour",
    "nearest_neighbor",
    "cheapest_insertion",
    # DQN Actions
    "N_ACTIONS",
    "ACTION_DECODE",
    # DQN State
    "DQNState",
    "normalize_gap",
    "compute_delta_reward",
    # DQN Network
    "QNetwork",
    # DQN Buffer
    "ReplayBuffer",
    # DQN Environment
    "DQNEnv",
    # DQN Trainer
    "DQNConfig",
    "TrainingStats",
    "train_dqn",
    "evaluate_dqn",
    "compute_time_budget",
    "save_model",
    "load_model",
]
