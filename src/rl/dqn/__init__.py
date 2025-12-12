"""DQN components for DQN-ILS.

This module provides Deep Q-Network components for learning
(perturbation, local_search) selection policies.

Modules:
    state: DQNState representation and normalization
    network: QNetwork neural network architecture
    buffer: ReplayBuffer for experience replay
    env: DQNEnv environment wrapper
    trainer: Training and evaluation functions
"""

from src.rl.dqn.buffer import Batch, ReplayBuffer, Transition
from src.rl.dqn.env import ACTION_DECODE, N_ACTIONS, DQNEnv
from src.rl.dqn.network import QNetwork
from src.rl.dqn.state import DQNState, compute_delta_reward, normalize_gap
from src.rl.dqn.trainer import (
    DQNConfig,
    TrainingStats,
    compute_time_budget,
    evaluate_dqn,
    load_model,
    save_model,
    train_dqn,
)

__all__ = [
    # Actions
    "N_ACTIONS",
    "ACTION_DECODE",
    # State
    "DQNState",
    "normalize_gap",
    "compute_delta_reward",
    # Network
    "QNetwork",
    # Buffer
    "ReplayBuffer",
    "Transition",
    "Batch",
    # Environment
    "DQNEnv",
    # Trainer
    "DQNConfig",
    "TrainingStats",
    "train_dqn",
    "evaluate_dqn",
    "compute_time_budget",
    "save_model",
    "load_model",
]
