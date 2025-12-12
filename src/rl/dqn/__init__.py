"""DQN components for Q-ILS.

This module provides Deep Q-Network components for learning
(perturbation, local_search) selection policies.

Modules:
    state: DQNState representation and normalization
    network: QNetwork neural network architecture
    buffer: ReplayBuffer for experience replay
    env: DQNEnv environment wrapper
    trainer: Training and evaluation functions
"""

from src.rl.dqn.state import DQNState, normalize_gap, compute_delta_reward

__all__ = [
    "DQNState",
    "normalize_gap",
    "compute_delta_reward",
]
