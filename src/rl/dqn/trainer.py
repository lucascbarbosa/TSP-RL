"""DQN training and evaluation functions."""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.rl.dqn.buffer import ReplayBuffer
from src.rl.dqn.env import DQNEnv, N_ACTIONS
from src.rl.dqn.network import QNetwork
from src.tsp.instance import TSPInstance


@dataclass
class DQNConfig:
    """Configuration for DQN training."""

    # Environment
    time_budget: float = 10.0  # Base time budget (scales with n)
    history_len: int = 2  # Number of past actions in state

    # DQN hyperparameters
    gamma: float = 0.99  # Discount factor
    lr: float = 0.001  # Learning rate
    batch_size: int = 64
    buffer_size: int = 50000
    target_update_freq: int = 50  # Episodes between target updates

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995  # Per episode

    # Training
    n_episodes: int = 2000
    updates_per_episode: int = 5
    min_buffer_size: int = 100  # Minimum transitions before training

    # Network
    hidden_dim: int = 64

    # Device
    device: str = "cpu"


@dataclass
class TrainingStats:
    """Statistics from DQN training."""

    episode_rewards: list[float] = field(default_factory=list)
    episode_best_gaps: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)


def compute_time_budget(n: int, base_budget: float = 10.0) -> float:
    """
    Compute time budget scaled by instance size.

    Uses O(n²) scaling: T(n) = (n/100)² × base_budget

    Args:
        n: Instance size (number of cities).
        base_budget: Base time budget for n=100.

    Returns:
        Scaled time budget in seconds.
    """
    return (n / 100) ** 2 * base_budget


def train_dqn(
    instances: list[TSPInstance],
    config: DQNConfig,
    verbose: bool = True,
) -> tuple[QNetwork, TrainingStats]:
    """
    Train DQN on a set of TSP instances.

    Args:
        instances: List of training instances.
        config: Training configuration.
        verbose: Print progress information.

    Returns:
        Tuple of (trained Q-network, training statistics).
    """
    if not instances:
        raise ValueError("No instances provided")

    # Infer instance size from first instance
    n = instances[0].dimension
    time_budget = compute_time_budget(n, config.time_budget)
    state_dim = 3 + config.history_len * N_ACTIONS

    if verbose:
        print(f"Training DQN: n={n}, time_budget={time_budget:.2f}s")
        print(f"State dim: {state_dim}, Actions: {N_ACTIONS}, Episodes: {config.n_episodes}")

    # Initialize networks
    q_net = QNetwork(state_dim, n_actions=N_ACTIONS, hidden_dim=config.hidden_dim).to(config.device)
    target_net = QNetwork(state_dim, n_actions=N_ACTIONS, hidden_dim=config.hidden_dim).to(config.device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = Adam(q_net.parameters(), lr=config.lr)
    replay_buffer = ReplayBuffer(config.buffer_size)

    stats = TrainingStats()
    epsilon = config.epsilon_start

    for episode in range(config.n_episodes):
        # Sample random instance
        instance = random.choice(instances)
        env = DQNEnv(instance, time_budget, config.history_len)

        # Collect episode
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randrange(env.n_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state.to_numpy(), dtype=torch.float32, device=config.device)
                    q_values = q_net(state_tensor)
                    action = int(q_values.argmax().item())

            # Execute action
            next_state, reward, done = env.step(action)

            # Store transition
            replay_buffer.push(
                state.to_numpy(),
                action,
                reward,
                next_state.to_numpy(),
                done,
            )

            episode_reward += reward
            episode_steps += 1
            state = next_state

        # Record episode stats
        stats.episode_rewards.append(episode_reward)
        stats.episode_best_gaps.append(env.best_gap)
        stats.episode_lengths.append(episode_steps)
        stats.epsilons.append(epsilon)

        # Update network (if buffer has enough samples)
        if len(replay_buffer) >= config.min_buffer_size:
            for _ in range(config.updates_per_episode):
                batch = replay_buffer.sample(config.batch_size, config.device)

                # Compute Q(s, a)
                q_values = q_net(batch.states)
                q_selected = q_values.gather(1, batch.actions).squeeze(1)

                # Compute target: r + γ max Q_target(s', a')
                with torch.no_grad():
                    next_q = target_net(batch.next_states).max(dim=1)[0]
                    targets = batch.rewards + config.gamma * next_q * (1 - batch.dones)

                # Update
                loss = F.mse_loss(q_selected, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                stats.losses.append(loss.item())

        # Decay epsilon
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

        # Update target network
        if (episode + 1) % config.target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Progress logging
        if verbose and (episode + 1) % 100 == 0:
            recent_gaps = stats.episode_best_gaps[-100:]
            avg_gap = np.mean(recent_gaps)
            print(
                f"Episode {episode + 1}/{config.n_episodes} | "
                f"Avg gap (last 100): {avg_gap:.2f}% | "
                f"ε: {epsilon:.3f}"
            )

    return q_net, stats


def evaluate_dqn(
    model: QNetwork,
    instances: list[TSPInstance],
    config: DQNConfig,
    verbose: bool = False,
) -> list[float]:
    """
    Evaluate trained DQN on test instances.

    Args:
        model: Trained Q-network.
        instances: List of test instances.
        config: Configuration (for time budget and history length).
        verbose: Print per-instance results.

    Returns:
        List of final gaps (%) for each instance.
    """
    model.eval()
    gaps = []

    n = instances[0].dimension if instances else 0
    time_budget = compute_time_budget(n, config.time_budget)

    for i, instance in enumerate(instances):
        env = DQNEnv(instance, time_budget, config.history_len)
        state = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state.to_numpy(), dtype=torch.float32, device=config.device)
                q_values = model(state_tensor)
                action = int(q_values.argmax().item())

            state, _, done = env.step(action)

        gaps.append(env.best_gap)

        if verbose:
            print(f"Instance {i + 1}/{len(instances)}: gap = {env.best_gap:.2f}%")

    return gaps


def save_model(model: QNetwork, path: str | Path) -> None:
    """Save model weights to file."""
    torch.save(model.state_dict(), path)


def load_model(
    path: str | Path,
    state_dim: int,
    n_actions: int = N_ACTIONS,
    hidden_dim: int = 64,
) -> QNetwork:
    """Load model weights from file."""
    model = QNetwork(state_dim, n_actions=n_actions, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


__all__ = [
    "DQNConfig",
    "TrainingStats",
    "train_dqn",
    "evaluate_dqn",
    "compute_time_budget",
    "save_model",
    "load_model",
    "N_ACTIONS",
]
