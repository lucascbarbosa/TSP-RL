"""Replay buffer for DQN experience replay."""

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    """Single transition (s, a, r, s', done)."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class Batch:
    """Batch of transitions as tensors."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions and provides random sampling for training.
    Uses a deque for O(1) append and automatic size limiting.
    """

    def __init__(self, capacity: int = 50000) -> None:
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
        """
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add transition to buffer.

        Args:
            state: State vector.
            action: Action index (0-8).
            reward: Reward received.
            next_state: Next state vector.
            done: Whether episode ended.
        """
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: str = "cpu") -> Batch:
        """
        Sample random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.
            device: Torch device for tensors.

        Returns:
            Batch of tensors.

        Raises:
            ValueError: If buffer has fewer than batch_size transitions.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has {len(self.buffer)} < {batch_size} transitions")

        transitions = random.sample(list(self.buffer), batch_size)

        states = np.stack([t.state for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.stack([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        return Batch(
            states=torch.tensor(states, dtype=torch.float32, device=device),
            actions=torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
            next_states=torch.tensor(next_states, dtype=torch.float32, device=device),
            dones=torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


__all__ = ["ReplayBuffer", "Transition", "Batch"]
