"""Neural network architectures for DQN."""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Q-Network for DQN.

    Simple MLP that maps state vectors to Q-values for each action.

    Architecture: state_dim → hidden → hidden → n_actions
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 9,
        hidden_dim: int = 64,
    ) -> None:
        """
        Initialize Q-Network.

        Args:
            state_dim: Input state dimension (e.g., 30 for R=3).
            n_actions: Number of actions (default: 9).
            hidden_dim: Hidden layer size (default: 64).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor of shape (batch, state_dim) or (state_dim,).

        Returns:
            Q-values tensor of shape (batch, n_actions) or (n_actions,).
        """
        return self.net(state)


__all__ = ["QNetwork"]
