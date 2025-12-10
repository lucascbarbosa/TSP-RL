"""Q-Table implementation for tabular Q-Learning."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from src.rl.mdp import build_mdp_from_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QTable:
    """
    Classic Q-Table for tabular reinforcement learning.

    Attributes:
        table: Q-values tensor of shape (n_states, n_actions).
        n_states: Number of states.
        n_actions: Number of actions.
        epsilon: Exploration rate for epsilon-greedy policy.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        epsilon: float = 0.1,
        table: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize Q-Table.

        Args:
            n_states: Number of states.
            n_actions: Number of actions.
            epsilon: Initial exploration rate.
            table: Pre-existing Q-table tensor (optional).
        """
        if table is not None:
            self.table = table.to(device)
        else:
            self.table = torch.zeros(
                (n_states, n_actions),
                dtype=torch.float32,
                device=device,
            )
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.init_epsilon = epsilon

    def get_policy(self, state: int) -> torch.Tensor:
        """
        Get action probabilities for epsilon-greedy policy.

        Args:
            state: Current state.

        Returns:
            Tensor of action probabilities.
        """
        best_actions = self.table[state, :].max(dim=0)

        prob_greedy = (1 - self.epsilon) / self.n_actions
        prob_explore = self.epsilon / self.n_actions

        policy = torch.zeros((self.n_actions,), device=device)
        policy[self.table[state] == best_actions.values] = prob_greedy + prob_explore
        policy[self.table[state] != best_actions.values] = prob_greedy

        return policy

    def decay_epsilon(
        self,
        factor: float = 0.995,
        min_epsilon: float = 0.01,
    ) -> None:
        """Decay exploration rate by factor, clamped to min_epsilon."""
        self.epsilon = max(min_epsilon, self.epsilon * factor)

    def to_txt(self, filename: Union[str, Path]) -> None:
        """
        Save Q-Table to text file.

        Format:
            Line 1: "n_states n_actions"
            Remaining lines: Q-values matrix (space-separated)

        Args:
            filename: Output file path.
        """
        with open(filename, "w") as f:
            f.write(f"{self.n_states} {self.n_actions}\n")
            np.savetxt(f, self.table.cpu().numpy(), delimiter=" ", fmt="%g")

    @classmethod
    def from_txt(cls, filename: Union[str, Path]) -> QTable:
        """
        Load Q-Table from text file.

        Args:
            filename: Path to Q-table file.

        Returns:
            Loaded QTable instance.
        """
        with open(filename, "r") as f:
            header = f.readline().strip().split()
            if len(header) != 2:
                raise ValueError("Invalid Q-table header (expected: 'n_states n_actions').")

            n_states, n_actions = map(int, header)

            data = []
            for i in range(n_states):
                line = f.readline()
                if not line:
                    raise ValueError(f"Q-table has fewer rows than n_states={n_states}.")
                row_vals = list(map(float, line.strip().split()))
                if len(row_vals) != n_actions:
                    raise ValueError(f"Row {i+2} has {len(row_vals)} columns, expected {n_actions}.")
                data.append(row_vals)

        table = torch.tensor(data, dtype=torch.float32, device=device)
        return cls(n_states=n_states, n_actions=n_actions, table=table)

    def copy(self) -> QTable:
        """Deep copy."""
        return QTable(
            n_states=self.n_states,
            n_actions=self.n_actions,
            epsilon=self.epsilon,
            table=self.table.clone(),
        )

    def rollout(
        self,
        transition_filename: Union[str, Path],
        n_simulations: int,
        initial_state: int,
        max_steps: int = 1000,
    ) -> dict[str, list[float]]:
        """
        Run rollout simulations using the learned policy.

        Args:
            transition_filename: Path to transition file for MDP.
            n_simulations: Number of simulations to run.
            initial_state: Starting state.
            max_steps: Maximum steps per simulation.

        Returns:
            Dictionary with 'rewards' and 'length' lists.
        """
        history: dict[str, list[float]] = {"rewards": [], "length": []}
        mdp = build_mdp_from_file(transition_filename)
        P = mdp.transition_matrix
        R = mdp.reward_matrix

        for _ in range(n_simulations):
            state = initial_state
            total_reward = 0.0

            for step in range(max_steps):
                policy = self.get_policy(state)
                action = torch.multinomial(policy, 1).item()

                transition_probs = P[state, action, :]
                next_state = torch.multinomial(transition_probs, 1).item()
                total_reward += R[state, action].item()
                state = next_state

                if state == self.n_states - 1:
                    break

            history["length"].append(float(step + 1))
            history["rewards"].append(total_reward)

        return history
