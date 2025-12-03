"""Q Table class."""
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QTable:
    """Classic Q Table class."""
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        epsilon: float = 0.1,
    ):
        """Initialize Q-Table parameters."""
        self.table: torch.Tensor = torch.zeros(
            (n_states, n_actions),
            dtype=torch.float32,
            device=device
        )
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.init_epsilon = epsilon

    def get_policy(self, state: int) -> torch.Tensor:
        """Get action probabilities for a state (epsilon-greedy policy).

        Returns:
            Dictionary mapping actions to probabilities
        """
        # Get best actions
        best_actions = self.table[state, :].max(axis=0)

        # Epsilon-greedy probabilities
        prob_greedy = (1 - self.epsilon) / self.n_actions
        prob_explore = self.epsilon / self.n_actions

        policy = torch.zeros((self.n_actions), device=device)
        policy[self.table == best_actions] = prob_greedy + prob_explore
        policy[self.table != best_actions] = prob_greedy
        return policy

    def decay_epsilon(
        self, factor: float = 0.995, min_epsilon: float = 0.01
    ) -> None:
        """Decay exploration rate."""
        self.epsilon = max(min_epsilon, self.epsilon * factor)

    def reset_epsilon(self) -> None:
        """Reset exploration rate."""
        self.epsilon = self.init_epsilon

    def to_txt(self, filename: str) -> None:
        """Save Q-Table to text file."""
        with open(filename, 'w') as f:
            # Write header line with n_states and n_actions
            f.write(f"{self.n_states} {self.n_actions}\n")
            # Write Q-table matrix with space-separated values
            np.savetxt(f, self.table.cpu().numpy(), delimiter=' ', fmt='%g')
