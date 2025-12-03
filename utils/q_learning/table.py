"""Q Table class."""
import numpy as np
import torch
from typing import Dict, List
from utils.mdp import build_mdp_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QTable:
    """Classic Q Table class."""
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        epsilon: float = 0.1,
        table: torch.Tensor = None,
    ):
        """Initialize Q-Table parameters."""
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

    def to_txt(self, filename: str) -> None:
        """Save Q-Table to text file."""
        with open(filename, 'w') as f:
            # Write header line with n_states and n_actions
            f.write(f"{self.n_states} {self.n_actions}\n")
            # Write Q-table matrix with space-separated values
            np.savetxt(f, self.table.cpu().numpy(), delimiter=' ', fmt='%g')

    def copy(self) -> 'QTable':
        """Copy Q-Table."""
        return QTable(
            n_states=self.n_states,
            n_actions=self.n_actions,
            epsilon=self.epsilon,
            table=self.table.clone(),
        )

    def rollout(
        self,
        transition_filename: str,
        n_simulations: int,
        initial_state: int,
        max_steps: int = 1000,
    ) -> Dict[str, List]:
        """Run rollout simulations using the policy."""
        history = {'rewards': [], 'length': []}
        mdp = build_mdp_model(transition_filename)
        P = mdp.transition_matrix
        R = mdp.reward_matrix
        for _ in range(n_simulations):
            state = initial_state
            total_reward = 0.0
            for step in range(max_steps):
                # Get policy for current state
                policy = self.get_policy(state)

                # Sample action from policy
                action = torch.multinomial(policy, 1).item()

                # Sample next state from transition probabilities
                transition_probs = P[state, action, :]
                next_state = torch.multinomial(transition_probs, 1).item()
                total_reward += R[state, action].item()
                state = next_state
                if state == self.n_states - 1:
                    break

            history['length'].append(step + 1)
            history['rewards'].append(total_reward)
        return history
