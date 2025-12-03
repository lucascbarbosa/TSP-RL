"""Basic Q-Learning implementation."""
import torch
from utils.q_learning.table import QTable
from utils.transition import load_transition_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def single_q_learning(
    transition_filename: str,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    max_iter: int = 100,
) -> QTable:
    """Train Q-table from transition matrix using Q-Learning."""
    # Load transition data
    (
        current_states,
        actions,
        rewards,
        next_states,
        n_states,
        n_actions,
    ) = load_transition_file(transition_filename, return_torch=True)

    # Convert to appropriate dtypes
    current_states = current_states.to(torch.int32)
    actions = actions.to(torch.int32)
    rewards = rewards.to(torch.float32)
    next_states = next_states.to(torch.int32)

    # Initialize Q-table as numpy array
    q_table = QTable(
        n_states=n_states,
        n_actions=n_actions,
        epsilon=epsilon,
    )

    for i in range(max_iter):
        # Shuffle transitions for each iteration
        indices = torch.randperm(len(current_states))

        # Reset epsilon
        q_table.reset_epsilon()

        for idx in indices:
            # Get state, action, reward, next state
            state = int(current_states[idx])
            action = int(actions[idx])
            reward = rewards[idx]
            next_state = int(next_states[idx])

            # Update Q-table using Q-Learning
            q_table.table[state, action] += (
                alpha * (
                    reward +
                    gamma * q_table.table[next_state].max(dim=0)[0] -
                    q_table.table[state, action]
                )
            )

            # Decay epsilon
            q_table.decay_epsilon()

        if i % 10 == 0:
            print(
                f"Iter [{i + 1}/{max_iter}]: "
                f"Avg Q-value: {q_table.table.mean().item():.4f}"
            )

    return q_table


if __name__ == "__main__":
    for i in range(100):
        q_table = single_q_learning(
            f"data/transitions/instance_{i + 1:02d}.txt",
            alpha=0.2,
        )
        q_table.to_txt(f"data/q_tables/single/instance_{i + 1:02d}.txt")
