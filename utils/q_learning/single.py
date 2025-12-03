"""Basic Q-Learning implementation."""
import numpy as np
from table import QTable


def single_q_learning(
    transition_filename: str,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
) -> QTable:
    """Train Q-table from transition matrix using Q-Learning."""
    # Read transition matrix
    transition_matrix = np.loadtxt(transition_filename)

    # Extract columns
    current_states = transition_matrix[:, 0].astype(int)
    actions = transition_matrix[:, 1].astype(int)
    rewards = transition_matrix[:, 2].astype(int)
    next_states = transition_matrix[:, 3].astype(int)

    # Determine number of states and actions
    max_state = max(current_states.max(), next_states.max())
    n_states = max_state + 1
    max_action = actions.max()
    n_actions = max_action + 1

    # Initialize Q-table as numpy array
    q_table = QTable(
        n_states=n_states,
        n_actions=n_actions,
        epsilon=epsilon,
    )

    # Shuffle transitions for each iteration
    indices = np.random.permutation(len(transition_matrix))

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
                gamma * np.argmax(q_table.table[next_state]) -
                q_table.table[state, action]
            )
        )

        # Decay epsilon
        q_table.decay_epsilon()

    return q_table


if __name__ == "__main__":
    for i in range(10):
        q_table = single_q_learning(
            f"data/transitions/instance_{i + 1:02d}.txt")
        q_table.to_txt(f"data/q_tables/single/instance_{i + 1:02d}.txt")
