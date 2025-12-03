"""Double Q-Learning implementation."""
import numpy as np
from table import QTable
from typing import Tuple


def double_q_learning(
    transition_filename: str,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
) -> Tuple[QTable, QTable]:
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

    # Initialize Q-tables as numpy arrays
    q_table_1 = QTable(
        n_states=n_states,
        n_actions=n_actions,
        epsilon=epsilon,
    )
    q_table_2 = QTable(
        n_states=n_states,
        n_actions=n_actions,
        epsilon=epsilon,
    )

    # Shuffle transitions for each iteration
    indices = np.random.permutation(len(transition_matrix))

    for idx in indices:
        state = int(current_states[idx])
        action = int(actions[idx])
        reward = rewards[idx]
        next_state = int(next_states[idx])

        # ---- Double Q-Learning update ----
        # Randomly select Q1 or Q2
        if np.random.rand() < 0.5:
            # action selection by Q1
            a_star = np.argmax(q_table_1.table[next_state])
            # evaluation by Q2
            target = reward + gamma * q_table_2.table[next_state, a_star]
            # Update Q1
            q_table_1.table[state, action] += (
                alpha * (target - q_table_1.table[state, action])
            )
        else:
            # action selection by Q2
            a_star = np.argmax(q_table_2.table[next_state])
            # evaluation by Q1
            target = reward + gamma * q_table_1.table[next_state, a_star]
            # Update Q2
            q_table_2.table[state, action] += (
                alpha * (target - q_table_2.table[state, action])
            )

        # Decay epsilon in both tables
        q_table_1.decay_epsilon()
        q_table_2.decay_epsilon()

    return q_table_1, q_table_2


if __name__ == "__main__":
    for i in range(10):
        q_table_1, q_table_2 = double_q_learning(
            f"data/transitions/instance_{i + 1:02d}.txt")
        q_table_1.to_txt(
            f"data/q_tables/double/instance_{i + 1:02d}_q_table_1.txt")
        q_table_2.to_txt(
            f"data/q_tables/double/instance_{i + 1:02d}_q_table_2.txt")
