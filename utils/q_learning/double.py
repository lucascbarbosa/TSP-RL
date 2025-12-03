"""Double Q-Learning implementation."""
import torch
from utils.q_learning.table import QTable
from utils.transition import load_transition_file
from typing import Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def double_q_learning(
    transition_filename: str,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    max_iter: int = 100,
) -> Tuple[QTable, QTable]:
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

    # Initialize Q-tables
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

    for i in range(max_iter):
        # Shuffle transitions for each iteration
        indices = torch.randperm(len(current_states))

        # Reset epsilon
        q_table_1.reset_epsilon()
        q_table_2.reset_epsilon()

        # Update q tables
        for idx in indices:
            state = int(current_states[idx])
            action = int(actions[idx])
            reward = rewards[idx]
            next_state = int(next_states[idx])

            # ---- Double Q-Learning update ----
            # Randomly select Q1 or Q2
            if torch.rand(1).item() < 0.5:
                # action selection by Q1
                a_star = torch.argmax(q_table_1.table[next_state]).item()
                # evaluation by Q2
                target = reward + gamma * q_table_2.table[next_state, a_star]
                # Update Q1
                q_table_1.table[state, action] += (
                    alpha * (target - q_table_1.table[state, action])
                )

            else:
                # action selection by Q2
                a_star = torch.argmax(q_table_2.table[next_state]).item()
                # evaluation by Q1
                target = reward + gamma * q_table_1.table[next_state, a_star]
                # Update Q2
                q_table_2.table[state, action] += (
                    alpha * (target - q_table_2.table[state, action])
                )

            # Decay epsilon in both tables
            q_table_1.decay_epsilon()
            q_table_2.decay_epsilon()

        if i % 10 == 0:
            print(
                f"Iter [{i + 1}/{max_iter}]: "
                f"Avg Q-value Q1: {q_table_1.table.mean().item():.4f}, "
                f"Avg Q-value Q2: {q_table_2.table.mean().item():.4f}"
            )

    return q_table_1, q_table_2


if __name__ == "__main__":
    for i in range(100):
        q_table_1, q_table_2 = double_q_learning(
            f"data/transitions/instance_{i + 1:02d}.txt")
        q_table_1.to_txt(
            f"data/q_tables/double/instance_{i + 1:02d}_q_table_1.txt")
        q_table_2.to_txt(
            f"data/q_tables/double/instance_{i + 1:02d}_q_table_2.txt")
