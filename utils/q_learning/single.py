"""Basic Q-Learning implementation."""
import torch
from utils.q_learning.table import QTable
from utils.mdp import build_mdp_model
from typing import Tuple, Dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def single_q_learning(
    transition_filename: str,
    gamma: float = 0.99,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Tuple[QTable, Dict[str, list]]:
    """Train Q-table from MDP using Q-Learning.

    Returns:
        Tuple of (QTable, history) where history contains 'avg_q_value' list
    """
    # Build MDP from transition data
    mdp = build_mdp_model(transition_filename)
    n_states = mdp.n_states
    n_actions = mdp.n_actions

    # Get MDP matrices
    P = mdp.transition_matrix  # (n_states, n_actions, n_states)
    R = mdp.reward_matrix  # (n_states, n_actions)

    # Training history
    history = {'avg_q_value': []}

    q_current = QTable(
        n_states=n_states,
        n_actions=n_actions,
    )
    for i in range(max_iter):
        q_new = QTable(
            n_states=n_states,
            n_actions=n_actions,
        )

        # Vectorized computation
        # Q'(s, a)  = R(s,a) + gamma * E[max_{a'}Q(s',a')]
        max_q_values = q_current.table.max(dim=1)[0]
        expected_values = torch.einsum('san,n->sa', P, max_q_values)
        q_new.table = R + gamma * expected_values

        # Track average Q value
        avg_q = q_current.table.mean().item()
        history['avg_q_value'].append(avg_q)

        if (q_current.table - q_new.table).abs().max() < tol:
            break

        q_current = q_new.copy()
        if i % 10 == 0:
            print(
                f"Iter [{i + 1}/{max_iter}]: "
                f"Avg Q-value: {avg_q:.4f}"
            )

    return q_current, history


if __name__ == "__main__":
    from utils.plot import plot_single_q_learning

    for i in range(70):
        q_table, history = single_q_learning(
            f"data/transitions/instance_{i + 1:02d}.txt",
            max_iter=5000,
        )
        q_table.to_txt(f"data/q_tables/single/instance_{i + 1:02d}.txt")
        plot_path = f"data/plots/single/instance_{i + 1:02d}.png"
        plot_single_q_learning(history, plot_path)

    for i in range(70, 100):
        history = q_table.rollout(
            f"data/transitions/instance_{i + 1:02d}.txt",
            n_simulations=100,
            initial_state=0,
            max_steps=1000,
        )
        plot_path = f"data/plots/single/instance_{i + 1:02d}.png"
        plot_single_q_learning(history, plot_path)