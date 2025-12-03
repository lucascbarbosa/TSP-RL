"""Double Q-Learning implementation."""
import random
import torch
from pathlib import Path
from typing import Tuple, Dict
from utils.mdp import build_mdp_model
from utils.plot import plot_double_q_learning, plot_rollout
from utils.q_learning.table import QTable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def double_q_learning(
    transition_filename: str,
    gamma: float = 0.99,
    max_iter: int = 100,
    tol: float = 1e-6,
    q_table_1: QTable | None = None,
    q_table_2: QTable | None = None,
) -> Tuple[QTable, QTable, Dict[str, list]]:
    """Train Q-table from MDP using Double Q-Learning.

    Returns:
        Tuple of (QTable1, QTable2, history) where history contains
        'avg_q_value_q1' and 'avg_q_value_q2' lists
    """
    # Build MDP from transition data
    mdp = build_mdp_model(transition_filename)
    n_states = mdp.n_states
    n_actions = mdp.n_actions

    # Initialize Q-tables
    if q_table_1 is None:
        q_current_1 = QTable(
            n_states=n_states,
            n_actions=n_actions,
        )
    else:
        q_current_1 = q_table_1.copy()
    if q_table_2 is None:
        q_current_2 = QTable(
            n_states=n_states,
            n_actions=n_actions,
        )
    else:
        q_current_2 = q_table_2.copy()

    # Get MDP matrices
    P = mdp.transition_matrix  # (n_states, n_actions, n_states)
    R = mdp.reward_matrix  # (n_states, n_actions)

    # Training history
    history = {'avg_q_value_q1': [], 'avg_q_value_q2': []}

    for i in range(max_iter):
        q_new_1 = QTable(
            n_states=n_states,
            n_actions=n_actions,
        )
        q_new_2 = QTable(
            n_states=n_states,
            n_actions=n_actions,
        )
        # Vectorized computation: compute max Q-values for all next states
        max_q_values_1 = q_current_1.table.max(dim=1)[0]  # (n_states,)
        max_q_values_2 = q_current_2.table.max(dim=1)[0]  # (n_states,)

        # Compute expected values for all (state, action) pairs at once
        # Q1 uses Q2's max values, Q2 uses Q1's max values
        expected_values_1 = torch.einsum('san,n->sa', P, max_q_values_2)
        expected_values_2 = torch.einsum('san,n->sa', P, max_q_values_1)

        # Update Q-tables:
        # Q1'(s, a) = R(s, a) + gamma * E[max_{a'}Q2(s', a')]
        # Q2'(s, a) = R(s, a) + gamma * E[max_{a'}Q1(s', a')]
        q_new_1.table = R + gamma * expected_values_1
        q_new_2.table = R + gamma * expected_values_2

        # Track average Q values
        avg_q1 = q_current_1.table.mean().item()
        avg_q2 = q_current_2.table.mean().item()
        history['avg_q_value_q1'].append(avg_q1)
        history['avg_q_value_q2'].append(avg_q2)

        if (
            (q_current_1.table - q_new_1.table).abs().max() < tol and
            (q_current_2.table - q_new_2.table).abs().max() < tol
        ):
            break

        q_current_1 = q_new_1.copy()
        q_current_2 = q_new_2.copy()

        if i % 100 == 0:
            print(
                f"Iter [{i + 1}/{max_iter}]: "
                f"Avg Q-value Q1: {avg_q1:.4f}, "
                f"Avg Q-value Q2: {avg_q2:.4f}"
            )

    return q_current_1, q_current_2, history


if __name__ == "__main__":
    import shutil

    # Clear previous results
    plots_dir = Path("data/plots/double")
    q_tables_dir = Path("data/q_tables/double")
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
    if q_tables_dir.exists():
        shutil.rmtree(q_tables_dir)
        q_tables_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    eval_ratio = 0.3  # 30% for evaluation, 70% for training
    transitions_dir = Path("data/transitions")

    # List all available instance files
    all_files = sorted(transitions_dir.glob("instance_*.txt"))
    all_instances = []
    for file in all_files:
        # Extract instance number from filename
        try:
            instance_num = int(file.stem.split("_")[1])
            all_instances.append(instance_num)
        except (ValueError, IndexError):
            continue

    # Split instances based on eval_ratio
    random.shuffle(all_instances)
    n_eval = int(len(all_instances) * eval_ratio)
    eval_instances = sorted(all_instances[:n_eval])
    train_instances = sorted(all_instances[n_eval:])

    # Train on training instances
    print(f"Training on {len(train_instances)} instances")
    q_table_1 = None
    q_table_2 = None
    for i, instance_idx in enumerate(train_instances):
        print(f"==== Instance {instance_idx:02d} ====")
        q_table_1, q_table_2, history = double_q_learning(
            f"data/transitions/instance_{instance_idx:02d}.txt",
            max_iter=5000,
            q_table_1=q_table_1,
            q_table_2=q_table_2,
        )
        q_table_1.to_txt(
            f"data/q_tables/double/instance_{instance_idx:02d}_q_table_1.txt"
        )
        q_table_2.to_txt(
            f"data/q_tables/double/instance_{instance_idx:02d}_q_table_2.txt"
        )
        plot_path = f"data/plots/double/instance_{instance_idx:02d}_train.png"
        plot_double_q_learning(history, plot_path)

    # Evaluate with rollout on eval instances using Q1
    print(f"Evaluating on {len(eval_instances)} instances")
    for i, instance_idx in enumerate(eval_instances):
        print(f"==== Instance {instance_idx:02d} ====")
        rollout_history = q_table_1.rollout(
            f"data/transitions/instance_{instance_idx:02d}.txt",
            n_simulations=100,
            initial_state=0,
            max_steps=1000,
        )
        plot_path = f"data/plots/double/instance_{instance_idx:02d}_eval.png"
        plot_rollout(rollout_history, plot_path)
