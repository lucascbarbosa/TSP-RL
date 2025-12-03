"""Basic Q-Learning implementation."""
import torch
import random
from pathlib import Path
from typing import Tuple, Dict
from utils.mdp import build_mdp_model
from utils.plot import plot_single_q_learning, plot_rollout
from utils.q_learning.table import QTable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def single_q_learning(
    transition_filename: str,
    gamma: float = 0.99,
    max_iter: int = 1000,
    tol: float = 1e-6,
    q_table: QTable | None = None,
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

    if q_table is None:
        q_current = QTable(
            n_states=n_states,
            n_actions=n_actions,
        )
    else:
        q_current = q_table.copy()

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
        if i % 100 == 0:
            print(
                f"Iter [{i + 1}/{max_iter}]: "
                f"Avg Q-value: {avg_q:.4f}"
            )

    return q_current, history


if __name__ == "__main__":
    import shutil

    # Clear previous results
    plots_dir = Path("data/plots/single")
    q_tables_dir = Path("data/q_tables/single")
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
    random.shuffle(all_instances[:10])
    n_eval = int(len(all_instances) * eval_ratio)
    eval_instances = sorted(all_instances[:n_eval])
    train_instances = sorted(all_instances[n_eval:])

    # Train on training instances
    q_table = None
    print(f"Training on {len(train_instances)} instances")
    for i, instance_idx in enumerate(train_instances):
        print(f"==== Instance {instance_idx:02d} ====")
        q_table, history = single_q_learning(
            f"data/transitions/instance_{instance_idx:02d}.txt",
            max_iter=5000,
            q_table=q_table,
        )
        q_table.to_txt(
            f"data/q_tables/single/instance_{instance_idx:02d}.txt"
        )
        plot_path = f"data/plots/single/instance_{instance_idx:02d}_train.png"
        plot_single_q_learning(history, plot_path)

    # Evaluate with rollout on eval instances
    print(f"Evaluating on {len(eval_instances)} instances")
    for i, instance_idx in enumerate(eval_instances):
        print(f"==== Instance {instance_idx:02d} ====")
        rollout_history = q_table.rollout(
            f"data/transitions/instance_{instance_idx:02d}.txt",
            n_simulations=100,
            initial_state=0,
            max_steps=1000,
        )
        plot_path = f"data/plots/single/instance_{instance_idx:02d}_eval.png"
        plot_rollout(rollout_history, plot_path)
