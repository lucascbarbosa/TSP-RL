"""Basic Q-Learning implementation."""

import torch
import random
import json
import re
import argparse
from pathlib import Path
from typing import Tuple, Dict

from tqdm import tqdm

from utils.mdp import build_mdp_model, build_mdp_model_from_folder, build_mdp_model_from_paths
from utils.plot import plot_single_q_learning, plot_heatmap
from utils.q_learning.table import QTable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    return single_q_learning_from_mdp(mdp, gamma, max_iter, tol, q_table)


def single_q_learning_from_paths(
    paths: list,
    gamma: float = 0.99,
    max_iter: int = 1000,
    tol: float = 1e-6,
    q_table: QTable | None = None,
    min_n_states: int = 0,
    min_n_actions: int = 0,
) -> Tuple[QTable, Dict[str, list]]:
    """Train Q-table from MDP using Q-Learning.

    Returns:
        Tuple of (QTable, history) where history contains 'avg_q_value' list
    """
    # Build MDP from transition data
    mdp = build_mdp_model_from_paths(paths, min_n_states=min_n_states, min_n_actions=min_n_actions)

    return single_q_learning_from_mdp(mdp, gamma, max_iter, tol, q_table)


def single_q_learning_from_folder(
    transition_folder: str,
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
    mdp = build_mdp_model_from_folder(transition_folder)

    return single_q_learning_from_mdp(mdp, gamma, max_iter, tol, q_table)


def single_q_learning_from_mdp(
    mdp,
    gamma: float = 0.99,
    max_iter: int = 1000,
    tol: float = 1e-6,
    q_table: QTable | None = None,
) -> Tuple[QTable, Dict[str, list]]:
    n_states = mdp.n_states
    n_actions = mdp.n_actions

    # Get MDP matrices
    P = mdp.transition_matrix  # (n_states, n_actions, n_states)
    R = mdp.reward_matrix  # (n_states, n_actions)

    # Training history
    history = {"avg_q_value": []}

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
        expected_values = torch.einsum("san,n->sa", P, max_q_values)
        q_new.table = R + gamma * expected_values

        # Track average Q value
        avg_q = q_current.table.mean().item()
        history["avg_q_value"].append(avg_q)

        if (q_current.table - q_new.table).abs().max() < tol:
            break

        q_current = q_new.copy()
        if i % 100 == 0:
            print(f"Iter [{i + 1}/{max_iter}]: " f"Avg Q-value: {avg_q:.4f}")

    return q_current, history


if __name__ == "__main__":
    import shutil

    parser = argparse.ArgumentParser(description="Train Q-tables from transition data")
    parser.add_argument(
        "--types",
        type=str,
        nargs="+",
        default=["EUC_2D", "GEO", "ATT"],
        help="Instance types to train (default: EUC_2D GEO ATT)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        help="Instance sizes to train (default: 10 20 ... 100)",
    )
    parser.add_argument("--max_iter", type=int, default=5000, help="Max iterations for Q-learning (default: 5000)")
    args = parser.parse_args()

    # Process each instance type
    for instance_type in tqdm(args.types, desc="Instance types", ncols=120):
        plots_dir = Path(f"data/plots/{instance_type}")
        q_tables_dir = Path(f"data/q_tables/{instance_type}")
        if plots_dir.exists():
            shutil.rmtree(plots_dir)

        plots_dir.mkdir(parents=True, exist_ok=True)

        if q_tables_dir.exists():
            shutil.rmtree(q_tables_dir)

        q_tables_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        transitions_dir = Path(f"data/train/{instance_type}")

        if not transitions_dir.exists():
            print(f"  [!] No transitions found for {instance_type}, skipping...")
            continue

        with open("data/splits.json", "r") as f:
            splits = json.load(f)

        # List all available instance files
        all_files = sorted(transitions_dir.glob("*.txt"))
        train_instances = []
        for file in all_files:
            # Extract instance number from filename
            match = re.search(r"random_instance_(\d+)_nodes_(\d+)", str(file))
            if match:
                instance_num = int(match.group(1))
                n_nodes = int(match.group(2))

                if instance_num in splits[f"data/{instance_type}.json"]["train"]:
                    train_instances.append((instance_num, str(file), n_nodes))

        # Train on training instances
        q_table = None
        print(f"\n  Training {instance_type} on {len(train_instances)} transition files")

        for n_cities in tqdm(args.sizes, desc=f"  {instance_type} sizes", ncols=120, leave=False):
            paths = [i[1] for i in train_instances if i[2] == n_cities]
            if not paths:
                continue

            q_table, history = single_q_learning_from_paths(
                paths,
                max_iter=args.max_iter,
                q_table=q_table,
                min_n_actions=8,  # 8 ações (5 perturbações × 2 buscas locais, com omissões)
                min_n_states=5,
            )
            q_table.to_txt(f"{q_tables_dir}/instance_size_{n_cities:02d}.txt")
            plot1_path = f"{plots_dir}/instance_size_{instance_type}_{n_cities:02d}_train.png"
            plot2_path = f"{plots_dir}/instance_size_{instance_type}_{n_cities:02d}_heatmap.png"
            plot_single_q_learning(history, plot1_path)
            plot_heatmap(
                q_table.table.detach().cpu().numpy(),
                title=f"Q-table heatmap {instance_type} {n_cities}",
                x_labels=[str(i) for i in range(q_table.n_actions)],
                y_labels=[str(i) for i in range(q_table.n_states)],
                save_path=plot2_path,
            )

    print("\nDone.")
