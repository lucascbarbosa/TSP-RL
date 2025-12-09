"""Basic Q-Learning implementation."""
import torch
import random
import json
import re
from pathlib import Path
from typing import Tuple, Dict
from utils.mdp import build_mdp_model, build_mdp_model_from_folder, build_mdp_model_from_paths
from utils.plot import plot_single_q_learning, plot_heatmap
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

    return single_q_learning_from_mdp(mdp,
        gamma,
        max_iter,
        tol,
        q_table)

def single_q_learning_from_paths(
    paths: list,
    gamma: float = 0.99,
    max_iter: int = 1000,
    tol: float = 1e-6,
    q_table: QTable | None = None,
    min_n_states:int = 0,
    min_n_actions:int = 0
) -> Tuple[QTable, Dict[str, list]]:
    """Train Q-table from MDP using Q-Learning.

    Returns:
        Tuple of (QTable, history) where history contains 'avg_q_value' list
    """
    # Build MDP from transition data
    mdp = build_mdp_model_from_paths(paths, min_n_states=min_n_states, min_n_actions=min_n_actions)

    return single_q_learning_from_mdp(mdp,
        gamma,
        max_iter,
        tol,
        q_table)

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

    return single_q_learning_from_mdp(mdp,
        gamma,
        max_iter,
        tol,
        q_table)

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

    # Clear previous result
    for instance_type in ["EUC_2D", "GEO", "ATT"]:
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

        with open("data/splits.json", "r") as f:
            splits = json.load(f)

        # List all available instance files
        all_files = sorted(transitions_dir.glob("*.txt"))
        all_instances = []
        eval_instances = []
        train_instances = []
        for file in all_files:
            # Extract instance number from filename
            match = re.search(r"random_instance_(\d+)_nodes_(\d+)", str(file))
            if match:
                # We assume the ID is the first group captured by the regex
                instance_num = int(match.group(1))
                n_nodes = int(match.group(2))
                all_instances.append(instance_num)

                if instance_num in splits[f"data/{instance_type}.json"]["train"]:
                    train_instances.append((instance_num, str(file), n_nodes))
                else:
                    eval_instances.append((instance_num, str(file), n_nodes))

        # Train on training instances
        q_table = None
        print(f"Training on {len(train_instances)} instances")
        for a in range(10):
            n_cities = (a+1)*10
            paths = [i[1] for i in train_instances if i[2] == n_cities]
            print(f"==== Instances with {n_cities:02d} nodes ====")
            q_table, history = single_q_learning_from_paths(
                paths,
                max_iter=5000,
                q_table=q_table,
                min_n_actions=15,  # 5 perturbações x 3 buscas locais
                min_n_states=5
            )
            q_table.to_txt(
                f"{q_tables_dir}/instance_size_{n_cities:02d}.txt"
            )
            plot1_path = f"{plots_dir}/instance_size_{instance_type}_{n_cities:02d}_train.png"
            plot2_path = f"{plots_dir}/instance_size_{instance_type}_{n_cities:02d}_heatmap.png"
            plot_single_q_learning(history, plot1_path)
            plot_heatmap(q_table.table.detach().cpu().numpy(), 
                        title=f"Q-table heatmap {instance_type} {n_cities}", 
                        x_labels=[str(i) for i in range(q_table.n_actions)],
                        y_labels=[str(i) for i in range(q_table.n_states)],
                        save_path=plot2_path)