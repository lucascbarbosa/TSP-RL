#!/usr/bin/env python
"""
Train Q-tables from transition data using value iteration.

Usage:
    python scripts/train_qtable.py
    python scripts/train_qtable.py --types EUC_2D --sizes 10 20 30
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import re
import shutil
import time
from pathlib import Path

from tqdm import tqdm

from src.rl.q_learning import train_q_table_from_paths
from src.rl.q_table import QTable

# Optional: import plotting utilities
try:
    from utils.plot import plot_q_convergence, plot_q_heatmap

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def get_transition_files(
    transitions_dir: Path,
    train_ids: list[int],
) -> list[tuple[int, str, int]]:
    """
    Get list of transition files matching training instances.

    Args:
        transitions_dir: Directory containing transition files.
        train_ids: List of training instance IDs.

    Returns:
        List of (instance_num, file_path, n_nodes) tuples.
    """
    all_files = sorted(transitions_dir.glob("*.txt"))
    train_instances: list[tuple[int, str, int]] = []

    for file in all_files:
        match = re.search(r"random_instance_(\d+)_nodes_(\d+)", str(file))
        if match:
            instance_num = int(match.group(1))
            n_nodes = int(match.group(2))

            if instance_num in train_ids:
                train_instances.append((instance_num, str(file), n_nodes))

    return train_instances


def main() -> None:
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
    parser.add_argument(
        "--max_iter",
        type=int,
        default=5000,
        help="Max iterations for Q-learning (default: 5000)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="data/splits.json",
        help="Path to splits JSON file",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable plot generation",
    )
    args = parser.parse_args()

    # Load splits
    with open(args.splits, "r") as f:
        splits = json.load(f)

    # Training log
    training_log: list[dict] = []

    # Process each instance type
    for instance_type in tqdm(args.types, desc="Instance types", ncols=120):
        plots_dir = Path(f"data/plots/{instance_type}")
        q_tables_dir = Path(f"data/q_tables/{instance_type}")

        # Clean and create directories
        if plots_dir.exists():
            shutil.rmtree(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        if q_tables_dir.exists():
            shutil.rmtree(q_tables_dir)
        q_tables_dir.mkdir(parents=True, exist_ok=True)

        transitions_dir = Path(f"data/train/{instance_type}")

        if not transitions_dir.exists():
            print(f"  [!] No transitions found for {instance_type}, skipping...")
            continue

        # Get training instance IDs
        split_key = f"data/{instance_type}.json"
        if split_key not in splits:
            print(f"  [!] No splits found for {instance_type}, skipping...")
            continue

        train_ids = splits[split_key]["train"]

        # Get transition files
        train_instances = get_transition_files(transitions_dir, train_ids)
        print(f"\n  Training {instance_type} on {len(train_instances)} transition files")

        # Train incrementally by size
        q_table: QTable | None = None

        for n_cities in tqdm(args.sizes, desc=f"  {instance_type} sizes", ncols=120, leave=False):
            paths = [i[1] for i in train_instances if i[2] == n_cities]

            if not paths:
                continue

            t_start = time.perf_counter()
            q_table, history = train_q_table_from_paths(
                paths,
                gamma=args.gamma,
                max_iter=args.max_iter,
                q_table=q_table,
                min_n_actions=8,  # 8 actions (5 perturbations x 2 local searches, with omissions)
                min_n_states=5,  # 5 states (gap-based)
            )
            train_time_ms = (time.perf_counter() - t_start) * 1000

            # Record training stats
            iterations_used = len(history.get("avg_q_value", []))
            final_avg_q = history["avg_q_value"][-1] if history.get("avg_q_value") else 0.0
            training_log.append(
                {
                    "type": instance_type,
                    "size": n_cities,
                    "n_files": len(paths),
                    "iterations": iterations_used,
                    "time_ms": train_time_ms,
                    "final_avg_q": final_avg_q,
                    "gamma": args.gamma,
                }
            )

            # Save Q-table
            q_table.to_txt(f"{q_tables_dir}/instance_size_{n_cities:02d}.txt")

            # Generate plots if available and enabled
            if HAS_PLOTTING and not args.no_plots:
                convergence_path = f"{plots_dir}/{instance_type}_{n_cities:02d}_convergence.png"
                heatmap_path = f"{plots_dir}/{instance_type}_{n_cities:02d}_qtable.png"

                plot_q_convergence(
                    history,
                    title=f"Q-Learning Convergence ({instance_type}, n={n_cities})",
                    save_path=convergence_path,
                )
                plot_q_heatmap(
                    q_table.table.detach().cpu().numpy(),
                    title=f"Q-table ({instance_type}, n={n_cities})",
                    save_path=heatmap_path,
                )

    # Save training log
    if training_log:
        log_path = Path("data/q_tables/training_log.csv")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
            writer.writeheader()
            writer.writerows(training_log)
        print(f"\nTraining log saved to {log_path}")

    print("Done.")


if __name__ == "__main__":
    main()
