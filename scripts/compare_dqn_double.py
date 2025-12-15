#!/usr/bin/env python
"""
Compare DQN vs Double DQN on TSP instances.

Trains both variants, evaluates on test set, and generates comparative plots.

Usage:
    python scripts/compare_dqn_double.py --type EUC_2D --sizes 30 50 --episodes 1000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from dataclasses import asdict

import numpy as np

from src.rl.dqn import DQNConfig, train_dqn, evaluate_dqn, save_model, N_ACTIONS, get_default_workers
from src.rl.dqn.env import ACTION_DECODE
from src.tsp.instance import TSPDataset
from utils.plot import (
    plot_learning_curves_comparison,
    plot_q_values_comparison,
    plot_action_distribution_comparison,
)


def stats_to_dict(stats) -> dict:
    """Convert TrainingStats to dict for JSON serialization and plotting."""
    return {
        "episode_rewards": stats.episode_rewards,
        "episode_best_gaps": stats.episode_best_gaps,
        "episode_lengths": stats.episode_lengths,
        "losses": stats.losses,
        "epsilons": stats.epsilons,
        "action_counts": {str(k): v for k, v in stats.action_counts.items()},
        "q_values_mean": stats.q_values_mean,
        "q_values_max": stats.q_values_max,
        "use_double_dqn": stats.use_double_dqn,
    }


def get_action_labels() -> list[str]:
    """Get human-readable action labels."""
    labels = []
    for i in range(N_ACTIONS):
        pert, ls = ACTION_DECODE[i]
        # Shorten names
        pert_short = pert.replace("segment_reverse", "seg_rev").replace("grasp_", "G")
        ls_short = ls.replace("two_opt_", "2o_").replace("lin_kernighan", "LK")
        labels.append(f"{i}: {pert_short}+{ls_short}")
    return labels


def run_comparison(
    inst_type: str,
    sizes: list[int],
    splits: dict,
    episodes: int = 1000,
    time_budget: float = 10.0,
    gamma: float = 0.99,
    lr: float = 0.001,
    hidden_dim: int = 64,
    history_len: int = 1,
    device: str = "cpu",
    workers: int = 1,
    train_limit: int | None = None,
    output_dir: str = "data/results/comparison",
    verbose: bool = True,
) -> dict:
    """
    Run DQN vs Double DQN comparison.

    Trains both variants for each size, evaluates, and generates plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    dataset_path = f"data/{inst_type}.json"
    if dataset_path not in splits:
        if verbose:
            print(f"No splits for {dataset_path}")
        return {}

    train_ids = splits[dataset_path]["train"]
    test_ids = splits[dataset_path]["test"]
    results = {}
    action_labels = get_action_labels()

    for size in sizes:
        # Filter by size
        size_start = (size // 10 - 1) * 1111
        size_end = (size // 10) * 1111
        size_train_ids = [i for i in train_ids if size_start <= i < size_end]
        size_test_ids = [i for i in test_ids if size_start <= i < size_end]

        if train_limit:
            size_train_ids = size_train_ids[:train_limit]

        if not size_train_ids:
            if verbose:
                print(f"No training instances for size {size}, skipping...")
            continue

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Comparing DQN variants: {inst_type} n={size}")
            print(f"{'=' * 60}")
            print(f"  Train instances: {len(size_train_ids)}")
            print(f"  Test instances: {len(size_test_ids)}")

        # Load instances
        train_dataset = TSPDataset(dataset_path, size_train_ids)
        train_instances = list(train_dataset)

        test_instances = []
        if size_test_ids:
            test_dataset = TSPDataset(dataset_path, size_test_ids)
            test_instances = list(test_dataset)

        variants = [
            ("DQN", False),
            ("Double DQN", True),
        ]

        all_stats = []
        all_labels = []
        size_results = {}

        for variant_name, use_double in variants:
            if verbose:
                print(f"\n--- Training {variant_name} ---")

            config = DQNConfig(
                time_budget=time_budget,
                history_len=history_len,
                n_episodes=episodes,
                gamma=gamma,
                lr=lr,
                hidden_dim=hidden_dim,
                device=device,
                n_workers=workers,
                use_double_dqn=use_double,
            )

            model, stats = train_dqn(
                train_instances,
                config,
                verbose=verbose,
                dataset_path=dataset_path,
                instance_ids=size_train_ids,
            )

            # Evaluate
            test_gaps = []
            if test_instances:
                test_gaps = evaluate_dqn(
                    model,
                    test_instances,
                    config,
                    verbose=False,
                    dataset_path=dataset_path,
                    instance_ids=size_test_ids,
                )

            stats_dict = stats_to_dict(stats)
            stats_dict["test_gaps"] = test_gaps
            stats_dict["test_avg_gap"] = float(np.mean(test_gaps)) if test_gaps else None
            stats_dict["test_std_gap"] = float(np.std(test_gaps)) if test_gaps else None

            all_stats.append(stats_dict)
            all_labels.append(variant_name)

            # Save model
            variant_suffix = "double" if use_double else "standard"
            model_path = output_dir / f"{inst_type}_n{size:03d}_{variant_suffix}.pt"
            save_model(model, model_path)

            # Save stats
            stats_path = output_dir / f"{inst_type}_n{size:03d}_{variant_suffix}_stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats_dict, f, indent=2)

            size_results[variant_name] = {
                "model_path": str(model_path),
                "final_avg_gap": float(np.mean(stats.episode_best_gaps[-100:])),
                "test_avg_gap": stats_dict["test_avg_gap"],
                "test_std_gap": stats_dict["test_std_gap"],
            }

            if verbose and stats_dict["test_avg_gap"] is not None:
                print(f"  Test gap: {stats_dict['test_avg_gap']:.2f}% +/- {stats_dict['test_std_gap']:.2f}%")

        # Generate comparison plots
        if verbose:
            print(f"\nGenerating comparison plots...")

        # 1. Learning curves
        plot_learning_curves_comparison(
            all_stats,
            all_labels,
            title=f"Learning Curves: {inst_type} n={size}",
            save_path=plots_dir / f"{inst_type}_n{size:03d}_learning_curves.png",
        )

        # 2. Q-values evolution
        plot_q_values_comparison(
            all_stats,
            all_labels,
            title=f"Q-values Evolution: {inst_type} n={size}",
            save_path=plots_dir / f"{inst_type}_n{size:03d}_q_values.png",
        )

        # 3. Action distributions
        plot_action_distribution_comparison(
            all_stats,
            all_labels,
            action_labels=action_labels,
            title=f"Action Distribution (last 10% episodes): {inst_type} n={size}",
            save_path=plots_dir / f"{inst_type}_n{size:03d}_actions.png",
        )

        results[size] = size_results

        # Print summary
        if verbose:
            print(f"\n--- Summary for n={size} ---")
            for variant_name, res in size_results.items():
                test_str = f"{res['test_avg_gap']:.2f}%" if res["test_avg_gap"] else "N/A"
                print(f"  {variant_name}: train={res['final_avg_gap']:.2f}%, test={test_str}")

    # Save overall results
    results_path = output_dir / f"{inst_type}_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {output_dir}")
        print(f"Plots saved to: {plots_dir}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DQN vs Double DQN")
    parser.add_argument("--type", type=str, required=True, choices=["EUC_2D", "ATT", "GEO"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[50])
    parser.add_argument("--split_path", type=str, default="data/splits.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit training instances per size")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--time_budget", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--history_len", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--workers", type=int, default=1, help=f"Parallel workers (max recommended: {get_default_workers()})"
    )
    parser.add_argument("--output_dir", type=str, default="data/results/comparison")
    args = parser.parse_args()

    # Load splits
    print(f"Loading splits from: {args.split_path}")
    with open(args.split_path) as f:
        splits = json.load(f)

    run_comparison(
        inst_type=args.type,
        sizes=args.sizes,
        splits=splits,
        episodes=args.episodes,
        time_budget=args.time_budget,
        gamma=args.gamma,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        history_len=args.history_len,
        device=args.device,
        workers=args.workers,
        train_limit=args.limit,
        output_dir=args.output_dir,
    )

    print("\nComparison complete!")


if __name__ == "__main__":
    main()
