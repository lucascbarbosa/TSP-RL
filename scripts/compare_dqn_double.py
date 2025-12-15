#!/usr/bin/env python
"""
Compare DQN vs Double DQN on TSP instances.

When called from pipeline (after training step), loads existing stats and generates plots.
When called standalone, trains both variants from scratch if stats don't exist.

Usage:
    python scripts/compare_dqn_double.py --type EUC_2D --sizes 30 50 --episodes 1000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

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


def format_hyperparams(episodes: int, lr: float, gamma: float) -> str:
    """Format hyperparameters for plot subtitle."""
    return f"ep={episodes}, lr={lr}, γ={gamma}"


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
    models_dir: str = "models/dqn",
    verbose: bool = True,
) -> dict:
    """
    Run DQN vs Double DQN comparison.

    If stats already exist (from training step), loads them and generates plots.
    If stats don't exist, trains both variants from scratch (standalone mode).
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("data/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

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

        # Check if stats already exist (from training step)
        standard_stats_path = models_dir / f"{inst_type}_n{size:03d}_standard_stats.json"
        double_stats_path = models_dir / f"{inst_type}_n{size:03d}_double_stats.json"

        variants = [
            ("DQN", "standard", False),
            ("Double DQN", "double", True),
        ]

        all_stats = []
        all_labels = []
        size_results = {}
        stats_loaded = False

        # Try to load existing stats
        if standard_stats_path.exists() and double_stats_path.exists():
            if verbose:
                print("  Loading existing stats from training step...")
            stats_loaded = True

            for variant_name, variant_suffix, use_double in variants:
                stats_path = models_dir / f"{inst_type}_n{size:03d}_{variant_suffix}_stats.json"
                with open(stats_path) as f:
                    stats_dict = json.load(f)

                all_stats.append(stats_dict)
                all_labels.append(variant_name)

                # Extract hyperparams from loaded stats (for plot titles)
                episodes = stats_dict.get("n_episodes", episodes)
                lr = stats_dict.get("lr", lr)
                gamma = stats_dict.get("gamma", gamma)

                size_results[variant_name] = {
                    "model_path": str(models_dir / f"{inst_type}_n{size:03d}_{variant_suffix}.pt"),
                    "final_avg_gap": stats_dict.get("final_avg_gap", 0),
                    "test_avg_gap": stats_dict.get("test_avg_gap"),
                    "test_std_gap": stats_dict.get("test_std_gap"),
                }

        # If stats don't exist, train from scratch (standalone mode)
        if not stats_loaded:
            if verbose:
                print(f"  Train instances: {len(size_train_ids)}")
                print(f"  Test instances: {len(size_test_ids)}")
                print("  Training variants from scratch...")

            # Load instances
            train_dataset = TSPDataset(dataset_path, size_train_ids)
            train_instances = list(train_dataset)

            test_instances = []
            if size_test_ids:
                test_dataset = TSPDataset(dataset_path, size_test_ids)
                test_instances = list(test_dataset)

            for variant_name, variant_suffix, use_double in variants:
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

                # Evaluate (returns unsupervised and supervised gaps)
                test_gaps = []
                test_gaps_unsup = []
                if test_instances:
                    test_gaps_unsup, test_gaps = evaluate_dqn(
                        model,
                        test_instances,
                        config,
                        verbose=False,
                        dataset_path=dataset_path,
                        instance_ids=size_test_ids,
                    )

                stats_dict = stats_to_dict(stats)
                stats_dict["test_gaps"] = test_gaps
                stats_dict["test_gaps_unsup"] = test_gaps_unsup
                stats_dict["test_avg_gap"] = float(np.mean(test_gaps)) if test_gaps else None
                stats_dict["test_std_gap"] = float(np.std(test_gaps)) if test_gaps else None
                stats_dict["test_avg_gap_unsup"] = float(np.mean(test_gaps_unsup)) if test_gaps_unsup else None
                # Add hyperparams for plot titles
                stats_dict["n_episodes"] = episodes
                stats_dict["lr"] = lr
                stats_dict["gamma"] = gamma
                stats_dict["time_budget"] = time_budget
                stats_dict["hidden_dim"] = hidden_dim

                all_stats.append(stats_dict)
                all_labels.append(variant_name)

                # Save model
                model_path = models_dir / f"{inst_type}_n{size:03d}_{variant_suffix}.pt"
                save_model(model, model_path)

                # Save stats
                stats_path = models_dir / f"{inst_type}_n{size:03d}_{variant_suffix}_stats.json"
                with open(stats_path, "w") as f:
                    json.dump(stats_dict, f, indent=2)

                size_results[variant_name] = {
                    "model_path": str(model_path),
                    "final_avg_gap": float(np.mean(stats.episode_best_gaps[-100:])),
                    "test_avg_gap": stats_dict["test_avg_gap"],
                    "test_std_gap": stats_dict["test_std_gap"],
                }

                if verbose and stats_dict["test_avg_gap"] is not None:
                    unsup = stats_dict["test_avg_gap_unsup"]
                    print(
                        f"  Test: sup={stats_dict['test_avg_gap']:.2f}% ± {stats_dict['test_std_gap']:.2f}%, unsup={unsup:.2f}%"
                    )

        # Generate comparison plots with hyperparameters in title
        if verbose:
            print(f"\nGenerating comparison plots...")

        hp_str = format_hyperparams(episodes, lr, gamma)

        # 1. Learning curves comparison
        plot_learning_curves_comparison(
            all_stats,
            all_labels,
            title=f"Learning Curves: {inst_type} n={size}\n({hp_str})",
            save_path=plots_dir / f"{inst_type}_n{size:03d}_learning_curve_comparison.png",
        )

        # 2. Q-values evolution comparison
        plot_q_values_comparison(
            all_stats,
            all_labels,
            title=f"Q-values Evolution: {inst_type} n={size}\n({hp_str})",
            save_path=plots_dir / f"{inst_type}_n{size:03d}_q_values_comparison.png",
        )

        # 3. Action distributions comparison
        plot_action_distribution_comparison(
            all_stats,
            all_labels,
            action_labels=action_labels,
            title=f"Action Distribution (last 10% episodes): {inst_type} n={size}\n({hp_str})",
            save_path=plots_dir / f"{inst_type}_n{size:03d}_action_dist_comparison.png",
        )

        results[size] = size_results

        # Print summary
        if verbose:
            print(f"\n--- Summary for n={size} ---")
            for variant_name, res in size_results.items():
                test_str = f"{res['test_avg_gap']:.2f}%" if res["test_avg_gap"] else "N/A"
                print(f"  {variant_name}: train={res['final_avg_gap']:.2f}%, test={test_str}")

    if verbose:
        print(f"\nPlots saved to: {plots_dir}/")

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
    parser.add_argument("--models_dir", type=str, default="models/dqn")
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
        models_dir=args.models_dir,
    )

    print("\nComparison complete!")


if __name__ == "__main__":
    main()
