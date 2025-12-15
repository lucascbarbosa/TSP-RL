#!/usr/bin/env python
"""
Train DQN models for DQN-ILS.

Usage:
    python scripts/train_dqn.py --type EUC_2D --sizes 10 20 --episodes 2000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np

from src.rl.dqn import DQNConfig, train_dqn, evaluate_dqn, save_model, N_ACTIONS, get_default_workers
from src.tsp.instance import TSPDataset


def run_training(
    inst_type: str,
    sizes: list[int],
    splits: dict,
    episodes: int = 2000,
    time_budget: float = 10.0,
    gamma: float = 0.99,
    lr: float = 0.001,
    hidden_dim: int = 64,
    history_len: int = 1,
    device: str = "cpu",
    workers: int = 1,
    train_limit: int | None = None,
    val_limit: int | None = None,
    output_dir: str = "models/dqn",
    verbose: bool = True,
    compare_variants: bool = False,
) -> dict:
    """
    Train DQN models for a given instance type and sizes.

    Args:
        inst_type: Instance type (EUC_2D, ATT, GEO).
        sizes: List of instance sizes to train.
        splits: Dictionary with train/val/test splits.
        episodes: Number of training episodes.
        time_budget: Base time budget in seconds.
        gamma: Discount factor.
        lr: Learning rate.
        hidden_dim: Hidden layer dimension.
        history_len: Number of past actions in state.
        device: Device for training (cpu/cuda).
        workers: Number of parallel workers.
        train_limit: Limit training instances per size (None = no limit).
        val_limit: Limit validation instances per size (None = no limit).
        output_dir: Output directory for models.
        verbose: Print progress.
        compare_variants: If True, train both DQN and Double DQN for comparison.

    Returns:
        Dictionary with training results per size.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = f"data/{inst_type}.json"
    if dataset_path not in splits:
        if verbose:
            print(f"No splits for {dataset_path}")
        return {}

    train_ids = splits[dataset_path]["train"]
    # Use 'val' if available (train/val/test split), fallback to 'test' for compatibility
    val_ids = splits[dataset_path].get("val", splits[dataset_path].get("test", []))
    results = {}

    for size in sizes:
        # Filter by size
        size_start = (size // 10 - 1) * 1111
        size_end = (size // 10) * 1111
        size_train_ids = [i for i in train_ids if size_start <= i < size_end]
        size_val_ids = [i for i in val_ids if size_start <= i < size_end]

        if train_limit:
            size_train_ids = size_train_ids[:train_limit]
        if val_limit:
            size_val_ids = size_val_ids[:val_limit]

        if not size_train_ids:
            if verbose:
                print(f"No training instances for size {size}, skipping...")
            continue

        if verbose:
            print(f"\nTraining {inst_type} n={size}")
            print(f"  Train instances: {len(size_train_ids)}")
            print(f"  Val instances: {len(size_val_ids)}")

        # Load instances
        train_dataset = TSPDataset(dataset_path, size_train_ids)
        train_instances = list(train_dataset)

        val_instances = []
        if size_val_ids:
            val_dataset = TSPDataset(dataset_path, size_val_ids)
            val_instances = list(val_dataset)

        # Define variants to train
        if compare_variants:
            variants = [
                ("standard", False),  # DQN
                ("double", True),  # Double DQN
            ]
        else:
            variants = [("double", True)]  # Default: Double DQN only

        size_results = {}

        for variant_suffix, use_double in variants:
            variant_name = "Double DQN" if use_double else "DQN"
            if verbose:
                print(f"\n  --- Training {variant_name} ---")

            # Configure training
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

            # Train
            model, stats = train_dqn(
                train_instances,
                config,
                verbose=verbose,
                dataset_path=dataset_path,
                instance_ids=size_train_ids,
            )

            # Evaluate on validation set (parallel if workers > 1)
            # Returns (gaps_vs_base, gaps_vs_opt): unsupervised and supervised gaps
            val_avg_gap = None
            val_std_gap = None
            val_avg_gap_unsup = None
            val_gaps = []
            val_gaps_unsup = []
            if val_instances:
                val_gaps_unsup, val_gaps = evaluate_dqn(
                    model,
                    val_instances,
                    config,
                    verbose=False,
                    dataset_path=dataset_path,
                    instance_ids=size_val_ids,
                )
                val_avg_gap = float(np.mean(val_gaps))
                val_std_gap = float(np.std(val_gaps))
                val_avg_gap_unsup = float(np.mean(val_gaps_unsup))
                if verbose:
                    print(
                        f"  Val: Supervised={val_avg_gap:.2f}% ± {val_std_gap:.2f}%, Unsupervised={val_avg_gap_unsup:.2f}%"
                    )

            # Determine filename suffix
            if compare_variants:
                model_path = output_dir / f"{inst_type}_n{size:03d}_{variant_suffix}.pt"
                stats_path = output_dir / f"{inst_type}_n{size:03d}_{variant_suffix}_stats.json"
            else:
                model_path = output_dir / f"{inst_type}_n{size:03d}.pt"
                stats_path = output_dir / f"{inst_type}_n{size:03d}_stats.json"

            # Save model
            save_model(model, model_path)
            if verbose:
                print(f"  Model saved to: {model_path}")

            # Save stats (include Q-value stats and hyperparams for plotting)
            stats_dict = {
                "type": inst_type,
                "size": size,
                "n_actions": N_ACTIONS,
                "n_episodes": episodes,
                "use_double_dqn": use_double,
                "gamma": gamma,
                "lr": lr,
                "hidden_dim": hidden_dim,
                "time_budget": time_budget,
                "final_avg_gap": float(np.mean(stats.episode_best_gaps[-100:])),
                "val_avg_gap": val_avg_gap,  # supervised (vs optimal)
                "val_std_gap": val_std_gap,
                "val_avg_gap_unsup": val_avg_gap_unsup,  # unsupervised (vs baseline)
                "val_gaps": val_gaps,
                "val_gaps_unsup": val_gaps_unsup,
                "episode_best_gaps": stats.episode_best_gaps,
                "episode_rewards": stats.episode_rewards,
                "episode_lengths": stats.episode_lengths,
                "losses": stats.losses,
                "epsilons": stats.epsilons,
                "action_counts": {str(k): v for k, v in stats.action_counts.items()},
                "q_values_mean": stats.q_values_mean,
                "q_values_max": stats.q_values_max,
            }
            with open(stats_path, "w") as f:
                json.dump(stats_dict, f, indent=2)

            size_results[variant_suffix] = {
                "model_path": str(model_path),
                "stats_path": str(stats_path),
                "final_avg_gap": stats_dict["final_avg_gap"],
                "val_avg_gap": val_avg_gap,
                "use_double_dqn": use_double,
            }

        # Store results
        if compare_variants:
            results[size] = size_results
        else:
            # Single variant: flatten structure for backward compatibility
            results[size] = size_results["double"]

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN for DQN-ILS")
    parser.add_argument("--type", type=str, required=True, choices=["EUC_2D", "ATT", "GEO"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[50])
    parser.add_argument("--split_path", type=str, default="data/splits.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit train instances per size")
    parser.add_argument("--val_limit", type=int, default=None, help="Limit validation instances per size")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--time_budget", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--history_len", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--workers", type=int, default=1, help=f"Parallel workers (max recommended: {get_default_workers()})"
    )
    parser.add_argument("--output_dir", type=str, default="models/dqn")
    args = parser.parse_args()

    # Load splits
    print(f"Loading splits from: {args.split_path}")
    with open(args.split_path) as f:
        splits = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"Training DQN: type={args.type}, sizes={args.sizes}")
    print(f"{'=' * 60}")

    run_training(
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
        val_limit=args.val_limit,
        output_dir=args.output_dir,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
