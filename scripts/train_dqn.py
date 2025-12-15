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
    history_len: int = 2,
    device: str = "cpu",
    workers: int = 1,
    train_limit: int | None = None,
    output_dir: str = "models/dqn",
    verbose: bool = True,
) -> dict:
    """
    Train DQN models for a given instance type and sizes.

    Args:
        inst_type: Instance type (EUC_2D, ATT, GEO).
        sizes: List of instance sizes to train.
        splits: Dictionary with train/test splits.
        episodes: Number of training episodes.
        time_budget: Base time budget in seconds.
        gamma: Discount factor.
        lr: Learning rate.
        hidden_dim: Hidden layer dimension.
        history_len: Number of past actions in state.
        device: Device for training (cpu/cuda).
        workers: Number of parallel workers.
        train_limit: Limit training instances per size (None = no limit).
        output_dir: Output directory for models.
        verbose: Print progress.

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
    test_ids = splits[dataset_path]["test"]
    results = {}

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
            print(f"\nTraining {inst_type} n={size}")
            print(f"  Train instances: {len(size_train_ids)}")
            print(f"  Test instances: {len(size_test_ids)}")

        # Load instances
        train_dataset = TSPDataset(dataset_path, size_train_ids)
        train_instances = list(train_dataset)

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
        )

        # Train
        model, stats = train_dqn(
            train_instances,
            config,
            verbose=verbose,
            dataset_path=dataset_path,
            instance_ids=size_train_ids,
        )

        # Evaluate on test set (parallel if workers > 1)
        test_avg_gap = None
        test_std_gap = None
        if size_test_ids:
            test_dataset = TSPDataset(dataset_path, size_test_ids)
            test_instances = list(test_dataset)
            test_gaps = evaluate_dqn(
                model,
                test_instances,
                config,
                verbose=False,
                dataset_path=dataset_path,
                instance_ids=size_test_ids,
            )
            test_avg_gap = float(np.mean(test_gaps))
            test_std_gap = float(np.std(test_gaps))
            if verbose:
                print(f"Test gap: {test_avg_gap:.2f}% ± {test_std_gap:.2f}%")

        # Save model
        model_path = output_dir / f"{inst_type}_n{size:03d}.pt"
        save_model(model, model_path)
        if verbose:
            print(f"Model saved to: {model_path}")

        # Save stats
        stats_path = output_dir / f"{inst_type}_n{size:03d}_stats.json"
        stats_dict = {
            "type": inst_type,
            "size": size,
            "n_actions": N_ACTIONS,
            "n_episodes": episodes,
            "final_avg_gap": float(np.mean(stats.episode_best_gaps[-100:])),
            "test_avg_gap": test_avg_gap,
            "test_std_gap": test_std_gap,
            "episode_best_gaps": stats.episode_best_gaps,
            "episode_rewards": stats.episode_rewards,
            "losses": stats.losses,
            "action_counts": {str(k): v for k, v in stats.action_counts.items()},
        }
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=2)

        results[size] = {
            "model_path": str(model_path),
            "final_avg_gap": stats_dict["final_avg_gap"],
            "test_avg_gap": test_avg_gap,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN for DQN-ILS")
    parser.add_argument("--type", type=str, required=True, choices=["EUC_2D", "ATT", "GEO"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[50])
    parser.add_argument("--split_path", type=str, default="data/splits.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--time_budget", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--history_len", type=int, default=2)
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
        output_dir=args.output_dir,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
