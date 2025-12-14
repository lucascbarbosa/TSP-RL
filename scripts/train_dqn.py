#!/usr/bin/env python
"""
Train DQN models for Q-ILS.

Trains a DQN agent to select (perturbation, local_search) pairs
for each instance type and size combination.

Usage:
    python scripts/train_dqn.py --type EUC_2D --size 50 --episodes 2000
    python scripts/train_dqn.py --type EUC_2D --sizes 10 20 30 --episodes 1000
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np

from src.rl.dqn import DQNConfig, train_dqn, evaluate_dqn, save_model, N_ACTIONS, get_default_workers
from src.tsp.instance import TSPDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN for Q-ILS")

    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["EUC_2D", "ATT", "GEO"],
        help="Instance type",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[50],
        help="Instance sizes to train (default: 50)",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default="data/splits.json",
        help="Path to splits JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit training instances per size (for testing)",
    )

    # DQN hyperparameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="Number of training episodes (default: 2000)",
    )
    parser.add_argument(
        "--time_budget",
        type=float,
        default=10.0,
        help="Base time budget in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden layer dimension (default: 64)",
    )
    parser.add_argument(
        "--history_len",
        type=int,
        default=2,
        help="Number of past actions in state (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for training (default: cpu)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=f"Parallel workers for training (default: 1, max recommended: {get_default_workers()})",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/dqn",
        help="Output directory for models (default: models/dqn)",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    print(f"Loading splits from: {args.split_path}")
    with open(args.split_path, "r") as f:
        splits = json.load(f)

    dataset_path = f"data/{args.type}.json"
    if dataset_path in splits:
        train_ids = splits[dataset_path]["train"]
        test_ids = splits[dataset_path]["test"]
    else:
        train_ids = splits.get("train", [])
        test_ids = splits.get("test", [])

    # Train for each size
    for size in args.sizes:
        print(f"\n{'=' * 60}")
        print(f"Training DQN: type={args.type}, size={size}")
        print(f"{'=' * 60}")

        # Filter instances by size
        # Instance structure: IDs 0-1110 = size 10, 1111-2221 = size 20, etc.
        size_start = (size // 10 - 1) * 1111
        size_end = (size // 10) * 1111
        size_train_ids = [i for i in train_ids if size_start <= i < size_end]
        size_test_ids = [i for i in test_ids if size_start <= i < size_end]

        if args.limit:
            size_train_ids = size_train_ids[: args.limit]

        if not size_train_ids:
            print(f"No training instances for size {size}, skipping...")
            continue

        print(f"Training instances: {len(size_train_ids)}")
        print(f"Test instances: {len(size_test_ids)}")

        # Load instances
        train_dataset = TSPDataset(dataset_path, size_train_ids)
        test_dataset = TSPDataset(dataset_path, size_test_ids)

        train_instances = list(train_dataset)
        test_instances = list(test_dataset)

        # Configure training
        config = DQNConfig(
            time_budget=args.time_budget,
            history_len=args.history_len,
            n_episodes=args.episodes,
            gamma=args.gamma,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            device=args.device,
            n_workers=args.workers,
        )

        # Train (pass dataset info for parallel training)
        model, stats = train_dqn(
            train_instances,
            config,
            verbose=True,
            dataset_path=dataset_path,
            instance_ids=size_train_ids,
        )

        # Evaluate on test set
        if test_instances:
            print(f"\nEvaluating on {len(test_instances)} test instances...")
            test_gaps = evaluate_dqn(model, test_instances, config, verbose=False)
            avg_gap = np.mean(test_gaps)
            std_gap = np.std(test_gaps)
            print(f"Test gap: {avg_gap:.2f}% ± {std_gap:.2f}%")

        # Save model
        model_path = output_dir / f"{args.type}_n{size:03d}.pt"
        save_model(model, model_path)
        print(f"Model saved to: {model_path}")

        # Save training stats (summary + full data for plotting)
        stats_path = output_dir / f"{args.type}_n{size:03d}_stats.json"
        stats_dict = {
            "type": args.type,
            "size": size,
            "n_actions": N_ACTIONS,
            "n_episodes": args.episodes,
            "final_avg_gap": float(np.mean(stats.episode_best_gaps[-100:])),
            "test_avg_gap": float(avg_gap) if test_instances else None,
            "test_std_gap": float(std_gap) if test_instances else None,
            # Full data for plotting
            "episode_best_gaps": stats.episode_best_gaps,
            "episode_rewards": stats.episode_rewards,
            "losses": stats.losses,
        }
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=2)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
