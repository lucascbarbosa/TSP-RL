#!/usr/bin/env python
"""
DQN-ILS Pipeline: train and evaluate models.

Orchestrates the training pipeline by calling module functions directly.

Usage:
    python scripts/pipeline.py                     # Use defaults
    python scripts/pipeline.py --types EUC_2D ATT --sizes 10 20
    python scripts/pipeline.py --episodes 500 --train_limit 50
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class PipelineConfig:
    """Pipeline configuration."""

    types: list[str] = field(default_factory=lambda: ["EUC_2D"])
    sizes: list[int] = field(default_factory=lambda: [10, 20])
    episodes: int = 200
    time_budget: float = 5.0
    gamma: float = 0.99
    lr: float = 0.001
    hidden_dim: int = 64
    history_len: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    reward_type: str = "delta"
    train_limit: int = 100
    val_limit: int = 40  # Validation instances during training (step 2)
    test_limit: int = 40  # Test instances for final evaluation (step 3)
    baseline: bool = True
    device: str = "cpu"
    workers: int = 16
    compare_double: bool = True  # Run DQN vs Double DQN comparison


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def run_step(name: str, func, *args, **kwargs):
    """Run a pipeline step with timing."""
    print(f"\n{'=' * 50}")
    print(f"{name}")
    print("=" * 50)

    start = time.time()
    result = func(*args, **kwargs)
    duration = time.time() - start

    print(f"   Duration: {format_duration(duration)}")
    return duration, result


def main():
    parser = argparse.ArgumentParser(
        description="DQN-ILS Pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--types", nargs="+", default=["EUC_2D"], choices=["EUC_2D", "ATT", "GEO"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--time_budget", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--history_len", type=int, default=1)
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Final exploration rate")
    parser.add_argument(
        "--reward_type",
        default="delta",
        choices=["delta", "sparse"],
        help="Reward type: delta (improvement) or sparse (end-only)",
    )
    parser.add_argument("--train_limit", type=int, default=100)
    parser.add_argument("--val_limit", type=int, default=40, help="Validation instances during training (step 2)")
    parser.add_argument("--test_limit", type=int, default=40, help="Test instances for final evaluation (step 3)")
    parser.add_argument("--no_baseline", action="store_true")
    parser.add_argument("--no_compare_double", action="store_true", help="Skip DQN vs Double DQN comparison")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    config = PipelineConfig(
        types=args.types,
        sizes=args.sizes,
        episodes=args.episodes,
        time_budget=args.time_budget,
        gamma=args.gamma,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        history_len=args.history_len,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        reward_type=args.reward_type,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        baseline=not args.no_baseline,
        compare_double=not args.no_compare_double,
        device=args.device,
        workers=args.workers,
    )

    os.chdir(Path(__file__).resolve().parent.parent)

    # Import run functions from modules
    from scripts.generate_splits import run_generate_splits
    from scripts.train_dqn import run_training
    from scripts.evaluate_dqn import run_grouped_evaluation
    from scripts.compare_dqn_double import run_comparison
    from scripts.generate_plots import run_plots

    start_time = datetime.now()
    print("=" * 50)
    print("DQN-ILS Pipeline")
    print("=" * 50)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuration:")
    print(f"  Instance types:    {config.types}")
    print(f"  Instance sizes:    {config.sizes}")
    print(f"  Episodes:          {config.episodes}")
    print(f"  Time budget:       {config.time_budget}s")
    print(f"  Gamma:             {config.gamma}")
    print(f"  Learning rate:     {config.lr}")
    print(f"  Hidden dim:        {config.hidden_dim}")
    print(f"  History len:       {config.history_len}")
    print(f"  Epsilon start:     {config.epsilon_start}")
    print(f"  Epsilon end:       {config.epsilon_end}")
    print(f"  Reward type:       {config.reward_type}")
    print(f"  Train limit:       {config.train_limit}")
    print(f"  Val limit:         {config.val_limit}")
    print(f"  Test limit:        {config.test_limit}")
    print(f"  Baseline:          {config.baseline}")
    print(f"  Compare double:    {config.compare_double}")
    print(f"  Device:            {config.device}")
    print(f"  Workers:           {config.workers}")

    timings = {}

    # Step 1: Generate splits (train/val/test)
    def step_splits():
        if Path("data/splits.json").exists():
            print("   data/splits.json exists, skipping.")
            return
        run_generate_splits(with_val=True)

    timings["splits"], _ = run_step("1. Generating splits", step_splits)

    # Load splits for subsequent steps
    with open("data/splits.json") as f:
        splits = json.load(f)

    # Step 2: Train models (uses train + val sets)
    # If compare_double is True, trains both DQN and Double DQN variants
    def step_train():
        for inst_type in config.types:
            run_training(
                inst_type=inst_type,
                sizes=config.sizes,
                splits=splits,
                episodes=config.episodes,
                time_budget=config.time_budget,
                gamma=config.gamma,
                lr=config.lr,
                hidden_dim=config.hidden_dim,
                history_len=config.history_len,
                epsilon_start=config.epsilon_start,
                epsilon_end=config.epsilon_end,
                reward_type=config.reward_type,
                device=config.device,
                workers=config.workers,
                train_limit=config.train_limit,
                val_limit=config.val_limit,
                compare_variants=config.compare_double,
            )

    timings["training"], _ = run_step("2. Training models", step_train)

    # Step 3: Evaluate models on test set (baseline computed once per instance group)
    def step_evaluate():
        for inst_type in config.types:
            for size in config.sizes:
                # Find all models for this (type, size)
                models = {}
                standard_path = Path(f"models/dqn/{inst_type}_n{size:03d}_standard.pt")
                double_path = Path(f"models/dqn/{inst_type}_n{size:03d}_double.pt")
                single_path = Path(f"models/dqn/{inst_type}_n{size:03d}.pt")

                if standard_path.exists():
                    models["DQN"] = str(standard_path)
                if double_path.exists():
                    models["Double DQN"] = str(double_path)
                if single_path.exists() and not models:
                    models["Double DQN"] = str(single_path)

                if not models:
                    continue

                run_grouped_evaluation(
                    inst_type=inst_type,
                    size=size,
                    models=models,
                    splits=splits,
                    time_budget=config.time_budget,
                    workers=config.workers,
                    test_limit=config.test_limit,
                    report_baseline=config.baseline,
                )

    timings["evaluation"], _ = run_step("3. Evaluating models", step_evaluate)

    # Step 4: Generate comparison plots (uses stats from step 2)
    def step_compare():
        if not config.compare_double:
            print("   Skipped (--no_compare_double)")
            return
        for inst_type in config.types:
            run_comparison(
                inst_type=inst_type,
                sizes=config.sizes,
                splits=splits,
                episodes=config.episodes,
                time_budget=config.time_budget,
                gamma=config.gamma,
                lr=config.lr,
                hidden_dim=config.hidden_dim,
                history_len=config.history_len,
                device=config.device,
                workers=config.workers,
                train_limit=config.train_limit,
                models_dir="models/dqn",
            )

    timings["comparison"], _ = run_step("4. DQN vs Double DQN comparison plots", step_compare)

    # Step 5: Generate plots
    def step_plots():
        run_plots(
            models_pattern="models/dqn/*.pt",
            results_pattern="data/results/*.csv",
        )

    timings["plots"], _ = run_step("5. Generating plots", step_plots)

    # Summary
    end_time = datetime.now()
    total = (end_time - start_time).total_seconds()

    print(f"\n{'=' * 50}")
    print("Pipeline complete!")
    print("=" * 50)
    print(f"Total: {format_duration(total)}")
    for name, dur in timings.items():
        print(f"  {name:12s}: {format_duration(dur):>8s} ({dur/total*100:5.1f}%)")
    print("\nOutputs: models/dqn/, data/results/, data/plots/")


if __name__ == "__main__":
    main()
