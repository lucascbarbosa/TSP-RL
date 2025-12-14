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
    time_budget: float = 10.0
    gamma: float = 0.99
    lr: float = 0.001
    hidden_dim: int = 64
    history_len: int = 2
    train_limit: int = 100
    eval_limit: int = 20
    baseline: bool = True
    device: str = "cpu"
    workers: int = 16


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
    parser.add_argument("--time_budget", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--history_len", type=int, default=2)
    parser.add_argument("--train_limit", type=int, default=100)
    parser.add_argument("--eval_limit", type=int, default=20)
    parser.add_argument("--no_baseline", action="store_true")
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
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        baseline=not args.no_baseline,
        device=args.device,
        workers=args.workers,
    )

    os.chdir(Path(__file__).resolve().parent.parent)

    # Import run functions from modules
    from scripts.generate_splits import run_generate_splits
    from scripts.train_dqn import run_training
    from scripts.evaluate_dqn import run_evaluation
    from scripts.generate_plots import run_plots
    from glob import glob

    start_time = datetime.now()
    print("=" * 50)
    print("DQN-ILS Pipeline")
    print("=" * 50)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: types={config.types}, sizes={config.sizes}, episodes={config.episodes}, workers={config.workers}")

    timings = {}

    # Step 1: Generate splits
    def step_splits():
        if Path("data/splits.json").exists():
            print("   data/splits.json exists, skipping.")
            return
        run_generate_splits()

    timings["splits"], _ = run_step("1. Generating splits", step_splits)

    # Load splits for subsequent steps
    with open("data/splits.json") as f:
        splits = json.load(f)

    # Step 2: Train models
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
                device=config.device,
                workers=config.workers,
                train_limit=config.train_limit,
            )

    timings["training"], _ = run_step("2. Training models", step_train)

    # Step 3: Evaluate models
    def step_evaluate():
        for inst_type in config.types:
            for model_path in sorted(glob(f"models/dqn/{inst_type}_*.pt")):
                run_evaluation(
                    model_path=model_path,
                    splits=splits,
                    time_budget=config.time_budget,
                    workers=config.workers,
                    eval_limit=config.eval_limit,
                    baseline=config.baseline,
                )

    timings["evaluation"], _ = run_step("3. Evaluating models", step_evaluate)

    # Step 4: Generate plots
    def step_plots():
        run_plots(
            models_pattern="models/dqn/*.pt",
            results_pattern="data/results/*.csv",
        )

    timings["plots"], _ = run_step("4. Generating plots", step_plots)

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
