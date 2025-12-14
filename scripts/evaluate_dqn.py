#!/usr/bin/env python
"""
Evaluate trained DQN models on test instances.

Supports:
- Evaluating single or multiple models (glob patterns)
- Comparing with GRASP+2opt baseline (time-based)
- Parallel evaluation using multiple CPU cores
- Generating CSV results for analysis

Usage:
    # Evaluate a single model
    python scripts/evaluate_dqn.py --model models/dqn/EUC_2D_n050.pt

    # Evaluate all models for a type
    python scripts/evaluate_dqn.py --model "models/dqn/EUC_2D_*.pt"

    # Evaluate with baseline comparison
    python scripts/evaluate_dqn.py --model models/dqn/EUC_2D_n050.pt --baseline

    # Parallel evaluation with 16 workers
    python scripts/evaluate_dqn.py --model models/dqn/EUC_2D_n050.pt --workers 16

    # Limit instances for quick testing
    python scripts/evaluate_dqn.py --model models/dqn/EUC_2D_n050.pt --limit 10
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np

from src.rl.dqn import DQNConfig, N_ACTIONS, load_model, compute_time_budget
from src.rl.dqn.env import DQNEnv
from src.tsp.instance import TSPDataset
from src.tsp.local_search import two_opt_full
from src.tsp.constructive import grasp
from src.tsp.solution import Solution


def get_default_workers() -> int:
    """Get default number of workers (n_cpus - 2, minimum 1)."""
    n_cpus = os.cpu_count() or 1
    return max(1, n_cpus - 2)


def parse_model_name(model_path: str) -> tuple[str, int]:
    """
    Extract instance type and size from model filename.

    Expected format: {type}_n{size:03d}.pt
    Examples: EUC_2D_n050.pt -> ("EUC_2D", 50)
              ATT_n100.pt -> ("ATT", 100)
    """
    filename = Path(model_path).stem
    match = re.match(r"(.+)_n(\d+)", filename)
    if not match:
        raise ValueError(f"Cannot parse model name: {filename}")
    return match.group(1), int(match.group(2))


def evaluate_baseline_grasp(instance, time_budget: float) -> tuple[float, float, int]:
    """
    Baseline: GRASP + 2-opt full with time budget.

    Runs GRASP construction followed by intensive 2-opt local search
    repeatedly until time budget is exhausted, keeping the best solution.

    Args:
        instance: TSP instance to solve.
        time_budget: Time budget in seconds (same as DQN evaluation).

    Returns: (best_gap, time_seconds, iterations)
    """
    t0 = time.perf_counter()

    # Compute optimal cost
    opt_tour = instance.opt_tour
    opt_cost = sum(instance.get_weight(opt_tour[i], opt_tour[(i + 1) % len(opt_tour)]) for i in range(len(opt_tour)))

    best_cost = float("inf")
    iterations = 0

    while time.perf_counter() - t0 < time_budget:
        # GRASP construction (alpha=0.2 default)
        tour, _ = grasp(instance)
        sol = Solution(tour, instance.dist_matrix, is_closed=True)

        # Intensive local search (2-opt full)
        improved = two_opt_full(sol)

        if improved.cost < best_cost:
            best_cost = improved.cost
        iterations += 1

    elapsed = time.perf_counter() - t0
    gap = ((best_cost - opt_cost) / opt_cost) * 100

    return gap, elapsed, iterations


def evaluate_dqn_instance(model, instance, config: DQNConfig) -> tuple[float, float, int]:
    """
    Evaluate DQN on a single instance.

    Returns: (gap, time_seconds, iterations)
    """
    time_budget = compute_time_budget(instance.dimension, config.time_budget)

    t0 = time.perf_counter()
    env = DQNEnv(instance, time_budget, config.history_len)
    state = env.reset()
    done = False
    iterations = 0

    import torch

    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(state.to_numpy(), dtype=torch.float32, device=config.device)
            q_values = model(state_tensor)
            action = int(q_values.argmax().item())

        state, _, done = env.step(action)
        iterations += 1

    elapsed = time.perf_counter() - t0

    return env.best_gap, elapsed, iterations


# Worker functions for parallel evaluation (must be top-level for pickling)


def _eval_worker_dqn(args: tuple) -> dict:
    """
    Worker function for parallel DQN evaluation.

    Args:
        args: Tuple of (instance_id, dataset_path, model_path, config_dict)

    Returns:
        Dictionary with evaluation results.
    """
    instance_id, dataset_path, model_path, config_dict = args

    # Load instance
    dataset = TSPDataset(dataset_path, [instance_id])
    instance = next(iter(dataset))

    # Recreate config and load model (each worker loads its own copy)
    config = DQNConfig(**config_dict)
    state_dim = 3 + config.history_len * N_ACTIONS
    model = load_model(model_path, state_dim=state_dim, hidden_dim=config_dict.get("hidden_dim", 64))

    # Evaluate
    gap, elapsed, iters = evaluate_dqn_instance(model, instance, config)

    return {
        "instance_id": instance_id,
        "method": "DQN-ILS",
        "gap": gap,
        "time": elapsed,
        "iterations": iters,
    }


def _eval_worker_baseline(args: tuple) -> dict:
    """
    Worker function for parallel baseline evaluation.

    Args:
        args: Tuple of (instance_id, dataset_path, time_budget)

    Returns:
        Dictionary with evaluation results.
    """
    instance_id, dataset_path, time_budget = args

    # Load instance
    dataset = TSPDataset(dataset_path, [instance_id])
    instance = next(iter(dataset))

    # Evaluate baseline
    gap, elapsed, iters = evaluate_baseline_grasp(instance, time_budget)

    return {
        "instance_id": instance_id,
        "method": "GRASP+2opt",
        "gap": gap,
        "time": elapsed,
        "iterations": iters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN models")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or glob pattern (e.g., 'models/dqn/EUC_2D_*.pt')",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default="data/splits.json",
        help="Path to splits JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: data/results/eval_{type}_{size}.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test instances (for quick testing)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Include GRASP+2opt baseline comparison (5 restarts)",
    )
    parser.add_argument(
        "--time_budget",
        type=float,
        default=10.0,
        help="Base time budget in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--history_len",
        type=int,
        default=2,
        help="History length for state (must match trained model)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension (must match trained model)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for evaluation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: n_cpus-2 = {get_default_workers()})",
    )

    args = parser.parse_args()

    # Set number of workers
    n_workers = args.workers if args.workers is not None else get_default_workers()

    # Find model files
    model_paths = sorted(glob(args.model))
    if not model_paths:
        print(f"No models found matching: {args.model}")
        sys.exit(1)

    print(f"Found {len(model_paths)} model(s) to evaluate")

    # Load splits
    print(f"Loading splits from: {args.split_path}")
    with open(args.split_path, "r") as f:
        splits = json.load(f)

    # Process each model
    for model_path in model_paths:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_path}")
        print(f"{'=' * 60}")

        # Parse model info
        try:
            instance_type, size = parse_model_name(model_path)
        except ValueError as e:
            print(f"  Skipping: {e}")
            continue

        print(f"Type: {instance_type}, Size: {size}")

        # Get test instances
        dataset_path = f"data/{instance_type}.json"
        if dataset_path in splits:
            test_ids = splits[dataset_path]["test"]
        else:
            print(f"  No split found for {dataset_path}, skipping...")
            continue

        # Filter by size
        size_start = (size // 10 - 1) * 1111
        size_end = (size // 10) * 1111
        size_test_ids = [i for i in test_ids if size_start <= i < size_end]

        if args.limit:
            size_test_ids = size_test_ids[: args.limit]

        if not size_test_ids:
            print(f"  No test instances for size {size}, skipping...")
            continue

        print(f"Test instances: {len(size_test_ids)}")
        print(f"Using {n_workers} parallel workers")

        # Config dict for workers (must be serializable)
        config_dict = {
            "time_budget": args.time_budget,
            "history_len": args.history_len,
            "device": args.device,
            "hidden_dim": args.hidden_dim,
        }
        time_budget = compute_time_budget(size, args.time_budget)

        # Prepare worker arguments
        dqn_args = [(inst_id, dataset_path, model_path, config_dict) for inst_id in size_test_ids]

        # Parallel DQN evaluation
        results = []
        completed = 0
        total = len(size_test_ids)

        print(f"  Evaluating DQN-ILS...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_eval_worker_dqn, arg): arg[0] for arg in dqn_args}

            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"    {completed}/{total} instances...", end="\r")

                results.append(
                    {
                        "Type": instance_type,
                        "Dimension": size,
                        "Instance": result["instance_id"],
                        "Method": result["method"],
                        "Gap": f"{result['gap']:.4f}%",
                        "Time (s)": f"{result['time']:.3f}",
                        "Iterations": result["iterations"],
                    }
                )

        print()  # Clear progress line

        # Parallel baseline evaluation
        if args.baseline:
            baseline_args = [(inst_id, dataset_path, time_budget) for inst_id in size_test_ids]

            completed = 0
            print(f"  Evaluating GRASP+2opt baseline...")
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_eval_worker_baseline, arg): arg[0] for arg in baseline_args}

                for future in as_completed(futures):
                    result = future.result()
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print(f"    {completed}/{total} instances...", end="\r")

                    results.append(
                        {
                            "Type": instance_type,
                            "Dimension": size,
                            "Instance": result["instance_id"],
                            "Method": result["method"],
                            "Gap": f"{result['gap']:.4f}%",
                            "Time (s)": f"{result['time']:.3f}",
                            "Iterations": result["iterations"],
                        }
                    )

            print()  # Clear progress line

        # Save results
        output_dir = Path("data/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.output:
            output_path = Path(args.output)
        else:
            suffix = "_baseline" if args.baseline else ""
            output_path = output_dir / f"eval_{instance_type}_n{size:03d}{suffix}.csv"

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to: {output_path}")

        # Summary statistics
        dqn_results = [r for r in results if r["Method"] == "DQN-ILS"]
        dqn_gaps = [float(r["Gap"].replace("%", "")) for r in dqn_results]

        print(f"\nDQN-ILS Results:")
        print(f"  Mean gap: {np.mean(dqn_gaps):.2f}%")
        print(f"  Std gap:  {np.std(dqn_gaps):.2f}%")
        print(f"  Min gap:  {np.min(dqn_gaps):.2f}%")
        print(f"  Max gap:  {np.max(dqn_gaps):.2f}%")

        if args.baseline:
            bl_results = [r for r in results if r["Method"] == "GRASP+2opt"]
            bl_gaps = [float(r["Gap"].replace("%", "")) for r in bl_results]
            print(f"\nGRASP+2opt Baseline:")
            print(f"  Mean gap: {np.mean(bl_gaps):.2f}%")
            print(f"  Std gap:  {np.std(bl_gaps):.2f}%")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
