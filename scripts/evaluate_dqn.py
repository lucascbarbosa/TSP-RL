#!/usr/bin/env python
"""
Evaluate trained DQN models on test instances.

Usage:
    python scripts/evaluate_dqn.py --model models/dqn/EUC_2D_n050.pt
    python scripts/evaluate_dqn.py --model "models/dqn/EUC_2D_*.pt" --baseline
"""

import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import re
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
    """Extract instance type and size from model filename."""
    filename = Path(model_path).stem
    match = re.match(r"(.+)_n(\d+)", filename)
    if not match:
        raise ValueError(f"Cannot parse model name: {filename}")
    return match.group(1), int(match.group(2))


# GRASP alphas available to DQN-ILS agent (actions 10-15)
BASELINE_GRASP_ALPHAS = [0.03, 0.1, 0.3]


def evaluate_baseline_grasp(instance, time_budget: float) -> tuple[float, float, int, float]:
    """
    Baseline: GRASP + 2-opt full with time budget.

    Randomly selects alpha from the same pool available to DQN-ILS agent
    for fair comparison (alpha ∈ {0.03, 0.1, 0.3}).

    Returns:
        (gap, elapsed_time, iterations, best_cost)
    """
    t0 = time.perf_counter()
    opt_cost = instance.opt_cost

    best_cost = float("inf")
    iterations = 0
    while time.perf_counter() - t0 < time_budget:
        # Randomly select alpha each iteration (same pool as DQN-ILS)
        alpha = random.choice(BASELINE_GRASP_ALPHAS)
        tour, _ = grasp(instance, alpha=alpha)
        sol = Solution(tour, instance.dist_matrix, is_closed=True)
        improved = two_opt_full(sol)
        if improved.cost < best_cost:
            best_cost = improved.cost
        iterations += 1

    elapsed = time.perf_counter() - t0

    # Compute gap (vs optimal if available, else vs initial baseline)
    if opt_cost is not None:
        gap = ((best_cost - opt_cost) / opt_cost) * 100
    else:
        gap = ((best_cost - instance.baseline_cost) / instance.baseline_cost) * 100

    return gap, elapsed, iterations, best_cost


def evaluate_dqn_instance(
    model, instance, config: DQNConfig, use_baseline: bool = False
) -> tuple[float, float, int, float]:
    """
    Evaluate DQN on a single instance.

    Args:
        model: Trained DQN model.
        instance: TSP instance.
        config: DQN configuration.
        use_baseline: If True, use baseline_cost as reference for state computation.
                     Gap is still reported relative to opt_cost for comparison.

    Returns:
        (gap_vs_opt, elapsed_time, iterations, best_cost)
    """
    import torch

    time_budget = compute_time_budget(instance.dimension, config.time_budget)

    t0 = time.perf_counter()
    env = DQNEnv(instance, time_budget, config.history_len, use_baseline=use_baseline)
    state = env.reset()
    done = False
    iterations = 0

    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(state.to_numpy(), dtype=torch.float32, device=config.device)
            q_values = model(state_tensor)
            action = int(q_values.argmax().item())
        state, _, done = env.step(action)
        iterations += 1

    elapsed = time.perf_counter() - t0

    # Report gap relative to optimal (not baseline) for fair comparison
    best_cost = env.solution.cost
    opt_cost = instance.opt_cost
    if opt_cost is not None:
        gap = ((best_cost - opt_cost) / opt_cost) * 100
    else:
        # If no optimal available, report relative to baseline
        gap = ((best_cost - instance.baseline_cost) / instance.baseline_cost) * 100

    # Update baseline if we found a better solution (improves future evaluations)
    instance.update_baseline_cost(best_cost)

    return gap, elapsed, iterations, best_cost


# Worker functions for parallel evaluation
def _eval_worker_unified(args: tuple) -> list[dict]:
    """
    Unified worker: loads instance once, runs baseline then DQN.

    By running baseline first, we use its result (from full time budget)
    as the reference for DQN's state computation. This is more accurate
    than the quick single-shot baseline_cost.
    """
    instance_id, dataset_path, model_path, time_budget, run_baseline = args

    # Load instance once
    dataset = TSPDataset(dataset_path, [instance_id], verbose=False)
    instance = next(iter(dataset))

    results = []

    # Run baseline first (if requested) - use result as reference for DQN
    if run_baseline:
        bl_gap, bl_elapsed, bl_iters, bl_best_cost = evaluate_baseline_grasp(instance, time_budget)
        results.append(
            {
                "instance_id": instance_id,
                "method": "GRASP+2opt",
                "gap": bl_gap,
                "time": bl_elapsed,
                "iterations": bl_iters,
            }
        )
        # Update baseline_cost with the result from full evaluation
        instance.update_baseline_cost(bl_best_cost)

    # Run DQN (uses baseline as reference if use_baseline=True)
    model = load_model(model_path)
    history_len = (model.state_dim - 3) // N_ACTIONS
    config = DQNConfig(time_budget=time_budget, history_len=history_len)
    dqn_gap, dqn_elapsed, dqn_iters, _ = evaluate_dqn_instance(model, instance, config, use_baseline=run_baseline)
    results.append(
        {
            "instance_id": instance_id,
            "method": "DQN-ILS",
            "gap": dqn_gap,
            "time": dqn_elapsed,
            "iterations": dqn_iters,
        }
    )

    return results


def _eval_worker_dqn_only(args: tuple) -> dict:
    """Worker for DQN-only evaluation (no baseline comparison)."""
    instance_id, dataset_path, model_path, time_budget, use_baseline = args
    dataset = TSPDataset(dataset_path, [instance_id], verbose=False)
    instance = next(iter(dataset))
    model = load_model(model_path)
    history_len = (model.state_dim - 3) // N_ACTIONS
    config = DQNConfig(time_budget=time_budget, history_len=history_len)
    gap, elapsed, iters, _ = evaluate_dqn_instance(model, instance, config, use_baseline=use_baseline)
    return {"instance_id": instance_id, "method": "DQN-ILS", "gap": gap, "time": elapsed, "iterations": iters}


def run_evaluation(
    model_path: str,
    splits: dict,
    time_budget: float = 10.0,
    workers: int = 1,
    eval_limit: int | None = None,
    baseline: bool = False,
    output_dir: str = "data/results",
    verbose: bool = True,
    use_baseline_reference: bool = True,
) -> dict:
    """
    Evaluate a trained DQN model.

    Args:
        model_path: Path to model file (architecture inferred from checkpoint).
        splits: Dictionary with train/test splits.
        time_budget: Base time budget in seconds.
        workers: Number of parallel workers.
        eval_limit: Limit test instances (None = no limit).
        baseline: Include GRASP+2opt baseline comparison.
        output_dir: Output directory for results.
        verbose: Print progress.
        use_baseline_reference: If True, DQN uses baseline_cost as reference for
                               state computation (doesn't need opt_cost). Gap is
                               still reported vs optimal for fair comparison.

    Returns:
        Dictionary with evaluation results.
    """
    try:
        instance_type, size = parse_model_name(model_path)
    except ValueError as e:
        if verbose:
            print(f"Skipping {model_path}: {e}")
        return {}

    dataset_path = f"data/{instance_type}.json"
    if dataset_path not in splits:
        if verbose:
            print(f"No splits for {dataset_path}")
        return {}

    test_ids = splits[dataset_path]["test"]

    # Filter by size
    size_start = (size // 10 - 1) * 1111
    size_end = (size // 10) * 1111
    size_test_ids = [i for i in test_ids if size_start <= i < size_end]

    if eval_limit:
        size_test_ids = size_test_ids[:eval_limit]

    if not size_test_ids:
        if verbose:
            print(f"No test instances for size {size}")
        return {}

    if verbose:
        print(f"Evaluating {instance_type} n={size} ({len(size_test_ids)} instances, {workers} workers)")

    tb = compute_time_budget(size, time_budget)
    results = []

    if baseline:
        # Unified evaluation: baseline first, then DQN (uses baseline result as reference)
        # This loads each instance only once and reuses the baseline result
        unified_args = [(inst_id, dataset_path, model_path, tb, True) for inst_id in size_test_ids]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_eval_worker_unified, arg): arg[0] for arg in unified_args}
            for future in as_completed(futures):
                for result in future.result():  # Each worker returns list of 2 results
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
    else:
        # DQN-only evaluation
        dqn_args = [(inst_id, dataset_path, model_path, tb, use_baseline_reference) for inst_id in size_test_ids]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_eval_worker_dqn_only, arg): arg[0] for arg in dqn_args}
            for future in as_completed(futures):
                result = future.result()
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

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_baseline" if baseline else ""
    output_path = output_dir / f"eval_{instance_type}_n{size:03d}{suffix}.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Summary
    dqn_gaps = [float(r["Gap"].replace("%", "")) for r in results if r["Method"] == "DQN-ILS"]
    summary = {"dqn_mean": np.mean(dqn_gaps), "dqn_std": np.std(dqn_gaps)}

    if verbose:
        print(f"  DQN: {summary['dqn_mean']:.2f}% ± {summary['dqn_std']:.2f}%")

    if baseline:
        bl_gaps = [float(r["Gap"].replace("%", "")) for r in results if r["Method"] == "GRASP+2opt"]
        summary["baseline_mean"] = np.mean(bl_gaps)
        summary["baseline_std"] = np.std(bl_gaps)
        if verbose:
            print(f"  Baseline: {summary['baseline_mean']:.2f}% ± {summary['baseline_std']:.2f}%")

    if verbose:
        print(f"  Results saved to: {output_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN models")
    parser.add_argument("--model", type=str, required=True, help="Model path or glob pattern")
    parser.add_argument("--split_path", type=str, default="data/splits.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--time_budget", type=float, default=10.0)
    parser.add_argument(
        "--workers", type=int, default=None, help=f"Parallel workers (default: {get_default_workers()})"
    )
    args = parser.parse_args()

    n_workers = args.workers if args.workers is not None else get_default_workers()

    model_paths = sorted(glob(args.model))
    if not model_paths:
        print(f"No models found matching: {args.model}")
        sys.exit(1)

    print(f"Found {len(model_paths)} model(s)")
    print(f"Loading splits from: {args.split_path}")

    with open(args.split_path) as f:
        splits = json.load(f)

    for model_path in model_paths:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_path}")
        print(f"{'=' * 60}")

        run_evaluation(
            model_path=model_path,
            splits=splits,
            time_budget=args.time_budget,
            workers=n_workers,
            eval_limit=args.limit,
            baseline=args.baseline,
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
