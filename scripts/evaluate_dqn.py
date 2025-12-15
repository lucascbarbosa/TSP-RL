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


def parse_model_name(model_path: str) -> tuple[str, int, str | None]:
    """
    Extract instance type, size, and variant from model filename.

    Returns:
        (instance_type, size, variant) where variant is 'standard', 'double', or None.
    """
    filename = Path(model_path).stem

    # Try to match with variant suffix first (e.g., EUC_2D_n010_standard)
    match = re.match(r"(.+)_n(\d+)_(standard|double)$", filename)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)

    # Fallback: no variant suffix (e.g., EUC_2D_n010)
    match = re.match(r"(.+)_n(\d+)$", filename)
    if match:
        return match.group(1), int(match.group(2)), None

    raise ValueError(f"Cannot parse model name: {filename}")


# GRASP alphas available to DQN-ILS agent (actions 10-15)
BASELINE_GRASP_ALPHAS = [0.03, 0.1, 0.3]


# =============================================================================
# Worker functions for parallel execution
# =============================================================================


def _baseline_worker(args: tuple) -> dict:
    """
    Compute baseline (GRASP+2opt) for a single instance.

    Args:
        args: (instance_id, dataset_path, time_budget)

    Returns:
        Dict with instance_id, gap, time, iterations, cost.
    """
    instance_id, dataset_path, time_budget = args

    dataset = TSPDataset(dataset_path, [instance_id], verbose=False)
    instance = next(iter(dataset))

    t0 = time.perf_counter()
    best_cost = float("inf")
    iterations = 0

    while time.perf_counter() - t0 < time_budget:
        alpha = random.choice(BASELINE_GRASP_ALPHAS)
        tour, _ = grasp(instance, alpha=alpha)
        sol = Solution(tour, instance.dist_matrix, is_closed=True)
        improved = two_opt_full(sol)
        if improved.cost < best_cost:
            best_cost = improved.cost
        iterations += 1

    elapsed = time.perf_counter() - t0
    gap = ((best_cost - instance.opt_cost) / instance.opt_cost) * 100

    return {
        "instance_id": instance_id,
        "gap": gap,
        "time": elapsed,
        "iterations": iterations,
        "cost": best_cost,
    }


def _dqn_worker(args: tuple) -> dict:
    """
    Evaluate DQN on a single instance using pre-computed baseline.

    Args:
        args: (instance_id, dataset_path, model_path, time_budget, baseline_cost)

    Returns:
        Dict with instance_id, gap, time, iterations.
    """
    import torch

    instance_id, dataset_path, model_path, time_budget, baseline_cost = args

    dataset = TSPDataset(dataset_path, [instance_id], verbose=False)
    instance = next(iter(dataset))
    instance.set_baseline_cost(baseline_cost)

    model = load_model(model_path)
    history_len = (model.state_dim - 3) // N_ACTIONS
    config = DQNConfig(time_budget=time_budget, history_len=history_len)

    t0 = time.perf_counter()
    env = DQNEnv(instance, time_budget, config.history_len, use_baseline=True)
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
    best_cost = env.solution.cost
    gap = ((best_cost - instance.opt_cost) / instance.opt_cost) * 100

    return {
        "instance_id": instance_id,
        "gap": gap,
        "time": elapsed,
        "iterations": iterations,
    }


# =============================================================================
# Main evaluation functions
# =============================================================================


def run_grouped_evaluation(
    inst_type: str,
    size: int,
    models: dict[str, str],
    splits: dict,
    time_budget: float = 10.0,
    workers: int = 1,
    test_limit: int | None = None,
    report_baseline: bool = False,
    output_dir: str = "data/results",
    verbose: bool = True,
) -> dict:
    """
    Evaluate multiple models on the same instances (baseline computed once).

    This is the main evaluation function for the pipeline. It computes baselines
    once per instance, then evaluates all models using cached baselines.

    Args:
        inst_type: Instance type (EUC_2D, ATT, GEO).
        size: Instance size.
        models: Dict mapping variant name to model path, e.g.:
                {"DQN": "models/dqn/EUC_2D_n010_standard.pt",
                 "Double DQN": "models/dqn/EUC_2D_n010_double.pt"}
        splits: Dictionary with train/test splits.
        time_budget: Base time budget in seconds.
        workers: Number of parallel workers.
        test_limit: Limit test instances (None = no limit).
        report_baseline: Include baseline results in output CSV.
        output_dir: Output directory for results.
        verbose: Print progress.

    Returns:
        Dictionary with evaluation summaries per method.
    """
    dataset_path = f"data/{inst_type}.json"
    if dataset_path not in splits:
        if verbose:
            print(f"No splits for {dataset_path}")
        return {}

    test_ids = splits[dataset_path]["test"]

    # Filter by size
    size_start = (size // 10 - 1) * 1111
    size_end = (size // 10) * 1111
    size_test_ids = [i for i in test_ids if size_start <= i < size_end]

    if test_limit:
        size_test_ids = size_test_ids[:test_limit]

    if not size_test_ids:
        if verbose:
            print(f"No test instances for {inst_type} n={size}")
        return {}

    tb = compute_time_budget(size, time_budget)

    if verbose:
        print(f"Evaluating {inst_type} n={size} ({len(size_test_ids)} instances, {workers} workers)")
        print(f"  Models: {list(models.keys())}")

    # Step 1: Compute baselines (once for all models)
    if verbose:
        print("  Computing baselines...")

    baseline_args = [(inst_id, dataset_path, tb) for inst_id in size_test_ids]
    baselines = {}  # instance_id -> baseline result dict

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_baseline_worker, arg): arg[0] for arg in baseline_args}
        for future in as_completed(futures):
            result = future.result()
            baselines[result["instance_id"]] = result

    if verbose:
        bl_gaps = [baselines[i]["gap"] for i in size_test_ids]
        print(f"  Baseline: {np.mean(bl_gaps):.2f}% ± {np.std(bl_gaps):.2f}%")

    # Step 2: Evaluate each model using cached baselines
    all_results = []
    summaries = {}

    # Add baseline results if requested
    if report_baseline:
        for inst_id in size_test_ids:
            bl = baselines[inst_id]
            all_results.append(
                {
                    "Type": inst_type,
                    "Dimension": size,
                    "Instance": inst_id,
                    "Method": "GRASP+2opt",
                    "Gap": f"{bl['gap']:.4f}%",
                    "Time (s)": f"{bl['time']:.3f}",
                    "Iterations": bl["iterations"],
                }
            )
        bl_gaps = [baselines[i]["gap"] for i in size_test_ids]
        summaries["GRASP+2opt"] = {"mean": np.mean(bl_gaps), "std": np.std(bl_gaps)}

    # Evaluate each model
    for method_name, model_path in models.items():
        if verbose:
            print(f"  Evaluating {method_name}...")

        dqn_args = [(inst_id, dataset_path, model_path, tb, baselines[inst_id]["cost"]) for inst_id in size_test_ids]

        dqn_results = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_dqn_worker, arg): arg[0] for arg in dqn_args}
            for future in as_completed(futures):
                result = future.result()
                dqn_results[result["instance_id"]] = result

        gaps = []
        for inst_id in size_test_ids:
            res = dqn_results[inst_id]
            gaps.append(res["gap"])
            all_results.append(
                {
                    "Type": inst_type,
                    "Dimension": size,
                    "Instance": inst_id,
                    "Method": method_name,
                    "Gap": f"{res['gap']:.4f}%",
                    "Time (s)": f"{res['time']:.3f}",
                    "Iterations": res["iterations"],
                }
            )

        summaries[method_name] = {"mean": np.mean(gaps), "std": np.std(gaps)}
        if verbose:
            print(f"  {method_name}: {np.mean(gaps):.2f}% ± {np.std(gaps):.2f}%")

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_suffix = "_baseline" if report_baseline else ""
    output_path = output_dir / f"eval_{inst_type}_n{size:03d}{baseline_suffix}.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    if verbose:
        print(f"  Results saved to: {output_path}")

    return summaries


def run_evaluation(
    model_path: str,
    splits: dict,
    time_budget: float = 10.0,
    workers: int = 1,
    test_limit: int | None = None,
    baseline: bool = False,
    output_dir: str = "data/results",
    verbose: bool = True,
) -> dict:
    """
    Evaluate a single trained DQN model (standalone mode).

    For pipeline use, prefer run_grouped_evaluation which computes baselines once.

    Args:
        model_path: Path to model file.
        splits: Dictionary with train/test splits.
        time_budget: Base time budget in seconds.
        workers: Number of parallel workers.
        test_limit: Limit test instances (None = no limit).
        baseline: Include GRASP+2opt baseline comparison in results.
        output_dir: Output directory for results.
        verbose: Print progress.

    Returns:
        Dictionary with evaluation results.
    """
    try:
        instance_type, size, variant = parse_model_name(model_path)
    except ValueError as e:
        if verbose:
            print(f"Skipping {model_path}: {e}")
        return {}

    # Determine method name
    if variant == "double":
        method_name = "Double DQN"
    elif variant == "standard":
        method_name = "DQN"
    else:
        method_name = "Double DQN"  # Default

    models = {method_name: model_path}
    return run_grouped_evaluation(
        inst_type=instance_type,
        size=size,
        models=models,
        splits=splits,
        time_budget=time_budget,
        workers=workers,
        test_limit=test_limit,
        report_baseline=baseline,
        output_dir=output_dir,
        verbose=verbose,
    )


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

    # Group models by (type, size) for efficient evaluation
    grouped: dict[tuple[str, int], dict[str, str]] = {}
    for model_path in model_paths:
        try:
            inst_type, size, variant = parse_model_name(model_path)
            key = (inst_type, size)
            if key not in grouped:
                grouped[key] = {}

            # Determine method name
            if variant == "double":
                method_name = "Double DQN"
            elif variant == "standard":
                method_name = "DQN"
            else:
                method_name = "Double DQN"

            grouped[key][method_name] = model_path
        except ValueError as e:
            print(f"Skipping {model_path}: {e}")

    # Evaluate each group
    for (inst_type, size), models in sorted(grouped.items()):
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {inst_type} n={size}")
        print("=" * 60)

        run_grouped_evaluation(
            inst_type=inst_type,
            size=size,
            models=models,
            splits=splits,
            time_budget=args.time_budget,
            workers=n_workers,
            test_limit=args.limit,
            report_baseline=args.baseline,
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
