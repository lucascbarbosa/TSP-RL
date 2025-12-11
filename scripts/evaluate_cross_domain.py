#!/usr/bin/env python
"""
Evaluate Q-ILS on test instances using trained Q-tables.

Usage:
    python scripts/evaluate_cross_domain.py
    python scripts/evaluate_cross_domain.py --types EUC_2D GEO --workers 8

Supports cross-domain evaluation:
    python scripts/evaluate_cross_domain.py --source-type EUC_2D --source-size 50
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

from src.tsp.instance import TSPInstance, _load_json
from src.ils.q_ils import QILS

# Optional plotting utilities
try:
    from utils.plot import generate_gap_violin_plots, generate_time_analysis_plots

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def get_available_sizes(instance_type: str) -> set[int]:
    """
    Detect available Q-table sizes for a given instance type.

    Args:
        instance_type: Instance type (e.g., 'EUC_2D').

    Returns:
        Set of available instance sizes (dimensions).
    """
    q_tables_dir = Path(f"data/q_tables/{instance_type}")
    if not q_tables_dir.exists():
        return set()

    sizes: set[int] = set()
    for f in q_tables_dir.glob("instance_size_*.txt"):
        match = re.search(r"instance_size_(\d+)\.txt", f.name)
        if match:
            sizes.add(int(match.group(1)))
    return sizes


def get_instance_dimensions(dataset_path: str, instance_ids: list[int]) -> dict[int, int]:
    """
    Get dimensions for specific instances from dataset.

    Args:
        dataset_path: Path to dataset JSON file (supports .json.zip).
        instance_ids: List of instance IDs to check.

    Returns:
        Dict mapping instance_id -> dimension.
    """
    data = _load_json(dataset_path)

    dimensions: dict[int, int] = {}
    for idx in instance_ids:
        if 0 <= idx < len(data):
            dimensions[idx] = len(data[idx]["coords"])
    return dimensions


# Configuration
DEFAULT_OUTPUT = "data/results/results.csv"
DEFAULT_MAX_ITER = 50
DEFAULT_EPSILON = 0.1


def get_default_workers() -> int:
    """Return cpu_count - 2, minimum 1."""
    return max(1, os.cpu_count() - 2)


def get_processed_instances(filename: str) -> set[str]:
    """Load already processed instance IDs from results file."""
    processed: set[str] = set()
    if os.path.exists(filename):
        with open(filename, mode="r", newline="") as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            for row in reader:
                if row:
                    processed.add(row[0])  # full_id in first column
    return processed


def initialize_csv(filename: str) -> None:
    """Create CSV file with header if it doesn't exist."""
    # Ensure parent directory exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.isfile(filename):
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    # Instance info
                    "Full ID",
                    "Name",
                    "ID",
                    "Type",
                    "Dimension",
                    # Solution quality
                    "Optimal Cost",
                    "Best Cost",
                    "Gap",
                    # Timing (seconds)
                    "Time (s)",
                    "Init Time (s)",
                    # Iteration stats
                    "Total Iterations",
                    "Best Iteration",
                    "Improvements",
                    "Early Stopped",
                    # Initial solution
                    "Initial Cost",
                    "Initial Gap",
                    # Best tour
                    "Best Tour",
                ]
            )


def process_instance(args: tuple[int, str, int, float, bool, Optional[str]]) -> Optional[list]:
    """
    Process a single TSP instance.

    Args:
        args: Tuple of (instance_id, instance_type, max_iter, epsilon, allow_early_stop, q_table_path).

    Returns:
        Row data for CSV or None on failure.
    """
    instance_id, instance_type, max_iter, epsilon, allow_early_stop, q_table_path = args

    try:
        # Load instance
        problem = TSPInstance(f"data/{instance_type}.json", instance_id=instance_id)

        # Calculate optimal cost (primal value from MIP solver)
        opt_tour = problem.opt_tour
        if opt_tour is None:
            print(f"SKIP: {instance_type}{instance_id} - No optimal tour")
            return None

        primal_cost = sum(
            problem.get_weight(opt_tour[i], opt_tour[(i + 1) % len(opt_tour)]) for i in range(len(opt_tour))
        )

        # Calculate early_stop_target based on MIP gap (fraction: 0.03 = 3%)
        # - gap == 0.0: tour is provably optimal, use primal_cost
        # - gap > 0: use lower_bound = primal / (1 + gap)
        # - gap is None: no guarantee, disable early stop
        mip_gap = problem.mip_gap
        if not allow_early_stop:
            early_stop_target = None  # Globally disabled
        elif mip_gap is None:
            early_stop_target = None  # No gap info, can't guarantee optimality
        elif mip_gap == 0.0:
            early_stop_target = primal_cost  # Tour is optimal
        else:
            # mip_gap > 0: lower_bound = primal / (1 + gap)
            early_stop_target = primal_cost / (1 + mip_gap)

        # Setup Q-ILS solver
        solver = QILS(problem)

        # Load Q-table: either per-instance (if q_table_path is None, fallback to old behavior),
        # or use the provided q_table_path (cross-domain scenario).
        if q_table_path is None:
            q_table_path_local = f"data/q_tables/{instance_type}/instance_size_{problem.dimension:02d}.txt"
        else:
            q_table_path_local = q_table_path

        # Ensure Q-table exists
        if not Path(q_table_path_local).exists():
            print(f"SKIP: {instance_type}{instance_id} - Q-table not found at {q_table_path_local}")
            return None

        solver.load_q_table(q_table_path_local)

        # Run Q-ILS (stats are collected internally)
        # opt_cost is always primal_cost (for gap calculation)
        # early_stop_target controls when to stop early
        best_solution = solver.run(
            max_iter=max_iter,
            opt_cost=primal_cost,
            epsilon=epsilon,
            verbose=False,
            early_stop=allow_early_stop,
            early_stop_target=early_stop_target,
        )

        # Get stats from solver
        stats = solver.last_stats
        full_id = f"{instance_type}{instance_id}"

        early_str = " [EARLY]" if stats.early_stopped else ""
        print(
            f"DONE: {full_id} | Gap: {stats.final_gap:.2f}% | "
            f"Time: {stats.total_time:.2f}s | Iter: {stats.total_iterations}{early_str}"
        )

        return [
            # Instance info
            full_id,
            problem.name,
            instance_id,
            instance_type,
            problem.dimension,
            # Solution quality
            primal_cost,
            best_solution.cost,
            f"{stats.final_gap:.4f}%",
            # Timing (seconds)
            f"{stats.total_time:.4f}",
            f"{stats.init_time:.4f}",
            # Iteration stats
            stats.total_iterations,
            stats.best_iteration,
            stats.improvements,
            stats.early_stopped,
            # Initial solution
            f"{stats.initial_cost:.4f}",
            f"{stats.initial_gap:.4f}%",
            # Best tour
            str(best_solution.tour),
        ]

    except Exception as e:
        print(f"ERROR processing {instance_type}-{instance_id}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Q-ILS on test instances")
    parser.add_argument(
        "--types",
        nargs="+",
        default=["EUC_2D", "GEO", "ATT"],
        help="Instance types to evaluate (default: EUC_2D GEO ATT)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Instance sizes to evaluate (default: all available Q-tables)",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default=None,
        help="(Optional) Source instance type where Q-table was trained (enables cross-domain evaluation).",
    )
    parser.add_argument(
        "--source-size",
        type=int,
        default=None,
        help="(Optional) Source instance size where Q-table was trained (enables cross-domain evaluation).",
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=None,
        help="(Optional) Target sizes for cross-domain evaluation (default: 10 20 ... 100).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: cpu_count - 2)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help=f"Max iterations without improvement (default: {DEFAULT_MAX_ITER})",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"Exploration rate (default: {DEFAULT_EPSILON})",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="data/splits.json",
        help="Path to splits JSON file (default: data/splits.json)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation after evaluation",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stop when reaching optimal cost",
    )
    args = parser.parse_args()

    # Load processed instances and initialize CSV
    processed = get_processed_instances(args.output)
    initialize_csv(args.output)

    # Load splits
    with open(args.splits, "r") as f:
        splits = json.load(f)

    # Determine if cross-domain mode is requested
    cross_domain = args.source_type is not None and args.source_size is not None

    # If cross-domain and target_sizes were not specified, defaults to 10, 20, ..., 100
    if cross_domain:
        if args.target_sizes is None:
            target_sizes = list(range(10, 101, 10))  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        else:
            target_sizes = args.target_sizes
        # build source q-table path and check existence
        source_qtable = Path(f"data/q_tables/{args.source_type}/instance_size_{args.source_size:02d}.txt")
        if not source_qtable.exists():
            print(f"ERROR: Source Q-table not found at {source_qtable}")
            return
        print(f"Cross-domain evaluation mode: using Q-table {source_qtable} as source")
    else:
        target_sizes = None
        source_qtable = None

    # Build job list (instance_id, instance_type, max_iter, epsilon, early_stop, q_table_path)
    early_stop = not args.no_early_stop
    jobs: list[tuple[int, str, int, float, bool, Optional[str]]] = []

    for instance_type in args.types:
        key = f"data/{instance_type}.json"
        if key not in splits:
            print(f"Warning: No splits found for {instance_type}")
            continue

        test_instances = splits[key]["test"]

        # Get dimensions for all test instances
        dataset_path = f"data/{instance_type}.json"
        dimensions = get_instance_dimensions(dataset_path, test_instances)

        if cross_domain:
            # Cross-domain: use single source Q-table for ALL evaluations
            for instance_id in test_instances:
                full_id = f"{instance_type}{instance_id}"

                if full_id in processed:
                    continue

                dim = dimensions.get(instance_id)
                if dim is None:
                    continue

                if dim in target_sizes:
                    jobs.append(
                        (instance_id, instance_type, args.max_iter, args.epsilon, early_stop, str(source_qtable))
                    )
        else:
            # Original behavior: only evaluate instances that have a Q-table for their size
            available_sizes = get_available_sizes(instance_type)
            if not available_sizes:
                print(f"Warning: No Q-tables found for {instance_type}, skipping...")
                continue

            # Filter by requested sizes if specified
            if args.sizes is not None:
                available_sizes = available_sizes & set(args.sizes)

            # Filter to instances with available Q-tables
            filtered_count = 0
            for instance_id in test_instances:
                full_id = f"{instance_type}{instance_id}"

                if full_id in processed:
                    continue

                dim = dimensions.get(instance_id)
                if dim is None or dim not in available_sizes:
                    filtered_count += 1
                    continue

                q_table_path = f"data/q_tables/{instance_type}/instance_size_{dim:02d}.txt"
                jobs.append((instance_id, instance_type, args.max_iter, args.epsilon, early_stop, q_table_path))

            if filtered_count > 0:
                print(f"  {instance_type}: skipped {filtered_count} instances (no Q-table for their size)")

    # Resolve workers: use cli arg or default to cpu_count - 2
    num_workers = args.workers if args.workers is not None else get_default_workers()
    print(f"Starting evaluation of {len(jobs)} instances with {num_workers} processes...")

    # Run parallel evaluation and collect results
    # Use 'spawn' context to avoid CUDA re-initialization issues with fork
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        results = list(executor.map(process_instance, jobs))

    # Write results to CSV
    valid_results = [r for r in results if r is not None]
    with open(args.output, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(valid_results)

    print(f"Evaluation complete. {len(valid_results)}/{len(jobs)} instances processed.")

    # Generate plots if enabled
    if HAS_PLOTTING and not args.no_plots:
        print("\nGenerating gap distribution plots...")
        generate_gap_violin_plots(
            csv_path=args.output,
            output_dir="data/plots",
            types=args.types,
        )
        print("\nGenerating time analysis plots...")
        generate_time_analysis_plots(
            csv_path=args.output,
            output_dir="data/plots",
            types=args.types,
        )


if __name__ == "__main__":
    main()
