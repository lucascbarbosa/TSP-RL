#!/usr/bin/env python
"""
Evaluate Q-ILS on test instances using trained Q-tables.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --types EUC_2D GEO --workers 8
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.tsp.instance import TSPInstance
from src.ils.q_ils import QILS

# Optional plotting utilities
try:
    from utils.plot import generate_gap_violin_plots

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def get_available_sizes(instance_type: str) -> Set[int]:
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

    sizes: Set[int] = set()
    for f in q_tables_dir.glob("instance_size_*.txt"):
        match = re.search(r"instance_size_(\d+)\.txt", f.name)
        if match:
            sizes.add(int(match.group(1)))
    return sizes


def get_instance_dimensions(dataset_path: str, instance_ids: List[int]) -> Dict[int, int]:
    """
    Get dimensions for specific instances from dataset.

    Args:
        dataset_path: Path to dataset JSON file.
        instance_ids: List of instance IDs to check.

    Returns:
        Dict mapping instance_id -> dimension.
    """
    with open(dataset_path, "r") as f:
        data = json.load(f)

    dimensions: Dict[int, int] = {}
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


# Thread-safe file writing
csv_lock = threading.Lock()


def get_processed_instances(filename: str) -> Set[str]:
    """Load already processed instance IDs from results file."""
    processed: Set[str] = set()
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
                ["Full ID", "Name", "ID", "Type", "Dimension", "Optimal Cost", "Best Cost", "Gap", "Time", "Best Tour"]
            )


def process_instance(args: Tuple[int, str, str, int, float]) -> None:
    """
    Process a single TSP instance.

    Args:
        args: Tuple of (instance_id, instance_type, output_filename, max_iter, epsilon).
    """
    instance_id, instance_type, output_filename, max_iter, epsilon = args

    try:
        # Load instance
        problem = TSPInstance(f"data/{instance_type}.json", instance_id=instance_id)

        # Calculate optimal cost
        opt_tour = problem.opt_tour
        if opt_tour is None:
            print(f"SKIP: {instance_type}{instance_id} - No optimal tour")
            return

        opt_cost = sum(problem.get_weight(opt_tour[i], opt_tour[(i + 1) % len(opt_tour)]) for i in range(len(opt_tour)))

        # Setup Q-ILS solver
        solver = QILS(problem)
        q_table_path = f"data/q_tables/{instance_type}/instance_size_{problem.dimension:02d}.txt"
        solver.load_q_table(q_table_path)

        # Run Q-ILS
        time_start = time.time()
        best_solution = solver.run(
            max_iter=max_iter,
            opt_cost=opt_cost,
            epsilon=epsilon,
            verbose=False,
        )
        exec_time = time.time() - time_start

        # Calculate gap
        gap_value = ((best_solution.cost - opt_cost) / opt_cost) * 100
        full_id = f"{instance_type}{instance_id}"

        row_data = [
            full_id,
            problem.name,
            instance_id,
            instance_type,
            problem.dimension,
            opt_cost,
            best_solution.cost,
            f"{gap_value:.4f}%",
            exec_time,
            str(best_solution.tour),
        ]

        # Thread-safe write to CSV
        with csv_lock:
            with open(output_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row_data)

        print(f"DONE: {full_id} | Gap: {gap_value:.2f}% | Time: {exec_time:.2f}s")

    except Exception as e:
        print(f"ERROR processing {instance_type}-{instance_id}: {e}")


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
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads (default: cpu_count - 2)",
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
    args = parser.parse_args()

    # Load processed instances and initialize CSV
    processed = get_processed_instances(args.output)
    initialize_csv(args.output)

    # Load splits
    with open(args.splits, "r") as f:
        splits = json.load(f)

    # Build job list
    jobs: List[Tuple[int, str, str, int, float]] = []

    for instance_type in args.types:
        key = f"data/{instance_type}.json"
        if key not in splits:
            print(f"Warning: No splits found for {instance_type}")
            continue

        # Detect available Q-table sizes for this type
        available_sizes = get_available_sizes(instance_type)
        if not available_sizes:
            print(f"Warning: No Q-tables found for {instance_type}, skipping...")
            continue

        # Filter by requested sizes if specified
        if args.sizes is not None:
            available_sizes = available_sizes & set(args.sizes)

        test_instances = splits[key]["test"]

        # Get dimensions for all test instances
        dataset_path = f"data/{instance_type}.json"
        dimensions = get_instance_dimensions(dataset_path, test_instances)

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

            jobs.append((instance_id, instance_type, args.output, args.max_iter, args.epsilon))

        if filtered_count > 0:
            print(f"  {instance_type}: skipped {filtered_count} instances (no Q-table for their size)")

    # Resolve workers: use cli arg or default to cpu_count - 2
    num_workers = args.workers if args.workers is not None else get_default_workers()
    print(f"Starting evaluation of {len(jobs)} instances with {num_workers} threads...")

    # Run parallel evaluation
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_instance, jobs)

    print("Evaluation complete.")

    # Generate violin plots if enabled
    if HAS_PLOTTING and not args.no_plots:
        print("\nGenerating gap distribution plots...")
        generate_gap_violin_plots(
            csv_path=args.output,
            output_dir="data/plots",
            types=args.types,
        )


if __name__ == "__main__":
    main()
