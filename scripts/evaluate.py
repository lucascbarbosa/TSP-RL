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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.tsp.instance import TSPInstance
from src.ils.q_ils import QILS

# Configuration
DEFAULT_OUTPUT = "results.csv"
DEFAULT_WORKERS = 10
DEFAULT_MAX_ITER = 50
DEFAULT_EPSILON = 0.1

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

        if not os.path.exists(q_table_path):
            print(f"SKIP: {instance_type}{instance_id} - Q-table not found: {q_table_path}")
            return

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
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of worker threads (default: {DEFAULT_WORKERS})",
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

        test_instances = splits[key]["test"]

        for instance_id in test_instances:
            full_id = f"{instance_type}{instance_id}"

            if full_id in processed:
                continue

            jobs.append((instance_id, instance_type, args.output, args.max_iter, args.epsilon))

    print(f"Starting evaluation of {len(jobs)} instances with {args.workers} threads...")

    # Run parallel evaluation
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        executor.map(process_instance, jobs)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
