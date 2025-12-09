#!/usr/bin/env python
"""
Generate transition data for MDP training.

Runs ILS with random actions on training instances and saves
(state, action, reward, next_state) transitions.

Usage:
    python scripts/train_transitions.py \\
        --split_path data/splits.json \\
        --dataset_path data/EUC_2D.json \\
        --output_dir data/train/EUC_2D
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.tsp.instance import TSPDataset, TSPInstance
from src.ils.q_ils import QILS


def process_instance(
    tsp_instance: TSPInstance,
    output_dir: str,
    max_iter: int,
) -> str:
    """
    Generate transitions for a single instance.

    Args:
        tsp_instance: TSP instance to process.
        output_dir: Directory for output files.
        max_iter: Max iterations without improvement.

    Returns:
        Status message.
    """
    out_name = os.path.join(output_dir, f"{tsp_instance.name}.txt")

    # Skip if already exists
    if os.path.isfile(out_name):
        return f"Skipped {tsp_instance.name}"

    # Calculate optimal cost
    opt_tour = tsp_instance.opt_tour
    if opt_tour:
        opt_cost = sum(
            tsp_instance.get_weight(opt_tour[i], opt_tour[(i + 1) % len(opt_tour)]) for i in range(len(opt_tour))
        )
    else:
        opt_cost = 1.0  # Fallback if no optimal tour

    # Initialize solver and generate transitions
    solver = QILS(tsp_instance)
    solver.generate_transitions(max_iter=max_iter, opt_cost=opt_cost, out_path=out_name)

    return f"Processed {tsp_instance.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate transition data for training")

    parser.add_argument(
        "--split_path",
        type=str,
        required=True,
        help="Path to splits JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output transition files",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of instances (for testing)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=50,
        help="Max iterations without improvement (default: 50)",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load splits
    print(f"Loading split from: {args.split_path}")
    import json

    with open(args.split_path, "r") as f:
        splits = json.load(f)

    # Get training indices
    dataset_key = args.dataset_path
    if dataset_key not in splits:
        if "train" in splits:
            train_ids = splits["train"]
        else:
            raise KeyError(f"Could not find key '{dataset_key}' or 'train' in split file.")
    else:
        train_ids = splits[dataset_key]["train"]

    # Apply limit if specified
    if args.limit is not None:
        train_ids = train_ids[: args.limit]

    # Initialize dataset
    print(f"Loading dataset: {args.dataset_path} with {len(train_ids)} instances...")
    train_set = TSPDataset(args.dataset_path, train_ids)

    # Setup multiprocessing
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"\n--- Starting Transition Generation on {num_cores} cores ---")
    print(f"Saving results to: {args.output_dir}")
    print(f"Max iterations per instance: {args.max_iter}\n")

    # Create worker function with fixed parameters
    worker_func = partial(
        process_instance,
        output_dir=args.output_dir,
        max_iter=args.max_iter,
    )

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(
            tqdm(
                executor.map(worker_func, train_set),
                total=len(train_set),
                desc="Generating transitions",
                ncols=120,
            )
        )

    print("Done.")


if __name__ == "__main__":
    main()
