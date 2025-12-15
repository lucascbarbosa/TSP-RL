#!/usr/bin/env python
"""
Gera splits train/test para as instancias TSP.

Compativel com o split usado pelo outro grupo da disciplina (seed=42, 90/10).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random

from src.tsp.instance import _resolve_json_path, _load_json


def generate_split(n_instances: int, seed: int = 42, train_ratio: float = 0.9):
    """
    Gera indices de train e test.

    Args:
        n_instances: Total de instancias
        seed: Seed para reproducibilidade
        train_ratio: Fracao para treino (default 0.9)

    Returns:
        Tupla (train_indices, test_indices)
    """
    indices = list(range(n_instances))
    random.seed(seed)
    random.shuffle(indices)
    n_train = int(n_instances * train_ratio)
    return sorted(indices[:n_train]), sorted(indices[n_train:])


def run_generate_splits(
    data_dir: str = "data",
    output: str = "data/splits.json",
    seed: int = 42,
    train_ratio: float = 0.9,
    verbose: bool = True,
) -> dict:
    """
    Generate train/test splits for all instance files.

    Args:
        data_dir: Directory containing JSON instance files.
        output: Output path for splits JSON.
        seed: Random seed.
        train_ratio: Fraction for training.
        verbose: Print progress.

    Returns:
        Dictionary with splits.
    """
    data_dir = Path(data_dir)
    instance_files = ["EUC_2D.json", "ATT.json", "GEO.json"]
    splits = {}

    for filename in instance_files:
        filepath = data_dir / filename
        try:
            resolved = _resolve_json_path(filepath)
        except FileNotFoundError:
            if verbose:
                print(f"[!] {filepath} (or .zip) not found, skipping...")
            continue

        data = _load_json(resolved)
        n_instances = len(data)
        train_ids, test_ids = generate_split(n_instances, seed=seed, train_ratio=train_ratio)

        key = f"data/{filename}"
        splits[key] = {"train": train_ids, "test": test_ids}

        if verbose:
            print(f"{filename} (from {resolved.name}):")
            print(f"  Total:  {n_instances}")
            print(f"  Train:  {len(train_ids)} ({len(train_ids)/n_instances*100:.1f}%)")
            print(f"  Test:   {len(test_ids)} ({len(test_ids)/n_instances*100:.1f}%)")

    # Save
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    if verbose:
        print(f"\nSplits saved to {output}")

    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate train/test splits")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="data/splits.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    run_generate_splits(
        data_dir=args.data_dir,
        output=args.output,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
