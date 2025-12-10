#!/usr/bin/env python
"""
Gera splits train/test para as instancias TSP.

Compativel com o split usado pelo outro grupo da disciplina (seed=42, 90/10).
"""

import json
import random
import argparse
from pathlib import Path


def generate_split(n_instances: int, seed: int = 42, train_ratio: float = 0.9):
    """
    Gera indices de train e test.

    Args:
        n_instances: Total de instancias
        seed: Seed para reproducibilidade
        train_ratio: Fracao para treino (default 0.9 = 80% train + 10% val)

    Returns:
        Tupla (train_indices, test_indices)
    """
    indices = list(range(n_instances))

    random.seed(seed)
    random.shuffle(indices)

    n_train = int(n_instances * train_ratio)

    train_indices = sorted(indices[:n_train])
    test_indices = sorted(indices[n_train:])

    return train_indices, test_indices


def main():
    parser = argparse.ArgumentParser(description="Generate train/test splits")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing JSON instance files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/splits.json",
        help="Output path for splits JSON",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train ratio (default: 0.9 = 90%% train, 10%% test)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    instance_files = ["EUC_2D.json", "ATT.json", "GEO.json"]

    splits = {}

    for filename in instance_files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"[!] {filepath} not found, skipping...")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)

        n_instances = len(data)
        train_ids, test_ids = generate_split(n_instances, seed=args.seed, train_ratio=args.train_ratio)

        key = f"data/{filename}"
        splits[key] = {"train": train_ids, "test": test_ids}

        print(f"{filename}:")
        print(f"  Total:  {n_instances}")
        print(f"  Train:  {len(train_ids)} ({len(train_ids)/n_instances*100:.1f}%)")
        print(f"  Test:   {len(test_ids)} ({len(test_ids)/n_instances*100:.1f}%)")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSplits saved to {args.output}")


if __name__ == "__main__":
    main()
