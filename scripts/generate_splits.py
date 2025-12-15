#!/usr/bin/env python
"""
Gera splits train/test (ou train/val/test) para as instancias TSP.

Compativel com o split usado pelo outro grupo da disciplina (seed=42, 90/10).
Com --with_val, gera split 80/10/10 preservando o conjunto de teste original.
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
    Gera indices de train e test (compatível com outras equipes).

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


def generate_split_with_val(
    n_instances: int,
    seed: int = 42,
    val_seed: int = 123,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """
    Gera indices de train, val e test.

    Preserva o conjunto de teste do split original (seed=42, 90/10),
    e divide o antigo train em train + val.

    Args:
        n_instances: Total de instancias
        seed: Seed para o split original train/test (default 42)
        val_seed: Seed para dividir train em train/val (default 123)
        train_ratio: Fracao final para treino (default 0.8)
        val_ratio: Fracao para validacao (default 0.1)

    Returns:
        Tupla (train_indices, val_indices, test_indices)
    """
    # Primeiro, gera split original 90/10 para preservar test
    original_train, test_ids = generate_split(n_instances, seed=seed, train_ratio=0.9)

    # Agora divide original_train em train + val
    # val_ratio de 0.1 do total = ~1/9 do original_train
    random.seed(val_seed)
    random.shuffle(original_train)

    # Calcular quantos vão para val (10% do total)
    n_val = int(n_instances * val_ratio)
    val_ids = sorted(original_train[:n_val])
    train_ids = sorted(original_train[n_val:])

    return train_ids, val_ids, test_ids


def run_generate_splits(
    data_dir: str = "data",
    output: str = "data/splits.json",
    seed: int = 42,
    train_ratio: float = 0.9,
    with_val: bool = False,
    val_seed: int = 123,
    val_ratio: float = 0.1,
    verbose: bool = True,
) -> dict:
    """
    Generate train/test (or train/val/test) splits for all instance files.

    Args:
        data_dir: Directory containing JSON instance files.
        output: Output path for splits JSON.
        seed: Random seed for original train/test split.
        train_ratio: Fraction for training (only used without --with_val).
        with_val: If True, generate train/val/test split (80/10/10).
        val_seed: Seed for train/val subdivision.
        val_ratio: Fraction for validation (default 0.1).
        verbose: Print progress.

    Returns:
        Dictionary with splits.
    """
    data_dir = Path(data_dir)
    instance_files = ["EUC_2D.json", "ATT.json", "GEO.json"]
    splits = {}

    if verbose and with_val:
        print("Generating train/val/test split (preserving original test set)\n")

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

        key = f"data/{filename}"

        if with_val:
            train_ids, val_ids, test_ids = generate_split_with_val(
                n_instances, seed=seed, val_seed=val_seed, val_ratio=val_ratio
            )
            splits[key] = {"train": train_ids, "val": val_ids, "test": test_ids}

            if verbose:
                print(f"{filename} (from {resolved.name}):")
                print(f"  Total:  {n_instances}")
                print(f"  Train:  {len(train_ids)} ({len(train_ids)/n_instances*100:.1f}%)")
                print(f"  Val:    {len(val_ids)} ({len(val_ids)/n_instances*100:.1f}%)")
                print(f"  Test:   {len(test_ids)} ({len(test_ids)/n_instances*100:.1f}%)")
        else:
            train_ids, test_ids = generate_split(n_instances, seed=seed, train_ratio=train_ratio)
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
    parser = argparse.ArgumentParser(description="Generate train/test (or train/val/test) splits")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="data/splits.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument(
        "--with_val",
        action="store_true",
        help="Generate train/val/test split (80/10/10), preserving original test set",
    )
    parser.add_argument("--val_seed", type=int, default=123, help="Seed for train/val split")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio (default 0.1)")
    args = parser.parse_args()

    run_generate_splits(
        data_dir=args.data_dir,
        output=args.output,
        seed=args.seed,
        train_ratio=args.train_ratio,
        with_val=args.with_val,
        val_seed=args.val_seed,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
