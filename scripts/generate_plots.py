#!/usr/bin/env python
"""
Generate plots from training stats and evaluation results.

Reads stats JSON files and evaluation CSVs to generate:
- Learning curves (gaps over episodes)
- Q-values heatmaps
- Evaluation result comparisons

Usage:
    python scripts/generate_plots.py --models "models/dqn/*.pt"
    python scripts/generate_plots.py --models "models/dqn/EUC_2D_*.pt" --results "data/results/*.csv"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from glob import glob

from src.rl.dqn import load_model, compute_q_matrix, N_ACTIONS, ACTION_DECODE
from utils import (
    plot_learning_curve,
    plot_q_values_heatmap,
    plot_gap_violins_by_size,
    load_gaps_from_csv,
)


def generate_action_labels() -> dict[int, str]:
    """Generate short action labels from ACTION_DECODE."""
    labels = {}
    for action_id, (pert, ls) in ACTION_DECODE.items():
        # Shorten names
        pert_short = {
            "two_swap": "swap",
            "segment_reverse": "rev",
            "random": "rand",
            "nearest": "near",
            "cheapest": "cheap",
            "grasp": "grasp",
        }.get(pert, pert[:4])

        ls_short = {
            "two_opt_full": "2opt",
            "two_opt_nn": "2nn",
            "two_opt_dlb": "2dlb",
            "lin_kernighan": "LK",
        }.get(ls, ls[:4])

        labels[action_id] = f"{pert_short}+{ls_short}"
    return labels


def run_plots(
    models_pattern: str = "models/dqn/*.pt",
    results_pattern: str = "data/results/*.csv",
    output_dir: str = "data/plots",
    history_len: int = 2,
    hidden_dim: int = 64,
    verbose: bool = True,
) -> dict:
    """
    Generate plots from training stats and evaluation results.

    Args:
        models_pattern: Glob pattern for model files.
        results_pattern: Glob pattern for result CSVs.
        output_dir: Output directory for plots.
        history_len: History length (must match models).
        hidden_dim: Hidden dimension (must match models).
        verbose: Print progress.

    Returns:
        Dictionary with generated plot paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    action_labels = generate_action_labels()
    state_dim = 3 + history_len * N_ACTIONS
    generated = {"learning_curves": [], "heatmaps": [], "violins": []}

    # Find model files
    model_paths = sorted(glob(models_pattern))
    if verbose:
        print(f"Found {len(model_paths)} model(s)")

    # Training plots (learning curves, Q-values heatmaps)
    for model_path in model_paths:
        model_path = Path(model_path)
        stats_path = model_path.with_name(model_path.stem + "_stats.json")
        name = model_path.stem

        if verbose:
            print(f"\nProcessing: {name}")

        # Learning curve
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)

            if "episode_best_gaps" in stats:
                gaps = stats["episode_best_gaps"]
                plot_path = output_dir / f"{name}_learning_curve.png"
                plot_learning_curve(
                    gaps,
                    window=min(100, len(gaps) // 5) if len(gaps) > 10 else 1,
                    title=f"Learning Curve: {name}",
                    save_path=plot_path,
                )
                generated["learning_curves"].append(str(plot_path))
                if verbose:
                    print(f"  Saved: {plot_path}")

        # Q-values heatmap
        try:
            model = load_model(model_path, state_dim, N_ACTIONS, hidden_dim)
            q_matrix = compute_q_matrix(model, N_ACTIONS, history_len)
            gap_labels = ["0%", "1%", "2%", "5%", "10%", "20%", "50%"]
            act_labels = [action_labels.get(i, f"A{i}") for i in range(N_ACTIONS)]

            plot_path = output_dir / f"{name}_q_heatmap.png"
            plot_q_values_heatmap(
                q_matrix,
                gap_labels,
                act_labels,
                title=f"Q-values: {name}",
                save_path=plot_path,
            )
            generated["heatmaps"].append(str(plot_path))
            if verbose:
                print(f"  Saved: {plot_path}")
        except Exception as e:
            if verbose:
                print(f"  Error generating Q-heatmap: {e}")

    # Evaluation plots (gap distributions)
    result_files = sorted(glob(results_pattern))
    if verbose:
        print(f"\nFound {len(result_files)} result file(s)")

    for result_path in result_files:
        result_path = Path(result_path)
        name = result_path.stem

        if verbose:
            print(f"\nProcessing: {name}")

        try:
            gaps_data = load_gaps_from_csv(result_path)
            for instance_type, gaps_by_size in gaps_data.items():
                if not gaps_by_size:
                    continue
                plot_path = output_dir / f"{name}_violins.png"
                plot_gap_violins_by_size(
                    gaps_by_size,
                    title=f"Gap Distribution: {name}",
                    save_path=plot_path,
                )
                generated["violins"].append(str(plot_path))
                if verbose:
                    print(f"  Saved: {plot_path}")
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")

    if verbose:
        print(f"\nPlots saved to: {output_dir}/")

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from training/evaluation data")
    parser.add_argument("--models", type=str, default="models/dqn/*.pt")
    parser.add_argument("--results", type=str, default="data/results/*.csv")
    parser.add_argument("--output_dir", type=str, default="data/plots")
    parser.add_argument("--history_len", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    args = parser.parse_args()

    run_plots(
        models_pattern=args.models,
        results_pattern=args.results,
        output_dir=args.output_dir,
        history_len=args.history_len,
        hidden_dim=args.hidden_dim,
    )


if __name__ == "__main__":
    main()
