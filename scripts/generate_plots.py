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
    plot_action_distribution,
    plot_q_values_heatmap,
    plot_gap_violins_by_size_method,
    plot_time_vs_gap_scatter,
    load_gaps_by_type_method,
    load_results_by_type,
)


def generate_action_labels() -> dict[int, str]:
    """Generate short action labels from ACTION_DECODE."""
    labels = {}
    for action_id, (pert, ls) in ACTION_DECODE.items():
        # Shorten perturbation names
        if pert.startswith("grasp_"):
            # grasp_0.03 -> gr1, grasp_0.1 -> gr2, grasp_0.3 -> gr3
            alpha = pert.split("_")[1]
            pert_short = {"0.03": "gr1", "0.1": "gr2", "0.3": "gr3"}.get(alpha, "gr?")
        else:
            pert_short = {
                "two_swap": "swap",
                "segment_reverse": "rev",
                "random": "rand",
                "nearest": "near",
                "cheapest": "cheap",
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
    verbose: bool = True,
) -> dict:
    """
    Generate plots from training stats and evaluation results.

    Args:
        models_pattern: Glob pattern for model files.
        results_pattern: Glob pattern for result CSVs.
        output_dir: Output directory for plots.
        verbose: Print progress.

    Returns:
        Dictionary with generated plot paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    action_labels = generate_action_labels()
    action_labels_list = [action_labels.get(i, f"A{i}") for i in range(N_ACTIONS)]
    generated = {"learning_curves": [], "action_dists": [], "heatmaps": [], "violins": [], "scatters": []}

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

            if "action_counts" in stats:
                # Convert string keys to int (JSON serializes dict keys as strings)
                action_counts = {int(k): v for k, v in stats["action_counts"].items()}
                plot_path = output_dir / f"{name}_action_dist.png"
                plot_action_distribution(
                    action_counts,
                    action_labels=action_labels_list,
                    title=f"Action Distribution: {name}",
                    save_path=plot_path,
                )
                generated["action_dists"].append(str(plot_path))
                if verbose:
                    print(f"  Saved: {plot_path}")

        # Q-values heatmap
        try:
            model = load_model(model_path)
            q_matrix = compute_q_matrix(model)
            gap_labels = ["0%", "1%", "2%", "5%", "10%", "20%", "50%"]
            act_labels = [action_labels.get(i, f"A{i}") for i in range(model.n_actions)]

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

    # Evaluation plots (gap distributions) - one plot per type with all sizes
    result_files = sorted(glob(results_pattern))
    if verbose:
        print(f"\nFound {len(result_files)} result file(s)")

    if result_files:
        try:
            # Aggregate all results by type and method
            all_gaps = load_gaps_by_type_method(result_files)

            for instance_type, gaps_by_method in all_gaps.items():
                if not gaps_by_method:
                    continue

                plot_path = output_dir / f"{instance_type}_gaps_violin.png"
                plot_gap_violins_by_size_method(
                    gaps_by_method,
                    title=f"Gap Distribution: {instance_type}",
                    save_path=plot_path,
                )
                generated["violins"].append(str(plot_path))
                if verbose:
                    print(f"  Saved: {plot_path}")

            # Time vs Gap scatter plots (one per type)
            all_results = load_results_by_type(result_files)

            for instance_type, data_by_size in all_results.items():
                if not data_by_size:
                    continue

                plot_path = output_dir / f"{instance_type}_time_vs_gap.png"
                plot_time_vs_gap_scatter(
                    data_by_size,
                    title=f"Time vs Gap: {instance_type}",
                    save_path=plot_path,
                )
                generated["scatters"].append(str(plot_path))
                if verbose:
                    print(f"  Saved: {plot_path}")

        except Exception as e:
            if verbose:
                print(f"  Error loading results: {e}")

    if verbose:
        print(f"\nPlots saved to: {output_dir}/")

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from training/evaluation data")
    parser.add_argument("--models", type=str, default="models/dqn/*.pt")
    parser.add_argument("--results", type=str, default="data/results/*.csv")
    parser.add_argument("--output_dir", type=str, default="data/plots")
    args = parser.parse_args()

    run_plots(
        models_pattern=args.models,
        results_pattern=args.results,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
