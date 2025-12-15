"""Plotting utilities for TSP-RL visualization."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Style Configuration
# =============================================================================

COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "tertiary": "#F18F01",
    "quaternary": "#C73E1D",
}


def setup_style() -> None:
    """Configure matplotlib for academic-style plots."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
        }
    )


setup_style()


# =============================================================================
# Data Loading
# =============================================================================


def load_gaps_by_type_method(
    csv_paths: List[Union[str, Path]],
) -> Dict[str, Dict[str, Dict[int, List[float]]]]:
    """
    Load and aggregate gap data from multiple CSV files.

    Args:
        csv_paths: List of paths to results CSV files.

    Returns:
        Nested dict: {instance_type: {method: {size: [gaps]}}}
    """
    data: Dict[str, Dict[str, Dict[int, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for csv_path in csv_paths:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                instance_type = row["Type"]
                method = row["Method"]
                dimension = int(row["Dimension"])
                gap_str = row["Gap"].replace("%", "")
                gap_value = float(gap_str)
                data[instance_type][method][dimension].append(gap_value)

    return {t: {m: dict(sizes) for m, sizes in methods.items()} for t, methods in data.items()}


def load_results_by_type(
    csv_paths: List[Union[str, Path]],
) -> Dict[str, Dict[int, Dict[str, List[float]]]]:
    """
    Load gap and time data from CSV files, grouped by type and size.

    Args:
        csv_paths: List of paths to results CSV files.

    Returns:
        Nested dict: {instance_type: {size: {"gaps": [...], "times": [...]}}}
    """
    data: Dict[str, Dict[int, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"gaps": [], "times": []})
    )

    for csv_path in csv_paths:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                instance_type = row["Type"]
                dimension = int(row["Dimension"])
                gap = float(row["Gap"].replace("%", ""))
                time_s = float(row.get("Time (s)", row.get("Time", 0)))
                data[instance_type][dimension]["gaps"].append(gap)
                data[instance_type][dimension]["times"].append(time_s)

    return {t: dict(sizes) for t, sizes in data.items()}


# =============================================================================
# Training Plots
# =============================================================================


def plot_learning_curve(
    episode_gaps: List[float],
    window: int = 100,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot DQN learning curve showing gap evolution during training.

    Args:
        episode_gaps: List of best gaps per episode.
        window: Moving average window size.
        title: Plot title.
        save_path: Path to save figure (displays if None).
    """
    episodes = np.arange(1, len(episode_gaps) + 1)
    gaps = np.array(episode_gaps)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(episodes, gaps, color=COLORS["primary"], alpha=0.2, linewidth=0.8, label="Per episode")

    if len(gaps) >= window:
        ma = np.convolve(gaps, np.ones(window) / window, mode="valid")
        ma_episodes = episodes[window - 1 :]
        ax.plot(ma_episodes, ma, color=COLORS["primary"], linewidth=2, label=f"Moving avg ({window} ep)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Best Gap (%)")
    ax.set_xlim(0, len(episodes))
    ax.set_ylim(bottom=0)
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper right", framealpha=0.9)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_action_distribution(
    action_counts: Union[Dict[int, int], List[int]],
    action_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot distribution of actions chosen by the agent as horizontal bars.

    Args:
        action_counts: Dict {action_id: count} or list of counts per action.
        action_labels: Labels for each action (default: "A0", "A1", ...).
        title: Plot title.
        save_path: Path to save figure (displays if None).
    """
    if isinstance(action_counts, dict):
        n_actions = max(action_counts.keys()) + 1
        counts = [action_counts.get(i, 0) for i in range(n_actions)]
    else:
        counts = list(action_counts)
        n_actions = len(counts)

    total = sum(counts)
    if total == 0:
        return
    percentages = [100 * c / total for c in counts]

    if action_labels is None:
        action_labels = [f"A{i}" for i in range(n_actions)]

    fig, ax = plt.subplots(figsize=(8, max(4, n_actions * 0.25)))

    y_pos = np.arange(n_actions)
    bars = ax.barh(y_pos, percentages, color=COLORS["primary"], alpha=0.7, edgecolor="white")

    # Highlight top 3 actions
    top_indices = np.argsort(percentages)[-3:]
    for idx in top_indices:
        bars[idx].set_color(COLORS["secondary"])
        bars[idx].set_alpha(0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(action_labels, fontsize=9)
    ax.set_xlabel("Frequency (%)")
    ax.set_ylabel("Action")
    ax.invert_yaxis()

    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        if pct > 1:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=8)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_learning_curves_comparison(
    stats_list: List[dict],
    labels: List[str],
    window: int = 100,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot learning curves comparing multiple DQN variants.

    Args:
        stats_list: List of dicts with 'episode_best_gaps' key.
        labels: Label for each variant (e.g., ["DQN", "Double DQN"]).
        window: Moving average window size.
        title: Plot title.
        save_path: Path to save figure (displays if None).
    """
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"], COLORS["quaternary"]]
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (stats, label) in enumerate(zip(stats_list, labels)):
        gaps = np.array(stats["episode_best_gaps"])
        episodes = np.arange(1, len(gaps) + 1)
        color = colors[i % len(colors)]

        ax.plot(episodes, gaps, color=color, alpha=0.4, linewidth=0.8)

        if len(gaps) >= window:
            ma = np.convolve(gaps, np.ones(window) / window, mode="valid")
            ma_episodes = episodes[window - 1 :]
            ax.plot(ma_episodes, ma, color=color, linewidth=2.5, label=f"{label} (MA-{window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Best Gap (%)")
    ax.set_xlim(0, max(len(s["episode_best_gaps"]) for s in stats_list))
    ax.set_ylim(bottom=0)
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper right", framealpha=0.9)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_q_values_comparison(
    stats_list: List[dict],
    labels: List[str],
    window: int = 500,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot Q-value evolution comparing multiple DQN variants.

    Shows mean and max Q-values over training to analyze overestimation.

    Args:
        stats_list: List of dicts with 'q_values_mean' and 'q_values_max' keys.
        labels: Label for each variant.
        window: Moving average window size.
        title: Plot title.
        save_path: Path to save figure (displays if None).
    """
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"], COLORS["quaternary"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i, (stats, label) in enumerate(zip(stats_list, labels)):
        color = colors[i % len(colors)]

        # Mean Q-values
        q_mean = np.array(stats.get("q_values_mean", []))
        if len(q_mean) > 0:
            updates = np.arange(1, len(q_mean) + 1)
            axes[0].plot(updates, q_mean, color=color, alpha=0.35, linewidth=0.5)
            if len(q_mean) >= window:
                ma = np.convolve(q_mean, np.ones(window) / window, mode="valid")
                axes[0].plot(np.arange(window, len(q_mean) + 1), ma, color=color, linewidth=2, label=label)

        # Max Q-values
        q_max = np.array(stats.get("q_values_max", []))
        if len(q_max) > 0:
            updates = np.arange(1, len(q_max) + 1)
            axes[1].plot(updates, q_max, color=color, alpha=0.35, linewidth=0.5)
            if len(q_max) >= window:
                ma = np.convolve(q_max, np.ones(window) / window, mode="valid")
                axes[1].plot(np.arange(window, len(q_max) + 1), ma, color=color, linewidth=2, label=label)

    axes[0].set_xlabel("Update Step")
    axes[0].set_ylabel("Mean Q-value")
    axes[0].set_title("Mean Q-values")
    axes[0].legend(loc="upper left", framealpha=0.9)

    axes[1].set_xlabel("Update Step")
    axes[1].set_ylabel("Max Q-value")
    axes[1].set_title("Max Q-values (overestimation indicator)")
    axes[1].legend(loc="upper left", framealpha=0.9)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_action_distribution_comparison(
    stats_list: List[dict],
    labels: List[str],
    action_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot action distributions as a butterfly chart (diverging horizontal bars).

    Left side shows first variant (e.g., DQN), right side shows second (e.g., Double DQN).
    Central vertical line at zero, with frequencies growing outward in both directions.

    Args:
        stats_list: List of dicts with 'action_counts' key (expects exactly 2).
        labels: Label for each variant.
        action_labels: Labels for each action.
        title: Plot title.
        save_path: Path to save figure (displays if None).
    """
    if len(stats_list) != 2:
        # Fallback for non-binary comparisons
        return _plot_action_distribution_comparison_fallback(stats_list, labels, action_labels, title, save_path)

    colors = [COLORS["primary"], COLORS["secondary"]]

    # Get action counts
    all_counts = []
    for stats in stats_list:
        counts = stats.get("action_counts", {})
        if isinstance(counts, dict):
            # Handle string keys (from JSON serialization)
            int_counts = {int(k): v for k, v in counts.items()}
            n_actions = max(int_counts.keys()) + 1 if int_counts else 0
            counts = [int_counts.get(i, 0) for i in range(n_actions)]
        all_counts.append(counts)

    if not all_counts or not all_counts[0]:
        return

    n_actions = len(all_counts[0])

    if action_labels is None:
        action_labels = [f"A{i}" for i in range(n_actions)]

    # Convert to percentages
    percentages = []
    for counts in all_counts:
        total = sum(counts)
        if total > 0:
            percentages.append([100 * c / total for c in counts])
        else:
            percentages.append([0] * n_actions)

    fig, ax = plt.subplots(figsize=(10, max(5, n_actions * 0.35)))

    y_positions = np.arange(n_actions)
    bar_height = 0.7

    # Left side (first variant) - negative values for left-growing bars
    left_pcts = [-p for p in percentages[0]]
    ax.barh(
        y_positions,
        left_pcts,
        height=bar_height,
        color=colors[0],
        alpha=0.8,
        label=labels[0],
    )

    # Right side (second variant) - positive values for right-growing bars
    right_pcts = percentages[1]
    ax.barh(
        y_positions,
        right_pcts,
        height=bar_height,
        color=colors[1],
        alpha=0.8,
        label=labels[1],
    )

    # Central vertical line at zero
    ax.axvline(x=0, color="#333333", linewidth=1.2)

    # Symmetric x-axis limits
    max_pct = max(max(percentages[0]), max(percentages[1]))
    limit = max(10, max_pct * 1.15)  # At least 10%, with some padding
    ax.set_xlim(-limit, limit)

    # X-axis: show absolute values (no negative signs)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{abs(x):.0f}"))
    ax.set_xlabel("Frequency (%)")

    # Y-axis
    ax.set_yticks(y_positions)
    ax.set_yticklabels(action_labels, fontsize=9)
    ax.set_ylabel("Action")
    ax.invert_yaxis()

    # Legend at top
    ax.legend(loc="upper center", ncol=2, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

    # Add variant labels on each side
    ax.text(-limit * 0.5, -0.8, labels[0], ha="center", va="bottom", fontsize=10, fontweight="bold", color=colors[0])
    ax.text(limit * 0.5, -0.8, labels[1], ha="center", va="bottom", fontsize=10, fontweight="bold", color=colors[1])

    if title:
        ax.set_title(title, pad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def _plot_action_distribution_comparison_fallback(
    stats_list: List[dict],
    labels: List[str],
    action_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Fallback for non-binary comparisons (clustered horizontal bars)."""
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"], COLORS["quaternary"]]

    all_counts = []
    for stats in stats_list:
        counts = stats.get("action_counts", {})
        if isinstance(counts, dict):
            int_counts = {int(k): v for k, v in counts.items()}
            n_actions = max(int_counts.keys()) + 1 if int_counts else 0
            counts = [int_counts.get(i, 0) for i in range(n_actions)]
        all_counts.append(counts)

    if not all_counts or not all_counts[0]:
        return

    n_actions = len(all_counts[0])
    n_methods = len(stats_list)

    if action_labels is None:
        action_labels = [f"A{i}" for i in range(n_actions)]

    percentages = []
    for counts in all_counts:
        total = sum(counts)
        if total > 0:
            percentages.append([100 * c / total for c in counts])
        else:
            percentages.append([0] * n_actions)

    fig, ax = plt.subplots(figsize=(10, max(4, n_actions * 0.3)))

    bar_height = 0.8 / n_methods
    y_positions = np.arange(n_actions)

    for i, (pcts, label) in enumerate(zip(percentages, labels)):
        offset = (i - (n_methods - 1) / 2) * bar_height
        ax.barh(
            y_positions + offset, pcts, height=bar_height * 0.9, color=colors[i % len(colors)], alpha=0.7, label=label
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(action_labels, fontsize=9)
    ax.set_xlabel("Frequency (%)")
    ax.set_ylabel("Action")
    ax.invert_yaxis()
    ax.legend(loc="lower right", framealpha=0.9)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_q_values_heatmap(
    q_matrix: NDArray[np.float64],
    gap_labels: Optional[List[str]] = None,
    action_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot Q-values as a heatmap over discretized gap levels.

    Args:
        q_matrix: Matrix of shape (n_gaps, n_actions) with Q-values.
        gap_labels: Labels for Y-axis.
        action_labels: Labels for X-axis.
        title: Plot title.
        save_path: Path to save figure (displays if None).
    """
    n_gaps, n_actions = q_matrix.shape

    if gap_labels is None:
        default_gaps = [0, 1, 2, 5, 10, 20, 50]
        gap_labels = [f"{g}%" for g in default_gaps[:n_gaps]]
    if action_labels is None:
        action_labels = [f"A{i}" for i in range(n_actions)]

    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.imshow(q_matrix, aspect="auto", cmap="YlOrRd")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Q-value", rotation=270, labelpad=15)

    ax.set_xticks(range(n_actions))
    ax.set_xticklabels(action_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_gaps))
    ax.set_yticklabels(gap_labels)
    ax.set_xlabel("Action")
    ax.set_ylabel("Gap Level")

    for i in range(n_gaps):
        best_action = int(np.argmax(q_matrix[i, :]))
        ax.add_patch(plt.Rectangle((best_action - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2))

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_q_values_heatmap_time_comparison(
    q_matrix_early: NDArray[np.float64],
    q_matrix_late: NDArray[np.float64],
    gap_labels: Optional[List[str]] = None,
    action_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    t_early: float = 0.8,
    t_late: float = 0.2,
) -> None:
    """
    Plot Q-values heatmaps comparing early vs late episode (time remaining).

    Shows how the learned policy changes based on remaining time budget.
    Displays two heatmaps stacked vertically with shared colorbar.

    Args:
        q_matrix_early: Q-values with high time remaining (e.g., 80%).
        q_matrix_late: Q-values with low time remaining (e.g., 20%).
        gap_labels: Labels for Y-axis (gap levels).
        action_labels: Labels for X-axis (actions).
        title: Base plot title.
        save_path: Path to save figure (displays if None).
        t_early: Time ratio for early episode (for subtitle).
        t_late: Time ratio for late episode (for subtitle).
    """
    n_gaps, n_actions = q_matrix_early.shape

    if gap_labels is None:
        default_gaps = [0, 1, 2, 5, 10, 20, 50]
        gap_labels = [f"{g}%" for g in default_gaps[:n_gaps]]
    if action_labels is None:
        action_labels = [f"A{i}" for i in range(n_actions)]

    # Use shared color scale
    vmin = min(q_matrix_early.min(), q_matrix_late.min())
    vmax = max(q_matrix_early.max(), q_matrix_late.max())

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)

    # Early episode (top)
    im1 = axes[0].imshow(q_matrix_early, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    axes[0].set_yticks(range(n_gaps))
    axes[0].set_yticklabels(gap_labels)
    axes[0].set_ylabel("Gap Level")
    axes[0].set_title(f"Early episode ({int(t_early * 100)}% time remaining)", fontsize=10)

    for i in range(n_gaps):
        best_action = int(np.argmax(q_matrix_early[i, :]))
        axes[0].add_patch(plt.Rectangle((best_action - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2))

    # Late episode (bottom)
    im2 = axes[1].imshow(q_matrix_late, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    axes[1].set_xticks(range(n_actions))
    axes[1].set_xticklabels(action_labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(n_gaps))
    axes[1].set_yticklabels(gap_labels)
    axes[1].set_xlabel("Action")
    axes[1].set_ylabel("Gap Level")
    axes[1].set_title(f"Late episode ({int(t_late * 100)}% time remaining)", fontsize=10)

    for i in range(n_gaps):
        best_action = int(np.argmax(q_matrix_late[i, :]))
        axes[1].add_patch(plt.Rectangle((best_action - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2))

    # Shared colorbar
    cbar = fig.colorbar(im2, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Q-value", rotation=270, labelpad=15)

    if title:
        fig.suptitle(title, fontsize=12)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


# =============================================================================
# Results Plots
# =============================================================================


def plot_gap_violins_by_size_method(
    gaps_by_method: Dict[str, Dict[int, List[float]]],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot gap distributions with sizes on X-axis and methods side by side.

    Args:
        gaps_by_method: Dict mapping method -> {size: [gaps]}.
        title: Plot title.
        save_path: Path to save figure (displays if None).
    """
    methods = sorted(gaps_by_method.keys())
    all_sizes = set()
    for method_data in gaps_by_method.values():
        all_sizes.update(method_data.keys())
    sizes = sorted(all_sizes)

    if not sizes or not methods:
        return

    n_methods = len(methods)
    n_sizes = len(sizes)
    width = 0.35
    method_colors = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"], COLORS["quaternary"]]

    fig_width = max(8, n_sizes * 1.5 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    for m_idx, method in enumerate(methods):
        method_data = gaps_by_method.get(method, {})
        positions = []
        data = []

        for s_idx, size in enumerate(sizes):
            gaps = method_data.get(size, [])
            if gaps:
                offset = (m_idx - (n_methods - 1) / 2) * width
                positions.append(s_idx + offset)
                data.append(gaps)

        if not data:
            continue

        color = method_colors[m_idx % len(method_colors)]
        parts = ax.violinplot(
            data,
            positions=positions,
            showmeans=True,
            showmedians=False,
            widths=width * 0.9,
        )

        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.6)

        parts["cmeans"].set_color(color)
        parts["cmeans"].set_linewidth(2)
        for partname in ["cbars", "cmins", "cmaxes"]:
            parts[partname].set_color(color)
            parts[partname].set_linewidth(1)

    ax.set_xticks(range(n_sizes))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Instance Size (n)")
    ax.set_ylabel("Gap (%)")
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=method_colors[i % len(method_colors)], alpha=0.6, label=m) for i, m in enumerate(methods)
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_time_vs_gap_scatter(
    data_by_size: Dict[int, Dict[str, List[float]]],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Scatter plot of execution time vs gap, colored by instance size.

    Shows relationship between computation time and solution quality,
    with all sizes of an instance type in a single plot.

    Args:
        data_by_size: Dict {size: {"gaps": [...], "times": [...]}}.
        title: Plot title.
        save_path: Path to save figure (displays if None).
    """
    sizes = sorted(data_by_size.keys())
    if not sizes:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Color palette for sizes
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(sizes) - 1)) for i in range(len(sizes))]

    for size, color in zip(sizes, colors):
        gaps = np.array(data_by_size[size]["gaps"])
        times = np.array(data_by_size[size]["times"])

        if len(gaps) == 0:
            continue

        ax.scatter(times, gaps, c=[color], s=20, alpha=0.6, label=f"n={size}", edgecolor="none")

    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7, label="Optimal")
    ax.set_xlabel("Execution Time (s)")
    ax.set_ylabel("Gap (%)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


# =============================================================================
# TSP Visualization
# =============================================================================


def plot_tour(
    coords: NDArray[np.float64],
    tour: List[int],
    title: Optional[str] = None,
    highlight_depot: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot a TSP tour on 2D coordinates.

    Args:
        coords: Array of (x, y) coordinates.
        tour: Tour as list of city indices (0-based).
        title: Plot title.
        highlight_depot: Whether to highlight the starting city.
        save_path: Path to save figure (displays if None).
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    closed_tour = list(tour) + [tour[0]]
    xs = [coords[i][0] for i in closed_tour]
    ys = [coords[i][1] for i in closed_tour]
    ax.plot(xs, ys, color=COLORS["primary"], linewidth=1.2, zorder=1)

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=50,
        c=COLORS["primary"],
        edgecolor="white",
        linewidth=1,
        zorder=2,
    )

    if highlight_depot and len(tour) > 0:
        depot = tour[0]
        ax.scatter(
            coords[depot, 0],
            coords[depot, 1],
            s=100,
            c=COLORS["secondary"],
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
            marker="s",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()
