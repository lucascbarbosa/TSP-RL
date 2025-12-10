"""Plotting utilities for TSP-RL visualization.

Academic-style plots with consistent formatting for publication.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray


# =============================================================================
# Style Configuration
# =============================================================================

# Academic color palette (colorblind-friendly)
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "tertiary": "#F18F01",
    "quaternary": "#C73E1D",
}

# State and action labels for Q-table visualization
STATE_LABELS = ["EXCELLENT", "GOOD", "REGULAR", "POOR", "BETTER"]
ACTION_LABELS = [
    "swap+2opt",
    "swap+LK",
    "rev+2opt",
    "rev+LK",
    "rand+2opt",
    "near+2opt",
    "cheap+2opt",
    "near+LK",
]


def setup_style() -> None:
    """Configure matplotlib for academic-style plots."""
    plt.rcParams.update(
        {
            # Figure
            "figure.facecolor": "white",
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            # Font
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            # Axes
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            # Lines
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
        }
    )


# Apply style on import
setup_style()


# =============================================================================
# Q-Learning Plots
# =============================================================================


def plot_q_convergence(
    history: dict[str, list[float]],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot Q-Learning convergence curve.

    Args:
        history: Dictionary with 'avg_q_value' list.
        title: Plot title (auto-generated if None).
        save_path: Path to save figure (displays if None).
    """
    avg_q = history["avg_q_value"]
    iterations = range(1, len(avg_q) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(iterations, avg_q, color=COLORS["primary"], linewidth=1.5)
    ax.fill_between(iterations, avg_q, alpha=0.15, color=COLORS["primary"])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Q-value")
    ax.set_xlim(0, len(avg_q))
    ax.set_ylim(bottom=0)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_q_heatmap(
    matrix: NDArray[np.float64],
    title: Optional[str] = None,
    state_labels: Optional[List[str]] = None,
    action_labels: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot Q-table as annotated heatmap.

    Args:
        matrix: Q-table array (states x actions).
        title: Plot title.
        state_labels: Labels for states (y-axis).
        action_labels: Labels for actions (x-axis).
        save_path: Path to save figure (displays if None).
    """
    state_labels = state_labels or STATE_LABELS[: matrix.shape[0]]
    action_labels = action_labels or ACTION_LABELS[: matrix.shape[1]]

    # Mask zeros for better visualization
    matrix_display = matrix.copy()
    mask = matrix_display == 0

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.heatmap(
        matrix_display,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        mask=mask,
        cbar_kws={"label": "Q-value"},
        xticklabels=action_labels,
        yticklabels=state_labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    # Show zeros in gray
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if mask[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color="#E0E0E0"))
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "0",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#888888",
                )

    ax.set_xlabel("Action")
    ax.set_ylabel("State")
    ax.set_xticklabels(action_labels, rotation=45, ha="right")

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
# Results Analysis Plots
# =============================================================================


def plot_gap_distribution(
    gaps: List[float],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot distribution of optimality gaps using violin + strip plot.

    Args:
        gaps: List of gap percentages.
        title: Plot title.
        save_path: Path to save figure (displays if None).
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Violin plot
    parts = ax.violinplot(gaps, positions=[0], showmeans=True, showmedians=True)

    # Style violin
    for pc in parts["bodies"]:
        pc.set_facecolor(COLORS["primary"])
        pc.set_alpha(0.3)
    parts["cmeans"].set_color(COLORS["secondary"])
    parts["cmedians"].set_color(COLORS["tertiary"])

    # Overlay strip plot for individual points
    jitter = np.random.normal(0, 0.04, len(gaps))
    ax.scatter(jitter, gaps, alpha=0.5, s=20, color=COLORS["primary"], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Gap (%)")
    ax.set_xticks([])
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)

    # Statistics annotation
    mean_gap = np.mean(gaps)
    median_gap = np.median(gaps)
    ax.text(
        0.95,
        0.95,
        f"Mean: {mean_gap:.2f}%\nMedian: {median_gap:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC"),
    )

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_gap_by_size(
    gaps_by_size: dict[int, List[float]],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot gap distribution grouped by instance size using box plots.

    Args:
        gaps_by_size: Dictionary mapping size -> list of gaps.
        title: Plot title.
        save_path: Path to save figure (displays if None).
    """
    sizes = sorted(gaps_by_size.keys())
    data = [gaps_by_size[s] for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(
        data,
        positions=range(len(sizes)),
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor=COLORS["secondary"], markeredgecolor="white", markersize=5),
    )

    # Style boxes
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["primary"])
        patch.set_alpha(0.4)
    for median in bp["medians"]:
        median.set_color(COLORS["tertiary"])
        median.set_linewidth(1.5)

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Instance Size (cities)")
    ax.set_ylabel("Gap (%)")
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_gap_comparison(
    gaps_by_method: dict[str, List[float]],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Compare gap distributions across methods using violin plots.

    Args:
        gaps_by_method: Dictionary mapping method name -> list of gaps.
        title: Plot title.
        save_path: Path to save figure (displays if None).
    """
    methods = list(gaps_by_method.keys())
    data = [gaps_by_method[m] for m in methods]

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.5), 5))

    parts = ax.violinplot(data, positions=range(len(methods)), showmeans=True, showmedians=True)

    # Style
    colors = list(COLORS.values())
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.4)

    parts["cmeans"].set_color("#333333")
    parts["cmedians"].set_color("#333333")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Gap (%)")
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)

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

    # Plot edges
    closed_tour = list(tour) + [tour[0]]
    xs = [coords[i][0] for i in closed_tour]
    ys = [coords[i][1] for i in closed_tour]
    ax.plot(xs, ys, color=COLORS["primary"], linewidth=1.2, zorder=1)

    # Plot nodes
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=50,
        c=COLORS["primary"],
        edgecolor="white",
        linewidth=1,
        zorder=2,
    )

    # Highlight depot
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
