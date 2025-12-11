"""Plotting utilities for TSP-RL visualization.

Academic-style plots with consistent formatting for publication.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
# Results Loading
# =============================================================================


def load_gaps_from_csv(
    csv_path: Union[str, Path],
) -> Dict[str, Dict[int, List[float]]]:
    """
    Load gap data from results CSV file.

    Args:
        csv_path: Path to results CSV file.

    Returns:
        Nested dict: {instance_type: {size: [gaps]}}
        Example: {"EUC_2D": {10: [0.0, 1.2, ...], 20: [...]}, ...}
    """
    gaps_by_type_size: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance_type = row["Type"]
            dimension = int(row["Dimension"])
            # Parse gap: remove "%" and convert to float
            gap_str = row["Gap"].replace("%", "")
            gap_value = float(gap_str)

            gaps_by_type_size[instance_type][dimension].append(gap_value)

    # Convert defaultdicts to regular dicts
    return {t: dict(sizes) for t, sizes in gaps_by_type_size.items()}


def load_results_from_csv(
    csv_path: Union[str, Path],
) -> Dict[str, Dict[int, Dict[str, List[float]]]]:
    """
    Load full results data from CSV file (gaps, times, iterations).

    Args:
        csv_path: Path to results CSV file.

    Returns:
        Nested dict: {instance_type: {size: {"gaps": [...], "times": [...], ...}}}
    """
    data: Dict[str, Dict[int, Dict[str, List]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance_type = row["Type"]
            dimension = int(row["Dimension"])

            # Parse gap
            gap_str = row["Gap"].replace("%", "")
            gap_value = float(gap_str)

            # Parse time (ms) - handle both old "Time" (seconds) and new "Time (ms)" formats
            if "Time (ms)" in row:
                time_ms = float(row["Time (ms)"])
            elif "Time" in row:
                time_ms = float(row["Time"]) * 1000  # Convert s to ms
            else:
                time_ms = 0.0

            # Parse iterations if available
            iterations = int(row.get("Total Iterations", 0))

            data[instance_type][dimension]["gaps"].append(gap_value)
            data[instance_type][dimension]["times"].append(time_ms)
            data[instance_type][dimension]["iterations"].append(iterations)

    # Convert defaultdicts to regular dicts
    return {t: {s: dict(metrics) for s, metrics in sizes.items()} for t, sizes in data.items()}


def generate_gap_violin_plots(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    types: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate violin plots for all instance types from results CSV.

    Creates one plot per instance type showing gap distribution by size.

    Args:
        csv_path: Path to results CSV file.
        output_dir: Directory to save plots.
        types: Instance types to plot (default: all found in CSV).

    Returns:
        List of generated plot file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gaps_data = load_gaps_from_csv(csv_path)

    if types is not None:
        # Filter to requested types
        gaps_data = {t: gaps_data[t] for t in types if t in gaps_data}

    generated_files: List[str] = []

    for instance_type, gaps_by_size in gaps_data.items():
        if not gaps_by_size:
            continue

        # Create type-specific output directory
        type_dir = output_dir / instance_type
        type_dir.mkdir(parents=True, exist_ok=True)

        # Generate violin plot
        plot_path = type_dir / f"{instance_type}_gap_violins.png"
        plot_gap_violins_by_size(
            gaps_by_size,
            title=f"Q-ILS Gap Distribution ({instance_type})",
            save_path=plot_path,
        )
        generated_files.append(str(plot_path))

        print(f"  Generated: {plot_path}")

    return generated_files


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


def plot_gap_violins_by_size(
    gaps_by_size: dict[int, List[float]],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_points: bool = False,
) -> None:
    """
    Plot gap distribution grouped by instance size using violin plots.

    Shows empirical probability density for each instance size.

    Args:
        gaps_by_size: Dictionary mapping size -> list of gaps.
        title: Plot title.
        save_path: Path to save figure (displays if None).
        show_points: If True, overlay individual data points.
    """
    sizes = sorted(gaps_by_size.keys())
    data = [gaps_by_size[s] for s in sizes]

    # Adjust figure width based on number of sizes
    fig_width = max(6, len(sizes) * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    # Create violin plot
    parts = ax.violinplot(
        data,
        positions=range(len(sizes)),
        showmeans=True,
        showmedians=True,
        widths=0.7,
    )

    # Style violin bodies
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS["primary"])
        pc.set_edgecolor(COLORS["primary"])
        pc.set_alpha(0.4)

    # Style mean and median lines
    parts["cmeans"].set_color(COLORS["secondary"])
    parts["cmeans"].set_linewidth(1.5)
    parts["cmedians"].set_color(COLORS["tertiary"])
    parts["cmedians"].set_linewidth(1.5)

    # Style min/max bars
    for partname in ["cbars", "cmins", "cmaxes"]:
        parts[partname].set_color("#666666")
        parts[partname].set_linewidth(0.8)

    # Overlay individual points if requested
    if show_points:
        for i, (size, gaps) in enumerate(zip(sizes, data)):
            jitter = np.random.normal(0, 0.05, len(gaps))
            ax.scatter(
                i + jitter,
                gaps,
                alpha=0.3,
                s=8,
                color=COLORS["primary"],
                edgecolor="none",
            )

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Instance Size (n)")
    ax.set_ylabel("Gap (%)")
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)

    # Add legend for mean/median
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=COLORS["secondary"], linewidth=1.5, label="Mean"),
        Line2D([0], [0], color=COLORS["tertiary"], linewidth=1.5, label="Median"),
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


# =============================================================================
# Time Analysis Plots
# =============================================================================


def plot_gap_and_time_violins(
    results_by_size: Dict[int, Dict[str, List[float]]],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot gap and time distributions side-by-side as violin plots.

    Shows gap (left, blue) and time (right, orange) for each instance size.

    Args:
        results_by_size: {size: {"gaps": [...], "times": [...] (ms internally)}}
        title: Plot title.
        save_path: Path to save figure (displays if None).
    """
    sizes = sorted(results_by_size.keys())
    gaps_data = [results_by_size[s]["gaps"] for s in sizes]
    # Convert ms to seconds for display
    times_data = [[t / 1000 for t in results_by_size[s]["times"]] for s in sizes]

    # Adjust figure width based on number of sizes
    fig_width = max(8, len(sizes) * 1.2 + 2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 8), sharex=True)

    # Gap violin (top)
    parts_gap = ax1.violinplot(
        gaps_data,
        positions=range(len(sizes)),
        showmeans=True,
        showmedians=True,
        widths=0.7,
    )
    for pc in parts_gap["bodies"]:
        pc.set_facecolor(COLORS["primary"])
        pc.set_alpha(0.4)
    parts_gap["cmeans"].set_color(COLORS["secondary"])
    parts_gap["cmedians"].set_color(COLORS["tertiary"])
    for partname in ["cbars", "cmins", "cmaxes"]:
        parts_gap[partname].set_color("#666666")

    ax1.set_ylabel("Gap (%)")
    ax1.axhline(y=0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_title("Gap Distribution" if not title else f"{title} - Gap")

    # Time violin (bottom)
    parts_time = ax2.violinplot(
        times_data,
        positions=range(len(sizes)),
        showmeans=True,
        showmedians=True,
        widths=0.7,
    )
    for pc in parts_time["bodies"]:
        pc.set_facecolor(COLORS["tertiary"])
        pc.set_alpha(0.4)
    parts_time["cmeans"].set_color(COLORS["secondary"])
    parts_time["cmedians"].set_color(COLORS["primary"])
    for partname in ["cbars", "cmins", "cmaxes"]:
        parts_time[partname].set_color("#666666")

    ax2.set_ylabel("Time (s)")
    ax2.set_xlabel("Instance Size (n)")
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.set_title("Time Distribution" if not title else f"{title} - Time")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_time_vs_gap_scatter(
    gaps: List[float],
    times: List[float],
    title: Optional[str] = None,
    highlight_suboptimal: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Scatter plot of execution time vs gap.

    Useful to analyze if longer runs tend to find worse solutions.

    Args:
        gaps: List of gap percentages.
        times: List of execution times (ms internally, displayed in seconds).
        title: Plot title.
        highlight_suboptimal: Highlight instances with gap > 0.
        save_path: Path to save figure (displays if None).
    """
    gaps_arr = np.array(gaps)
    # Convert ms to seconds for display
    times_arr = np.array(times) / 1000

    fig, ax = plt.subplots(figsize=(8, 5))

    if highlight_suboptimal:
        # Split into optimal (gap <= 0) and suboptimal (gap > 0)
        optimal_mask = gaps_arr <= 0
        suboptimal_mask = ~optimal_mask

        ax.scatter(
            times_arr[optimal_mask],
            gaps_arr[optimal_mask],
            alpha=0.5,
            s=30,
            c=COLORS["primary"],
            label=f"Optimal (n={optimal_mask.sum()})",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.scatter(
            times_arr[suboptimal_mask],
            gaps_arr[suboptimal_mask],
            alpha=0.7,
            s=40,
            c=COLORS["quaternary"],
            label=f"Suboptimal (n={suboptimal_mask.sum()})",
            edgecolor="white",
            linewidth=0.3,
        )

        # Stats for suboptimal only
        if suboptimal_mask.sum() > 0:
            mean_time_sub = times_arr[suboptimal_mask].mean()
            mean_gap_sub = gaps_arr[suboptimal_mask].mean()
            ax.axvline(
                x=mean_time_sub,
                color=COLORS["quaternary"],
                linestyle=":",
                alpha=0.7,
                label=f"Subopt. mean time: {mean_time_sub:.2f}s",
            )
    else:
        ax.scatter(
            times_arr,
            gaps_arr,
            alpha=0.5,
            s=30,
            c=COLORS["primary"],
            edgecolor="white",
            linewidth=0.3,
        )

    ax.set_xlabel("Execution Time (s)")
    ax.set_ylabel("Gap (%)")
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


def plot_suboptimal_time_analysis(
    results_by_size: Dict[int, Dict[str, List[float]]],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Analyze execution time specifically for suboptimal instances (gap > 0).

    Shows time distribution only for instances that didn't reach the optimum.

    Args:
        results_by_size: {size: {"gaps": [...], "times": [...] (ms internally)}}
        title: Plot title.
        save_path: Path to save figure (displays if None).
    """
    sizes = sorted(results_by_size.keys())

    # Filter to suboptimal instances only, convert ms to seconds
    subopt_times: Dict[int, List[float]] = {}
    subopt_counts: Dict[int, Tuple[int, int]] = {}  # (suboptimal, total)

    for size in sizes:
        gaps = np.array(results_by_size[size]["gaps"])
        times = np.array(results_by_size[size]["times"]) / 1000  # Convert to seconds
        mask = gaps > 0.001  # Small threshold for floating point
        subopt_times[size] = times[mask].tolist()
        subopt_counts[size] = (mask.sum(), len(gaps))

    # Only plot sizes with suboptimal instances
    sizes_with_subopt = [s for s in sizes if len(subopt_times[s]) > 0]

    if not sizes_with_subopt:
        print("No suboptimal instances found - skipping plot")
        return

    fig_width = max(6, len(sizes_with_subopt) * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    times_data = [subopt_times[s] for s in sizes_with_subopt]

    parts = ax.violinplot(
        times_data,
        positions=range(len(sizes_with_subopt)),
        showmeans=True,
        showmedians=True,
        widths=0.7,
    )

    for pc in parts["bodies"]:
        pc.set_facecolor(COLORS["quaternary"])
        pc.set_alpha(0.4)
    parts["cmeans"].set_color(COLORS["secondary"])
    parts["cmedians"].set_color(COLORS["tertiary"])
    for partname in ["cbars", "cmins", "cmaxes"]:
        parts[partname].set_color("#666666")

    # Add count annotations
    for i, size in enumerate(sizes_with_subopt):
        n_sub, n_total = subopt_counts[size]
        pct = 100 * n_sub / n_total if n_total > 0 else 0
        ax.annotate(
            f"{n_sub}/{n_total}\n({pct:.0f}%)",
            xy=(i, ax.get_ylim()[1]),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#666666",
        )

    ax.set_xticks(range(len(sizes_with_subopt)))
    ax.set_xticklabels([str(s) for s in sizes_with_subopt])
    ax.set_xlabel("Instance Size (n)")
    ax.set_ylabel("Execution Time (s)")

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Execution Time for Suboptimal Instances (gap > 0)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def generate_time_analysis_plots(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    types: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate time analysis plots from results CSV.

    Creates per-type plots: gap+time violins, time vs gap scatter, suboptimal analysis.

    Args:
        csv_path: Path to results CSV file.
        output_dir: Directory to save plots.
        types: Instance types to plot (default: all found in CSV).

    Returns:
        List of generated plot file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = load_results_from_csv(csv_path)

    if types is not None:
        results_data = {t: results_data[t] for t in types if t in results_data}

    generated_files: List[str] = []

    for instance_type, results_by_size in results_data.items():
        if not results_by_size:
            continue

        type_dir = output_dir / instance_type
        type_dir.mkdir(parents=True, exist_ok=True)

        # 1. Gap + Time violins
        gap_time_path = type_dir / f"{instance_type}_gap_time_violins.png"
        plot_gap_and_time_violins(
            results_by_size,
            title=f"Q-ILS Performance ({instance_type})",
            save_path=gap_time_path,
        )
        generated_files.append(str(gap_time_path))
        print(f"  Generated: {gap_time_path}")

        # 2. Time vs Gap scatter (aggregated across all sizes)
        all_gaps = []
        all_times = []
        for size_data in results_by_size.values():
            all_gaps.extend(size_data["gaps"])
            all_times.extend(size_data["times"])

        scatter_path = type_dir / f"{instance_type}_time_vs_gap.png"
        plot_time_vs_gap_scatter(
            all_gaps,
            all_times,
            title=f"Time vs Gap ({instance_type})",
            save_path=scatter_path,
        )
        generated_files.append(str(scatter_path))
        print(f"  Generated: {scatter_path}")

        # 3. Suboptimal time analysis
        subopt_path = type_dir / f"{instance_type}_suboptimal_time.png"
        plot_suboptimal_time_analysis(
            results_by_size,
            title=f"Suboptimal Instance Times ({instance_type})",
            save_path=subopt_path,
        )
        generated_files.append(str(subopt_path))
        print(f"  Generated: {subopt_path}")

    return generated_files
