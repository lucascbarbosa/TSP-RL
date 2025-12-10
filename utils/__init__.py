"""Utility functions for TSP-RL."""

from utils.plot import (
    # Style
    COLORS,
    STATE_LABELS,
    ACTION_LABELS,
    setup_style,
    # Results loading
    load_gaps_from_csv,
    generate_gap_violin_plots,
    # Q-Learning plots
    plot_q_convergence,
    plot_q_heatmap,
    # Results analysis
    plot_gap_distribution,
    plot_gap_by_size,
    plot_gap_violins_by_size,
    plot_gap_comparison,
    # TSP visualization
    plot_tour,
)

__all__ = [
    "COLORS",
    "STATE_LABELS",
    "ACTION_LABELS",
    "setup_style",
    "load_gaps_from_csv",
    "generate_gap_violin_plots",
    "plot_q_convergence",
    "plot_q_heatmap",
    "plot_gap_distribution",
    "plot_gap_by_size",
    "plot_gap_violins_by_size",
    "plot_gap_comparison",
    "plot_tour",
]
