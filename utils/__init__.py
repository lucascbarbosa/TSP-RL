"""Utility functions for TSP-RL."""

from utils.plot import (
    # Style
    COLORS,
    setup_style,
    # Results loading
    load_gaps_from_csv,
    load_results_from_csv,
    generate_gap_violin_plots,
    # Results analysis
    plot_gap_distribution,
    plot_gap_by_size,
    plot_gap_violins_by_size,
    plot_gap_comparison,
    # TSP visualization
    plot_tour,
    # Time analysis
    plot_gap_and_time_violins,
    plot_time_vs_gap_scatter,
    plot_suboptimal_time_analysis,
    generate_time_analysis_plots,
)

__all__ = [
    "COLORS",
    "setup_style",
    "load_gaps_from_csv",
    "load_results_from_csv",
    "generate_gap_violin_plots",
    "plot_gap_distribution",
    "plot_gap_by_size",
    "plot_gap_violins_by_size",
    "plot_gap_comparison",
    "plot_tour",
    "plot_gap_and_time_violins",
    "plot_time_vs_gap_scatter",
    "plot_suboptimal_time_analysis",
    "generate_time_analysis_plots",
]
