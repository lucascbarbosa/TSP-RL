"""Plotting utilities for TSP-RL visualization."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray


def plot_heatmap(
    matrix: NDArray[np.float64],
    title: str = "Q-table heatmap",
    x_labels: List[str] | None = None,
    y_labels: List[str] | None = None,
    cmap: str = "viridis",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot a heatmap with zeros rendered as black.

    Args:
        matrix: 2D array to visualize.
        title: Plot title.
        x_labels: Labels for x-axis (actions).
        y_labels: Labels for y-axis (states).
        cmap: Matplotlib colormap name.
        save_path: Path to save figure (displays if None).
    """
    plt.figure(figsize=(10, 8))

    # Replace zeros with NaN for proper colormap handling
    matrix_to_plot = matrix.copy()
    matrix_to_plot[matrix_to_plot == 0.0] = np.nan

    # Create colormap with black for NaN values
    current_cmap = copy.copy(plt.get_cmap(cmap))
    current_cmap.set_bad("black")

    sns.heatmap(
        matrix_to_plot,
        annot=True,
        fmt=".1f",
        cmap=current_cmap,
        cbar=True,
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Actions", fontsize=12)
    plt.ylabel("States", fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_history(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot training history with loss and Q-value.

    Args:
        history: Dictionary with 'loss' and 'avg_q_value' lists.
        save_path: Path to save figure (displays if None).
    """
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(epochs, history["loss"], "b-", label="Training Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # Plot average Q-value
    ax2.plot(epochs, history["avg_q_value"], "r-", label="Avg Q-Value", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Average Q-Value", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_single_q_learning(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot single Q-Learning training history.

    Args:
        history: Dictionary with 'avg_q_value' list.
        save_path: Path to save figure (displays if None).
    """
    iterations = range(1, len(history["avg_q_value"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, history["avg_q_value"], "b-", linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Average Q-Value", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_double_q_learning(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot double Q-Learning training history.

    Args:
        history: Dictionary with 'avg_q_value_q1' and 'avg_q_value_q2' lists.
        save_path: Path to save figure (displays if None).
    """
    iterations = range(1, len(history["avg_q_value_q1"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, history["avg_q_value_q1"], "b-", label="Avg Q-Value Q1", linewidth=2)
    plt.plot(iterations, history["avg_q_value_q2"], "r-", label="Avg Q-Value Q2", linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Average Q-Value", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim(left=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_rollout(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot rollout evaluation results.

    Args:
        history: Dictionary with 'rewards' and 'length' lists.
        save_path: Path to save figure (displays if None).
    """
    n_simulations = len(history["rewards"])
    simulations = range(1, n_simulations + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot rewards
    ax1.plot(simulations, history["rewards"], "b-", linewidth=2, alpha=0.7)
    ax1.set_xlabel("Simulation", fontsize=12)
    ax1.set_ylabel("Total Reward", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # Plot episode lengths
    ax2.plot(simulations, history["length"], "r-", linewidth=2, alpha=0.7)
    ax2.set_xlabel("Simulation", fontsize=12)
    ax2.set_ylabel("Episode Length", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_tour(
    coords: NDArray[np.float64],
    tour: List[int],
    title: str = "TSP Tour",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot a TSP tour on 2D coordinates.

    Args:
        coords: Array of (x, y) coordinates.
        tour: Tour as list of city indices (0-based).
        title: Plot title.
        save_path: Path to save figure (displays if None).
    """
    closed_tour = list(tour) + [tour[0]]
    xs = [coords[i][0] for i in closed_tour]
    ys = [coords[i][1] for i in closed_tour]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker="o")

    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=10)

    plt.title(title)
    plt.axis("equal")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()
