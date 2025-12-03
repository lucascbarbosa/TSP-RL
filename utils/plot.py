"""Plotting functions."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List


def plot_history(
    history: Dict[str, list],
    save_path: Optional[str] = None
) -> None:
    """Plot training history.

    Args:
        history: Dictionary containing 'loss' and 'avg_q_value' lists
        save_path: Optional path to save the plot. If None, display the plot.
    """
    epochs = range(1, len(history['loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # Plot average Q-value
    ax2.plot(
        epochs, history['avg_q_value'], 'r-', label='Avg Q-Value', linewidth=2
    )
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Average Q-Value', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_single_q_learning(
    history: Dict[str, list],
    save_path: Optional[str] = None
) -> None:
    """Plot single Q-Learning training history.

    Args:
        history: Dictionary containing 'avg_q_value' list
        save_path: Optional path to save the plot. If None, display the plot.
    """
    iterations = range(1, len(history['avg_q_value']) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(
        iterations, history['avg_q_value'], 'b-',
        linewidth=2
    )
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Average Q-Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_double_q_learning(
    history: Dict[str, list],
    save_path: Optional[str] = None
) -> None:
    """Plot double Q-Learning training history.

    Args:
        history: Dictionary containing 'avg_q_value_q1' and
            'avg_q_value_q2' lists
        save_path: Optional path to save the plot. If None, display the plot.
    """
    iterations = range(1, len(history['avg_q_value_q1']) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(
        iterations, history['avg_q_value_q1'], 'b-',
        label='Avg Q-Value Q1', linewidth=2
    )
    plt.plot(
        iterations, history['avg_q_value_q2'], 'r-',
        label='Avg Q-Value Q2', linewidth=2
    )
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Average Q-Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim(left=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_rollout(
    history: Dict[str, list],
    save_path: Optional[str] = None
) -> None:
    """Plot rollout evaluation results.

    Args:
        history: Dictionary containing 'rewards' and 'length' lists
        save_path: Optional path to save the plot. If None, display the plot.
    """
    n_simulations = len(history['rewards'])
    simulations = range(1, n_simulations + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot rewards
    ax1.plot(
        simulations, history['rewards'], 'b-',
        linewidth=2, alpha=0.7
    )
    ax1.set_xlabel('Simulation', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # Plot episode lengths
    ax2.plot(
        simulations, history['length'], 'r-',
        linewidth=2, alpha=0.7
    )
    ax2.set_xlabel('Simulation', fontsize=12)
    ax2.set_ylabel('Episode Length', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_tour(
    coords: np.ndarray,
    tour: List[int],
    title: str = "TSP Tour",
    save_path: Optional[str] = None
) -> None:
    """Plots a TSP tour."""
    tour = np.array(tour + [tour[0]])
    xs = [coords[i][0] for i in tour]
    ys = [coords[i][1] for i in tour]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker='o')
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=10)
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
