"""Plotting functions."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy # Crucial for modifying colormaps safely

def plot_heatmap(matrix, title:str="Q-table heatmap", x_labels:list=[], y_labels:list=[], cmap:str="viridis", save_path: Optional[str] = None):
    """
    Plots a heatmap where 0.0 values are rendered black, and the
    colormap stretches only across non-zero values.
    Assuming input matrix contains floats.
    """
    plt.figure(figsize=(10, 8))
    
    # --- STEP 1: Prepare Data ---
    # Create a copy to avoid modifying the original data outside the function
    matrix_to_plot = matrix.copy()
    
    # Replace exact zeros with NaN (Not a Number)
    # Matplotlib ignores NaNs when calculating the color scale range.
    matrix_to_plot[matrix_to_plot == 0.0] = np.nan
    
    # --- STEP 2: Prepare Custom Colormap ---
    # Fetch the desired matplotlib colormap object (e.g., "viridis")
    # We MUST use copy.copy(), otherwise, we permanently change the standard
    # 'viridis' map for the rest of the Python session.
    current_cmap = copy.copy(plt.get_cmap(cmap))
    
    # Set the color for NaN (bad) values to solid black
    current_cmap.set_bad("black")

    # --- STEP 3: Plot ---
    # Note: We use an fmt that handles floats nicely. 
    # If annot=True, NaNs usually show up as 'nan' text, which can be cluttered.
    # Sometimes it's better to turn annot=False if you have many zeros.
    sns.heatmap(matrix_to_plot, 
                annot=True,       # Set to False if the 'nan' text annoys you
                fmt=".1f",        # Format floats to 1 decimal place
                cmap=current_cmap, 
                cbar=True)

    # Add titles and labels
    plt.title(title, fontsize=16)
    plt.xlabel("Actions", fontsize=12)
    plt.ylabel("States", fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

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
