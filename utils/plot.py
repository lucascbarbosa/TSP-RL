"""Plotting functions."""
import matplotlib.pyplot as plt
from typing import Dict, Optional


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
