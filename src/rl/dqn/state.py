"""DQN state representation and utilities."""

from dataclasses import dataclass, field

import numpy as np


def normalize_gap(gap: float) -> float:
    """
    Normalize gap percentage to [-1, 1] scale using symmetric log transform.

    Log transform compresses large gaps while preserving resolution
    for small gaps (which matter more for optimization).

    Supports negative gaps (when solution beats the reference, e.g., baseline).

    Args:
        gap: Gap percentage (e.g., 5.0 for 5%, -3.0 for -3%).

    Returns:
        Normalized value in [-1, 1]. gap=-100% → -1.0, gap=0% → 0.0, gap=100% → 1.0.

    Examples:
        >>> normalize_gap(0.0)
        0.0
        >>> 0.38 < normalize_gap(5.0) < 0.40  # ~0.39
        True
        >>> 0.99 < normalize_gap(100.0) < 1.01  # ~1.0
        True
        >>> -0.40 < normalize_gap(-5.0) < -0.38  # ~-0.39
        True
    """
    # Symmetric log transform: handles both positive and negative gaps
    # log1p(x) = log(1 + x), numerically stable for small x
    # Scale: gap=-100% → -1.0, gap=0% → 0.0, gap=100% → 1.0
    if gap >= 0:
        return float(np.log1p(gap) / np.log1p(100))
    else:
        # Mirror transformation for negative gaps
        return float(-np.log1p(-gap) / np.log1p(100))


def compute_delta_reward(g_best_old: float, g_best_new: float) -> float:
    """
    Compute reward as improvement in best gap.

    Rewards the agent only when it improves the best solution found.
    This aligns the reward signal with the actual optimization objective.

    Args:
        g_best_old: Previous best gap (percentage).
        g_best_new: New best gap (percentage).

    Returns:
        Reward = g_best_old - g_best_new (always >= 0 since g_best only decreases).

    Examples:
        >>> compute_delta_reward(5.0, 3.0)  # Improved from 5% to 3%
        2.0
        >>> compute_delta_reward(3.0, 3.0)  # No improvement
        0.0
        >>> compute_delta_reward(3.0, 5.0)  # g_best can't increase (would be 0)
        -2.0
    """
    return g_best_old - g_best_new


def compute_sparse_reward(best_gap: float) -> float:
    """
    Compute end-of-episode reward based on final best gap.

    Returns negative normalized gap (lower gap = higher reward).

    Args:
        best_gap: Best gap achieved during the episode (percentage).

    Returns:
        Reward in [-1, 1]. gap=-100% → +1.0, gap=0% → 0.0, gap=100% → -1.0.

    Examples:
        >>> compute_sparse_reward(0.0)  # Matched baseline
        0.0
        >>> compute_sparse_reward(-5.0)  # Beat baseline by 5%
        0.39...
        >>> compute_sparse_reward(5.0)  # 5% worse than baseline
        -0.39...
    """
    return -normalize_gap(best_gap)


@dataclass
class DQNState:
    """
    Continuous state representation for DQN.

    Attributes:
        g: Current gap (normalized).
        g_best: Best gap found in episode (normalized).
        t_ratio: Remaining time ratio (t_remaining / T) in [0, 1].
        history: Tuple of last R action indices, or -1 for empty slots.
        n_actions: Number of available actions (for one-hot encoding).
    """

    g: float
    g_best: float
    t_ratio: float
    history: tuple[int, ...] = field(default_factory=lambda: (-1, -1))
    n_actions: int = 21  # Default to current action space size

    def to_numpy(self) -> np.ndarray:
        """
        Convert state to numpy array for network input.

        Returns:
            1D array of shape (3 + R*n_actions,) with:
            - [0:3]: g, g_best, t_ratio
            - [3:]: one-hot encoded history (R actions × n_actions options)
        """
        base = [self.g, self.g_best, self.t_ratio]
        history_onehot = []
        for action_idx in self.history:
            oh = [0.0] * self.n_actions
            if 0 <= action_idx < self.n_actions:
                oh[action_idx] = 1.0
            history_onehot.extend(oh)
        return np.array(base + history_onehot, dtype=np.float32)

    @staticmethod
    def dim(history_len: int = 1, n_actions: int = 16) -> int:
        """
        Get state vector dimension.

        Args:
            history_len: Number of past actions to track.
            n_actions: Number of available actions.

        Returns:
            Total dimension: 3 + history_len * n_actions.
        """
        return 3 + history_len * n_actions
