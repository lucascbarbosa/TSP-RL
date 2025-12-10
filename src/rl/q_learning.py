"""Q-Learning algorithm implementation using value iteration."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from src.rl.mdp import MDP, build_mdp_from_file, build_mdp_from_folder, build_mdp_from_paths
from src.rl.q_table import QTable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_q_table(
    transition_filename: Union[str, Path],
    gamma: float = 0.99,
    max_iter: int = 1000,
    tol: float = 1e-6,
    q_table: QTable | None = None,
) -> tuple[QTable, dict[str, list[float]]]:
    """
    Train Q-table from transition file using value iteration.

    Args:
        transition_filename: Path to transition file.
        gamma: Discount factor.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        q_table: Existing Q-table to continue training (optional).

    Returns:
        Tuple of (trained_q_table, training_history).
    """
    mdp = build_mdp_from_file(transition_filename)
    return train_q_table_from_mdp(mdp, gamma, max_iter, tol, q_table)


def train_q_table_from_folder(
    transition_folder: Union[str, Path],
    gamma: float = 0.99,
    max_iter: int = 1000,
    tol: float = 1e-6,
    q_table: QTable | None = None,
) -> tuple[QTable, dict[str, list[float]]]:
    """
    Train Q-table from all transition files in a folder.

    Args:
        transition_folder: Directory containing transition files.
        gamma: Discount factor.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        q_table: Existing Q-table to continue training (optional).

    Returns:
        Tuple of (trained_q_table, training_history).
    """
    mdp = build_mdp_from_folder(transition_folder)
    return train_q_table_from_mdp(mdp, gamma, max_iter, tol, q_table)


def train_q_table_from_paths(
    paths: list[Union[str, Path]],
    gamma: float = 0.99,
    max_iter: int = 1000,
    tol: float = 1e-6,
    q_table: QTable | None = None,
    min_n_states: int = 0,
    min_n_actions: int = 0,
) -> tuple[QTable, dict[str, list[float]]]:
    """
    Train Q-table from specific transition file paths.

    Args:
        paths: List of paths to transition files.
        gamma: Discount factor.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        q_table: Existing Q-table to continue training (optional).
        min_n_states: Minimum state space size.
        min_n_actions: Minimum action space size.

    Returns:
        Tuple of (trained_q_table, training_history).
    """
    mdp = build_mdp_from_paths(paths, min_n_states=min_n_states, min_n_actions=min_n_actions)
    return train_q_table_from_mdp(mdp, gamma, max_iter, tol, q_table)


def train_q_table_from_mdp(
    mdp: MDP,
    gamma: float = 0.99,
    max_iter: int = 1000,
    tol: float = 1e-6,
    q_table: QTable | None = None,
) -> tuple[QTable, dict[str, list[float]]]:
    """
    Train Q-table from MDP using vectorized value iteration.

    The update rule is:
        Q'(s, a) = R(s, a) + gamma * E[max_{a'} Q(s', a')]

    Args:
        mdp: MDP model.
        gamma: Discount factor.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        q_table: Existing Q-table to continue training (optional).

    Returns:
        Tuple of (trained_q_table, training_history).
    """
    n_states = mdp.n_states
    n_actions = mdp.n_actions

    P = mdp.transition_matrix  # (n_states, n_actions, n_states)
    R = mdp.reward_matrix  # (n_states, n_actions)

    history: dict[str, list[float]] = {"avg_q_value": []}

    if q_table is None:
        q_current = QTable(n_states=n_states, n_actions=n_actions)
    else:
        q_current = q_table.copy()

    for i in range(max_iter):
        q_new = QTable(n_states=n_states, n_actions=n_actions)

        # Vectorized computation:
        # Q'(s, a) = R(s,a) + gamma * E[max_{a'} Q(s', a')]
        max_q_values = q_current.table.max(dim=1)[0]
        expected_values = torch.einsum("san,n->sa", P, max_q_values)
        q_new.table = R + gamma * expected_values

        # Track convergence
        avg_q = q_current.table.mean().item()
        history["avg_q_value"].append(avg_q)

        # Check convergence
        if (q_current.table - q_new.table).abs().max() < tol:
            break

        q_current = q_new.copy()

        if i % 100 == 0:
            print(f"Iter [{i + 1}/{max_iter}]: Avg Q-value: {avg_q:.4f}")

    return q_current, history
