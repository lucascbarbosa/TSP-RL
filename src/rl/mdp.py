"""Markov Decision Process model construction."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import torch

from src.rl.transition import (
    load_transition_file,
    load_transition_folder,
    load_transition_from_paths,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MDP:
    """
    Markov Decision Process model.

    Attributes:
        transition_matrix: P(s'|s,a) tensor of shape (n_states, n_actions, n_states).
        reward_matrix: R(s,a) tensor of shape (n_states, n_actions).
        n_states: Number of states.
        n_actions: Number of actions.
    """

    def __init__(
        self,
        transition_matrix: torch.Tensor,
        reward_matrix: torch.Tensor,
    ) -> None:
        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix
        self.n_states = transition_matrix.shape[0]
        self.n_actions = transition_matrix.shape[1]


def build_mdp_from_file(transition_filename: Union[str, Path]) -> MDP:
    """
    Build MDP from a single transition file.

    Args:
        transition_filename: Path to transition file.

    Returns:
        MDP instance.
    """
    (
        current_states,
        actions,
        rewards,
        next_states,
        n_states,
        n_actions,
    ) = load_transition_file(transition_filename, return_torch=True)

    return _build_mdp_from_data(current_states, actions, rewards, next_states, n_states, n_actions)


def build_mdp_from_folder(
    transition_folder: Union[str, Path],
    filter_pattern: str = r".*",
) -> MDP:
    """
    Build MDP from all transition files in a folder.

    Args:
        transition_folder: Directory containing transition files.
        filter_pattern: Regex to filter filenames.

    Returns:
        MDP instance.
    """
    (
        current_states,
        actions,
        rewards,
        next_states,
        n_states,
        n_actions,
    ) = load_transition_folder(transition_folder, return_torch=True, filter_pattern=filter_pattern)

    return _build_mdp_from_data(current_states, actions, rewards, next_states, n_states, n_actions)


def build_mdp_from_paths(
    paths: List[Union[str, Path]],
    min_n_states: int = 0,
    min_n_actions: int = 0,
) -> MDP:
    """
    Build MDP from specific transition file paths.

    Args:
        paths: List of paths to transition files.
        min_n_states: Minimum number of states (pads if needed).
        min_n_actions: Minimum number of actions (pads if needed).

    Returns:
        MDP instance.
    """
    (
        current_states,
        actions,
        rewards,
        next_states,
        n_states,
        n_actions,
    ) = load_transition_from_paths(paths, return_torch=True)

    n_states = max(n_states, min_n_states)
    n_actions = max(n_actions, min_n_actions)

    return _build_mdp_from_data(current_states, actions, rewards, next_states, n_states, n_actions)


def _build_mdp_from_data(
    current_states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    n_states: int,
    n_actions: int,
) -> MDP:
    """
    Build MDP model from transition data.

    Constructs transition probability matrix P(s'|s,a) and reward matrix R(s,a)
    from empirical transition samples.

    Args:
        current_states: Tensor of current states.
        actions: Tensor of actions.
        rewards: Tensor of rewards.
        next_states: Tensor of next states.
        n_states: Number of states.
        n_actions: Number of actions.

    Returns:
        MDP instance.
    """
    current_states = current_states.to(torch.int32)
    actions = actions.to(torch.int32)
    rewards = rewards.to(torch.float32)
    next_states = next_states.to(torch.int32)

    transition_matrix = torch.zeros(
        (n_states, n_actions, n_states),
        dtype=torch.float32,
        device=device,
    )
    reward_matrix = torch.zeros(
        (n_states, n_actions),
        dtype=torch.float32,
        device=device,
    )
    visit_count = torch.zeros(
        (n_states, n_actions),
        dtype=torch.float32,
        device=device,
    )

    # Populate matrices from samples
    for state, action, reward, next_state in zip(current_states, actions, rewards, next_states):
        transition_matrix[state, action, next_state] += 1
        reward_matrix[state, action] += reward
        visit_count[state, action] += 1

    # Normalize by visit count
    visits_mask = visit_count > 0
    visit_count_expanded = visit_count.unsqueeze(-1)
    visits_mask_expanded = visits_mask.unsqueeze(-1)

    transition_matrix = torch.where(
        visits_mask_expanded,
        transition_matrix / visit_count_expanded,
        torch.zeros_like(transition_matrix),
    )
    reward_matrix = torch.where(
        visits_mask,
        reward_matrix / visit_count,
        torch.zeros_like(reward_matrix),
    )

    return MDP(transition_matrix, reward_matrix)
