"""Transition file utilities for MDP construction."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TransitionData = Tuple[
    Union[NDArray[np.int32], torch.Tensor],  # current_states
    Union[NDArray[np.int32], torch.Tensor],  # actions
    Union[NDArray[np.float32], torch.Tensor],  # rewards
    Union[NDArray[np.int32], torch.Tensor],  # next_states
    int,  # n_states
    int,  # n_actions
]


def load_transition_file(
    transition_filename: Union[str, Path],
    return_torch: bool = False,
) -> TransitionData:
    """
    Load transition data from a single file.

    File format: Each line contains "state action reward next_state".

    Args:
        transition_filename: Path to transition file.
        return_torch: If True, return PyTorch tensors on GPU/CPU.

    Returns:
        Tuple of (current_states, actions, rewards, next_states, n_states, n_actions).
    """
    transition_matrix = np.loadtxt(transition_filename)

    current_states = transition_matrix[:, 0].astype(np.int32)
    actions = transition_matrix[:, 1].astype(np.int32)
    rewards = transition_matrix[:, 2].astype(np.float32)
    next_states = transition_matrix[:, 3].astype(np.int32)

    max_state = max(current_states.max(), next_states.max())
    n_states = int(max_state + 1)
    max_action = actions.max()
    n_actions = int(max_action + 1)

    if return_torch:
        current_states = torch.from_numpy(current_states).to(device)
        actions = torch.from_numpy(actions).to(device)
        rewards = torch.from_numpy(rewards).to(device)
        next_states = torch.from_numpy(next_states).to(device)

    return current_states, actions, rewards, next_states, n_states, n_actions


def load_transition_folder(
    transition_folder: Union[str, Path],
    return_torch: bool = False,
    filter_pattern: str = r".*",
) -> TransitionData:
    """
    Load and aggregate transition data from all .txt files in a folder.

    Args:
        transition_folder: Directory containing transition files.
        return_torch: If True, return PyTorch tensors.
        filter_pattern: Regex pattern to filter filenames.

    Returns:
        Aggregated transition data tuple.
    """
    matrix_list: List[NDArray] = []

    for filename in sorted(os.listdir(transition_folder)):
        if re.search(filter_pattern, filename) and filename.endswith(".txt"):
            file_path = os.path.join(transition_folder, filename)
            try:
                data = np.loadtxt(file_path)

                if len(data.shape) == 1:
                    data = data.reshape(1, -1)

                if data.shape[1] >= 4:
                    matrix_list.append(data)
                else:
                    print(f"Warning: Skipping {filename}, invalid shape {data.shape}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if not matrix_list:
        raise ValueError(f"No valid transition files found in {transition_folder}")

    return _process_transition_matrices(matrix_list, return_torch)


def load_transition_from_paths(
    paths: List[Union[str, Path]],
    return_torch: bool = False,
) -> TransitionData:
    """
    Load and aggregate transition data from specific file paths.

    Args:
        paths: List of paths to transition files.
        return_torch: If True, return PyTorch tensors.

    Returns:
        Aggregated transition data tuple.
    """
    matrix_list: List[NDArray] = []

    for file_path in sorted(paths):
        try:
            data = np.loadtxt(file_path)

            if len(data.shape) == 1:
                data = data.reshape(1, -1)

            if data.shape[1] >= 4:
                matrix_list.append(data)
            else:
                print(f"Warning: Skipping {file_path}, invalid shape {data.shape}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not matrix_list:
        raise ValueError("No valid transition files found in paths")

    return _process_transition_matrices(matrix_list, return_torch)


def _process_transition_matrices(
    matrix_list: List[NDArray],
    return_torch: bool,
) -> TransitionData:
    """
    Process and aggregate transition matrices.

    Args:
        matrix_list: List of raw transition matrices.
        return_torch: If True, convert to PyTorch tensors.

    Returns:
        Aggregated transition data tuple.
    """
    full_transition_matrix = np.concatenate(matrix_list, axis=0)

    current_states = full_transition_matrix[:, 0].astype(np.int32)
    actions = full_transition_matrix[:, 1].astype(np.int32)
    rewards = full_transition_matrix[:, 2].astype(np.float32)
    next_states = full_transition_matrix[:, 3].astype(np.int32)

    max_state = max(current_states.max(), next_states.max())
    n_states = int(max_state + 1)
    max_action = actions.max()
    n_actions = int(max_action + 1)

    if return_torch:
        current_states = torch.from_numpy(current_states).to(device)
        actions = torch.from_numpy(actions).to(device)
        rewards = torch.from_numpy(rewards).to(device)
        next_states = torch.from_numpy(next_states).to(device)

    return current_states, actions, rewards, next_states, n_states, n_actions
