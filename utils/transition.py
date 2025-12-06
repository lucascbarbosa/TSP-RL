"""Transition file utilities."""
import os
import numpy as np
import torch
from typing import Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_transition_folder(
    transition_folder: str,
    return_torch: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Load all transition files from a folder and aggregate them into a single MDP.
    Assumes files are readable by np.loadtxt (e.g., .txt files).
    """
    matrix_list = []

    # 1. Iterate over files and load raw data
    # Sort ensures deterministic order, though not strictly necessary for MDPs
    for filename in sorted(os.listdir(transition_folder)):
        if filename.endswith(".txt"): 
            file_path = os.path.join(transition_folder, filename)
            try:
                # Load the raw matrix
                data = np.loadtxt(file_path)
                
                # Handle case where file might be a single row (1D array)
                if len(data.shape) == 1:
                    data = data.reshape(1, -1)
                
                # Verify shape is correct (needs at least 4 columns: S, A, R, S')
                if data.shape[1] >= 4:
                    matrix_list.append(data)
                else:
                    print(f"Warning: Skipping {filename}, invalid shape {data.shape}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if not matrix_list:
        raise ValueError(f"No valid transition files found in {transition_folder}")

    # 2. Concatenate all matrices along the row axis (axis 0)
    # This creates one giant transition matrix [Total_Samples, Columns]
    full_transition_matrix = np.concatenate(matrix_list, axis=0)

    # 3. Extract columns (Identical logic to single-file version)
    current_states = full_transition_matrix[:, 0].astype(int)
    actions = full_transition_matrix[:, 1].astype(int)
    rewards = full_transition_matrix[:, 2].astype(float)
    next_states = full_transition_matrix[:, 3].astype(int)

    # 4. Determine global number of states and actions
    max_state = max(current_states.max(), next_states.max())
    n_states = max_state + 1
    max_action = actions.max()
    n_actions = max_action + 1

    # 5. Tensor Conversion
    if return_torch:
        current_states = torch.from_numpy(current_states).to(device)
        actions = torch.from_numpy(actions).to(device)
        rewards = torch.from_numpy(rewards).to(device)
        next_states = torch.from_numpy(next_states).to(device)

    return current_states, actions, rewards, next_states, int(n_states), int(n_actions)

def load_transition_file(
    transition_filename: str,
    return_torch: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Load transition file and extract data."""
    # Read transition matrix
    transition_matrix = np.loadtxt(transition_filename)

    # Extract columns
    current_states = transition_matrix[:, 0].astype(int)
    actions = transition_matrix[:, 1].astype(int)
    rewards = transition_matrix[:, 2].astype(float)
    next_states = transition_matrix[:, 3].astype(int)

    # Determine number of states and actions
    max_state = max(current_states.max(), next_states.max())
    n_states = max_state + 1
    max_action = actions.max()
    n_actions = max_action + 1

    if return_torch:
        # Convert to PyTorch tensors and move to device
        current_states = torch.from_numpy(current_states).to(device)
        actions = torch.from_numpy(actions).to(device)
        rewards = torch.from_numpy(rewards).to(device)
        next_states = torch.from_numpy(next_states).to(device)

    return current_states, actions, rewards, next_states, n_states, n_actions
