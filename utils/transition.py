"""Transition file utilities."""
import numpy as np
import torch
from typing import Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
