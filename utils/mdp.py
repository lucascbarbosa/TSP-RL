"""Markov Decision Process implementation."""
import torch
import math
from utils.transition import load_transition_file, load_transition_folder, load_transition_from_paths

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MDP:
    """Markov Decision Process class."""
    def __init__(
        self,
        transition_matrix: torch.Tensor,
        reward_matrix: torch.Tensor,
    ):
        """Initialize MDP."""
        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix
        self.n_states = transition_matrix.shape[0]
        self.n_actions = transition_matrix.shape[1]


def build_mdp_model(transition_filename: str):
    """Build MDP model from transitions."""
    # Load transition data
    (
        current_states,
        actions,
        rewards,
        next_states,
        n_states,
        n_actions,
    ) = load_transition_file(transition_filename, return_torch=True)

    return build_mdp_model_from_data(current_states,
        actions,
        rewards,
        next_states,
        n_states,
        n_actions)


def build_mdp_model_from_paths(paths: list, min_n_states:int = 0, min_n_actions:int = 0):
    """Build MDP model from transitions."""
    # Load transition data
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

    return build_mdp_model_from_data(current_states,
        actions,
        rewards,
        next_states,
        n_states,
        n_actions)

def build_mdp_model_from_folder(transition_folder: str, filter=r".*"):
    """Build MDP model from transitions."""
    # Load transition data
    (
        current_states,
        actions,
        rewards,
        next_states,
        n_states,
        n_actions,
    ) = load_transition_folder(transition_folder, return_torch=True, filter=filter)

    return build_mdp_model_from_data(current_states,
        actions,
        rewards,
        next_states,
        n_states,
        n_actions)

def build_mdp_model_from_data(current_states,
        actions,
        rewards,
        next_states,
        n_states,
        n_actions):
    """Build MDP model from transitions."""

    # Convert to appropriate dtypes
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

    # Populate matrices
    for state, action, reward, next_state in zip(
        current_states, actions, rewards, next_states
    ):
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
        torch.zeros_like(transition_matrix)
    )
    reward_matrix = torch.where(
        visits_mask,
        reward_matrix / visit_count,
        torch.zeros_like(reward_matrix)
    )
    return MDP(transition_matrix, reward_matrix)


if __name__ == "__main__":
    for i in range(1,11):
        mdp = build_mdp_model_from_folder("data/train/EUC_2D", filter=f"nodes_{i}0")
        print(mdp.transition_matrix, mdp.reward_matrix)
    """
    for i in range(10):
        filename = f"data/transitions/instance_{i + 1:02d}.txt"
        mdp = build_mdp_model(filename)
        print(mdp.transition_matrix, mdp.reward_matrix)
    """
