"""Q-Learning and Deep Q-Learning implementations for transition matrices."""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple
from table import QTable

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)


class TransitionDataset(Dataset):
    """PyTorch Dataset for transition data."""
    def __init__(
        self,
        current_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor
    ):
        """Initialize dataset."""
        self.current_states = current_states.to(device)
        self.actions = actions.to(device)
        self.rewards = rewards.to(device)
        self.next_states = next_states.to(device)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.current_states)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item at index."""
        current_state = self.current_states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_state = self.next_states[idx]
        return current_state, action, reward, next_state


class DQN(nn.Module):
    """Deep Q-Network (DQN) neural network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [128, 128],
        activation: str = 'relu'
    ):
        """Initialize DQN.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions (for discrete actions)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(DQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build network layers
        layers = []
        input_dim = state_dim

        act_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }.get(activation.lower(), nn.ReLU())

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act_fn)
            input_dim = hidden_dim

        # Output layer: Q-values for each action
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns Q-values for all actions."""
        return self.network(state)

    def to_q_table(self, epsilon: float = 0.1) -> QTable:
        """Convert DQN to QTable by extracting Q-values for all states.

        Args:
            epsilon: Exploration rate for the created QTable

        Returns:
            QTable object populated with Q-values from the neural network
        """
        # Create QTable with appropriate dimensions
        q_table = QTable(
            n_states=self.state_dim,
            n_actions=self.action_dim,
            epsilon=epsilon
        )

        # Set model to evaluation mode
        self.eval()

        # Generate Q-values for all states
        with torch.no_grad():
            for state in range(self.state_dim):
                # One-hot encode the state
                state_one_hot = torch.nn.functional.one_hot(
                    torch.tensor(state), num_classes=self.state_dim
                ).float().unsqueeze(0).to(device)

                # Get Q-values from the network
                q_values = self.forward(state_one_hot)

                # Convert to numpy and populate QTable
                q_table.table[state, :] = (
                    q_values.cpu().numpy().flatten().astype(np.float32)
                )

        # Set model back to training mode
        self.train()

        return q_table


def plot_history(history: Dict[str, list], save_path: str = None) -> None:
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
    ax1.legend()
    ax1.set_xlim(left=0)

    # Plot average Q-value
    ax2.plot(
        epochs, history['avg_q_value'], 'r-', label='Avg Q-Value', linewidth=2
    )
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Average Q-Value', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(left=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def deep_q_learning(
    transition_filename: str,
    hidden_dims: list = [64, 64],
    learning_rate: float = 0.001,
    gamma: float = 0.9,
    batch_size: int = 32,
    num_epochs: int = 10,
) -> Tuple[DQN, Dict[str, list]]:
    """Train Deep Q-Network (DQN) from transition matrix."""
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

    # Apply one-hot encoding to states (convert to float32 for neural network)
    current_states = torch.nn.functional.one_hot(
        torch.tensor(current_states), num_classes=n_states
    ).float()
    next_states = torch.nn.functional.one_hot(
        torch.tensor(next_states), num_classes=n_states
    ).float()

    # Keep actions as integer indices (not one-hot encoded, needed for gather)
    actions = torch.tensor(actions, dtype=torch.long)

    # Convert rewards to float tensor
    rewards = torch.from_numpy(rewards).float()

    # Create dataset and dataloader
    dataset = TransitionDataset(
        current_states, actions, rewards, next_states,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Initialize DQN
    model = DQN(
        state_dim=n_states,
        action_dim=n_actions,
        hidden_dims=hidden_dims
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training history
    history = {'loss': [], 'avg_q_value': []}

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_q_values = []

        for (
            batch_current_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
        ) in dataloader:
            # Current Q-values using current states
            current_q_values = model(batch_current_states)
            current_q = current_q_values.gather(
                1, batch_actions.unsqueeze(1)
            ).squeeze(1)

            # Next state Q-values (detached for stability)
            with torch.no_grad():
                next_q_values = model(batch_next_states)
                max_next_q = next_q_values.max(1)[0]
                target_q = batch_rewards + gamma * max_next_q

            # Compute loss
            loss = criterion(current_q, target_q)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_q_values.append(current_q.mean().item())

        avg_loss = np.mean(epoch_losses)
        avg_q = np.mean(epoch_q_values)
        history['loss'].append(avg_loss)
        history['avg_q_value'].append(avg_q)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Loss={avg_loss:.6f}, Avg Q={avg_q:.6f}"
        )

    return model, history


if __name__ == "__main__":
    for i in range(10):
        filename = f"data/transitions/instance_{i + 1:02d}.txt"
        model, history = deep_q_learning(filename)
        plot_path = f"data/plots/neural/instance_{i + 1:02d}.png"
        plot_history(history, plot_path)
        q_table = model.to_q_table()
        q_table.to_txt(f"data/q_tables/neural/instance_{i + 1:02d}.txt")
