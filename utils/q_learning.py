"""Q-Learning and Deep Q-Learning implementations for transition dataframes."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, Any
TORCH_AVAILABLE = True


class QTable:
    """Q-table for Q-Learning with generic state and action spaces.

    Supports both discrete states (as hashable types) and continuous states
    (by discretization or using state representations as keys).
    """
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        default_q_value: float = 0.0
    ):
        """Initialize Q-table.

        Args:
            alpha: Learning rate (0 < alpha <= 1)
            gamma: Discount factor (0 <= gamma <= 1)
            epsilon: Exploration rate for epsilon-greedy policy
            default_q_value: Default Q-value for unseen state-action pairs
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.default_q_value = default_q_value
        self.q_table: Dict[Tuple[Any, Any], float] = defaultdict(
            lambda: default_q_value
        )
        self.visit_count: Dict[Tuple[Any, Any], int] = defaultdict(int)

    def get_q_value(self, state: Any, action: Any) -> float:
        """Get Q-value for a state-action pair."""
        key = (state, action)
        return self.q_table[key]

    def set_q_value(self, state: Any, action: Any, value: float) -> None:
        """Set Q-value for a state-action pair."""
        key = (state, action)
        self.q_table[key] = value
        self.visit_count[key] += 1

    def update(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        next_actions: Optional[list] = None
    ) -> None:
        """Update Q-value using Q-learning update rule.

        Q(s,a) = Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state reached
            next_actions: Optional list of available actions in next_state.
                         If None, will use all actions seen for next_state.
        """
        current_q = self.get_q_value(state, action)

        # Find max Q-value for next state
        if next_actions is None:
            # Find all actions that have been tried for next_state
            next_actions = set()
            for (s, a), _ in self.q_table.items():
                if s == next_state:
                    next_actions.add(a)

        if next_actions:
            max_next_q = max(
                self.get_q_value(next_state, a) for a in next_actions
            )
        else:
            # Terminal state or no known actions
            max_next_q = 0.0

        # Q-learning update
        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )
        self.set_q_value(state, action, new_q)

    def choose_action(
        self,
        state: Any,
        available_actions: list,
        use_epsilon_greedy: bool = True
    ) -> Any:
        """Choose an action using epsilon-greedy policy.

        Args:
            state: Current state
            available_actions: List of available actions
            use_epsilon_greedy: If True, use epsilon-greedy; else use greedy

        Returns:
            Selected action
        """
        if not available_actions:
            raise ValueError("No available actions provided")

        if use_epsilon_greedy and np.random.random() < self.epsilon:
            return np.random.choice(available_actions)

        # Greedy action selection
        best_action = None
        best_q = float('-inf')

        for action in available_actions:
            q_val = self.get_q_value(state, action)
            if q_val > best_q:
                best_q = q_val
                best_action = action

        # If all actions have same Q-value, choose randomly
        if best_q == self.default_q_value:
            return np.random.choice(available_actions)

        return best_action

    def get_policy(
        self, state: Any, available_actions: list
    ) -> Dict[Any, float]:
        """Get action probabilities for a state (epsilon-greedy policy).

        Returns:
            Dictionary mapping actions to probabilities
        """
        n_actions = len(available_actions)
        if n_actions == 0:
            return {}

        # Get Q-values for all actions
        q_values = {
            action: self.get_q_value(state, action)
            for action in available_actions
        }
        max_q = max(q_values.values())
        best_actions = [
            a for a, q in q_values.items() if q == max_q
        ]

        # Epsilon-greedy probabilities
        prob_greedy = (1 - self.epsilon) / len(best_actions)
        prob_explore = self.epsilon / n_actions

        policy = {}
        for action in available_actions:
            if action in best_actions:
                policy[action] = prob_greedy + prob_explore
            else:
                policy[action] = prob_explore

        return policy

    def decay_epsilon(
        self, factor: float = 0.995, min_epsilon: float = 0.01
    ) -> None:
        """Decay exploration rate."""
        self.epsilon = max(min_epsilon, self.epsilon * factor)

    def get_table_size(self) -> int:
        """Get number of state-action pairs in the table."""
        return len(self.q_table)

    def to_dataframe(self, file_path: str) -> pd.DataFrame:
        """Convert Q-table to pandas DataFrame."""
        data = []
        for (state, action), q_value in self.q_table.items():
            data.append({
                'state': state,
                'action': action,
                'q_value': q_value,
                'visits': self.visit_count[(state, action)]
            })
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        return df

    def reset(self) -> None:
        """Reset the Q-table."""
        self.q_table.clear()
        self.visit_count.clear()


class TransitionDataset(Dataset):
    """PyTorch Dataset for transition data."""
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        state_encoder: Optional[Any] = None,
        action_encoder: Optional[Any] = None
    ):
        """Initialize dataset.

        Args:
            states: Array of states
            actions: Array of actions
            rewards: Array of rewards
            next_states: Array of next states
            state_encoder: Optional function to encode states to numeric vecs
            action_encoder: Optional function to encode actions to numeric vecs
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states

        # Simple encoders if not provided
        if state_encoder is None:
            self.state_encoder = self._default_encoder
        else:
            self.state_encoder = state_encoder

        if action_encoder is None:
            self.action_encoder = self._default_encoder
        else:
            self.action_encoder = action_encoder

    def _default_encoder(self, x):
        """Default encoder: convert to numpy array if not already."""
        if isinstance(x, (list, tuple)):
            return np.array(x, dtype=np.float32)
        elif isinstance(x, (int, float)):
            return np.array([x], dtype=np.float32)
        else:
            return np.array(x, dtype=np.float32)

    def __len__(self):
        """Return dataset size."""
        return len(self.states)

    def __getitem__(self, idx):
        """Get item at index."""
        state = torch.FloatTensor(self.state_encoder(self.states[idx]))
        action_idx = self.actions[idx]
        if isinstance(action_idx, (int, np.integer)):
            action = torch.LongTensor([action_idx])[0]
        else:
            action = torch.FloatTensor(
                self.action_encoder(action_idx)
            )
        reward = torch.FloatTensor([self.rewards[idx]])[0]
        next_state = torch.FloatTensor(
            self.state_encoder(self.next_states[idx])
        )

        return state, action, reward, next_state


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


def q_learning(
    transition_df: pd.DataFrame,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    num_iterations: int = 1,
    verbose: bool = True
) -> QTable:
    """Train Q-table from transition dataframe using Q-Learning.

    Args:
        transition_df: DataFrame with columns: current_state, action,
                       reward, next_state
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        num_iterations: Number of times to iterate over the dataframe
        verbose: Whether to print progress

    Returns:
        Trained QTable object
    """
    # Validate dataframe columns
    required_cols = ['current_state', 'action', 'reward', 'next_state']
    missing_cols = [
        col for col in required_cols
        if col not in transition_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"DataFrame missing required columns: {missing_cols}. "
            f"Found columns: {list(transition_df.columns)}"
        )

    # Initialize Q-table
    q_table = QTable(alpha=alpha, gamma=gamma, epsilon=epsilon)

    # Get unique states and actions for reference
    all_states = set(transition_df['current_state'].unique())
    all_states.update(transition_df['next_state'].unique())
    all_actions = set(transition_df['action'].unique())

    if verbose:
        print(f"Training Q-Learning on {len(transition_df)} transitions")
        msg = f"Unique states: {len(all_states)}, "
        msg += f"Unique actions: {len(all_actions)}"
        print(msg)

    # Train for multiple iterations
    for iteration in range(num_iterations):
        if verbose and num_iterations > 1:
            print(f"Iteration {iteration + 1}/{num_iterations}")

        # Shuffle dataframe for each iteration
        df_shuffled = transition_df.sample(frac=1.0).reset_index(drop=True)

        for _, row in df_shuffled.iterrows():
            state = row['current_state']
            action = row['action']
            reward = row['reward']
            next_state = row['next_state']

            # Get available actions for next state (all actions in data)
            next_actions = list(all_actions)

            # Update Q-table
            q_table.update(state, action, reward, next_state, next_actions)

        if verbose:
            size = q_table.get_table_size()
            print(f"  Q-table size: {size} state-action pairs")

    return q_table


def deep_q_learning(
    transition_df: pd.DataFrame,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    hidden_dims: list = [128, 128],
    learning_rate: float = 0.001,
    gamma: float = 0.9,
    batch_size: int = 32,
    num_epochs: int = 10,
    state_encoder: Optional[callable] = None,
    action_encoder: Optional[callable] = None,
    verbose: bool = True
) -> Tuple[DQN, Dict[str, list]]:
    """Train Deep Q-Network (DQN) from transition dataframe.

    Args:
        transition_df: DataFrame with columns: current_state, action,
                       reward, next_state
        state_dim: Dimension of state space (auto-detected if None)
        action_dim: Number of actions for discrete action space
                    (auto-detected if None)
        hidden_dims: List of hidden layer dimensions
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        state_encoder: Optional function to encode states to numeric vecs
        action_encoder: Optional function to encode actions to numeric vecs
        verbose: Whether to print progress

    Returns:
        Tuple of (trained DQN model, training history dictionary)
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for Deep Q-Learning. "
            "Install with: pip install torch"
        )

    # Validate dataframe columns
    required_cols = ['current_state', 'action', 'reward', 'next_state']
    missing_cols = [
        col for col in required_cols
        if col not in transition_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"DataFrame missing required columns: {missing_cols}. "
            f"Found columns: {list(transition_df.columns)}"
        )

    # Prepare data
    states = transition_df['current_state'].to_numpy()
    actions = transition_df['action'].to_numpy()
    rewards = transition_df['reward'].to_numpy()
    next_states = transition_df['next_state'].to_numpy()

    # Auto-detect dimensions if not provided
    if state_encoder is None:
        # Try to infer state dimension from first state
        sample_state = states[0]
        if isinstance(sample_state, (list, tuple, np.ndarray)):
            state_dim = len(sample_state)
        else:
            state_dim = 1

        def state_encoder(x):
            """Encode state to numpy array."""
            if not isinstance(x, (list, tuple, np.ndarray)):
                return np.array([x], dtype=np.float32)
            return np.array(x, dtype=np.float32)
    else:
        # Use provided encoder to determine dimension
        sample_encoded = state_encoder(states[0])
        if hasattr(sample_encoded, '__len__'):
            state_dim = len(sample_encoded)
        else:
            state_dim = 1

    if action_dim is None:
        # Assume discrete action space
        unique_actions = np.unique(actions)
        action_dim = len(unique_actions)
        # Create action mapping
        action_to_idx = {
            action: idx for idx, action in enumerate(unique_actions)
        }
        actions = np.array([action_to_idx[a] for a in actions])
    else:
        # Assume actions are already indices
        actions = np.array(actions, dtype=np.int64)

    if verbose:
        print(f"Training Deep Q-Learning on {len(transition_df)} transitions")
        msg = f"State dimension: {state_dim}, "
        msg += f"Action dimension: {action_dim}"
        print(msg)
        arch = [state_dim] + hidden_dims + [action_dim]
        print(f"Network architecture: {arch}")

    # Create dataset and dataloader
    dataset = TransitionDataset(
        states, actions, rewards, next_states,
        state_encoder=state_encoder,
        action_encoder=None  # Actions are already indices
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Initialize DQN
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    model = DQN(state_dim, action_dim, hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training history
    history = {'loss': [], 'avg_q_value': []}

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_q_values = []

        for batch in dataloader:
            (batch_states, batch_actions, batch_rewards,
             batch_next_states) = batch
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_rewards = batch_rewards.to(device)
            batch_next_states = batch_next_states.to(device)

            # Current Q-values
            current_q_values = model(batch_states)
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

        if verbose:
            msg = f"Epoch {epoch + 1}/{num_epochs}: "
            msg += f"Loss={avg_loss:.6f}, Avg Q={avg_q:.6f}"
            print(msg)

    return model, history


if __name__ == "__main__":
    # Load transition dataframe
    transition_df = pd.read_csv('data/transitions/teste.csv')

    # Train Q-table
    q_table = q_learning(transition_df)
    print(q_table)

    # Save Q-table to CSV
    q_table.to_dataframe('data/q_tables/teste.csv')
