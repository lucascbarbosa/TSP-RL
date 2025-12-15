"""DQN training and evaluation functions."""

import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.rl.dqn.buffer import ReplayBuffer
from src.rl.dqn.env import DQNEnv, N_ACTIONS
from src.rl.dqn.network import QNetwork
from src.tsp.constructive import grasp
from src.tsp.instance import TSPInstance, TSPDataset
from src.tsp.local_search import two_opt_full
from src.tsp.solution import Solution


# =============================================================================
# Baseline Computation
# =============================================================================

# GRASP alphas for baseline (same pool as DQN-ILS agent uses)
_BASELINE_GRASP_ALPHAS = [0.03, 0.1, 0.3]


def _compute_baseline(instance: TSPInstance, time_budget: float) -> float:
    """
    Compute baseline cost for an instance using GRASP+2opt.

    Runs multiple iterations within time budget, returns best cost found.
    Uses same GRASP alphas available to DQN-ILS agent for fairness.

    Args:
        instance: TSP instance.
        time_budget: Time budget in seconds.

    Returns:
        Best tour cost found.
    """
    t0 = time.perf_counter()
    best_cost = float("inf")

    while time.perf_counter() - t0 < time_budget:
        alpha = random.choice(_BASELINE_GRASP_ALPHAS)
        tour, _ = grasp(instance, alpha=alpha)
        sol = Solution(tour, instance.dist_matrix, is_closed=True)
        improved = two_opt_full(sol)
        if improved.cost < best_cost:
            best_cost = improved.cost

    return best_cost


def _baseline_worker(args: tuple) -> tuple[int, float]:
    """Worker function for parallel baseline computation."""
    instance_id, dataset_path, time_budget = args
    dataset = TSPDataset(dataset_path, [instance_id], verbose=False)
    instance = next(iter(dataset))
    baseline_cost = _compute_baseline(instance, time_budget)
    return instance_id, baseline_cost


def compute_baselines_parallel(
    dataset_path: str,
    instance_ids: list[int],
    time_budget: float,
    n_workers: int,
    verbose: bool = True,
) -> dict[int, float]:
    """
    Compute baselines for all instances in parallel.

    Args:
        dataset_path: Path to dataset JSON.
        instance_ids: List of instance IDs.
        time_budget: Time budget per instance.
        n_workers: Number of parallel workers.
        verbose: Print progress.

    Returns:
        Dict mapping instance_id to baseline_cost.
    """
    if verbose:
        print(f"  Computing baselines for {len(instance_ids)} instances ({n_workers} workers)...")

    worker_args = [(inst_id, dataset_path, time_budget) for inst_id in instance_ids]
    baselines = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_baseline_worker, args): args[0] for args in worker_args}
        for future in as_completed(futures):
            inst_id, baseline_cost = future.result()
            baselines[inst_id] = baseline_cost

    if verbose:
        print(f"  Baselines computed for {len(baselines)} instances")

    return baselines


def get_default_workers() -> int:
    """Get default number of workers (n_cpus - 2, minimum 1)."""
    n_cpus = os.cpu_count() or 1
    return max(1, n_cpus - 2)


@dataclass
class DQNConfig:
    """Configuration for DQN training."""

    # Environment
    time_budget: float = 10.0  # Base time budget (scales with n)
    history_len: int = 1  # Number of past actions in state

    # DQN hyperparameters
    gamma: float = 0.99  # Discount factor
    lr: float = 0.001  # Learning rate
    batch_size: int = 64
    buffer_size: int = 50000
    target_update_freq: int = 50  # Episodes between target updates

    # Double DQN: use online network for action selection, target for evaluation
    # Reduces overestimation bias in Q-values
    use_double_dqn: bool = True

    # Exploration (logarithmic decay from start to end over all episodes)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05

    # Training
    n_episodes: int = 2000
    updates_per_episode: int = 5
    min_buffer_size: int = 100  # Minimum transitions before training

    # Network
    hidden_dim: int = 64

    # Device
    device: str = "cpu"

    # Parallelization
    n_workers: int = 1  # Number of parallel workers (1 = sequential)


@dataclass
class TrainingStats:
    """Statistics from DQN training."""

    episode_rewards: list[float] = field(default_factory=list)
    episode_best_gaps: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    # Action counts from final 10% of episodes (low epsilon, learned policy)
    action_counts: dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(N_ACTIONS)})
    # Q-value statistics per update batch (for overestimation analysis)
    q_values_mean: list[float] = field(default_factory=list)
    q_values_max: list[float] = field(default_factory=list)
    # Metadata
    use_double_dqn: bool = False


def compute_epsilon(episode: int, n_episodes: int, eps_start: float, eps_end: float) -> float:
    """
    Compute epsilon with logarithmic decay.

    Guarantees epsilon goes from eps_start to eps_end over n_episodes.
    Formula: eps(t) = eps_start * (eps_end / eps_start) ^ (t / T)

    Args:
        episode: Current episode (0-indexed).
        n_episodes: Total number of episodes.
        eps_start: Initial epsilon.
        eps_end: Final epsilon.

    Returns:
        Epsilon value for the current episode.
    """
    if n_episodes <= 1:
        return eps_end
    progress = episode / (n_episodes - 1)
    return eps_start * (eps_end / eps_start) ** progress


def compute_time_budget(n: int, base_budget: float = 10.0) -> float:
    """
    Compute time budget scaled by instance size.

    Uses O(n²) scaling: T(n) = (n/100)² × base_budget

    Args:
        n: Instance size (number of cities).
        base_budget: Base time budget for n=100.

    Returns:
        Scaled time budget in seconds.
    """
    return (n / 100) ** 2 * base_budget


# Worker function for parallel training (must be top-level for pickling)
def _episode_worker(args: tuple) -> dict:
    """
    Run a single episode and return transitions.

    Args:
        args: Tuple of (instance_idx, dataset_path, instance_ids, weights, config_dict, baselines)

    Returns:
        Dictionary with transitions, reward, best_gap, and steps.
    """
    instance_idx, dataset_path, instance_ids, weights, config_dict, baselines = args

    # Load instance
    instance_id = instance_ids[instance_idx % len(instance_ids)]
    dataset = TSPDataset(dataset_path, [instance_id], verbose=False)
    instance = next(iter(dataset))

    # Set pre-computed baseline (avoids redundant computation)
    instance.set_baseline_cost(baselines[instance_id])

    # Create network and load weights
    state_dim = 3 + config_dict["history_len"] * N_ACTIONS
    q_net = QNetwork(state_dim, n_actions=N_ACTIONS, hidden_dim=config_dict["hidden_dim"])
    q_net.load_state_dict(weights)
    q_net.eval()

    # Compute epsilon for this episode (logarithmic decay)
    epsilon = compute_epsilon(
        instance_idx,
        config_dict["n_episodes"],
        config_dict["epsilon_start"],
        config_dict["epsilon_end"],
    )

    # Run episode (use baseline as reference for state computation)
    time_budget = compute_time_budget(instance.dimension, config_dict["time_budget"])
    env = DQNEnv(instance, time_budget, config_dict["history_len"], use_baseline=True)

    state = env.reset()
    transitions = []
    actions_taken = []
    episode_reward = 0.0
    done = False

    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randrange(N_ACTIONS)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state.to_numpy(), dtype=torch.float32)
                q_values = q_net(state_tensor)
                action = int(q_values.argmax().item())

        # Execute action
        next_state, reward, done = env.step(action)

        # Store transition and action
        transitions.append(
            (
                state.to_numpy(),
                action,
                reward,
                next_state.to_numpy(),
                done,
            )
        )
        actions_taken.append(action)

        episode_reward += reward
        state = next_state

    return {
        "episode": instance_idx,
        "transitions": transitions,
        "actions": actions_taken,
        "reward": episode_reward,
        "best_gap": env.best_gap,
        "steps": len(transitions),
        "epsilon": epsilon,
    }


def train_dqn(
    instances: list[TSPInstance],
    config: DQNConfig,
    verbose: bool = True,
    dataset_path: str | None = None,
    instance_ids: list[int] | None = None,
) -> tuple[QNetwork, TrainingStats]:
    """
    Train DQN on a set of TSP instances.

    Computes baseline for each instance first (parallel if n_workers > 1),
    then trains using baseline-relative gaps for consistent state distribution
    between training and evaluation.

    Args:
        instances: List of training instances.
        config: Training configuration.
        verbose: Print progress information.
        dataset_path: Path to dataset JSON (required for parallel training).
        instance_ids: List of instance IDs (required for parallel training).

    Returns:
        Tuple of (trained Q-network, training statistics).
    """
    if not instances:
        raise ValueError("No instances provided")

    # Compute baselines for all instances (once, reused across episodes)
    n = instances[0].dimension
    time_budget = compute_time_budget(n, config.time_budget)

    if dataset_path is not None and instance_ids is not None:
        # Parallel baseline computation
        n_workers = max(1, config.n_workers)
        baselines = compute_baselines_parallel(dataset_path, instance_ids, time_budget, n_workers, verbose)
        # Set baselines on instances
        for inst, inst_id in zip(instances, instance_ids):
            inst.set_baseline_cost(baselines[inst_id])
    else:
        # Sequential baseline computation
        if verbose:
            print(f"  Computing baselines for {len(instances)} instances...")
        for inst in instances:
            baseline_cost = _compute_baseline(inst, time_budget)
            inst.set_baseline_cost(baseline_cost)
        if verbose:
            print(f"  Baselines computed")

    # Check parallel requirements for training
    if config.n_workers > 1:
        if dataset_path is None or instance_ids is None:
            raise ValueError("Parallel training (n_workers > 1) requires dataset_path and instance_ids")
        baselines_dict = {inst_id: inst.baseline_cost for inst_id, inst in zip(instance_ids, instances)}
        return _train_dqn_parallel(instances, config, verbose, dataset_path, instance_ids, baselines_dict)

    # Sequential training
    return _train_dqn_sequential(instances, config, verbose)


def _train_dqn_sequential(
    instances: list[TSPInstance],
    config: DQNConfig,
    verbose: bool = True,
) -> tuple[QNetwork, TrainingStats]:
    """Sequential DQN training (single worker)."""
    # Infer instance size from first instance
    n = instances[0].dimension
    time_budget = compute_time_budget(n, config.time_budget)
    state_dim = 3 + config.history_len * N_ACTIONS

    if verbose:
        print(f"Training DQN: n={n}, time_budget={time_budget:.2f}s")
        print(f"State dim: {state_dim}, Actions: {N_ACTIONS}, Episodes: {config.n_episodes}")

    # Initialize networks
    q_net = QNetwork(state_dim, n_actions=N_ACTIONS, hidden_dim=config.hidden_dim).to(config.device)
    target_net = QNetwork(state_dim, n_actions=N_ACTIONS, hidden_dim=config.hidden_dim).to(config.device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = Adam(q_net.parameters(), lr=config.lr)
    replay_buffer = ReplayBuffer(config.buffer_size)

    stats = TrainingStats(use_double_dqn=config.use_double_dqn)
    final_episodes_start = int(config.n_episodes * 0.9)  # Last 10% for action stats

    for episode in range(config.n_episodes):
        # Compute epsilon for this episode (logarithmic decay)
        epsilon = compute_epsilon(episode, config.n_episodes, config.epsilon_start, config.epsilon_end)

        # Sample random instance (baseline already set in train_dqn)
        instance = random.choice(instances)
        env = DQNEnv(instance, time_budget, config.history_len, use_baseline=True)

        # Collect episode
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randrange(env.n_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state.to_numpy(), dtype=torch.float32, device=config.device)
                    q_values = q_net(state_tensor)
                    action = int(q_values.argmax().item())

            # Execute action
            next_state, reward, done = env.step(action)

            # Store transition
            replay_buffer.push(
                state.to_numpy(),
                action,
                reward,
                next_state.to_numpy(),
                done,
            )

            # Track action (only final 10% of episodes for learned policy stats)
            if episode >= final_episodes_start:
                stats.action_counts[action] += 1

            episode_reward += reward
            episode_steps += 1
            state = next_state

        # Record episode stats
        stats.episode_rewards.append(episode_reward)
        stats.episode_best_gaps.append(env.best_gap)
        stats.episode_lengths.append(episode_steps)
        stats.epsilons.append(epsilon)

        # Update network (if buffer has enough samples)
        if len(replay_buffer) >= config.min_buffer_size:
            for _ in range(config.updates_per_episode):
                batch = replay_buffer.sample(config.batch_size, config.device)

                # Compute Q(s, a)
                q_values = q_net(batch.states)
                q_selected = q_values.gather(1, batch.actions).squeeze(1)

                # Compute target
                with torch.no_grad():
                    if config.use_double_dqn:
                        # Double DQN: select action with online, evaluate with target
                        next_q_online = q_net(batch.next_states)
                        best_actions = next_q_online.argmax(dim=1, keepdim=True)
                        next_q = target_net(batch.next_states).gather(1, best_actions).squeeze(1)
                    else:
                        # Standard DQN: max Q_target(s', a')
                        next_q = target_net(batch.next_states).max(dim=1)[0]
                    targets = batch.rewards + config.gamma * next_q * (1 - batch.dones)

                # Update
                loss = F.mse_loss(q_selected, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                stats.losses.append(loss.item())
                # Track Q-value statistics for overestimation analysis
                stats.q_values_mean.append(float(q_values.mean().item()))
                stats.q_values_max.append(float(q_values.max().item()))

        # Update target network
        if (episode + 1) % config.target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Progress logging
        if verbose and (episode + 1) % 100 == 0:
            recent_gaps = stats.episode_best_gaps[-100:]
            avg_gap = np.mean(recent_gaps)
            print(
                f"Episode {episode + 1}/{config.n_episodes} | "
                f"Avg gap (last 100): {avg_gap:.2f}% | "
                f"ε: {epsilon:.3f}"
            )

    return q_net, stats


def _train_dqn_parallel(
    instances: list[TSPInstance],
    config: DQNConfig,
    verbose: bool,
    dataset_path: str,
    instance_ids: list[int],
    baselines: dict[int, float],
) -> tuple[QNetwork, TrainingStats]:
    """
    Parallel DQN training using batch episodes.

    Runs multiple episodes in parallel, aggregates transitions, then updates.
    Uses pre-computed baselines for consistent state distribution.
    """
    n = instances[0].dimension
    time_budget = compute_time_budget(n, config.time_budget)
    state_dim = 3 + config.history_len * N_ACTIONS
    n_workers = config.n_workers

    if verbose:
        print(f"Training DQN (parallel): n={n}, time_budget={time_budget:.2f}s")
        print(f"State dim: {state_dim}, Actions: {N_ACTIONS}, Episodes: {config.n_episodes}")
        print(f"Using {n_workers} parallel workers")

    # Initialize networks
    q_net = QNetwork(state_dim, n_actions=N_ACTIONS, hidden_dim=config.hidden_dim).to(config.device)
    target_net = QNetwork(state_dim, n_actions=N_ACTIONS, hidden_dim=config.hidden_dim).to(config.device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = Adam(q_net.parameters(), lr=config.lr)
    replay_buffer = ReplayBuffer(config.buffer_size)

    stats = TrainingStats(use_double_dqn=config.use_double_dqn)
    final_episodes_start = int(config.n_episodes * 0.9)  # Last 10% for action stats

    # Calculate number of batches
    n_batches = (config.n_episodes + n_workers - 1) // n_workers
    episode_counter = 0

    # Keep executor alive for entire training to avoid process spawn overhead
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for batch_idx in range(n_batches):
            # How many episodes in this batch
            episodes_in_batch = min(n_workers, config.n_episodes - episode_counter)
            if episodes_in_batch <= 0:
                break

            # Prepare worker arguments (each worker computes its own epsilon)
            weights = {k: v.cpu() for k, v in q_net.state_dict().items()}
            config_dict = {
                "time_budget": config.time_budget,
                "history_len": config.history_len,
                "hidden_dim": config.hidden_dim,
                "n_episodes": config.n_episodes,
                "epsilon_start": config.epsilon_start,
                "epsilon_end": config.epsilon_end,
            }

            worker_args = [
                (episode_counter + i, dataset_path, instance_ids, weights, config_dict, baselines)
                for i in range(episodes_in_batch)
            ]

            # Run episodes in parallel
            batch_results = []
            futures = [executor.submit(_episode_worker, args) for args in worker_args]
            for future in as_completed(futures):
                batch_results.append(future.result())

            # Aggregate results
            for result in batch_results:
                # Add transitions to buffer
                for transition in result["transitions"]:
                    replay_buffer.push(*transition)

                # Track actions (only final 10% of episodes for learned policy stats)
                if result["episode"] >= final_episodes_start:
                    for action in result["actions"]:
                        stats.action_counts[action] += 1

                # Record stats
                stats.episode_rewards.append(result["reward"])
                stats.episode_best_gaps.append(result["best_gap"])
                stats.episode_lengths.append(result["steps"])
                stats.epsilons.append(result["epsilon"])

                episode_counter += 1

            # Update network (if buffer has enough samples)
            if len(replay_buffer) >= config.min_buffer_size:
                # More updates for larger batches
                n_updates = config.updates_per_episode * episodes_in_batch
                for _ in range(n_updates):
                    batch = replay_buffer.sample(config.batch_size, config.device)

                    # Compute Q(s, a)
                    q_values = q_net(batch.states)
                    q_selected = q_values.gather(1, batch.actions).squeeze(1)

                    # Compute target
                    with torch.no_grad():
                        if config.use_double_dqn:
                            # Double DQN: select action with online, evaluate with target
                            next_q_online = q_net(batch.next_states)
                            best_actions = next_q_online.argmax(dim=1, keepdim=True)
                            next_q = target_net(batch.next_states).gather(1, best_actions).squeeze(1)
                        else:
                            # Standard DQN: max Q_target(s', a')
                            next_q = target_net(batch.next_states).max(dim=1)[0]
                        targets = batch.rewards + config.gamma * next_q * (1 - batch.dones)

                    # Update
                    loss = F.mse_loss(q_selected, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    stats.losses.append(loss.item())
                    # Track Q-value statistics for overestimation analysis
                    stats.q_values_mean.append(float(q_values.mean().item()))
                    stats.q_values_max.append(float(q_values.max().item()))

            # Update target network
            if episode_counter % config.target_update_freq < episodes_in_batch:
                target_net.load_state_dict(q_net.state_dict())

            # Progress logging
            if verbose and (episode_counter % 100 < episodes_in_batch or episode_counter == config.n_episodes):
                recent_gaps = stats.episode_best_gaps[-100:]
                avg_gap = np.mean(recent_gaps)
                current_epsilon = stats.epsilons[-1] if stats.epsilons else config.epsilon_start
                print(
                    f"Episode {episode_counter}/{config.n_episodes} | "
                    f"Avg gap (last 100): {avg_gap:.2f}% | "
                    f"ε: {current_epsilon:.3f}"
                )

    return q_net, stats


def _eval_worker(args: tuple) -> float:
    """Evaluate a single instance (worker function for parallel evaluation)."""
    instance_id, dataset_path, weights, config_dict, baselines = args

    # Load instance
    dataset = TSPDataset(dataset_path, [instance_id], verbose=False)
    instance = next(iter(dataset))

    # Set pre-computed baseline
    instance.set_baseline_cost(baselines[instance_id])

    # Create network and load weights
    state_dim = 3 + config_dict["history_len"] * N_ACTIONS
    q_net = QNetwork(state_dim, n_actions=N_ACTIONS, hidden_dim=config_dict["hidden_dim"])
    q_net.load_state_dict(weights)
    q_net.eval()

    # Run evaluation episode (greedy policy, baseline as state reference)
    time_budget = compute_time_budget(instance.dimension, config_dict["time_budget"])
    env = DQNEnv(instance, time_budget, config_dict["history_len"], use_baseline=True)

    state = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(state.to_numpy(), dtype=torch.float32)
            q_values = q_net(state_tensor)
            action = int(q_values.argmax().item())
        state, _, done = env.step(action)

    return env.best_gap


def evaluate_dqn(
    model: QNetwork,
    instances: list[TSPInstance],
    config: DQNConfig,
    verbose: bool = False,
    dataset_path: str | None = None,
    instance_ids: list[int] | None = None,
) -> list[float]:
    """
    Evaluate trained DQN on test instances.

    Computes baseline for each instance first, then runs DQN using baseline
    as state reference (consistent with training distribution).

    Args:
        model: Trained Q-network.
        instances: List of test instances.
        config: Configuration (for time budget and history length).
        verbose: Print per-instance results.
        dataset_path: Path to dataset JSON (required for parallel evaluation).
        instance_ids: List of instance IDs (required for parallel evaluation).

    Returns:
        List of final gaps (%) for each instance.
    """
    if not instances:
        return []

    # Compute baselines for all test instances
    n = instances[0].dimension
    time_budget = compute_time_budget(n, config.time_budget)

    if dataset_path is not None and instance_ids is not None:
        # Parallel baseline computation
        n_workers = max(1, config.n_workers)
        baselines = compute_baselines_parallel(dataset_path, instance_ids, time_budget, n_workers, verbose=False)
        # Set baselines on instances
        for inst, inst_id in zip(instances, instance_ids):
            inst.set_baseline_cost(baselines[inst_id])
    else:
        # Sequential baseline computation
        baselines = {}
        for i, inst in enumerate(instances):
            baseline_cost = _compute_baseline(inst, time_budget)
            inst.set_baseline_cost(baseline_cost)
            if instance_ids:
                baselines[instance_ids[i]] = baseline_cost

    # Use parallel evaluation if workers > 1 and we have dataset info
    if config.n_workers > 1 and dataset_path is not None and instance_ids is not None:
        weights = {k: v.cpu() for k, v in model.state_dict().items()}
        config_dict = {
            "time_budget": config.time_budget,
            "history_len": config.history_len,
            "hidden_dim": config.hidden_dim,
        }

        worker_args = [(inst_id, dataset_path, weights, config_dict, baselines) for inst_id in instance_ids]

        gaps = []
        with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
            futures = {executor.submit(_eval_worker, args): i for i, args in enumerate(worker_args)}
            results = [None] * len(worker_args)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
            gaps = results

        if verbose:
            for i, gap in enumerate(gaps):
                print(f"Instance {i + 1}/{len(gaps)}: gap = {gap:.2f}%")

        return gaps

    # Sequential evaluation (fallback)
    model.eval()
    gaps = []

    for i, instance in enumerate(instances):
        env = DQNEnv(instance, time_budget, config.history_len, use_baseline=True)
        state = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state.to_numpy(), dtype=torch.float32, device=config.device)
                q_values = model(state_tensor)
                action = int(q_values.argmax().item())

            state, _, done = env.step(action)

        gaps.append(env.best_gap)

        if verbose:
            print(f"Instance {i + 1}/{len(instances)}: gap = {env.best_gap:.2f}%")

    return gaps


def save_model(model: QNetwork, path: str | Path) -> None:
    """Save model with architecture metadata."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "state_dim": model.state_dim,
        "n_actions": model.n_actions,
        "hidden_dim": model.hidden_dim,
    }
    torch.save(checkpoint, path)


def load_model(path: str | Path) -> QNetwork:
    """Load model from checkpoint (architecture is inferred from metadata)."""
    checkpoint = torch.load(path, weights_only=False)
    model = QNetwork(
        state_dim=checkpoint["state_dim"],
        n_actions=checkpoint["n_actions"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def compute_q_matrix(
    model: QNetwork,
    gap_levels: list[float] | None = None,
    t_ratio: float = 0.5,
) -> np.ndarray:
    """
    Compute Q-values for discretized gap levels.

    Creates a matrix of Q-values by querying the network at synthetic
    states with varying gap levels. Useful for visualizing learned policy.

    Args:
        model: Trained Q-network (n_actions and history_len inferred from model).
        gap_levels: Gap percentages to evaluate (default: [0, 1, 2, 5, 10, 20, 50]).
        t_ratio: Fixed time ratio for states (default: 0.5).

    Returns:
        Matrix of shape (len(gap_levels), n_actions) with Q-values.
    """
    if gap_levels is None:
        gap_levels = [0, 1, 2, 5, 10, 20, 50]

    # Infer from model attributes
    n_actions = model.n_actions
    history_len = (model.state_dim - 3) // n_actions

    # Normalize gaps using same function as DQNState
    def normalize_gap(gap: float) -> float:
        return float(np.log1p(gap) / np.log1p(100))

    q_matrix = np.zeros((len(gap_levels), n_actions))

    model.eval()
    with torch.no_grad():
        for i, gap in enumerate(gap_levels):
            # Create state: g=g_best=gap, t_ratio fixed, empty history
            g_norm = normalize_gap(gap)
            state = [g_norm, g_norm, t_ratio]

            # Empty history (one-hot with no action selected)
            for _ in range(history_len):
                state.extend([0.0] * n_actions)

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = model(state_tensor).squeeze(0).numpy()
            q_matrix[i, :] = q_values

    return q_matrix


__all__ = [
    "DQNConfig",
    "TrainingStats",
    "train_dqn",
    "evaluate_dqn",
    "compute_time_budget",
    "save_model",
    "load_model",
    "compute_q_matrix",
    "get_default_workers",
    "N_ACTIONS",
]
