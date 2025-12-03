import os
import numpy as np
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import defaultdict

# ===========================================================
# 0. Metrics
# ===========================================================

def init_metrics():
    return defaultdict(lambda: {
        "episode_rewards": [],
        "episode_lengths": [],
        "q_tables": []
    })

def record_metrics(metrics, algo_name, rewards, lengths, qtable):
    m = metrics[algo_name]
    m["episode_rewards"].append(rewards)
    m["episode_lengths"].append(lengths)
    m["q_tables"].append(np.copy(qtable))


def smooth_curve(values, window=10):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode="valid")


def plot_learning_curves(metrics, title="Learning Curves"):
    plt.figure(figsize=(12,5))

    for algo, data in metrics.items():
        # average across runs
        runs = np.array(data["episode_rewards"], dtype=object)
        max_len = max(len(x) for x in runs)
        aligned = np.zeros((len(runs), max_len))
        aligned[:] = np.nan
        for i, r in enumerate(runs):
            aligned[i, :len(r)] = r

        mean_rewards = np.nanmean(aligned, axis=0)
        smoothed = smooth_curve(mean_rewards, window=10)
        plt.plot(smoothed, label=f"{algo}")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_final_performance(metrics, title="Final Performance"):
    algos = []
    values = []

    for algo, data in metrics.items():
        # final reward of each run
        final_rewards = [r[-1] for r in data["episode_rewards"]]
        algos.append(algo)
        values.append(np.mean(final_rewards))

    plt.figure(figsize=(8,5))
    plt.bar(algos, values)
    plt.ylabel("Final Average Reward")
    plt.title(title)
    plt.grid(axis="y")
    plt.show()

# ===========================================================
# 1. LOAD TRANSITION DATASET
# ===========================================================
def load_transition_file(path):
    transitions = []
    max_state = -1
    max_action = -1

    with open(path, "r") as f:
        for line in f:
            s, a, r, s2 = line.strip().split()
            s, a, s2 = int(s), int(a), int(s2)
            r = float(r)
            transitions.append((s, a, r, s2))
            max_state = max(max_state, s, s2)
            max_action = max(max_action, a)

    num_states = max_state + 1
    num_actions = max_action + 1

    return transitions, num_states, num_actions


# ===========================================================
# 2. BUILD EMPIRICAL MDP MODEL FROM TRANSITIONS
# ===========================================================
def build_mdp_model(transitions, num_states, num_actions):
    counts = np.zeros((num_states, num_actions, num_states))
    rewards = np.zeros((num_states, num_actions))
    visits = np.zeros((num_states, num_actions))

    for s, a, r, s2 in transitions:
        counts[s, a, s2] += 1
        rewards[s, a] += r
        visits[s, a] += 1

    # Normalize
    P = np.zeros_like(counts)
    R = np.zeros((num_states, num_actions))

    for s in range(num_states):
        for a in range(num_actions):
            if visits[s, a] > 0:
                P[s, a, :] = counts[s, a] / visits[s, a]
                R[s, a] = rewards[s, a] / visits[s, a]
            else:
                P[s, a, :] = 0
                R[s, a] = 0

    return P, R


# ===========================================================
# 3. VANILLA Q-LEARNING (VALUE-ITERATION STYLE)
# ===========================================================
def vanilla_q_learning(P, R, gamma=0.99, tol=1e-6, max_iter=5000):
    S, A, _ = P.shape
    Q = np.zeros((S, A))

    for it in range(max_iter):
        Q_new = np.zeros_like(Q)
        for s in range(S):
            for a in range(A):
                Q_new[s, a] = R[s, a] + gamma * np.dot(P[s, a], np.max(Q, axis=1))
        if np.max(np.abs(Q_new - Q)) < tol:
            print(f"Vanilla Q converged in {it} iterations")
            break
        Q = Q_new

    return Q


# ===========================================================
# 4. DOUBLE Q-LEARNING (SYNCHRONOUS)
# ===========================================================
def double_q_learning(P, R, gamma=0.99, tol=1e-6, max_iter=5000):
    S, A, _ = P.shape
    Q1 = np.zeros((S, A))
    Q2 = np.zeros((S, A))

    for it in range(max_iter):
        Q1_new = np.copy(Q1)
        Q2_new = np.copy(Q2)

        for s in range(S):
            for a in range(A):
                # Update Q1
                best = np.argmax(Q1[s])
                Q1_new[s, a] = R[s, a] + gamma * np.dot(P[s, a], Q2[:, best])
                # Update Q2
                best2 = np.argmax(Q2[s])
                Q2_new[s, a] = R[s, a] + gamma * np.dot(P[s, a], Q1[:, best2])

        if max(np.max(np.abs(Q1_new - Q1)), np.max(np.abs(Q2_new - Q2))) < tol:
            print(f"Double Q converged in {it} iterations")
            break

        Q1, Q2 = Q1_new, Q2_new

    return (Q1 + Q2) / 2


# ===========================================================
# 5. DQN MODEL (TINY NETWORK)
# ===========================================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ===========================================================
# DQN TRAINING FROM STATIC TRANSITION DATA
# ===========================================================
def dqn_offline(transitions, num_states, num_actions, gamma=0.99, lr=1e-3, steps=5000, batch_size=64):
    device = torch.device("cpu")

    net = DQN(num_states, num_actions).to(device)
    target = DQN(num_states, num_actions).to(device)
    target.load_state_dict(net.state_dict())

    opt = optim.Adam(net.parameters(), lr=lr)

    # Build replay buffer
    replay = transitions

    one_hot = torch.eye(num_states)

    for step in range(steps):
        batch = random.sample(replay, batch_size)
        s, a, r, s2 = zip(*batch)

        s = one_hot[list(s)].to(device)
        s2 = one_hot[list(s2)].to(device)
        a = torch.tensor(a).long().to(device)
        r = torch.tensor(r).float().to(device)

        q_values = net(s)
        q_selected = q_values[range(batch_size), a]

        with torch.no_grad():
            q_next = target(s2).max(dim=1)[0]
            q_target = r + gamma * q_next

        loss = ((q_selected - q_target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0:
            target.load_state_dict(net.state_dict())

    # return Q-table
    with torch.no_grad():
        Q = net(one_hot).cpu().numpy()

    return Q


# ===========================================================
# 6. GREEDY POLICY
# ===========================================================
def extract_policy(Q):
    return np.argmax(Q, axis=1)


# ===========================================================
# 7. ROLLOUT SIMULATION (FOR EVALUATION)
# ===========================================================
def rollout(policy, P, R, start_state=0, max_steps=100):
    s = start_state
    total_reward = 0

    for length in range(max_steps):
        a = policy[s]
        s2 = np.random.choice(len(P), p=P[s, a])
        total_reward += R[s, a]
        if s == len(P) - 1:  # final state
            break
        s = s2

    return total_reward, length


# ===========================================================
# 8. FULL PIPELINE FOR A SINGLE DATASET
# ===========================================================
def evaluate_one(metrics, path):
    print(f"\n=== Evaluating {path} ===")

    transitions, S, A = load_transition_file(path)
    P, R = build_mdp_model(transitions, S, A)

    # Tabular methods
    Q_vanilla = vanilla_q_learning(P, R)
    Q_double = double_q_learning(P, R)

    # DQN
    Q_dqn = dqn_offline(transitions, S, A)

    # Policies
    pi_v = extract_policy(Q_vanilla)
    pi_d = extract_policy(Q_double)
    pi_n = extract_policy(Q_dqn)

    # Evaluate
    result_v, len_v = zip(*[rollout(pi_v, P, R) for _ in range(200)])
    result_d, len_d = zip(*[rollout(pi_d, P, R) for _ in range(200)])
    result_n, len_n = zip(*[rollout(pi_n, P, R) for _ in range(200)])
    rew_v = np.mean(result_v)
    rew_d = np.mean(result_d)
    rew_n = np.mean(result_n)

    record_metrics(metrics, "q_learning", result_v, len_v, Q_vanilla)
    record_metrics(metrics, "double", result_d, len_d, Q_double)
    record_metrics(metrics, "DQN", result_n, len_n, Q_dqn)

    print("Average return (vanilla):", rew_v)
    print("Average return (double): ", rew_d)
    print("Average return (DQN):    ", rew_n)

    return {
        "vanilla": Q_vanilla,
        "double": Q_double,
        "dqn": Q_dqn,
        "return_vanilla": rew_v,
        "return_double": rew_d,
        "return_dqn": rew_n,
    }


# ===========================================================
# 9. BATCH EVALUATION FOR ALL DATASETS
# ===========================================================
def evaluate_all(metrics, root="datasets"):
    results = {}

    for sub in ["frozenlake", "taxi", "random"]:
        folder = os.path.join(root, sub)
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if fname.endswith(".txt"):
                path = os.path.join(folder, fname)
                results[f"{sub}/{fname}"] = evaluate_one(metrics, path)

    return results


# ===========================================================
# RUN
# ===========================================================
if __name__ == "__main__":
    metrics = init_metrics()
    start = time.time()
    #results = evaluate_all(metrics)
    evaluate_one(metrics, "datasets/frozenlake/frozenlake_0.txt")
    plot_learning_curves(metrics)
    plot_final_performance(metrics)
    end = time.time()
    print(end - start)
    print("\n=== Completed evaluation ===")
