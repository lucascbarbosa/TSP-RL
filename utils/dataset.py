import os
import random
import numpy as np
import gymnasium as gym


# =====================================================
# Utility: write transitions to TXT file
# =====================================================
def write_transition_dataset(path, transitions):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for s, a, r, s2 in transitions:
            f.write(f"{s} {a} {r} {s2}\n")


# =====================================================
# 1. FrozenLake exporter
# =====================================================
def generate_frozenlake_dataset(instance_id):
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P

    transitions = []
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    for s in range(num_states):
        for a in range(num_actions):
            for prob, s2, reward, done in P[s][a]:
                # include each (s,a,s2,r) multiple times based on probability
                count = max(1, int(prob * 20))  # amplify for dataset density
                for _ in range(count):
                    transitions.append((s, a, reward, s2))

    # ensure final state = highest state index
    # (FrozenLake does this automatically but we enforce)
    final_state = num_states - 1
    transitions = [(s, a, r, final_state if s2 == -1 else s2) for (s, a, r, s2) in transitions]

    path = f"datasets/frozenlake/frozenlake_{instance_id}.txt"
    write_transition_dataset(path, transitions)
    print(f"[✓] FrozenLake dataset {instance_id} saved")


# =====================================================
# 2. Taxi-v3 exporter
# =====================================================
def generate_taxi_dataset(instance_id):
    env = gym.make("Taxi-v3")
    P = env.unwrapped.P

    transitions = []
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    for s in range(num_states):
        for a in range(num_actions):
            for prob, s2, reward, done in P[s][a]:
                # deterministic transitions (prob=1), add 1 sample each
                transitions.append((s, a, reward, s2))

    # enforce final state rule
    final_state = num_states - 1
    transitions = [(s, a, r, final_state if s2 == -1 else s2) for (s, a, r, s2) in transitions]

    path = f"datasets/taxi/taxi_{instance_id}.txt"
    write_transition_dataset(path, transitions)
    print(f"[✓] Taxi dataset {instance_id} saved")


# =====================================================
# 3. Random synthetic MDP generator
# =====================================================
def generate_random_mdp_dataset(instance_id, num_states=30, num_actions=4):
    transitions = []

    final_state = num_states - 1

    for s in range(num_states):
        for a in range(num_actions):

            # synthetic terminal behavior
            if s == final_state:
                transitions.append((s, a, 0.0, final_state))
                continue

            # choose k possible next states
            k = random.randint(1, min(5, num_states))
            possible_next_states = random.sample(range(num_states), k)

            # random transition probabilities
            probs = np.random.dirichlet(np.ones(k))

            for s2, p in zip(possible_next_states, probs):
                reward = np.random.uniform(-1, 1)

                # amplify transitions for realism
                count = max(1, int(p * 20))
                for _ in range(count):
                    transitions.append((s, a, reward, s2))

    path = f"datasets/random/random_{instance_id}.txt"
    write_transition_dataset(path, transitions)
    print(f"[✓] Random MDP dataset {instance_id} saved")


# =====================================================
# MAIN: generate 10 datasets for each environment
# =====================================================
if __name__ == "__main__":
    for i in range(10):
        generate_frozenlake_dataset(i)
        generate_taxi_dataset(i)
        generate_random_mdp_dataset(i)

    print("\nAll datasets generated successfully!")
