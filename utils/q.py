"""Q-learning hyper-heuristic for TSP."""
import os
import math
import tempfile
import subprocess
from typing import List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# -------------------------
# Utilities and TSP primitives
# -------------------------
def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Calculate euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tour_length(tour: List[int], coords: List[Tuple[float, float]]) -> float:
    """Calculate the total length of a tour."""
    n = len(tour)
    s = 0.0
    for i in range(n):
        a = coords[tour[i]]
        b = coords[tour[(i + 1) % n]]
        s += euclidean_distance(a, b)
    return s


def greedy_tour(coords: List[Tuple[float, float]]) -> List[int]:
    """Build a greedy nearest-neighbor tour."""
    n = len(coords)
    unvisited = set(range(n))
    current = 0
    tour = [current]
    unvisited.remove(current)
    while unvisited:
        next_city = min(unvisited, key=lambda j: euclidean_distance(coords[current], coords[j]))
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    return tour


# -------------------------
# Low-level heuristics
# -------------------------
def two_opt_best_improvement(
    tour: List[int],
    coords: List[Tuple[float, float]],
    max_checks: int = 1000
) -> List[int]:
    """Apply 2-opt best improvement heuristic."""
    n = len(tour)
    best_delta = 0.0
    best_move = None
    checks = 0
    for _ in range(max_checks):
        i = np.random.randint(0, n - 1)
        j = np.random.randint(i + 1, n)
        if j == i + 1:
            continue
        a, b = coords[tour[i]], coords[tour[(i + 1) % n]]
        c, d = coords[tour[j]], coords[tour[(j + 1) % n]]
        delta = (euclidean_distance(a, c) + euclidean_distance(b, d)) - (euclidean_distance(a, b) + euclidean_distance(c, d))
        if delta < best_delta:
            best_delta = delta
            best_move = (i + 1, j)
        checks += 1
    if best_move is None:
        return tour[:]
    i1, j1 = best_move
    new_tour = tour[:i1] + list(reversed(tour[i1:j1 + 1])) + tour[j1 + 1:]
    return new_tour


def swap_two_nodes(
    tour: List[int],
    coords: List[Tuple[float, float]],
    tries: int = 50
) -> List[int]:
    """Swap two nodes in the tour."""
    n = len(tour)
    best_delta = 0.0
    best = None
    base = tour_length(tour, coords)
    for _ in range(tries):
        i, j = np.random.choice(n, size=2, replace=False)
        if i > j:
            i, j = j, i
        new = tour[:]
        new[i], new[j] = new[j], new[i]
        new_len = tour_length(new, coords)
        delta = new_len - base
        if delta < best_delta:
            best_delta = delta
            best = new
    if best is None:
        return tour[:]
    return best


def relocate_one_node(
    tour: List[int],
    coords: List[Tuple[float, float]],
    tries: int = 50
) -> List[int]:
    """Relocate one node to a different position."""
    n = len(tour)
    best_delta = 0.0
    best = None
    base = tour_length(tour, coords)
    for _ in range(tries):
        i, j = np.random.choice(n, size=2, replace=False)
        t = tour[:]
        city = t.pop(i)
        t.insert(j, city)
        new_len = tour_length(t, coords)
        delta = new_len - base
        if delta < best_delta:
            best_delta = delta
            best = t
    if best is None:
        return tour[:]
    return best


HEURISTICS = [
    ("2-opt", two_opt_best_improvement),
    ("swap", swap_two_nodes),
    ("relocate", relocate_one_node)
]


# -------------------------
# Lower bound computations
# -------------------------
def mst_lower_bound(coords: List[Tuple[float, float]]) -> float:
    """Return weight of MST on complete graph = simple LB (fast)."""
    n = len(coords)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            w = euclidean_distance(coords[i], coords[j])
            G.add_edge(i, j, weight=w)
    mst = nx.minimum_spanning_tree(G)
    return mst.size(weight='weight')


def one_tree_lower_bound(coords: List[Tuple[float, float]]) -> float:
    r"""Minimum 1-tree lower bound.

    For each possible root r:
      - compute MST on V\\{r}
      - add two smallest edges incident to r
    Return min over r (this is a common 1-tree LB; Held-Karp improvement
        requires Lagrangian iteration).
    Complexity: O(n * (m log n)) ~ O(n^2 log n) for dense graphs, OK for n up
        to a few hundreds.
    """
    n = len(coords)
    # Precompute pairwise distances
    dmat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dmat[i][j] = euclidean_distance(coords[i], coords[j])
    best = float('inf')
    # We'll reuse networkx's MST on n-1 nodes multiple times
    for r in range(n):
        # Build graph without r
        G = nx.Graph()
        for i in range(n):
            if i == r:
                continue
            G.add_node(i)
        for i in range(n):
            if i == r:
                continue
            for j in range(i + 1, n):
                if j == r:
                    continue
                G.add_edge(i, j, weight=dmat[i][j])
        mst = nx.minimum_spanning_tree(G)
        mst_w = mst.size(weight='weight')
        # add two smallest edges from r to other nodes
        edges_from_r = sorted((dmat[r][j] for j in range(n) if j != r))
        if len(edges_from_r) < 2:
            continue
        val = mst_w + edges_from_r[0] + edges_from_r[1]
        if val < best:
            best = val
    return best


# -------------------------
# Concorde / LKH wrappers (best-effort)
# -------------------------
def write_tsplib(coords: List[Tuple[float, float]], fname: str):
    """Write a minimal TSPLIB .tsp file for Concorde / LKH input (EUC_2D)."""
    n = len(coords)
    with open(fname, "w") as f:
        f.write("NAME: demo\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, start=1):
            f.write(f"{i} {x} {y}\n")
        f.write("EOF\n")


def run_concorde(
    coords: List[Tuple[float, float]],
    timeout_sec: int = 30
) -> Optional[float]:
    """Try to run 'concorde' on a TSPLIB file and parse optimal tour length."""
    try:
        # write instance
        with tempfile.TemporaryDirectory() as tmpd:
            tspfile = os.path.join(tmpd, "inst.tsp")
            write_tsplib(coords, tspfile)
            cmd = ["concorde", tspfile]
            print("Calling Concorde:", " ".join(cmd))
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec
            )
            out = p.stdout.decode() + "\n" + p.stderr.decode()
            for line in out.splitlines():
                line = line.strip()
                if (
                    "Optimal" in line and
                    ("cost" in line.lower() or "tour" in line.lower())
                ):
                    toks = [
                        t
                        for t in line.replace("=", " ").split()
                        if any(ch.isdigit() for ch in t)
                    ]
                    if toks:
                        return float(toks[-1])
            print("Concorde output (first 200 chars):", out[:200])
    except subprocess.TimeoutExpired:
        print("Concorde timed out.")
    except FileNotFoundError:
        print("Concorde binary not found.")
    except Exception as e:
        print("Error running concorde:", e)
    return None


def run_lkh(
    coords: List[Tuple[float, float]],
    max_trials: int = 1,
    timeout_sec: int = 15
) -> Optional[float]:
    """Run LKH to get a *tour cost* (upper bound). Not a lower bound."""
    try:
        with tempfile.TemporaryDirectory() as tmpd:
            tspfile = os.path.join(tmpd, "inst.tsp")
            write_tsplib(coords, tspfile)
            parfile = os.path.join(tmpd, "lkh.par")
            with open(parfile, "w") as f:
                f.write(f"PROBLEM_FILE = {tspfile}\n")
                f.write(f"OUTPUT_TOUR_FILE = {tmpd}/out.tour\n")
                f.write(f"MAX_TRIALS = {max_trials}\n")
            cmd = ["LKH", parfile]
            print("Calling LKH:", " ".join(cmd))
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec
            )
            out = p.stdout.decode() + "\n" + p.stderr.decode()
            best = None
            for line in out.splitlines():
                if "Cost" in line and "=" in line:
                    try:
                        val = float(line.split("=")[-1].strip().split()[0])
                        best = val
                        break
                    except (ValueError, IndexError):
                        pass
            if best is None:
                print("LKH output (first 200 chars):", out[:200])
            return best
    except subprocess.TimeoutExpired:
        print("LKH timed out.")
    except FileNotFoundError:
        print("LKH binary not found.")
    except Exception as e:
        print("Error running LKH:", e)
    return None


# -------------------------
# State abstraction and Q-learning agent
# -------------------------
def quality_bin(
    current_len: float,
    init_len: float,
    n_bins: int = 6
) -> int:
    """Bin the quality ratio into discrete states."""
    ratio = current_len / init_len
    capped = min(max(ratio, 0.9), 2.0)
    bin_idx = int((capped - 0.9) / (2.0 - 0.9) * (n_bins - 1))
    return max(0, min(n_bins - 1, bin_idx))


def improvement_bin(
    delta: float,
    thresholds: Tuple[float, float] = (-1e-6, -0.01)
) -> int:
    """Bin the improvement delta into discrete states."""
    if delta < thresholds[1]:
        return 2
    elif delta < thresholds[0]:
        return 1
    else:
        return 0


def state_from(
    tour: List[int],
    coords: List[Tuple[float, float]],
    init_len: float,
    last_delta: float
) -> Tuple[int, int]:
    """Extract state representation from current tour."""
    q = quality_bin(tour_length(tour, coords), init_len)
    ib = improvement_bin(last_delta)
    return (q, ib)


class QLearningHH:
    """Q-learning hyper-heuristic agent."""
    def __init__(
        self,
        n_quality_bins: int = 6,
        n_imp_bins: int = 3,
        n_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.9,
        eps: float = 0.2
    ):
        """Initialize Q-learning agent."""
        self.nq = n_quality_bins
        self.ni = n_imp_bins
        self.na = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Q = np.zeros((self.nq, self.ni, self.na), dtype=float)

    def choose(self, state: Tuple[int, int]) -> int:
        """Choose an action using epsilon-greedy policy."""
        qbin, ibin = state
        if np.random.random() < self.eps:
            return np.random.randint(0, self.na)
        vals = self.Q[qbin, ibin, :]
        maxv = vals.max()
        choices = np.flatnonzero(np.isclose(vals, maxv))
        return int(np.random.choice(choices))

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int]
    ) -> None:
        """Update Q-value using Q-learning update rule."""
        q0 = self.Q[state[0], state[1], action]
        qmax_next = self.Q[next_state[0], next_state[1], :].max()
        self.Q[state[0], state[1], action] = (
            q0 +
            self.alpha * (
                reward +
                self.gamma * qmax_next -
                q0
            )
        )

    def decay_epsilon(
        self,
        factor: float = 0.995,
        min_eps: float = 0.02
    ) -> None:
        """Decay exploration rate."""
        self.eps = max(min_eps, self.eps * factor)


# -------------------------
# Training loop with early stopping based on chosen bound
# -------------------------
def train_qhh_with_early_stopping(
    coords: np.ndarray,
    episodes: int = 400,
    steps_per_episode: int = 200,
    bound_mode: str = "mst",  # "mst", "1tree", "concorde"
    early_stop_gap: float = 0.01,  # relative gap threshold (e.g. 0.01 = 1%)
    patience: int = 30,
    verbose: bool = True,
    seed: Optional[int] = None,
    try_concorde_once: bool = True
):
    """Trains Q-learning HH and uses selected dual bound for early stopping."""
    rng = np.random.default_rng(seed)
    n = len(coords)
    init_tour = greedy_tour(coords)
    init_len = tour_length(init_tour, coords)
    agent = QLearningHH(
        n_quality_bins=6,
        n_imp_bins=3,
        n_actions=len(HEURISTICS),
        alpha=0.2,
        gamma=0.95,
        eps=0.4
    )
    history = []

    # pick LB function
    def compute_lb(mode):
        if mode == "mst":
            return mst_lower_bound(coords)
        elif mode == "1tree":
            return one_tree_lower_bound(coords)
        elif mode == "concorde":
            lb_concorde = run_concorde(coords, timeout_sec=60)
            return (
                lb_concorde
                if lb_concorde is not None
                else mst_lower_bound(coords)
            )
        else:
            return mst_lower_bound(coords)

    LB = compute_lb(bound_mode)

    best_cost = float('inf')
    no_improve = 0

    for ep in range(episodes):
        # initialize tour randomly by shuffling greedy slightly
        tour = init_tour[:]
        for _ in range(5):
            i, j = rng.choice(n, size=2, replace=False)
            tour[i], tour[j] = tour[j], tour[i]
        last_delta = 0.0
        total_reward = 0.0
        for step in range(steps_per_episode):
            state = state_from(tour, coords, init_len, last_delta)
            action = agent.choose(state)
            name, func = HEURISTICS[action]
            new_tour = func(tour, coords)
            old_len = tour_length(tour, coords)
            new_len = tour_length(new_tour, coords)
            delta = new_len - old_len
            reward = (old_len - new_len) / init_len
            total_reward += reward
            next_state = state_from(new_tour, coords, init_len, delta)
            agent.update(state, action, reward, next_state)
            tour = new_tour
            last_delta = delta
        agent.decay_epsilon()
        history.append(total_reward)

        # Check LB and early stopping at episode end
        tour_cost = tour_length(tour, coords)
        gap = (tour_cost - LB) / tour_cost if tour_cost != 0 else 0.0

        if tour_cost < best_cost:
            best_cost = tour_cost
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (ep % max(1, episodes // 10) == 0):
            print(
                f"Episode {ep + 1}/{episodes}, "
                f"total_reward={total_reward:.6f}, "
                f"eps={agent.eps:.4f}, "
                f"tour_len={tour_cost:.3f}, "
                f"gap={gap * 100:.4f}%"
            )
        if gap < early_stop_gap and no_improve >= patience:
            if verbose:
                print(
                    f"Early stopping at episode {ep + 1}: "
                    f"tour_cost={tour_cost:.2f}, "
                    f"LB={LB:.2f}, "
                    f"gap={gap * 100:.4f}%"
                )
            break

    return agent, history, init_len, init_tour, tour, LB


def evaluate_agent(
    agent: QLearningHH,
    coords: np.ndarray,
    episodes: int = 20,
    steps_per_episode: int = 200,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float, List[int]]:
    """Evaluate a trained agent."""
    rng = np.random.default_rng(seed)
    n = len(coords)
    init_tour = greedy_tour(coords)
    init_len = tour_length(init_tour, coords)
    results = []
    for ep in range(episodes):
        tour = init_tour[:]
        for _ in range(5):
            i, j = rng.choice(n, size=2, replace=False)
            tour[i], tour[j] = tour[j], tour[i]
        for step in range(steps_per_episode):
            state = state_from(tour, coords, init_len, 0.0)
            action = agent.choose(state)
            _, func = HEURISTICS[action]
            new_tour = func(tour, coords)
            tour = new_tour
        results.append(tour_length(tour, coords))
    return np.array(results), init_len, init_tour


def plot_tour(
    coords: np.ndarray,
    tour: List[int],
    title: str = "TSP Tour"
) -> None:
    """Plots a TSP tour."""
    tour = np.array(tour + [tour[0]])
    xs = [coords[i][0] for i in tour]
    ys = [coords[i][1] for i in tour]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker='o')
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=10)
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def demo_compare_bounds(
    coords: np.ndarray,
    seed: int = 123,
    episodes: int = 250,
    steps_per_episode: int = 150,
    early_stop_gap: float = 0.01,
    patience: int = 35,
    check_externals: bool = True
) -> dict[str, Any]:
    """Demo function to compare different bound modes."""
    bound_modes = ["mst", "1tree"]

    if check_externals:
        try:
            run_concorde(coords, timeout_sec=5)  # Quick check
            bound_modes.append("concorde")
        except Exception:
            print(
                "Concorde not available; "
                "skipping concorde run (will still record MST/1-tree)."
            )

    results = {}
    for mode in bound_modes:
        print("\n" + "=" * 60)
        print(f"Training using bound mode = {mode}")
        (
            agent,
            history,
            init_len,
            init_tour,
            best_tour,
            lb
        ) = train_qhh_with_early_stopping(
            coords,
            episodes=episodes,
            steps_per_episode=steps_per_episode,
            bound_mode=mode,
            early_stop_gap=early_stop_gap,
            patience=patience,
            verbose=True,
            seed=seed,
            try_concorde_once=True
        )
        plot_tour(coords, init_tour, "Tour inicial")
        plot_tour(coords, best_tour, "Tour final")
        (
            eval_res,
            base_len,
            base_tour
        ) = evaluate_agent(
            agent,
            coords,
            episodes=30,
            steps_per_episode=200,
            seed=seed
        )
        results[mode] = {
            "agent": agent,
            "history": history,
            "eval_results": eval_res,
            "base_len": base_len,
            "lb": lb
        }
        print(
            f"Mode {mode} finished: "
            f"base_len={base_len:.3f}, "
            f"eval_mean={eval_res.mean():.3f}, "
            f"best={eval_res.min():.3f}"
        )

    # Summarize
    print("\n" + "=" * 60)
    print("Summary of bounds and evaluation (None = not run / unavailable):")
    for mode in bound_modes:
        r = results.get(mode)
        if r is None:
            print(f"{mode}: None")
            continue
        eval_res = r["eval_results"]
        print(f"{mode}: Lower Bound={r['lb']:.2f}")
        print(
            f"eval_mean={eval_res.mean():.3f}, "
            f"std={eval_res.std():.3f}, "
            f"best={eval_res.min():.3f}")
    return results
