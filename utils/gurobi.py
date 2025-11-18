"""Gurobi solver configuration and TSP solving."""
import gurobipy as gp
import random
import time
from dataclasses import dataclass
from gurobipy import GRB
from settings import get_gurobi_credentials
from tsp import TSPInstance
from typing import List, Optional, Callable


@dataclass
class SolverConfig:
    """Configuration for Gurobi solver."""
    time_limit: Optional[float] = None  # seconds
    mip_gap: float = 0.01  # 1% gap
    threads: Optional[int] = None
    verbose: bool = True
    callback: Callable = None
    use_mtz: bool = False

    def apply_to_model(self, model: gp.Model):
        """Apply configuration to Gurobi model."""
        if self.time_limit:
            model.Params.TimeLimit = self.time_limit
        model.Params.MIPGap = self.mip_gap
        if self.threads:
            model.Params.Threads = self.threads
        model.Params.OutputFlag = 1 if self.verbose else 0
        model.Params.LazyConstraints = 1 if self.callback else 0


@dataclass
class Solution:
    """Represents a TSP solution."""
    tour: List[int]  # ordered list of city indices
    cost: float
    solve_time: float
    gap: float  # optimality gap
    status: str


def subtour_callback(model, where):
    """Callback to eliminate subtours using lazy constraints."""
    if where == GRB.Callback.MIPSOL:
        x_val = model.cbGetSolution(model._x)
        n = model._n

        # Find subtours
        subtours = find_subtours(x_val, n)

        # Add constraint for each subtour
        for tour in subtours:
            if len(tour) < n:
                model.cbLazy(
                    gp.quicksum(model._x[i, j] for i in tour for j in tour if i != j)
                    <= len(tour) - 1
                )


def find_subtours(x_val, n):
    """Find all subtours in current solution."""
    unvisited = set(range(n))
    subtours = []

    while unvisited:
        subtour = []
        node = unvisited.pop()
        subtour.append(node)

        while True:
            neighbors = [j for j in range(n) if x_val[node, j] > 0.5]
            if not neighbors:
                break
            node = neighbors[0]
            if node in subtour:
                break
            subtour.append(node)
            unvisited.discard(node)

        subtours.append(subtour)

    return subtours


def heuristic_random(n: int) -> list[int]:
    """Return a random Hamiltonian tour as a list of nodes."""
    tour = list(range(n))
    random.shuffle(tour)
    return tour


def heuristic_greedy(dist: list[list[float]]) -> list[int]:
    """Nearest-neighbor greedy heuristic."""
    n = len(dist)
    unvisited = set(range(1, n))
    tour = [0]
    while unvisited:
        last = tour[-1]
        next_node = min(unvisited, key=lambda j: dist[last][j])
        tour.append(next_node)
        unvisited.remove(next_node)
    return tour


def heuristic_two_opt(dist: list[list[float]], base_tour: list[int]) -> list[int]:
    """One pass of 2-opt improvement over a given tour."""
    n = len(dist)
    best_tour = base_tour[:]
    best_cost = tour_cost(dist, best_tour)
    for i in range(1, n - 2):
        for j in range(i + 1, n):
            new_tour = (
                base_tour[:i] +
                list(reversed(base_tour[i:j])) +
                base_tour[j:]
            )
            new_cost = tour_cost(dist, new_tour)
            if new_cost < best_cost:
                best_tour, best_cost = new_tour, new_cost
    return best_tour


def tour_cost(dist: list[list[float]], tour: list[int]) -> float:
    """Calculate the cost of a tour."""
    return sum(
        dist[tour[i]][tour[(i + 1) % len(tour)]]
        for i in range(len(tour))
    )


def tour_to_solution_dict(
    tour: list[int],
    x_vars: dict[tuple[int, int], gp.Var]
) -> dict[gp.Var, gp.Var.X]:
    """Convert a tour to a {var: value} dict suitable for cbSetSolution()."""
    n = len(tour)
    sol = {}
    for i in range(n):
        a, b = tour[i], tour[(i + 1) % n]
        for (ii, jj), var in x_vars.items():
            sol[var] = 1.0 if (ii == a and jj == b) else 0.0
    return sol


def hyper_callback(model, where):
    """Hyper-heuristic callback that chooses heuristics when stagnating."""
    if where == GRB.Callback.MIP:
        # Basic search info
        best_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
        bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        time = model.cbGet(GRB.Callback.RUNTIME)
        gap = abs(best_obj - bound) / (1e-10 + abs(best_obj))

        # Initialize state attributes
        if not hasattr(model, "_last_gap"):
            model._last_gap = gap
            model._stagnation = 0

        # Detect stagnation
        if abs(gap - model._last_gap) < 1e-4:
            model._stagnation += 1
        else:
            model._stagnation = 0
        model._last_gap = gap

        # If stagnating, pick a heuristic at random
        if model._stagnation > 10:
            heuristics = ["random", "greedy", "two_opt"]
            choice = random.choice(heuristics)
            if model.Params.OutputFlag:
                print(
                    f"[HH] Applying {choice} heuristic at time={time:.1f}s "
                    f"(gap={gap:.4f})"
                )
            model._stagnation = 0  # reset counter

            try:
                # Call the chosen heuristic
                dist = model._dist
                n = len(dist)
                if choice == "random":
                    tour = heuristic_random(n)
                elif choice == "greedy":
                    tour = heuristic_greedy(dist)
                elif choice == "two_opt":
                    base = heuristic_greedy(dist)
                    tour = heuristic_two_opt(dist, base)

                # Build solution dict
                sol = tour_to_solution_dict(tour, model._x)
                model.cbSetSolution(list(sol.keys()), list(sol.values()))
                if model.Params.OutputFlag:
                    print(
                        f"[HH] Injected solution from {choice}, "
                        f"cost={tour_cost(dist, tour):.2f}"
                    )

            except Exception as e:
                if model.Params.OutputFlag:
                    print(f"[HH] Failed to inject {choice}: {e}")


class GurobiTSPSolver:
    """Solves TSP using Gurobi."""
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()

    def solve(self, instance: TSPInstance) -> Solution:
        """Solve TSP instance."""
        start_time = time.time()
        n = instance.n_cities
        dist = instance.distances

        # Create model with credentials
        params = get_gurobi_credentials()
        env = gp.Env(params=params)
        model = gp.Model("TSP", env=env)
        self.config.apply_to_model(model)

        # Decision variables: x[i,j] = 1 if edge (i,j) is in tour
        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

        # Objective: minimize total distance
        model.setObjective(
            gp.quicksum(dist[i, j] * x[i, j] for i in range(n) for j in range(n) if i != j),
            GRB.MINIMIZE
        )

        # Constraints: each city visited exactly once (degree = 2)
        model.addConstrs((x.sum(i, '*') == 1 for i in range(n)), name="out")
        model.addConstrs((x.sum('*', j) == 1 for j in range(n)), name="in")

        # Constraints: MTZ
        if self.config.use_mtz:
            u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")
            model.addConstrs((u[i] - u[j] + n * x[i, j] <= n - 1
                          for i in range(1, n) for j in range(1, n) if i != j),
                          name="mtz")

        # No self-loops
        model.addConstrs((x[i, i] == 0 for i in range(n)), name="no_loop")

        model._dist = dist

        if self.config.callback:
            # Use callback for subtour elimination or hyper-heuristic
            model._x = x
            model._n = n
            model.optimize(self.config.callback)
        else:
            model.optimize()

        solve_time = time.time() - start_time

        # Extract solution
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            tour = self._extract_tour(x, n)
            gap = model.MIPGap if hasattr(model, 'MIPGap') else 0.0
            return Solution(
                tour=tour,
                cost=model.objVal,
                solve_time=solve_time,
                gap=gap,
                status="optimal" if model.status == GRB.OPTIMAL else "time_limit"
            )
        else:
            return Solution(
                tour=[],
                cost=float('inf'),
                solve_time=solve_time,
                gap=1.0,
                status="infeasible"
            )

    def _extract_tour(self, x, n):
        """Extract tour from solution"""
        tour = [0]
        current = 0

        for _ in range(n - 1):
            for j in range(n):
                if j != current and x[current, j].X > 0.5:
                    tour.append(j)
                    current = j
                    break

        return tour

