"""Main script for TSP-RL project."""
import numpy as np
from typing import Optional
from utils.gurobi import (
    GurobiTSPSolver,
    SolverConfig,
    subtour_callback,
    hyper_callback,
)
"""
from utils.q_learning import (
    evaluate_agent,
    plot_tour,
    tour_length,
    train_qhh_with_early_stopping,
)
"""
from utils.settings import (
    MODE,
    INSTANCE,
    RANDOM,
    SEED,
    TIME_LIMIT,
    MIP_GAP,
    THREADS,
    USE_MTZ,
    USE_CALLBACK,
    USE_HYPER_HEURISTIC,
    EPISODES,
    STEPS,
    BOUND_MODE,
    EARLY_STOP_GAP,
    PATIENCE,
    PLOT,
)
from utils.tsp import TSPInstance


def solve_with_gurobi(
    instance: TSPInstance,
    time_limit: Optional[float] = None,
    mip_gap: float = 0.01,
    threads: Optional[int] = None,
    use_mtz: bool = False,
    use_callback: bool = False,
    use_hyper_heuristic: bool = False
):
    """Solve TSP instance using Gurobi solver."""
    callback = None
    if use_hyper_heuristic:
        callback = hyper_callback
    elif use_callback:
        callback = subtour_callback
    config = SolverConfig(
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        verbose=True,
        callback=callback,
        use_mtz=use_mtz
    )
    solver = GurobiTSPSolver(config)
    solution = solver.solve(instance)
    return solution


def train_q_learning(
    coords: np.ndarray,
    episodes: int = 250,
    steps_per_episode: int = 150,
    bound_mode: str = "mst",
    early_stop_gap: float = 0.01,
    patience: int = 35,
    seed: Optional[int] = None
):
    """Train Q-learning hyper-heuristic agent."""
    coords_list = [(float(coord[0]), float(coord[1])) for coord in coords]
    (
        agent,
        history,
        init_len,
        init_tour,
        best_tour,
        lb
    ) = train_qhh_with_early_stopping(
        coords_list,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        bound_mode=bound_mode,
        early_stop_gap=early_stop_gap,
        patience=patience,
        verbose=True,
        seed=seed
    )
    print("\nTraining completed!")
    print(f"Initial tour length: {init_len:.2f}")
    print(f"Final tour length: {tour_length(best_tour, coords_list):.2f}")
    print(f"Lower bound: {lb:.2f}")
    print(
        f"Gap: {((tour_length(best_tour, coords_list) - lb) / lb * 100):.2f}%"
    )

    # Evaluate agent
    eval_res, base_len, base_tour = evaluate_agent(
        agent, coords_list, episodes=20, steps_per_episode=200, seed=seed
    )

    print("\nEvaluation results:")
    print(f"Mean tour length: {eval_res.mean():.2f}")
    print(f"Std tour length: {eval_res.std():.2f}")
    print(f"Best tour length: {eval_res.min():.2f}")

    return agent, history, init_tour, best_tour


def main():
    """Main entry point."""
    # Set random seed
    np.random.seed(SEED)

    # Load or generate instance
    if INSTANCE:
        print(f"Loading instance from {INSTANCE}")
        instance = TSPInstance.from_tsplib(INSTANCE)
        if instance is None:
            print(f"Failed to load instance from {INSTANCE}")
            return
        coords = None
    elif RANDOM:
        print(f"Generating random instance with {RANDOM} cities")
        coords = np.random.rand(RANDOM, 2) * 100
        instance = TSPInstance.from_coordinates(
            f"random_{RANDOM}",
            coords
        )
    else:
        # Default: random 30 cities
        print("Generating default random instance with 30 cities")
        coords = np.random.rand(30, 2) * 100
        instance = TSPInstance.from_coordinates("random_30", coords)

    print(f"\nInstance: {instance.name}")
    print(f"Cities: {instance.n_cities}")

    # Execute based on mode
    if MODE == 'gurobi':
        print("\n" + "=" * 60)
        print("Solving with Gurobi")
        print("=" * 60)
        solution = solve_with_gurobi(
            instance,
            time_limit=TIME_LIMIT,
            mip_gap=MIP_GAP,
            threads=THREADS,
            use_mtz=USE_MTZ,
            use_callback=USE_CALLBACK,
            use_hyper_heuristic=USE_HYPER_HEURISTIC
        )

        print("\nSolution:")
        print(f"Status: {solution.status}")
        print(f"Cost: {solution.cost:.2f}")
        print(f"Gap: {solution.gap * 100:.2f}%")
        print(f"Time: {solution.solve_time:.2f}s")
        if PLOT and solution.tour:
            coords_for_plot = coords if coords is not None else np.array([
                [0, 0] for _ in range(instance.n_cities)
            ])
            plot_tour(
                coords_for_plot.tolist(),
                solution.tour,
                f"Gurobi Solution - {instance.name}"
            )

    elif MODE == 'qlearning':
        print("\n" + "=" * 60)
        print("Training Q-learning hyper-heuristic")
        print("=" * 60)
        if coords is None:
            print(
                "Error: Q-learning requires coordinates. "
                "Set RANDOM in settings.py to generate an instance."
            )
            return

        agent, history, init_tour, best_tour = train_q_learning(
            coords,
            episodes=EPISODES,
            steps_per_episode=STEPS,
            bound_mode=BOUND_MODE,
            early_stop_gap=EARLY_STOP_GAP,
            patience=PATIENCE,
            seed=SEED
        )

        if PLOT:
            plot_tour(
                coords.tolist(),
                init_tour,
                f"Initial Tour - {instance.name}"
            )
            plot_tour(
                coords.tolist(),
                best_tour,
                f"Final Tour - {instance.name}"
            )

    elif MODE == 'compare':
        print("\n" + "=" * 60)
        print("Comparing Gurobi and Q-learning")
        print("=" * 60)
        if coords is None:
            print(
                "Error: Comparison requires coordinates. "
                "Set RANDOM in settings.py to generate an instance."
            )
            return

        # Solve with Gurobi
        print("\n--- Gurobi Solution ---")
        solution = solve_with_gurobi(
            instance,
            time_limit=TIME_LIMIT or 60,
            mip_gap=MIP_GAP,
            threads=THREADS
        )
        print(f"Gurobi cost: {solution.cost:.2f}")

        # Train Q-learning
        print("\n--- Q-learning Solution ---")
        agent, history, init_tour, best_tour = train_q_learning(
            coords,
            episodes=EPISODES,
            steps_per_episode=STEPS,
            bound_mode=BOUND_MODE,
            seed=SEED
        )
        coords_list = coords.tolist()
        q_cost = tour_length(best_tour, coords_list)
        print(f"Q-learning cost: {q_cost:.2f}")

        print("\n--- Comparison ---")
        print(f"Gurobi: {solution.cost:.2f}")
        print(f"Q-learning: {q_cost:.2f}")
        print(
            f"Difference: {abs(solution.cost - q_cost):.2f} "
            f"({abs(solution.cost - q_cost) / solution.cost * 100:.2f}%)"
        )


if __name__ == '__main__':
    main()
