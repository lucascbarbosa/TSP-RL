"""Main script for TSP-RL project."""
import argparse
import numpy as np
from typing import Optional
from utils.gurobi import (
    GurobiTSPSolver,
    SolverConfig,
    subtour_callback,
    hyper_callback,
)
from utils.q import (
    evaluate_agent,
    plot_tour,
    tour_length,
    train_qhh_with_early_stopping,
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
    parser = argparse.ArgumentParser(
        description='TSP-RL: TSP solving with Reinforcement Learning'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['gurobi', 'qlearning', 'compare'],
        default='gurobi',
        help='Solving mode'
    )
    parser.add_argument(
        '--instance',
        type=str,
        help='Path to TSPLIB instance file',
        default='data/tsplib/gr17.tsp'
    )
    parser.add_argument(
        '--random',
        type=int,
        metavar='N',
        help='Generate random instance with N cities',
        default=30
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )

    # Gurobi options
    parser.add_argument(
        '--time-limit',
        type=float,
        help='Time limit in seconds',
        default=60
    )
    parser.add_argument(
        '--mip-gap',
        type=float,
        help='MIP gap (default: 0.01)',
        default=0.01
    )
    parser.add_argument(
        '--threads',
        type=int,
        help='Number of threads',
        default=1
    )
    parser.add_argument(
        '--use-mtz',
        action='store_true',
        help='Use MTZ constraints',
        default=False
    )
    parser.add_argument(
        '--use-callback',
        action='store_true',
        help='Use subtour elimination callback',
        default=False
    )
    parser.add_argument(
        '--use-hyper-heuristic',
        action='store_true',
        help='Use hyper-heuristic callback',
        default=False
    )

    # Q-learning options
    parser.add_argument(
        '--episodes',
        type=int,
        help='Number of training episodes',
        default=250
    )
    parser.add_argument(
        '--steps',
        type=int,
        help='Steps per episode',
        default=150
    )
    parser.add_argument(
        '--bound-mode',
        type=str,
        choices=['mst', '1tree', 'concorde'],
        help='Lower bound mode for early stopping',
        default='mst'
    )
    parser.add_argument(
        '--early-stop-gap',
        type=float,
        help='Early stopping gap threshold',
        default=0.01
    )
    parser.add_argument(
        '--patience',
        type=int,
        help='Early stopping patience',
        default=35
    )

    # Output options
    parser.add_argument('--plot',
        action='store_true',
        help='Plot results',
        default=False
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load or generate instance
    if args.instance:
        print(f"Loading instance from {args.instance}")
        instance = TSPInstance.from_tsplib(args.instance)
        if instance is None:
            print(f"Failed to load instance from {args.instance}")
            return
        coords = None
    elif args.random:
        print(f"Generating random instance with {args.random} cities")
        coords = np.random.rand(args.random, 2) * 100
        instance = TSPInstance.from_coordinates(
            f"random_{args.random}",
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
    if args.mode == 'gurobi':
        print("\n" + "=" * 60)
        print("Solving with Gurobi")
        print("=" * 60)
        solution = solve_with_gurobi(
            instance,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            threads=args.threads,
            use_mtz=args.use_mtz,
            use_callback=args.use_callback,
            use_hyper_heuristic=args.use_hyper_heuristic
        )

        print("\nSolution:")
        print(f"Status: {solution.status}")
        print(f"Cost: {solution.cost:.2f}")
        print(f"Gap: {solution.gap * 100:.2f}%")
        print(f"Time: {solution.solve_time:.2f}s")
        if args.plot and solution.tour:
            coords_for_plot = coords if coords is not None else np.array([
                [0, 0] for _ in range(instance.n_cities)
            ])
            plot_tour(
                coords_for_plot.tolist(),
                solution.tour,
                f"Gurobi Solution - {instance.name}"
            )

    elif args.mode == 'qlearning':
        print("\n" + "=" * 60)
        print("Training Q-learning hyper-heuristic")
        print("=" * 60)
        if coords is None:
            print(
                "Error: Q-learning requires coordinates. "
                "Use --random to generate an instance."
            )
            return

        agent, history, init_tour, best_tour = train_q_learning(
            coords,
            episodes=args.episodes,
            steps_per_episode=args.steps,
            bound_mode=args.bound_mode,
            early_stop_gap=args.early_stop_gap,
            patience=args.patience,
            seed=args.seed
        )

        if args.plot:
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

    elif args.mode == 'compare':
        print("\n" + "=" * 60)
        print("Comparing Gurobi and Q-learning")
        print("=" * 60)
        if coords is None:
            print(
                "Error: Comparison requires coordinates. "
                "Use --random to generate an instance."
            )
            return

        # Solve with Gurobi
        print("\n--- Gurobi Solution ---")
        solution = solve_with_gurobi(
            instance,
            time_limit=args.time_limit or 60,
            mip_gap=args.mip_gap,
            threads=args.threads
        )
        print(f"Gurobi cost: {solution.cost:.2f}")

        # Train Q-learning
        print("\n--- Q-learning Solution ---")
        agent, history, init_tour, best_tour = train_q_learning(
            coords,
            episodes=args.episodes,
            steps_per_episode=args.steps,
            bound_mode=args.bound_mode,
            seed=args.seed
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
