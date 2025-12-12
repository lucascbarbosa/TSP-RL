"""DQN environment wrapper for Q-ILS operators.

TODO: Implement when ready for DQN training.

The DQNEnv will wrap the existing TSP operators:
- CONSTRUCTIVES for initial solution
- PERTURBATIONS for solution modification
- LOCAL_SEARCHES for improvement

Episode structure:
1. Reset: Sample instance, generate initial solution
2. Step: Apply (perturbation, local_search), return (next_state, reward, done)
3. Done when time budget exhausted
"""

# Placeholder for DQNEnv implementation

__all__: list[str] = []
