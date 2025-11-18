"""Settings and environment variable management for TSP-RL project."""
import os


# Default configuration values
DEFAULT_TIME_LIMIT = 60
DEFAULT_MIP_GAP = 0.01
DEFAULT_THREADS = 4
DEFAULT_VERBOSE = True

# Main script configuration
MODE = 'gurobi'  # Options: 'gurobi', 'qlearning', 'compare'
INSTANCE = ''  # Path to TSPLIB instance file
RANDOM = 30  # Generate random instance with N cities
SEED = 42  # Random seed

# Gurobi options
WLSACCESSID = os.getenv('WLSACCESSID')
WLSSECRET = os.getenv('WLSSECRET')
LICENSEID = os.getenv('LICENSEID')
TIME_LIMIT = 60.0  # Time limit in seconds
MIP_GAP = 0.01  # MIP gap
THREADS = 1  # Number of threads
USE_MTZ = False  # Use MTZ constraints
USE_CALLBACK = False  # Use subtour elimination callback
USE_HYPER_HEURISTIC = False  # Use hyper-heuristic callback

# Q-learning options
EPISODES = 250  # Number of training episodes
STEPS = 150  # Steps per episode
BOUND_MODE = 'mst'  # Lower bound mode: 'mst', '1tree', 'concorde'
EARLY_STOP_GAP = 0.01  # Early stopping gap threshold
PATIENCE = 35  # Early stopping patience

# Output options
PLOT = False  # Plot results
