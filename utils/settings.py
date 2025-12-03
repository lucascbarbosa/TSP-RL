"""Settings and environment variable management for TSP-RL project."""
import os


# Q-learning options
EPISODES = 250  # Number of training episodes
STEPS = 150  # Steps per episode
BOUND_MODE = 'mst'  # Lower bound mode: 'mst', '1tree', 'concorde'
EARLY_STOP_GAP = 0.01  # Early stopping gap threshold
PATIENCE = 35  # Early stopping patience

# Output options
PLOT = False  # Plot results
