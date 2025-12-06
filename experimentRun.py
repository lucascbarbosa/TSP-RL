import os
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from Q_learning_ILS import Q_ILS
from instance import TSPDataset, RandomTSPInstance

# --- WORKER FUNCTION ---
# This function must be defined at the top level to be "picklable"
def process_instance(tsp_instance):
    """
    Processes a single TSP instance: calculates opt_cost and runs Q_ILS.
    """
    out_name = f"data/train/EUC_2D/{tsp_instance.name}.txt"
    
    # Skip if already exists
    if os.path.isfile(out_name):
        return f"Skipped {tsp_instance.name}"

    # Calculate optimal cost
    opt_tour = tsp_instance.opt_tour
    if opt_tour:
        opt_cost = sum(
            tsp_instance.get_weight(opt_tour[i], opt_tour[(i + 1) % len(opt_tour)])
            for i in range(len(opt_tour))
        )
    else:
        # Handle cases where opt_tour might be None if your dataset allows it
        opt_cost = None 

    # Initialize Solver
    q_ils = Q_ILS(tsp_instance)
    
    # Generate transitions
    # Since each process writes to a UNIQUE file (out_name), 
    # we can let the worker write directly to disk safely.
    q_ils.generate_transition(max_iter=50, opt_cost=opt_cost, out_path=out_name)
    
    return f"Processed {tsp_instance.name}"

# --- MAIN BLOCK ---
if __name__ == "__main__":
    # 1. Load your index splits
    with open("data/splits.json", "r") as f:
        splits = json.load(f)

    filename = "data/EUC_2D.json"
    train_ids = splits[filename]["train"]

    # 2. Initialize the dataset
    # (Data is loaded into RAM in the main process, then copied to workers)
    print("Loading dataset...")
    train_set = TSPDataset(filename, train_ids)

    # 3. Multiprocessing Setup
    # Reserve 1 or 2 cores for the OS, use the rest for processing
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"\n--- Starting Training Loop on {num_cores} cores ---")

    # We use tqdm to wrap the executor to show a progress bar
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # submit all tasks to the pool
        results = list(
            executor.map(process_instance, train_set)
        )

    print("Processing complete.")