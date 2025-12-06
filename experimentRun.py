import os
import json
import argparse
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from Q_learning_ILS import Q_ILS 
from instance import TSPDataset, RandomTSPInstance

# --- WORKER FUNCTION ---
def process_instance(tsp_instance, output_dir):
    """
    Processes a single TSP instance.
    Args:
        tsp_instance: The data instance.
        output_dir (str): The directory where results should be saved.
    """
    # Construct the full output path dynamically
    # e.g., "data/train/EUC_2D/instance_123.txt"
    out_name = os.path.join(output_dir, f"{tsp_instance.name}.txt")
    
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
        opt_cost = None 

    # Initialize Solver
    q_ils = Q_ILS(tsp_instance)
    
    # Generate transitions
    q_ils.generate_transition(max_iter=50, opt_cost=opt_cost, out_path=out_name)
    
    return f"Processed {tsp_instance.name}"

# --- MAIN BLOCK ---
if __name__ == "__main__":
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Run TSP Training on Distributed Splits")
    
    # Argument for the Split File (e.g., data/split_computer_1.json)
    parser.add_argument(
        "--split_path", 
        type=str, 
        required=True, 
        help="Path to the JSON file containing the train/test splits."
    )
    
    # Argument for the Output Directory (e.g., data/train/EUC_2D/)
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory where output .txt files will be saved."
    )

    # Argument for the Dataset File (e.g., data/EUC_2D.json)
    # We default this so you don't always have to type it if it rarely changes
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="data/EUC_2D.json",
        help="Path to the original dataset JSON file."
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load the specific split
    print(f"Loading split from: {args.split_path}")
    with open(args.split_path, "r") as f:
        splits = json.load(f)

    # Note: We use args.dataset_path as the KEY to look up in the split file
    # Ensure your split file structure matches this key
    dataset_key = args.dataset_path 
    if dataset_key not in splits:
        # Fallback: sometimes split files just contain the dict directly without the filename key
        # Check if 'train' exists at the top level
        if "train" in splits:
             train_ids = splits["train"]
        else:
             raise KeyError(f"Could not find key '{dataset_key}' or 'train' in split file.")
    else:
        train_ids = splits[dataset_key]["train"]

    # 3. Initialize the dataset
    print(f"Loading dataset: {args.dataset_path} with {len(train_ids)} instances...")
    train_set = TSPDataset(args.dataset_path, train_ids)

    # 4. Multiprocessing Setup
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"\n--- Starting Training Loop on {num_cores} cores ---")
    print(f"Saving results to: {args.output_dir}\n")

    # Create a partial function to pass the constant 'output_dir' to every worker
    worker_func = partial(process_instance, output_dir=args.output_dir)

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Map the partial function (which already has output_dir) to the train_set
        results = list(
            executor.map(worker_func, train_set)
        )
        
    print("Done.")