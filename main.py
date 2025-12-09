import numpy as np
import json
import os
import csv
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Import your custom modules
from solution import Solution
from Q_learning_ILS import Q_ILS
from instance import RandomTSPInstance

# --- CONFIGURATION ---
output_filename = "results.csv"
# Limit threads to avoid crashing memory (adjust based on your CPU cores)
MAX_WORKERS = 10

# Create a lock object to synchronize file writing
csv_lock = threading.Lock()


def get_processed_instances(filename):
    processed = set()
    if os.path.exists(filename):
        with open(filename, mode="r", newline="") as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            for row in reader:
                if row:
                    # Assuming full_id is in the first column (index 0)
                    processed.add(row[0])
    return processed


def initialize_csv(filename):
    """Creates the CSV and writes the header if it doesn't exist."""
    if not os.path.isfile(filename):
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Full ID", "Name", "ID", "Type", "Dimension", "Optimal Cost", "Best Cost", "Gap", "Time", "Best Tour"]
            )


def process_instance(args):
    """
    Worker function to solve a single instance.
    args is a tuple: (instance_id, instance_type, splits_data)
    """
    instance_id, instance_type, splits = args

    # Re-instantiate the problem inside the thread
    # Note: We pass the path/id logic here just like your original code
    try:
        problem = RandomTSPInstance(f"data/{instance_type}.json", instance_id=instance_id)

        # Calculate optimal cost
        opt_tour = problem.opt_tour
        opt_cost = sum(problem.get_weight(opt_tour[i], opt_tour[(i + 1) % len(opt_tour)]) for i in range(len(opt_tour)))

        # Setup ILS
        q_ils = Q_ILS(problem)
        q_table_path = f"data/q_tables/{instance_type}/instance_size_{problem.dimension}.txt"
        q_ils.load_qtable(q_table_path)

        # Run Algorithm
        timeStart = time.time()
        best_solution_q = q_ils.exec_q_table(max_iter=50, opt_cost=opt_cost, epsilon=0.1)
        execTime = time.time() - timeStart

        # Prepare Data
        gap_value = ((best_solution_q.cost - opt_cost) / opt_cost) * 100
        full_id = f"{instance_type}{instance_id}"

        row_data = [
            full_id,
            problem.name,
            instance_id,
            instance_type,
            problem.dimension,
            opt_cost,
            best_solution_q.cost,
            f"{gap_value:.4f}%",
            execTime,
            str(best_solution_q.tour),
        ]

        # --- CRITICAL SECTION: WRITE TO FILE ---
        # We lock this block so threads don't write over each other
        with csv_lock:
            with open(output_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row_data)

        # Print status (print is thread-safe, but output might interleave slightly)
        print(f"DONE: {full_id} | Gap: {gap_value:.2f}% | Time: {execTime:.2f}s")

    except Exception as e:
        print(f"ERROR processing {instance_type}-{instance_id}: {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load config and history
    processed = get_processed_instances(output_filename)
    initialize_csv(output_filename)  # Ensure header exists before threads start

    with open("data/splits.json", "r") as f:
        splits = json.load(f)

    # 2. Build the list of jobs (Todo List)
    jobs = []

    # @TODO add "EUC_2D" ?
    for instance_type in ["GEO", "ATT"]:
        eval_instances = splits[f"data/{instance_type}.json"]["test"]

        for instance in eval_instances:
            full_id = f"{instance_type}{instance}"

            # Skip if already done
            if full_id in processed:
                continue

            # Add to job list
            jobs.append((instance, instance_type, splits))

    print(f"Starting processing of {len(jobs)} instances with {MAX_WORKERS} threads...")

    # 3. Start the Thread Pool
    # This automatically manages the threads for you
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_instance, jobs)
