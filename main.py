import numpy as np
from solution import Solution
from Q_learning_ILS import Q_ILS
from instance import RandomTSPInstance
import json
import os
import csv
import time


output_filename = "results.csv"


with open("data/splits.json", "r") as f:
    splits = json.load(f)

# List all available instance files
for instance_type in ["EUC_2D", "GEO", "ATT"]:
    eval_instances = splits[f"data/{instance_type}.json"]["test"]

    for instance in eval_instances:
        problem = RandomTSPInstance(f"data/{instance_type}.json", instance_id=instance)

        # tour ótimo das instâncias geradas aleatoriamente
        opt_tour = problem.opt_tour  # solução ótima conhecida (se existir)

        # custo da solução ótima (usando o próprio problem)
        opt_cost = sum(
            problem.get_weight(opt_tour[i], opt_tour[(i + 1) % len(opt_tour)])
            for i in range(len(opt_tour))
        )

        q_ils = Q_ILS(problem)

        # Codigo abaixo roda o ILS, usando a q-table treinada (Q-ILS)
        q_table_path = f"data/q_tables/{instance_type}/instance_size_{problem.dimension}.txt"
        q_ils.load_qtable(q_table_path)
        timeStart = time.time()
        best_solution_q = q_ils.exec_q_table(max_iter=50, opt_cost=opt_cost, epsilon=0.1)
        execTime = time.time() - timeStart

        # Write data

        # Calculate the gap variable explicitly so we can format it
        gap_value = ((best_solution_q.cost - opt_cost) / opt_cost) * 100

        # Prepare the data as a list.
        # Note: We convert the list/tour to a string to keep it in a single CSV cell.
        row_data = [
            problem.name,                        # Problem name
            instance,                        # Problem id (You used problem.name for both in your snippet)
            instance_type,                       # Problem type
            problem.dimension,                   # Number of nodes
            opt_cost,                            # Custo ótimo
            best_solution_q.cost,                # Best solution cost
            f"{gap_value:.4f}%",                 # Gap to optimal (formatted as string with %)
            execTime,                             # Exec Time
            str(best_solution_q.tour)           # Best solution tour (converted to string)
        ]

        # Check if file exists (to decide whether to write headers)
        file_exists = os.path.isfile(output_filename)

        # Open in 'a' (append) mode. newline='' is required to prevent blank lines on Windows
        with open(output_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header only if the file is new
            if not file_exists:
                writer.writerow([
                    "Name", 
                    "ID", 
                    "Type", 
                    "Dimension", 
                    "Optimal Cost",
                    "Best Cost", 
                    "Gap", 
                    "Time", 
                    "Best Tour"
                ])
            
            # Write the data row
            writer.writerow(row_data)
        
        print(f"Problem name: {problem.name}")
        print(f"Problem id: {instance}")
        print(f"Problem type: {instance_type}")
        print(f"Number of nodes/cities: {problem.dimension}")
        print("Custo ótimo:", opt_cost)
        print("Best solution tour:", best_solution_q.tour)
        print("Best solution cost:", best_solution_q.cost)
        print(f"Gap to optimal: {gap_value}%")
        print(f"Exec Time: {execTime}")

