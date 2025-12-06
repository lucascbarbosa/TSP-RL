import json
import random
import os

def generate_tsp_splits(json_files, output_file="tsp_splits.json", train_ratio=0.8, seed=42):
    """
    Reads multiple JSON files containing TSP instances, generates 
    train/test index splits, and saves them to a single JSON file.
    """
    random.seed(seed)
    
    # Structure to hold the splits
    # Format: { "filename.json": { "train": [id1, id2...], "test": [id3, id4...] } }
    all_splits = {}

    print(f"Generating splits (Train: {train_ratio:.0%}, Test: {1-train_ratio:.0%})...")

    for file_path in json_files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue

        print(f"Processing {file_path}...")
        
        # 1. Load the file efficiently just to get the length
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                num_instances = len(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # 2. Create indices [0, 1, ..., N-1]
        indices = list(range(num_instances))
        
        # 3. Shuffle
        random.shuffle(indices)

        # 4. Split
        split_point = int(num_instances * train_ratio)
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

        # 5. Store in dictionary
        # We use the filename as key
        all_splits[file_path] = {
            "total_instances": num_instances,
            "train": train_indices,
            "test": test_indices
        }
        
        print(f"  -> Found {num_instances} instances.")
        print(f"  -> Train: {len(train_indices)}, Test: {len(test_indices)}")

    # 6. Save the split manifest
    with open(output_file, "w") as f:
        json.dump(all_splits, f, indent=4)
    
    print(f"\nSuccess! Splits saved to {output_file}")

# ==========================================
# Configuration
# ==========================================
if __name__ == "__main__":
    # List your 3 json files here
    MY_FILES = [
        "data/EUC_2D.json",
        "data/ATT.json",
        "data/GEO.json"
    ]
    
    generate_tsp_splits(MY_FILES, output_file="data/splits.json", train_ratio=0.7)
