import json
import random
import os
import re
import math

def load_ids_from_split(json_path, split_key='train', source_file='data/EUC_2D.json'):
    """
    Loads a specific list of IDs (e.g., training set) from a JSON split file.
    
    Args:
        json_path (str): Path to the .json file containing the splits.
        split_key (str): The key in the JSON dictionary that holds the IDs 
                         (e.g., 'train', 'train_idx', 'training').
                         
    Returns:
        list: A list of IDs found under that key.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"The file {json_path} does not exist.")
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    data = data[source_file]
    # Check if the data is a dictionary and contains the key
    if isinstance(data, dict):
        if split_key not in data:
            raise KeyError(f"Key '{split_key}' not found in JSON. Available keys: {list(data.keys())}")
        return data[split_key]
    
    # Fallback: If the JSON is just a simple list, return it entirely
    elif isinstance(data, list):
        print("JSON is a direct list (not a dict). Returning all items.")
        return data
    
    else:
        raise ValueError("JSON structure not recognized (must be dict or list).")


def get_distributed_remaining_splits(
    all_ids, 
    output_dir, 
    n_computers, 
    filename_pattern=r"text(\d+)text\d+"
):
    """
    Reads existing files, filters out completed IDs, and splits the remaining
    IDs into n chunks for distribution.
    
    Args:
        all_ids (list): A list of all IDs (strings or ints) intended to be processed.
        output_dir (str): Path to the folder containing already generated files.
        n_computers (int): The number of new splits you want to create.
        filename_pattern (str): Regex pattern to capture the ID from the filename. 
                                Default assumes format like 'text123text456'.
                                The first capture group (...) must be the ID.
    
    Returns:
        list of lists: A list containing n lists of IDs.
    """
    
    # 1. Identify what has already been done
    processed_ids = set()
    
    try:
        existing_files = os.listdir(output_dir)
    except FileNotFoundError:
        print(f"Directory {output_dir} not found. Assuming no files generated yet.")
        existing_files = []

    for filename in existing_files:
        # Skip hidden files or non-files
        if filename.startswith('.'): 
            continue
            
        match = re.search(filename_pattern, filename)
        if match:
            # We assume the ID is the first group captured by the regex
            extracted_id = match.group(1) 
            processed_ids.add(str(extracted_id))
    
    print(f"Found {len(processed_ids)} already processed instances.")

    # 2. Filter the original list
    # We convert id to string to ensure matching works regardless of input type
    remaining_ids = [
        x for x in all_ids 
        if str(x) not in processed_ids
    ]
    
    total_remaining = len(remaining_ids)
    print(f"Remaining instances to process: {total_remaining}")
    
    if total_remaining == 0:
        return [[] for _ in range(n_computers)]

    # 3. Create N new splits (Chunking)
    # We use a math.ceil approach to distribute roughly evenly
    chunk_size = math.ceil(total_remaining / n_computers)
    
    new_splits = []
    for n, i in enumerate(range(0, total_remaining, chunk_size)):
        sp = remaining_ids[i:i + chunk_size]
        new_splits.append(sp)

        out_splits = dict()
        out_splits[file_path] = {
            "total_instances": len(sp),
            "train": sp
        }
        with open(f'data/split_computer_{n}.json', 'w') as f:
            json.dump(out_splits, f, indent=4)
    
    # Ensure we return exactly n_computers lists (even if empty)
    while len(new_splits) < n_computers:
        new_splits.append([])
        
    return new_splits


def generate_tsp_splits(json_files, output_file="splits.json", train_ratio=0.8, seed=42):
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
    file_path = "data/EUC_2D.json"
    #generate_tsp_splits(MY_FILES, output_file="data/splits.json", train_ratio=0.7)

    all_train_ids = load_ids_from_split('data/splits.json', split_key='train')


    distributed_work = get_distributed_remaining_splits(
        all_ids=all_train_ids,
        output_dir="data/train/EUC_2D", 
        n_computers=3,
        # Adjust this regex to match your EXACT file string structure.
        # The (\d+) part captures the ID.
        filename_pattern=r"random_instance_(\d+)_nodes_\d+" 
    )