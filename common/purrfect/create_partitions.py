import glob
import numpy as np
import os
import json

# Function to create and store partitions
def create_partitions():
    # Number of partitions (95 partitions of 1% each, and 1 partition of 5%)
    num_partitions = 95
    test_percentage = 0.05
    save_dir = "partitions/"
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare lists to store the paths for each partition
    partitions = [[] for _ in range(num_partitions + 1)]  # 95 partitions + 1 test partition
    
    # Loop through each dataset folder (from dataset_number 1 to 8)
    for dataset_number in range(1, 9):
        input_paths = sorted(glob.glob(f"dataset/{dataset_number}/EPSILON/*.npy"))
        target_paths = sorted(glob.glob(f"dataset/{dataset_number}/KAPPA/*.npy"))
        
        # Ensure input and target paths align
        assert len(input_paths) == len(target_paths), f"Mismatch between input and target files in dataset {dataset_number}"
        
        total_size = len(input_paths)
        one_percent_size = total_size // 100  # 1% of the data
        test_size = total_size * 5 // 100  # 5% for the test partition
        
        # Shuffle the data
        data = list(zip(input_paths, target_paths))
        np.random.shuffle(data)
        
        # Distribute 1% of the data to the first 95 partitions
        for i in range(num_partitions):
            partition_data = data[i * one_percent_size:(i + 1) * one_percent_size]
            partitions[i].extend(partition_data)
        
        # Distribute the remaining 5% of the data to the test partition
        remaining_data = data[num_partitions * one_percent_size:]
        partitions[-1].extend(remaining_data)
    
    # Save each partition to its corresponding JSON file
    for i in range(num_partitions):
        partition_file = os.path.join(save_dir, f"partition_{i + 1}.json")
        with open(partition_file, "w") as f:
            json.dump(partitions[i], f)
    
    # Save the test partition (with the 5% of data from each dataset)
    test_partition_file = os.path.join(save_dir, "partition_test.json")
    with open(test_partition_file, "w") as f:
        json.dump(partitions[-1], f)

# Run the partitioning process
create_partitions()
