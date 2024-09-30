from purrfect.metrics import MetricAccumulator
from purrfect.dataset import EPKADataset,save_partition,load_partition,create_loader_from_partition_name
from torch.utils.data import DataLoader
from purrfect.training import train_validate
from sklearn.model_selection import train_test_split
import random
import numpy as np
from purrfect.paths import PARTITIONS_PATH
def _create_new_partition(origin_partition,candidate_partition,candidate_metrics,new_partition_name,new_partition_path):
    total_ranks = np.zeros(len(candidate_partition))
    for key, value in candidate_metrics.items():
        arr = np.array(value)
        sorted_indices = np.argsort(-arr)  # Sort indices in descending order
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(arr) + 1)  # Assign ranks
        total_ranks += ranks

    candidates = np.array(candidate_partition)[np.argsort(total_ranks)].tolist()
    candidates = [(x[0],x[1]) for x in candidates]
    new_partition = join_lists(origin_partition,candidates)
    save_partition(new_partition_name,new_partition_path,new_partition)

def join_lists(list1, list2):
    # Create a local random number generator
    rng = random.Random(42)  # Use a fixed seed for reproducibility
    
    # Calculate half of the first list
    half_len1 = len(list1) // 2
    
    # Get the first half of the second list
    half_len2 = len(list2) // 2
    first_half_list2 = list2[:half_len2]
    
    # Randomly sample half of the first list using the local RNG
    random_half_list1 = rng.sample(list1, half_len1)
    
    # Combine the two parts
    new_list = random_half_list1 + first_half_list2
    
    return new_list

def create_new_partition(model,criterion, prev_partition, next_partition,new_partition_name,new_partition_path,device="cuda",epoch=None):
    metric_accumulator = MetricAccumulator.create_default()
    next_dataset = EPKADataset(next_partition)
    next_loader = DataLoader(next_dataset, batch_size=16, shuffle=False)

    train_validate(
        model,
        next_loader,
        criterion,
        optimizer=None,
        epoch=epoch,
        metric_accumulator=metric_accumulator,
        device=device,
    )
    _create_new_partition(prev_partition,next_partition,metric_accumulator.values,new_partition_name,new_partition_path)

def create_next_partitions(prev_partition_number,model,criterion,device="cuda"):
    next_partition_number = prev_partition_number+1
    next_partition_total = load_partition(f"partition_{next_partition_number}.json")
    next_train_partition, next_val_partition = train_test_split(next_partition_total, test_size=0.2, random_state=42)

    prev_partition = load_partition(f"partition_{prev_partition_number}_train.json",partitions_path="partitions")
    next_partition = next_train_partition
    create_new_partition(model,criterion, prev_partition, next_partition,f"partition_{next_partition_number}_train.json","partitions",device=device,epoch=f"partition_{next_partition_number}_train")

    prev_partition = load_partition(f"partition_{prev_partition_number}_val.json",partitions_path="partitions")
    next_partition = next_val_partition
    create_new_partition(model,criterion, prev_partition, next_partition,f"partition_{next_partition_number}_val.json","partitions",device=device,epoch=f"partition_{next_partition_number}_val")
def test_model(model,criterion,device="cuda",batch_size=16,experiment_name="experiment_1"):
    metric_accumulator = MetricAccumulator.create_default()
    test_loader = create_loader_from_partition_name("partition_test.json",PARTITIONS_PATH,shuffle=False,batch_size=batch_size)
    train_validate(
        model,
        test_loader,
        criterion,
        optimizer=None,
        epoch="test",
        metric_accumulator=metric_accumulator,
        device=device,
        use_autocast=False
    )
    with open("test_metrics.txt", "a") as f:
        f.write(f"{experiment_name}*{metric_accumulator.get_metrics_str()}\n")
