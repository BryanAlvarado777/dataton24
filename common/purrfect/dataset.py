import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from purrfect.paths import PARTITIONS_PATH,BASE_PROYECT_PATH,DATASET_PATH
import glob
def create_loader_from_partition_name(partition_name,partition_path,shuffle=True,batch_size=16,transform=None):
    partition = load_partition(partition_name,partitions_path=partition_path)
    dataset = EPKADataset(partition,transform= transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def create_train_valid_loaders(name_train_partition,name_val_partition,partition_path,batch_size=16,transform=None):
    train_loader = create_loader_from_partition_name(name_train_partition,partition_path, shuffle=True,batch_size=batch_size,transform=transform)
    val_loader = create_loader_from_partition_name(name_val_partition,partition_path, shuffle=False,batch_size=batch_size,transform=None)
    return train_loader, val_loader

def save_partition(partition_name,partition_path,partition):
    with open(os.path.join(partition_path,partition_name), "w") as f:
        json.dump(partition, f)
def load_partition(partition_name,partitions_path = PARTITIONS_PATH):
    """
    Load a partition from a JSON file and return a list of (input, target) tuples.

    :param partition_name: The name of the partition (e.g., 'partition_1.json').
    :return: A list of tuples where each tuple is (input_path, target_path).
    """
    partition_path = os.path.join(partitions_path, partition_name)
    # Open and load the JSON file containing the partition data
    with open(partition_path, "r") as f:
        data = json.load(f)

    # Return the list of tuples (input_path, target_path)
    return [(os.path.abspath(os.path.join(BASE_PROYECT_PATH,input_path)),os.path.abspath(os.path.join(BASE_PROYECT_PATH,target_path))) for input_path, target_path in data]

class TestDataset(Dataset):
    def __init__(self) -> None:
        self.partition = sorted(glob.glob(os.path.join(DATASET_PATH,"test_public/*.npy")))
    def __len__(self):
        return len(self.partition)
    def __getitem__(self, index):
        EP = np.load(self.partition[index])
        EP = torch.from_numpy(EP).float()
        return self.partition[index].split("/")[-1],EP

from torchvision.transforms import v2

class RandomTransform:
    def __init__(self, rotation_degrees=90, crop_scale=(0.8, 1.0), crop_ratio=(1.0, 1.0), image_size=(128, 128)):
        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(),  # Horizontal flip
            v2.RandomVerticalFlip(),    # Vertical flip
            v2.RandomResizedCrop(size=image_size, scale=crop_scale, ratio=crop_ratio)
        ])

    def __call__(self, x):
        return self.transform(x)
    
class EPKADataset(Dataset):
    def __init__(self, partition, transform=None):
        """
        Args:
            dataset_numbers (list): List of dataset numbers to load (e.g., [1, 2, 3, ..., 8])
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.data_pairs = partition

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        # Load input (epsilon) and target (kappa) data
        EP_pth, KA_pth = self.data_pairs[idx]
        EP = np.load(EP_pth)  # Input
        KA = np.load(KA_pth)  # Target

        # Convert numpy arrays to PyTorch tensors
        EP = torch.from_numpy(EP).float()  # Convert to float tensor
        KA = torch.from_numpy(KA).unsqueeze(0).float()  # Convert to float tensor

        # Check if the transform is provided
        if self.transform:
            seed = torch.randint(0, 2**32, (1,)).item()  # Generate a random seed
            torch.manual_seed(seed)  # Set seed for EP
            EP = self.transform(EP)

            torch.manual_seed(seed)  # Reuse the same seed for KA to ensure identical transforms
            KA = self.transform(KA)

        return EP, KA
