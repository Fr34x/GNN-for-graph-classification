import random
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def set_seed(seed=2025):
    """
    Ensure reproducibility by seeding all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ConstantNodeFeatures(object):
    """
    Transform to replace node features with a constant vector of 1.0 (for feature ablation).
    """
    def __call__(self, data):
        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
        return data

class NormalizeContinuousFeature(object):
    """
    Transform to normalize Feature 0 (continuous node attribute) in the dataset.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, data):
        data.x = data.x.clone()
        data.x[:, 0] = (data.x[:, 0] - self.mean) / (self.std + 1e-6)
        return data

def get_dataset(use_features=True):
    """
    Load the PROTEINS dataset.
    If use_features is False, node features are replaced with a constant vector of ones.
    """
    if use_features:
        dataset = TUDataset(root='data/TUDataset', name='PROTEINS', use_node_attr=True)
        # Calculate mean and std of Feature 0 across the entire dataset for z-score normalization
        feat0 = dataset.x[:, 0]
        mean = feat0.mean().item()
        std = feat0.std().item()
        dataset.transform = NormalizeContinuousFeature(mean, std)
    else:
        transform = ConstantNodeFeatures()
        dataset = TUDataset(root='data/TUDataset', name='PROTEINS', use_node_attr=True, transform=transform)
    return dataset

def get_dataloaders(dataset, batch_size=64, seed=2025):
    """
    Splits the dataset reproducibly into train (80%), validation (10%), and test (10%) splits.
    Returns PyG DataLoaders.
    """
    # Use PyTorch Generator for reproducible random_split
    g = torch.Generator().manual_seed(seed)
    num_graphs = len(dataset)
    train_len = int(0.8 * num_graphs)
    val_len = int(0.1 * num_graphs)
    test_len = num_graphs - train_len - val_len
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len], generator=g
    )
    
    # Use PyTorch Geometric DataLoader (handles batching of graph data structures)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
