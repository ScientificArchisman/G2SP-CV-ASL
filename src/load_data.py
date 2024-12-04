import torch
import numpy as np
from scipy.sparse import load_npz
import os
from torch.utils.data import Dataset, DataLoader, random_split
import yaml 
from tqdm import tqdm

def load_numpy_obj(file_path: str, extension: str, dtype = torch.float32) -> torch.Tensor:
    """Load a numpy object into a torch tensor
    Args:
        file_path: str, path to the file
        extension: str, file extension to load. Eithere  of .npy or .npz
        dtype: torch.dtype, data type to load the data
    Returns:
        matrix: torch.Tensor, tensor with the data loaded"""
    if extension == ".npy":
        matrix = torch.tensor(np.load(file_path), dtype = dtype)
    elif extension == ".npz":
        matrix = torch.tensor(load_npz(file_path).toarray(), dtype = dtype)
    return matrix



# Collect all paths and labels withouth loading the data
def load_data(features_dir, adj_matrix_dir):
    label_mapping = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "J": 9, 
            "K": 10,
            "L": 11,
            "M": 12,
            "N": 13,
            "O": 14,
            "P": 15,
            "Q": 16,
            "R": 17,
            "S": 18,
            "T": 19,
            "U": 20,
            "V": 21,
            "W": 22,
            "X": 23,
            "Y": 24,
            "Z": 25,
            "del": 26,
            "nothing": 27,
            "space": 28
        }
    adj_matrices, features, labels = [], [], []
    loop = tqdm(sorted(os.listdir(features_dir)))
    for label in loop:
            adj_label_dir = os.path.join(adj_matrix_dir, label)
            features_label_dir = os.path.join(features_dir, label)
            if os.path.isdir(adj_label_dir):
                for adj_matrix_file, feature_file in zip(sorted(os.listdir(adj_label_dir)), sorted(os.listdir(features_label_dir))):
                    adj_path = os.path.join(adj_label_dir, adj_matrix_file)
                    feature_path = os.path.join(features_label_dir, feature_file)
                    adj_matrices.append(load_numpy_obj(adj_path, extension = ".npz"))
                    features.append(load_numpy_obj(feature_path, extension = ".npy"))
                    labels.append(label_mapping[label])
    return adj_matrices, features, labels



class GraphDataset(Dataset):
    def __init__(self, adj_matrices, features, labels, transform = None):
        """ Initialize the dataset
        Args:
            adj_matrix_dir: str, path to the directory containing the adjacency matrices
            features_dir: str, path to the directory containing the features
            transform: callable, transformation to apply to the data
        Returns:
            (adj_matrix, feature_matrix, label)"""
        
        super().__init__()
        self.adj_matrices = adj_matrices
        self.features = features
        self.labels = labels
        self.transform = transform



    def __len__(self):
        return len(self.adj_matrices)
    
    
    def __getitem__(self, idx):
        
        # Load the adjacency matrix and feature matrix as float tensors
        adj_matrix, feature_matrix, label = self.adj_matrices[idx], self.features[idx], self.labels[idx]
        
        # Apply transformations if any
        if self.transform:
            adj_matrix, feature_matrix = self.transform(adj_matrix, feature_matrix)
    
        return adj_matrix, feature_matrix, label
    

def load_dataloader(adj_matrices: list, features: list, labels: list, config: dict):
    """ Load the data as dataloaders and split it into train, val and test sets
    Args:
        adj_matrix_dir: str, path to the directory containing the adjacency matrices
        features_dir: str, path to the directory containing the features
        config: dict, configuration dictionary
    Returns:
        train_data: DataLoader, data loader for the training data
        val_data: DataLoader, data loader for the validation data
        test_data: DataLoader, data loader for the test data
    Dataset format: (adj_matrix, feature_matrix, label)"""
    
    dataset = GraphDataset(adj_matrices, features, labels)
    train_split, val_split = config.get("dataset").get("split_ratio").get("train"), \
                             config.get("dataset").get("split_ratio").get("val")
    
    train_data, val_data, test_data = random_split(dataset = dataset,  lengths = [int(train_split*len(dataset)), int(val_split*len(dataset)), 
                                                                       len(dataset) - int(train_split*len(dataset)) - int(val_split*len(dataset))])
    
    info = f"""
{"-*"*10} Data Information {"-*"*10} \n
Total number of samples: {len(dataset)}
Number of training samples: {len(train_data)}
Number of validation samples: {len(val_data)}
Number of test samples: {len(test_data)}
{"-"*50}
"""
    print(info)

    batch_Size = config.get("training").get("batch_size")
    num_workers = config.get("training").get("num_workers", 0)

    train_loader = DataLoader(train_data, batch_size=batch_Size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size = batch_Size, shuffle = False)
    test_loader = DataLoader(test_data, batch_size = batch_Size, shuffle = False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)


    BASE_ADJ_MATRIX_PATH = config.get("dataset").get("adj_dir", "data/processed_data/adj_matrix")
    BASE_FEATURES_PATH = config.get("dataset").get("features_dir", "data/processed_data/features")
    
    adj_matrices, features, labels = load_data(features_dir = BASE_FEATURES_PATH, 
                                          adj_matrix_dir = BASE_ADJ_MATRIX_PATH)


    train_loader, val_loader, test_loader = load_dataloader(adj_matrices, features, labels, config)

    batch = next(iter(train_loader))
    adj_matrix, feature_matrix, label = batch
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Label: {label}")