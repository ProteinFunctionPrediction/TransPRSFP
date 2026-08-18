import torch
from torch.utils.data import Dataset


class MLPResidueClassifierDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {key: torch.as_tensor(value) for key, value in self.data[idx].items() if key != "prot_sequence"}
