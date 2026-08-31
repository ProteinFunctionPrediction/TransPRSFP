import torch
from torch.utils.data import Dataset


class GPT2Dataset(Dataset):
    def __init__(self, data):
        self.data = [{key: torch.tensor(value) for key, value in item.items() if key != "prot_sequence"} for item in data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]