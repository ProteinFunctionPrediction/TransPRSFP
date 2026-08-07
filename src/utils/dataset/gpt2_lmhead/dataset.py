import torch
from torch.utils.data import Dataset


class GPT2Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        result = {}
        for key, value in self.data[idx].items():
            if key != "prot_sequence":
                result[key] = torch.tensor(value)
            else:
                #print("PROT SEQUENCE OBSERVED!")
                #print(key, value)
                result[key] = value
        return result
        #return {key: torch.tensor(value) for key, value in self.data[idx].items() if key != "prot_sequence"}
