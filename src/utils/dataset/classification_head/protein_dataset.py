from torch.utils.data import Dataset
import torch

class ProteinDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, labels = self.data[idx]
        inputs = self.tokenizer(sequence, max_length=self.max_len, return_tensors="pt", truncation=True, padding="max_length")
        return inputs, torch.tensor(labels, dtype=torch.float32)