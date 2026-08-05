import torch
import torch.nn as nn
import os
from utils.utils import Utils
from model.custom_encoders.output import Output


class FixedRandomEncoder(nn.Module):
    def __init__(self, vocab_size=128, embed_size=1024, load_vectors_from=None, padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        if load_vectors_from is None:
            self.vocab_size = vocab_size
            self.embed_size = embed_size
            self.vectors = self.generate_random_vectors()
            self.vectors[padding_idx].zero_()
        else:
            self.vectors = torch.load(load_vectors_from)
            self.vectors[padding_idx].zero_()
            if vocab_size is None and embed_size is None:
                self.vocab_size = self.vectors.shape[0]
                self.embed_size = self.vectors.shape[1]
            
        self.embedding = nn.Embedding.from_pretrained(self.vectors, freeze=True, padding_idx=padding_idx)

    def forward(self, input_ids, attention_mask):
        result = self.embedding(input_ids)
        assert attention_mask.shape == result.shape[:2], f"Attention mask shape {attention_mask.shape} does not match result shape {result.shape[:2]}"
        mask = attention_mask.unsqueeze(-1).to(dtype=result.dtype, device=result.device)
        result = result * mask
        return Output(result)

    def generate_random_vectors(self):
        mean_values = Utils.get_rand_vector((self.vocab_size,), -0.003, -0.0006).unsqueeze(1)
        stddev_values = Utils.get_rand_vector((self.vocab_size,), 0.1450, 0.187).unsqueeze(1)
        
        return torch.randn(self.vocab_size, self.embed_size) * stddev_values + mean_values

    def save_vectors(self, path):
        assert not os.path.exists(path)
        torch.save(self.vectors, path)
