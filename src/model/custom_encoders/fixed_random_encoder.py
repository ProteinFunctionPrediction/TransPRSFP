import torch
import torch.nn as nn
import os
from utils.utils import Utils


class Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state

class FixedRandomEncoder(nn.Module):
    def __init__(self, vocab_size=128, embed_size=1024, load_vectors_from=None):
        super().__init__()
        if load_vectors_from is None:
            self.vocab_size = vocab_size
            self.embed_size = embed_size
            self.vectors = self.generate_random_vectors()
        else:
            self.vectors = torch.load(load_vectors_from)
            if vocab_size is None and embed_size is None:
                self.vocab_size = self.vectors.shape[0]
                self.embed_size = self.vectors.shape[1]
            
        self.embedding = nn.Embedding.from_pretrained(self.vectors, freeze=True)

    def forward(self, input_ids, attention_mask=None):
        return Output(self.embedding(input_ids))

    def generate_random_vectors(self):
        mean_values = Utils.get_rand_vector((self.vocab_size,), -0.003, -0.0006).unsqueeze(1)
        stddev_values = Utils.get_rand_vector((self.vocab_size,), 0.1450, 0.187).unsqueeze(1)
        
        return torch.randn(self.vocab_size, self.embed_size) * stddev_values + mean_values

    def save_vectors(self, path):
        assert not os.path.exists(path)
        torch.save(self.vectors, path)
