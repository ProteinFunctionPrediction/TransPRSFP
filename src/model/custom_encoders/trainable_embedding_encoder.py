import torch
import torch.nn as nn
import os
from model.custom_encoders.output import Output


class TrainableEmbeddingEncoder(nn.Module):
    def __init__(self, vocab_size=128, embed_size=1024, padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=padding_idx)


    def forward(self, input_ids, attention_mask):
        result = self.embedding(input_ids)
        assert attention_mask.shape == result.shape[:2], f"Attention mask shape {attention_mask.shape} does not match result shape {result.shape[:2]}"
        mask = attention_mask.unsqueeze(-1).to(dtype=result.dtype, device=result.device)
        result = result * mask

        return Output(result)
