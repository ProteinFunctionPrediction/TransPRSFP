import torch.nn as nn
from model.transformer.self_attention import SelfAttention
from model.transformer.transformer_block import TransformerBlock

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention, _ = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out, cross_attention = self.transformer_block(value, key, query, src_mask)
        return out, cross_attention