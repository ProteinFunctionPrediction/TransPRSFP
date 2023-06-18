import torch.nn as nn
import torch
from model.transformer.positional_encoding import PositionalEncoding
from model.transformer.decoder_block import DecoderBlock
import math

class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
        go_embedding_fetcher = None
    ):
        super(Decoder, self).__init__()
        self.device = device
        
        if go_embedding_fetcher:
            self.word_embedding = nn.Linear(go_embedding_fetcher.embedding_dim, embed_size)
        else:
            self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)        
        
        
        self.go_embedding_fetcher = go_embedding_fetcher
        self.embed_size = embed_size
        
        self.positional_encoder = PositionalEncoding(
            dim_model=embed_size, dropout_p=dropout, max_len=max_length
        )

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, self.device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask, output_attentions=False):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        if self.go_embedding_fetcher:
            x = self.go_embedding_fetcher.get_embeddings_for_batch(x, self.device)
            x = self.word_embedding(x) * math.sqrt(self.embed_size)
            x = self.positional_encoder(x)
            x = self.dropout(x)
        else:
            embedded = self.word_embedding(x) * math.sqrt(self.embed_size)
            embedded = self.positional_encoder(embedded)
            x = self.dropout(embedded)

        if output_attentions:
            attentions = []
            for layer in self.layers:
                x, attention = layer(x, enc_out, enc_out, src_mask, trg_mask)
                attentions.append(attention)

        else:
            for layer in self.layers:
                x, _ = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)


        if output_attentions:
            return out, attentions
        else:
            return out
