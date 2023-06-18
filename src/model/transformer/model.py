import torch
import torch.nn as nn
from model.transformer.decoder import Decoder
from model.model import Model

class Transformer(nn.Module, Model):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        prot_t5_model,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=10000,
        go_embedding_fetcher=None
    ):

        super(Transformer, self).__init__()

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
            go_embedding_fetcher
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.prot_t5_model = prot_t5_model

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
                N, 1, trg_len, trg_len
        ).to(self.device)

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        return (trg_mask * trg_pad_mask).to(self.device)

    def forward(self, src, trg, prediction_mask=None, output_attentions=False):
        src_mask = self.make_src_mask(src).to(torch.int)
        
        trg_mask = self.make_trg_mask(trg)
        
        first_dim = src_mask.shape[0]
        last_dim = src_mask.shape[-1]
        enc_src = self.prot_t5_model(input_ids=src,attention_mask=torch.reshape(src_mask, (first_dim, last_dim))).last_hidden_state
        if prediction_mask is not None:
            enc_src = torch.tensor(enc_src.cpu().numpy()[:, prediction_mask, :], device=self.device)

        if output_attentions:
            out, attentions = self.decoder(trg, enc_src, src_mask, trg_mask, output_attentions)
            return out, attentions
        else:
            out = self.decoder(trg, enc_src, src_mask, trg_mask, output_attentions)
            return out
        
        return out

    def save(self, model_save_dir: str) -> None:
        Model.save(self, model_save_dir)