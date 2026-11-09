import numpy as np
import torch

from transformers import default_data_collator

from universal.settings.settings import Settings


class PlainCollator:
    def __init__(self, embedding_dim=1024):
        self.embedding_dim = embedding_dim

    def __call__(self, features):
        batch = default_data_collator(features)

        src_length = int(batch["prot_attention_mask"].sum(dim=1).max().item())
        tgt_length = int((batch["go_input_ids"] != Settings.GO_LABEL_IGNORE_INDEX).sum(dim=1).max().item())

        if "labels" in batch:
            batch["labels"] = batch["labels"][:, :src_length]

        batch["prot_input_ids"] = batch["prot_input_ids"][:, :src_length]
        batch["prot_attention_mask"] = batch["prot_attention_mask"][:, :src_length]

        batch["go_input_ids"] = batch["go_input_ids"][:, :tgt_length]
        if "decoder_input_ids" in batch:
            batch["decoder_input_ids"] = batch["decoder_input_ids"][:, :tgt_length]

        return batch