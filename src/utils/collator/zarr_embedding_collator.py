import numpy as np
import torch
import zarr

from transformers import default_data_collator
from zarr.storage import NestedDirectoryStore

from universal.settings.settings import Settings


class ZarrEmbeddingCollator:
    def __init__(self, store_path, embedding_dim=1024, cache_bytes=None):
        self.store_path = store_path
        self.embedding_dim = embedding_dim
        self.cache_bytes = cache_bytes

        self.__embeddings = None
        self.__store = None
    
    def __open(self):
        if self.__embeddings is not None:
            return

        store = NestedDirectoryStore(self.store_path)

        if self.cache_bytes is not None:
            store = zarr.LRUStoreCache(store, max_size=self.cache_bytes)
        
        root = zarr.open_group(store=store, mode="r")
        self.__store = store
        self.__embeddings = root["embeddings"]

    def __call__(self, features):
        self.__open()

        offsets = [int(feature["embedding_offset"]) for feature in features]
        lengths = [int(feature["embedding_length"]) for feature in features]

        other_features = [{k: v for k, v in feature.items() if k not in ("embedding_offset", "embedding_length")} for feature in features]
        batch = default_data_collator(other_features)

        src_length = int(batch["prot_attention_mask"].sum(dim=1).max().item())
        tgt_length = int((batch["go_input_ids"] != Settings.TRANSFORMER_TRG_PAD_IDX).sum(dim=1).max().item())

        if "labels" in batch:
            batch["labels"] = batch["labels"][:, :src_length]

        batch["prot_input_ids"] = batch["prot_input_ids"][:, :src_length]
        batch["prot_attention_mask"] = batch["prot_attention_mask"][:, :src_length]

        batch["go_input_ids"] = batch["go_input_ids"][:, :tgt_length]

        batch_size = len(features)
        last_hidden_states = np.zeros((batch_size, src_length, self.embedding_dim), dtype=np.float32)

        for idx, (offset, length) in enumerate(zip(offsets, lengths)):
            l = min(length, src_length)
            last_hidden_states[idx, :l, :] = self.__embeddings[offset : offset + l, :]
        
        batch["last_hidden_states"] = torch.from_numpy(last_hidden_states)

        return batch