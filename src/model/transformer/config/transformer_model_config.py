import torch
import os
from transformers import T5EncoderModel

from model.config.model_config import ModelConfig
from model.transformer.model import Transformer
from universal.settings.settings import Settings


class TransformerModelConfig(ModelConfig):
    def __init__(self, filepath=None, src_vocab_size=None, trg_vocab_size=None,
                 sos_token=None, eos_token=None, max_length=None,
                 embed_size=None, num_layers=None, heads=None,
                 go_term_to_index_filepath=None, go_term_to_index=None) -> None:
        super().__init__(Settings.TRANSFORMER_MODEL_TYPE, filepath, go_term_to_index, go_term_to_index_filepath)
        
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_length = max_length
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.model: Transformer = None
        self.load_from_pretrained_model: bool = None
        self.loaded: bool = False
        
        self.register_key("src_vocab_size")
        self.register_key("trg_vocab_size")
        self.register_key("sos_token")
        self.register_key("eos_token")
        self.register_key("max_length")
        self.register_key("embed_size")
        self.register_key("num_layers")
        self.register_key("heads")

    def get_model(self, prot_t5_model: T5EncoderModel, device='cpu'):
        assert self.loaded == True
        self.model = Transformer(src_vocab_size=self.src_vocab_size,
                                 trg_vocab_size=self.trg_vocab_size,
                                 src_pad_idx=Settings.TRANSFORMER_SRC_PAD_IDX,
                                 trg_pad_idx=Settings.TRANSFORMER_TRG_PAD_IDX,
                                 prot_t5_model=prot_t5_model,
                                 embed_size=self.embed_size,
                                 num_layers=self.num_layers,
                                 heads=self.heads, device=device,
                                 max_length=self.max_length).to(device)
        self.model.set_config(self)
        if self.load_from_pretrained_model:
            self.model.load_state_dict(torch.load(os.path.join(self.config_directory, self.filepath), map_location=device))
        return self.model
