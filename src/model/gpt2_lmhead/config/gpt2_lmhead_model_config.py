from model.config.model_config import ModelConfig
from model.gpt2_lmhead.model import GPT2LMHead
from transformers import T5EncoderModel, GPT2Config
import torch
import os
from universal.settings.settings import Settings
from model.model import Model

class Gpt2LMHeadModelConfig(ModelConfig):
    def __init__(self, filepath=None, n_embd=None, heads=None, vocab_size=None,
                 n_positions=None, num_layers=None, sos_token=None, eos_token=None,
                 go_term_to_index_filepath=None, go_term_to_index=None) -> None:
        super().__init__(Settings.GPT2_MODEL_TYPE, filepath, go_term_to_index, go_term_to_index_filepath)

        self.n_embd = n_embd
        self.heads = heads
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.num_layers = num_layers
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.model: GPT2LMHead = None
        self.load_from_pretrained_model: bool = None
        self.loaded: bool = False

        self.register_key("n_embd")
        self.register_key("heads")
        self.register_key("vocab_size")
        self.register_key("n_positions")
        self.register_key("num_layers")
        self.register_key("sos_token")
        self.register_key("eos_token")

    def get_model(self, prot_t5_model: T5EncoderModel, device: str='cpu') -> Model:
        assert self.loaded == True
        configuration = GPT2Config(add_cross_attention=True, is_decoder=True, n_embd=self.n_embd,
                                n_head=self.heads, vocab_size=self.vocab_size, n_positions=self.n_positions,
                                n_layer=self.num_layers, bos_token_id=self.sos_token, eos_token_id=self.eos_token)
        self.model = GPT2LMHead(prot_t5_model, configuration).to(device)
        self.model.set_config(self)
        if self.load_from_pretrained_model:
            self.model.load_state_dict(torch.load(os.path.join(self.config_directory, self.filepath), map_location=device))
        return self.model
