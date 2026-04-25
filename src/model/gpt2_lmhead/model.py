from model.model import Model
from transformers import GPT2Config, GPT2LMHeadModel, T5EncoderModel
import torch.nn as nn
from typing import Union

class GPT2LMHead(GPT2LMHeadModel, Model):
    def __init__(self, encoder_model: Union[T5EncoderModel, nn.Module], configuration: GPT2Config):
        GPT2LMHeadModel.__init__(self, configuration)
        Model.__init__(self)
        
        self.encoder = encoder_model
        self.gpt2_config = configuration
