from model.model import Model
from transformers import GPT2Config, GPT2LMHeadModel, T5EncoderModel

class GPT2LMHead(GPT2LMHeadModel, Model):
    def __init__(self, prot_t5_model: T5EncoderModel, configuration: GPT2Config):
        GPT2LMHeadModel.__init__(self, configuration)
        Model.__init__(self)
        
        self.encoder = prot_t5_model
        self.gpt2_config = configuration
