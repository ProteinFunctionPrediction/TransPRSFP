from model.config.model_config import ModelConfig
from model.classification_head.model import MultiLabelProteinClassifier
from transformers import T5EncoderModel
import torch
import os
from universal.settings.settings import Settings

class ClassificationHeadModelConfig(ModelConfig):
    def __init__(self, filepath=None, go_term_count=None, max_length=None,
                 go_term_to_index_filepath=None, go_term_to_index=None) -> None:
        super().__init__(Settings.CLASSIFICATION_HEAD_MODEL_TYPE, filepath, go_term_to_index, go_term_to_index_filepath)
        
        self.go_term_count = go_term_count
        self.max_length = max_length
        self.model: MultiLabelProteinClassifier = None
        self.load_from_pretrained_model: bool = None
        self.loaded: bool = False
        
        self.register_key("go_term_count")
        self.register_key("max_length")

    def get_model(self, prot_t5_model: T5EncoderModel, device='cpu'):
        assert self.loaded == True
        self.model = MultiLabelProteinClassifier(prot_t5_model, self.go_term_count).to(device)
        self.model.set_config(self)
        if self.load_from_pretrained_model:
            self.model.load_state_dict(torch.load(os.path.join(self.config_directory, self.filepath), map_location=device))
        return self.model
