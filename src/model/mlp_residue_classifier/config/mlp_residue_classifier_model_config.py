from model.config.model_config import ModelConfig
from model.mlp_residue_classifier.model import MLPResidueClassifier
import torch
import torch.nn as nn
import os
from universal.settings.settings import Settings
from model.model import Model
from typing import Optional

class MLPResidueClassifierModelConfig(ModelConfig):
    def __init__(self, num_labels=None, input_size=None, hidden_sizes=None, dropout=None, ignore_index=None, filepath=None, go_term_to_index_filepath=None, go_term_to_index=None) -> None:
        super().__init__(Settings.MLP_RESIDUE_CLASSIFIER_MODEL_TYPE, filepath, go_term_to_index, go_term_to_index_filepath)

        self.num_labels = num_labels
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.ignore_index = ignore_index

        self.model: MLPResidueClassifier = None
        self.load_from_pretrained_model: bool = None
        self.loaded: bool = False

        self.register_key("input_size")
        self.register_key("hidden_size")
        self.register_key("dropout")
        self.register_key("ignore_index")

    def get_model(self, device='cpu'):
        assert self.loaded == True
        self.model = MLPResidueClassifier(num_labels=self.num_labels,
                                          input_size=self.input_size,
                                          hidden_sizes=self.hidden_sizes,
                                          dropout=self.dropout,
                                          ignore_index=self.ignore_index).to(device)
        self.model.set_config(self)
        if self.load_from_pretrained_model:
            self.model.load_state_dict(torch.load(os.path.join(self.config_directory, self.filepath), map_location=device))
        return self.model                                    