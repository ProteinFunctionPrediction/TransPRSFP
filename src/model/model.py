from abc import ABC
from model.config.model_config import ModelConfig
from universal.settings.settings import Settings
import os
import torch
import pickle

class Model(ABC):
    def __init__(self) -> None:
        self.config: ModelConfig = None
    
    def save(self, model_save_dir: str) -> None:
        os.mkdir(model_save_dir)
        
        self.config.set_config_dirpath(os.path.join(model_save_dir, Settings.CONFIG_FILENAME))
        self.config.save()
        torch.save(self.state_dict(), os.path.join(model_save_dir, Settings.MODEL_FILENAME))
        
        go_term_to_index_full_path = os.path.join(model_save_dir, self.config.go_term_to_index_filepath)
        
        if os.path.exists(go_term_to_index_full_path):
            raise RuntimeError(f"File already exists!: {go_term_to_index_full_path}")
        
        with open(go_term_to_index_full_path, "wb") as f:
            pickle.dump(self.config.go_term_to_index, f)
    
    def set_config(self, config: ModelConfig) -> None:
        self.config = config
    
    def get_config(self) -> ModelConfig:
        return self.config
