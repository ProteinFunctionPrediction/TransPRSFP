from __future__ import annotations
from abc import ABC, abstractmethod
from config.config import Config
from config.loader.config_loader import ConfigLoader
from config.loader.json.json_config_loader import JSONConfigLoader
from transformers import T5EncoderModel
import os
import pickle
from utils.utils import Utils

class ModelConfig(Config, ABC):
    def __init__(self, type:str, filepath:str, go_term_to_index: dict, go_term_to_index_filepath: str) -> None:
        super().__init__()
        
        self.type: str = type
        self.filepath: str = filepath
        self.go_term_to_index: dict = go_term_to_index
        self.go_term_to_index_filepath = go_term_to_index_filepath
        self.reverse_go_term_to_index: dict = None
        
        self.register_key("type")
        self.register_key("filepath")
        self.register_key("go_term_to_index_filepath")

    def _on_loading_completed(self) -> None:
        if self.go_term_to_index is None:
            with open(os.path.join(self.config_directory, self.go_term_to_index_filepath), "rb") as f:
                self.go_term_to_index = pickle.load(f)
            
        self.reverse_go_term_to_index = Utils.build_reverse_index(self.go_term_to_index)
        self.loaded = True
    
    def build_from_json_file(self, path) -> None:
        loader = JSONConfigLoader(path)
        loader.load(self)
        self.load_from_pretrained_model = True
        self._on_loading_completed()
        
    def build_from_config_loader(self, config_loader: ConfigLoader) -> None:
        config_loader.load(self)
        self.load_from_pretrained_model = True
        self._on_loading_completed()
        
    def build(self) -> None:
        self.load_from_pretrained_model = False
        self._on_loading_completed()
    
    @abstractmethod
    def get_model(prot_t5_model: T5EncoderModel, device: str='cpu') -> Model:
        pass