from model.model import Model
import os
from universal.settings.settings import Settings
from transformers import T5EncoderModel
from config.loader.json.json_config_loader import JSONConfigLoader
from model.config.model_config import ModelConfig

from model.classification_head.config.classification_head_model_config import ClassificationHeadModelConfig
from model.transformer.config.transformer_model_config import TransformerModelConfig
from model.gpt2_lmhead.config.gpt2_lmhead_model_config import Gpt2LMHeadModelConfig


class ModelNavigator:
    
    @staticmethod
    def create(config: ModelConfig, prot_t5_model: T5EncoderModel, device: str='cpu') -> Model:
        return config.get_model(prot_t5_model, device)
    
    @staticmethod
    def load_config(folder_path: str) -> None:
        if not os.path.exists(folder_path):
            raise RuntimeError(f"No such directory!: {folder_path}")
        
        if not Settings.CONFIG_FILENAME in os.listdir(folder_path):
            raise RuntimeError(f"No config file exists in directory!: {folder_path}")
        
        config_filepath = os.path.join(folder_path, Settings.CONFIG_FILENAME)
        json_config_loader = JSONConfigLoader(config_filepath)
        config = json_config_loader.config
        
        if "type" not in config:
            raise RuntimeError(f"Type is not set in the config file!: {config_filepath}")

        if config["type"] == Settings.TRANSFORMER_MODEL_TYPE:
            transformer_model_config = TransformerModelConfig()
            transformer_model_config.build_from_config_loader(json_config_loader)
            return transformer_model_config

        elif config["type"] == Settings.CLASSIFICATION_HEAD_MODEL_TYPE:
            classification_head_model_config = ClassificationHeadModelConfig()
            classification_head_model_config.build_from_config_loader(json_config_loader)
            return classification_head_model_config

        elif config["type"] == Settings.GPT2_MODEL_TYPE:
            gpt2_lmhead_model_config = Gpt2LMHeadModelConfig()
            gpt2_lmhead_model_config.build_from_config_loader(json_config_loader)
            return gpt2_lmhead_model_config

        else:
            raise RuntimeError(f"Model type is not recognized!: {config['type']}")
    
    @staticmethod
    def load(folder_path: str, prot_t5_model: T5EncoderModel, device: str='cpu') -> None:
        config = ModelNavigator.load_config(folder_path)
        return config.get_model(prot_t5_model, device)