from abc import ABC
import os
import json

class Config(ABC):
    def __init__(self, config_path: str = None, config_directory: str = None) -> None:
        self.config_path: str = config_path
        self.config_directory: str = config_directory
        self.registered_keys: list = list() # registered keys are saved in config.json when save is called
        self.registered_key_names: set = set()
            
    def set_config_path(self, config_path: str) -> None:
        self.config_path = config_path
    
    def set_config_directory(self, config_directory: str) -> None:
        self.config_directory = config_directory
    
    def set_config_dirpath(self, config_path: str) -> None:
        self.set_config_path(config_path)
        self.set_config_directory(os.path.split(config_path)[0])
    
    def register_key(self, key: str, appearing_key: str = None):
        if key in self.registered_key_names:
            raise RuntimeError(f"The key is already registered!: {key}")
        
        if appearing_key is None:
            appearing_key = key
            
        self.registered_keys.append({key: appearing_key})
        self.registered_key_names.add(key)

    
    def save(self) -> None:
        if self.config_path is None or self.config_directory is None:
            raise RuntimeError("Config path is not set!")
        
        if os.path.exists(self.config_path):
            raise RuntimeError(f"The config path already exists!: {self.config_path}")
        
        with open(self.config_path, "w") as f:
            f.write(json.dumps(self._get_registered_dict()))
    
    def _get_registered_dict(self) -> dict:
        result = dict()
        for key_dict in self.registered_keys:
            key, appearing_key = list(key_dict.items())[0]
            result[appearing_key] = getattr(self, key)
        
        return result
