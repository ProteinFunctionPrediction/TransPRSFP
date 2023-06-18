from config.loader.config_loader import ConfigLoader
from config.config import Config
import json

class JSONConfigLoader(ConfigLoader):
    def __init__(self, path) -> None:
        super().__init__(path)
        
        with open(path) as f:
            self.config = json.loads(f.read())
    
    def load(self, config: Config) -> None:
        super().load(config)
        for key, value in self.config.items():
            setattr(config, key, value)