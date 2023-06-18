from abc import ABC, abstractmethod
from config.config import Config
import os

class ConfigLoader(ABC):
    def __init__(self, path) -> None:
        super().__init__()
        self.path: str = path
        self.config: dict = None

    def load(self, config: Config) -> None:
        config.set_config_dirpath(self.path)
