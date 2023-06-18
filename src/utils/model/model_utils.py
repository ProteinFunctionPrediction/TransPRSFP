from abc import ABC

class ModelUtils(ABC):
    def __init__(self, device) -> None:
        self.device = device
