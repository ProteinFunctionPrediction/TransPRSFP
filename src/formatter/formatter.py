from abc import ABC, abstractmethod

class Formatter(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def format(self, message: str) -> str:
        return None