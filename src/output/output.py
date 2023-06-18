from abc import ABC, abstractmethod
from formatter.formatter import Formatter

class Output(ABC):
    def __init__(self, formatter: Formatter) -> None:
        super().__init__()
        self.formatter: Formatter = formatter
    
    @abstractmethod
    def write(self, message: str) -> None:
        pass