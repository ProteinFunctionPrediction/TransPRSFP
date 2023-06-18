from formatter.formatter import Formatter
import datetime

class DatetimeFormatter(Formatter):
    def __init__(self) -> None:
        super().__init__()
    
    def format(self, message: str) -> str:
        return str(datetime.datetime.now()) + ": " + message