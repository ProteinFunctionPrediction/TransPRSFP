from output.output import Output

class PrinterOutput(Output):
    def __init__(self, formatter=None) -> None:
        super().__init__(formatter)
        
    def write(self, message: str) -> None:
        if self.formatter:
            message = self.formatter.format(message)
        
        print(message)