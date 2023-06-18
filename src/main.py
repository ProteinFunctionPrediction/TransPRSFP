from argument_handler import ArgumentHandler
from universal.access.universal_access import UniversalAccess
from output.printer.printer_output import PrinterOutput
from formatter.datetime.datetime_formatter import DatetimeFormatter
from driver import Driver

class Main:
    def __init__(self) -> None:
        argument_handler = ArgumentHandler()
        
        UniversalAccess.output = PrinterOutput(DatetimeFormatter())
        driver = Driver(argument_handler.args)
        driver.run()


if __name__ == '__main__':
    main = Main()