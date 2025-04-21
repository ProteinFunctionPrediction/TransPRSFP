from argument_handler import ArgumentHandler
from universal.access.universal_access import UniversalAccess
from output.printer.printer_output import PrinterOutput
from formatter.datetime.datetime_formatter import DatetimeFormatter
from driver import Driver

import numpy as np
import random
import torch

class Main:
    def __init__(self) -> None:
        argument_handler = ArgumentHandler()
        
        UniversalAccess.output = PrinterOutput(DatetimeFormatter())
        args = argument_handler.args
        if not args.no_random_seed:
            torch.manual_seed(args.random_seed)
            torch.cuda.manual_seed(args.random_seed)
            np.random.seed(args.random_seed)
            random.seed(args.random_seed)
            UniversalAccess.output.write(f"The random seed has been set as {args.random_seed}")
        else:
            UniversalAccess.output.write("--no-random-seed flag is set: no random seed will be set")

        driver = Driver(argument_handler.args)
        driver.run()


if __name__ == '__main__':
    main = Main()