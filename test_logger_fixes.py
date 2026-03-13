import time
from src.log.log_manager import get_logger, init_logging, LogManager
import logging

# Issue 2: Early get_logger
early_logger = get_logger("src.api.early")
early_logger.info("This is early logger, before init.")

# Now init_logging with a new config
print("Calling init_logging...")
init_logging({"level": "DEBUG", "console_output": False})

early_logger.debug("This should be visible only in file if console_output=False, and level should be DEBUG.")
new_logger = get_logger("src.api.new")
new_logger.debug("New logger debug message.")

print("early handlers:", early_logger.handlers)
print("new handlers:", new_logger.handlers)

