import logging

# Create a logger object
logger = logging.getLogger('speed_harmo')

# Set up a file handler to write logs to a file
file_handler = logging.FileHandler('./logs/current_run.log')

# Set up a console handler to output logs to the console
console_handler = logging.StreamHandler()

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)