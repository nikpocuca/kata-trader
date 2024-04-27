import logging
from logging import DEBUG, Logger

def create_logger(name: str) -> Logger:
    """
    simple function to create logger class
    """
    logger = logging.Logger(name)
    logger.setLevel(DEBUG)

    # Create a StreamHandler and set its log level to DEBUG
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    # Create a formatter and set the format of log messages
    formatter = logging.Formatter('%(levelname)s<%(asctime)s> - %(message)s')
    stream_handler.setFormatter(formatter)

    # Add the StreamHandler to the logger
    logger.addHandler(stream_handler)
    return logger


    