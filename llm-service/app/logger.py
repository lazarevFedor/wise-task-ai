import logging


def get_logger(name) -> logging.Logger:
    """
    Create and configure a logger for the llm_service.

    This function creates a logger with the given name prefixed by 'llm_service.',
    sets the logging level to DEBUG, and adds a StreamHandler for console output.

    Args:
        name (str): The base name for the logger.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(f'llm_service.{name}')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    return logger
