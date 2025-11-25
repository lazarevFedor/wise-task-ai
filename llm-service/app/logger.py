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
    logging.basicConfig(
        handlers=[logging.StreamHandler()],
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    logger = logging.getLogger(f'llm_service.{name}')
    return logger
