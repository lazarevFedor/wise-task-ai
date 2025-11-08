import logging


def get_logger(name) -> logging.Logger:
    logger = logging.getLogger(f'llm_service.{name}')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    return logger
