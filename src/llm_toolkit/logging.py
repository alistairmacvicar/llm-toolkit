import logging

LOGGER_NAME = "llm_toolkit"


def get_logger(name: str | None = None):
    """
    Gets a logger in the llm_toolkit namespace.

    Args:
        name (str | None): Optional logger name. Defaults to None.

    Returns:
        logging.Logger: The requested logger.
    """

    if name is None:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(level=logging.INFO):
    """
    Configures basic logging for applications using llm_toolkit.

    Args:
        level (int | str): Logging level to use. Defaults to logging.INFO.
    """

    logging.basicConfig(level=level)
