import logging

# Setup logger
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logger = logging.getLogger("llm-benchmark")

formatter = logging.Formatter(log_format)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def setup_logging(debug):
    """Configure the logging level based on whether debug mode is enabled."""
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)
    handler.setLevel(log_level)
    if debug:
        logger.debug("Debug mode is now active.")
