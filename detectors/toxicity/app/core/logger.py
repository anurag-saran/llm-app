import logging


log_format = "%(asctime)s - %(levelname)s - %(message)s"
log_level = logging.INFO

logger = logging.getLogger("llm-benchmark")

logger.setLevel(log_level)
formatter = logging.Formatter(log_format)
handler = logging.StreamHandler()
handler.setLevel(log_level)
handler.setFormatter(formatter)
logger.addHandler(handler)
