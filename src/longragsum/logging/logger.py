from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/app_{time:YYYY-MM-DD}.log", rotation="7 days", retention="30 days", level="DEBUG")

__all__ = ["logger"]