import logging
import os

logger = logging.getLogger("blml1")
logger.setLevel(getattr(logging, os.environ.get("BLML1_LOG_LEVEL", "debug").upper()))
