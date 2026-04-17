import logging
import os
from functools import lru_cache

LOG_FILE = os.environ.get("EV_APP_LOG_FILE", "app_error.log")


@lru_cache(maxsize=1)
def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )


def get_logger(name: str) -> logging.Logger:
    _configure_logging()
    return logging.getLogger(name)
