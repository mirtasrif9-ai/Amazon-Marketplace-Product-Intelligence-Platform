import logging

from src.common.config import LOG_LEVEL


def setup_logging() -> None:
    """Configure application-wide logging."""

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )