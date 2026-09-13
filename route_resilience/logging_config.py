"""
Logging configuration utility for Route Resilience.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "route_resilience",
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configures and returns a standard logger instance.

    Args:
        name: Name of the logger.
        log_level: Desired log level (e.g. DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write output log file.

    Returns:
        Configured logging instance.
    """
    logger = logging.getLogger(name)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Stream handler (stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Optional file handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
