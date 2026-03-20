"""
shared/utils.py

Utility helpers shared across farm_gpu and fin_ai modules.
"""

import logging
import sys
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s – %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def safe_json_parse(text: str) -> Any:
    """Parse JSON string, stripping markdown fences if present."""
    import json
    clean = text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(clean)
