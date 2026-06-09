# -*- coding: utf-8 -*-
"""统一日志，替代 print，便于记录特征工程、训练、调参关键步骤。"""
import logging
import os
import sys
from typing import Optional

_LOGGER: Optional[logging.Logger] = None


def get_logger(
    name: str = "fracture_ml",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None and _LOGGER.name == name:
        return _LOGGER
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    logger.addHandler(h)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    _LOGGER = logger
    return logger
