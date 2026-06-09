# -*- coding: utf-8 -*-
import os
import yaml
from typing import Any, Optional

_CONFIG_CACHE = None


def load_config(config_path: Optional[str] = None) -> dict:
    """加载 config.yaml；若未传路径则从 program 目录查找。"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE
    if config_path is None:
        prog_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(prog_dir, "config.yaml")
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG_CACHE = yaml.safe_load(f) or {}
    return _CONFIG_CACHE
