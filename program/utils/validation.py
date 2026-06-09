# -*- coding: utf-8 -*-
"""路径、空值、数据类型校验，供特征工程与训练脚本使用。"""
import os
import pandas as pd
import numpy as np
from typing import List, Optional


def check_path_exists(path: str, label: str = "path") -> str:
    """校验文件存在，否则抛 ValueError。"""
    if not path or not os.path.isfile(path):
        raise ValueError(f"{label} 不存在或非文件: {path}")
    return path


def check_csv_has_columns(
    csv_path: str,
    required: Optional[List[str]] = None,
    at_least_one: Optional[List[str]] = None,
) -> pd.DataFrame:
    """校验 CSV 存在且包含所需列；返回 DataFrame。"""
    check_path_exists(csv_path, "CSV")
    df = pd.read_csv(csv_path, nrows=1)
    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV 缺少列: {missing}")
    if at_least_one:
        if not any(c in df.columns for c in at_least_one):
            raise ValueError(f"CSV 至少需包含下列之一: {at_least_one}")
    return pd.read_csv(csv_path)


def check_target_dtype(
    series: pd.Series,
    task: str = "regression",
) -> None:
    """校验标签列类型：回归应为数值，分类可为数值或类别。"""
    if task == "regression":
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("回归任务目标列应为数值型")
    # 分类不强制数值，允许字符串/整数类别
