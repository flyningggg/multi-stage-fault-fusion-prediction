# -*- coding: utf-8 -*-
"""
推理逻辑：加载已训练模型，对特征矩阵或 CSV 进行预测。
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Optional, List

_PROGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)


def load_xgboost_model(model_path: str, is_classifier: bool = False):
    """加载 XGBoost 模型（.json）。"""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("请安装 xgboost: pip install xgboost")
    if is_classifier:
        model = xgb.XGBClassifier()
    else:
        model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


def infer(
    model_path: str,
    X: np.ndarray,
    is_classifier: bool = False,
) -> np.ndarray:
    """对特征矩阵 X 做预测。"""
    model = load_xgboost_model(model_path, is_classifier=is_classifier)
    return model.predict(X)


def infer_from_csv(
    model_path: str,
    csv_path: str,
    feature_columns: Optional[List[str]] = None,
    is_classifier: bool = False,
    out_column: str = "pred",
) -> pd.DataFrame:
    """从 CSV 读取特征，预测后写回新列。"""
    from feature_engineering import build_feature_matrix
    r = build_feature_matrix(csv_path, feature_columns=feature_columns, out_processed_dir=None)
    pred = infer(model_path, r["X"], is_classifier=is_classifier)
    df = r["df"].copy()
    df[out_column] = pred
    return df


if __name__ == "__main__":
    model_dir = os.path.join(_PROGRAM_DIR, "model")
    path = os.path.join(model_dir, "xgboost_reg.json")
    if not os.path.isfile(path):
        print("请先运行 train.py 生成模型：", path)
        sys.exit(1)
    csv_path = os.path.join(_PROGRAM_DIR, "Yingmai 2 area in Tarim Basin.csv")
    if os.path.isfile(csv_path):
        df = infer_from_csv(path, csv_path, out_column="pred_reg")
        print(df["pred_reg"].head())
