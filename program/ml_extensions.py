# -*- coding: utf-8 -*-
"""
机器学习扩展：MLP 分类/回归、GNN 节点分类占位。
依赖 topology_fusion 的 load_and_prepare 与 DEFAULT_FEATURE_COLUMNS。
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

try:
    from topology_fusion import load_and_prepare, DEFAULT_FEATURE_COLUMNS
except ImportError:
    load_and_prepare = None
    DEFAULT_FEATURE_COLUMNS = []


def _get_X_y(
    csv_path: str,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    drop_na_target: bool = True,
    task: str = "classification",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    从网格 CSV 加载数据并提取特征矩阵 X 和标签 y。
    回归任务：y 为 float64；分类任务：字符串类别自动映射为整数编码。
    """
    if load_and_prepare is None:
        raise ImportError("请确保 topology_fusion 模块可用（与 main.py 同目录）")
    df, X, used_cols = load_and_prepare(
        csv_path, feature_columns=feature_columns, drop_all_nan=False
    )
    if target_column not in df.columns:
        raise ValueError(f"目标列不存在: {target_column}")
    y_raw = df[target_column].values
    if drop_na_target:
        valid = ~(pd.isna(y_raw) | (y_raw == ""))
        if not np.any(valid):
            raise ValueError(f"目标列 {target_column} 全部为空或无效")
        X = X[valid]
        y_raw = y_raw[valid]
        df = df.loc[valid].copy()
    if task == "regression":
        y = np.asarray(y_raw, dtype=np.float64)
        valid_f = ~np.isnan(y)
        if not np.all(valid_f):
            X, y = X[valid_f], y[valid_f]
            df = df.loc[valid_f].copy()
    else:
        # 分类：将非整数类别映射为整数编码
        if np.issubdtype(y_raw.dtype, np.integer):
            y = np.asarray(y_raw, dtype=np.int64)
        else:
            uniq = pd.unique(y_raw)
            uniq = uniq[~pd.isna(uniq)]
            label_to_id = {v: i for i, v in enumerate(uniq)}
            y = np.array([label_to_id.get(v, 0) for v in y_raw], dtype=np.int64)
    return df, X, y, used_cols


def train_mlp_classifier(
    csv_path: str,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    max_iter: int = 500,
) -> Tuple[Any, StandardScaler, Dict[str, Any]]:
    """
    MLP 分类器：sklearn.MLPClassifier 的简单封装。
    加载数据 → 标准化 → 训练 → 评估准确率与分类报告。
    """
    df, X, y, used_cols = _get_X_y(
        csv_path, target_column, feature_columns, task="classification"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
    )
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return clf, scaler, {"accuracy": acc, "report": report, "n_train": len(y_train), "n_test": len(y_test)}


def train_mlp_regressor(
    csv_path: str,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    max_iter: int = 500,
) -> Tuple[Any, StandardScaler, Dict[str, Any]]:
    """
    MLP 回归器：sklearn.MLPRegressor 的简单封装。
    加载数据 → 标准化 → 训练 → 评估 MSE/RMSE/R²。
    """
    df, X, y, used_cols = _get_X_y(
        csv_path, target_column, feature_columns, task="regression"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    reg = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
    )
    reg.fit(X_train_s, y_train)
    y_pred = reg.predict(X_test_s)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return reg, scaler, {"mse": mse, "rmse": np.sqrt(mse), "r2": r2, "n_train": len(y_train), "n_test": len(y_test)}


def build_grid_adjacency(n_rows: int, n_cols: int) -> np.ndarray:
    """
    构建规则网格的 4 邻域邻接矩阵（edge_index 格式）。
    每个网格单元连接到上下左右相邻单元（如果存在）。
    返回 shape (2, E) 的 edge_index，与 PyG 的 Data 格式兼容。
    """
    n = n_rows * n_cols
    edges = []
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            if r > 0:
                edges.append((idx, (r - 1) * n_cols + c))
            if r < n_rows - 1:
                edges.append((idx, (r + 1) * n_cols + c))
            if c > 0:
                edges.append((idx, r * n_cols + (c - 1)))
            if c < n_cols - 1:
                edges.append((idx, r * n_cols + (c + 1)))
    return np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
