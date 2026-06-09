# -*- coding: utf-8 -*-
"""
特征工程：勘探价值导向的拓扑属性预处理与筛选。
对不同维度的拓扑属性做「归一化 + 异常值处理 + 特征筛选」（方差分析 / 互信息法），
输出「裂缝拓扑特征矩阵」供融合与 ML 使用。
"""
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Set
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, SelectKBest, f_regression
from utils.config_loader import load_config
from utils.logging_utils import get_logger

# 基础属性（长度、密度、数量等）
BASE_ATTRS = [
    "Trace Mean Length", "Trace Min Length", "Trace Max Length",
    "Branch Mean Length",
    "Fracture Intensity B21", "Fracture Intensity P21",
    "Dimensionless Intensity B22", "Dimensionless Intensity P22",
    "Number of Traces (Real)", "Number of Branches (Real)",
    "Areal Frequency B20", "Areal Frequency P20",
]
# 高价值属性（连通率等，融合时可赋更高权重）
HIGH_VALUE_ATTRS = [
    "Connections per Branch", "Connections per Trace", "Connection Frequency",
]
# 与 HIGH_VALUE_ATTRS 一致：显式命名为「连通性特征组」，便于 SHAP/文档引用
CONNECTIVITY_FEATURE_COLUMNS: Tuple[str, ...] = tuple(HIGH_VALUE_ATTRS)
# 默认全部候选
DEFAULT_FEATURE_COLUMNS = BASE_ATTRS + HIGH_VALUE_ATTRS


def is_connectivity_feature(name: str) -> bool:
    """是否为网格 CSV 中与网络连通性直接相关的拓扑指标列。"""
    return name in set(CONNECTIVITY_FEATURE_COLUMNS)


def suggest_regression_target_columns(
    csv_path: str,
    *,
    max_suggestions: int = 16,
) -> List[str]:
    """
    从原始网格 CSV 中挑出适合作为回归目标的数值列（方差>0、非全缺失），
    供 GUI 下拉推荐；不替代用户最终选择。
    """
    from utils.export_utils import VERTEX_COLUMNS

    vertex_like: Set[str] = set(VERTEX_COLUMNS)
    df = load_raw(csv_path)
    out: List[str] = []
    for col in df.columns:
        if col in vertex_like:
            continue
        if str(col).lower().startswith("vertex"):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() < 3:
            continue
        v = s.dropna().values
        if len(v) > 1 and float(np.std(v)) < 1e-10:
            continue
        out.append(col)
    # 优先：常见目标与连通性相关列
    priority = (
        "Fracture Intensity B21",
        "Fracture Intensity P21",
        "Connections per Branch",
        "Connections per Trace",
        "Connection Frequency",
        "Areal Frequency B20",
        "Branch Mean Length",
    )
    ranked = sorted(
        out,
        key=lambda c: (priority.index(c) if c in priority else len(priority), c),
    )
    return ranked[: max(1, max_suggestions)]


def _runtime_cfg():
    cfg = load_config()
    fe_cfg = (cfg.get("feature_engineering") or {}) if isinstance(cfg, dict) else {}
    log_cfg = (cfg.get("logging") or {}) if isinstance(cfg, dict) else {}
    logger = get_logger(
        "feature_engineering",
        level=log_cfg.get("level", "INFO"),
        log_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), log_cfg.get("file", "logs/pipeline.log")),
    )
    return fe_cfg, logger


def _clean_numeric_string(s):
    """清理字符串形式的数值，如"[7.743774E-4]"，返回纯数值。"""
    if isinstance(s, str):
        # 移除括号和空格
        s = s.strip().strip('[]')
        try:
            return float(s)
        except:
            return 0.0
    elif pd.isna(s):
        return 0.0
    else:
        return float(s)

def load_raw(csv_path: str) -> pd.DataFrame:
    """加载原始网格 CSV。"""
    df = pd.read_csv(csv_path)
    # 清理所有列中的字符串数值并转换为数值类型
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(_clean_numeric_string)
        # 确保所有列都是数值类型
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df


def normalize(
    X: np.ndarray,
    method: str = "standard",
) -> Tuple[np.ndarray, object]:
    """归一化：standard（零均值单位方差）或 minmax（0-1）。返回 (X_norm, scaler)。"""
    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, scaler


def handle_outliers(
    X: np.ndarray,
    method: str = "iqr",
    iqr_factor: float = 1.5,
    z_threshold: float = 3.0,
) -> np.ndarray:
    """
    异常值处理：iqr（四分位距截断）或 zscore（Z 分数截断）。
    原地风格：将异常值替换为边界值。
    """
    X = X.copy()
    if method == "zscore":
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std < 1e-10] = 1e-10
        low = mean - z_threshold * std
        high = mean + z_threshold * std
        X = np.clip(X, low, high)
    else:
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        iqr[iqr < 1e-10] = 1e-10
        low = q1 - iqr_factor * iqr
        high = q3 + iqr_factor * iqr
        X = np.minimum(np.maximum(X, low), high)
    return X


def select_features(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    variance_threshold: float = 1e-6,
    n_select_mi: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[int], Optional[List[str]]]:
    """
    特征筛选：先方差过滤（去掉近常数列），再可选互信息选 TopK。
    若提供 y，则用互信息；否则只做方差筛选。
    返回 (X_selected, selected_indices, selected_names)。
    """
    n_samples = X.shape[0]
    if n_samples < 2:
        # 单样本无法计算方差，跳过 VarianceThreshold
        kept = list(range(X.shape[1]))
        names = list(feature_names) if feature_names is not None else None
        return X, kept, names
    # 方差筛选
    vt = VarianceThreshold(threshold=variance_threshold)
    X_var = vt.fit_transform(X)
    kept = np.where(vt.get_support())[0].tolist()
    names_var = [feature_names[i] for i in kept] if feature_names else None
    if y is None or n_select_mi is None or n_select_mi >= len(kept):
        return X_var, kept, names_var
    # 互信息选 TopK（top_idx 为方差筛选后的局部列索引）
    mi = mutual_info_regression(X_var, y, random_state=random_state)
    top_k = min(n_select_mi, len(kept))
    top_idx = np.argsort(mi)[::-1][:top_k]
    X_mi = X_var[:, top_idx]
    kept_mi = [kept[i] for i in top_idx]
    names_mi = [names_var[i] for i in top_idx] if names_var else None
    return X_mi, kept_mi, names_mi


def build_feature_matrix(
    csv_path: str,
    feature_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    normalize_method: Optional[str] = None,
    outlier_method: Optional[str] = None,
    variance_threshold: Optional[float] = None,
    n_select_mi: Optional[int] = None,
    drop_all_nan: bool = True,
    out_processed_dir: Optional[str] = None,
    random_state: int = 42,
) -> Dict:
    """
    完整特征工程流水线：加载 → 异常值处理 → 归一化 → 特征筛选 → 输出矩阵。
    若指定 out_processed_dir，将写入 data/processed/ 下 CSV 与元数据。
    返回 dict：X, y(可选), feature_names, scaler, selected_indices, df_meta。
    """
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS
    fe_cfg, logger = _runtime_cfg()
    if normalize_method is None:
        normalize_method = fe_cfg.get("normalize_method", "standard")
    if outlier_method is None:
        outlier_method = fe_cfg.get("outlier_method", "iqr")
    if variance_threshold is None:
        variance_threshold = float(fe_cfg.get("variance_threshold", 1e-6))
    if n_select_mi is None:
        n_select_mi = fe_cfg.get("n_select_mi")
    df = load_raw(csv_path)
    # 若目标列在特征中，排除以避免泄漏
    if target_column and target_column in feature_columns:
        feature_columns = [c for c in feature_columns if c != target_column]
    available = [c for c in feature_columns if c in df.columns]
    if not available:
        raise ValueError(f"CSV 中未找到任何特征列: {feature_columns}")
    X_raw = df[available].copy()
    # fillna 前记录真正全为 NaN 的行，避免把合法零值行误删
    _all_nan_mask = X_raw.isna().all(axis=1)
    # 确保所有数据都能正确转换为数值
    def convert_to_float(value):
        if isinstance(value, str):
            # 移除括号和空格
            value = value.strip().strip('[]')
            try:
                return float(value)
            except:
                return 0.0
        elif pd.isna(value):
            return 0.0
        else:
            return float(value)
    
    # 对每个元素应用转换函数
    for col in X_raw.columns:
        X_raw[col] = X_raw[col].apply(convert_to_float)
    
    X = X_raw.values.astype(np.float64)
    y = None
    if target_column and target_column in df.columns:
        # 对目标列也应用相同的转换
        df[target_column] = df[target_column].apply(convert_to_float)
        y = df[target_column].values.astype(np.float64)
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid]
        df = df.loc[valid].copy()
        # 目标列方差过小（如 Area 恒定）会导致 R² 爆炸、模型失效
        if y is not None and len(y) > 1 and np.std(y) < 1e-10:
            raise ValueError(
                f"目标列「{target_column}」几乎无变化（方差≈0），无法用于回归。"
                f"请选择有数值变化的列，如 Fracture Intensity B21、Connections per Branch 等。"
            )
    elif drop_all_nan:
        # 仅过滤特征全为 NaN 的行，不把合法的 0 值行当作无效行
        valid = ~_all_nan_mask.values
        X = X[valid]
        df = df.loc[valid].copy()
    X = handle_outliers(X, method=outlier_method)
    X_norm, scaler = normalize(X, method=normalize_method)
    X_sel, sel_idx, sel_names = select_features(
        X_norm, y,
        feature_names=available,
        variance_threshold=variance_threshold,
        n_select_mi=n_select_mi,
        random_state=random_state,
    )
    result = {
        "X": X_sel,
        "y": y,
        "feature_names": sel_names or [available[i] for i in sel_idx],
        "scaler": scaler,
        "selected_indices": sel_idx,
        "df": df,
        "n_samples": X_sel.shape[0],
        "n_features": X_sel.shape[1],
    }
    if out_processed_dir:
        os.makedirs(out_processed_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_csv = os.path.join(out_processed_dir, f"{base}_feature_matrix.csv")
        out_df = pd.DataFrame(X_sel, columns=result["feature_names"])
        if y is not None:
            out_df[target_column] = y
        out_df.to_csv(out_csv, index=False)
        result["path_feature_matrix"] = out_csv
    logger.info(
        "特征工程完成：csv=%s samples=%s features=%s target=%s",
        csv_path,
        result["n_samples"],
        result["n_features"],
        target_column,
    )
    return result


if __name__ == "__main__":
    import sys
    csv_path = "Yingmai 2 area in Tarim Basin.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    prog_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(prog_dir, "data", "processed")
    r = build_feature_matrix(
        csv_path,
        outlier_method="iqr",
        n_select_mi=10,
        out_processed_dir=out_dir,
    )
    print("特征矩阵形状:", r["X"].shape)
    print("选中特征:", r["feature_names"])
    if r.get("path_feature_matrix"):
        print("已写出:", r["path_feature_matrix"])
