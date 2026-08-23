# -*- coding: utf-8 -*-
"""
XGBoost 从拓扑属性 + 空间特征预测渗流关键性。
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from multiperiod_data import (
    load_all_periods, get_topology_matrix, TOPOLOGY_ATTRIBUTES,
)
from percolation import build_grid_graph, GRID_STEP

from utils.logging_utils import get_logger

logger = get_logger("agent_model")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# 诚实评估共享常量与工具
# ---------------------------------------------------------------------------
# 代理模型统一超参（train_agent_model 与 CV/LOPO 折内模型共用）
AGENT_XGB_PARAMS: Dict = {
    "n_estimators": 800,
    "max_depth": 8,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "min_child_weight": 3,
    "gamma": 0.05,
    "random_state": 42,
    "verbosity": 0,
}

# 训练 df 的元数据列：不作为特征进入模型
AGENT_EXCLUDE_COLS = {
    "period",
    "betweenness",
    "log1p_betweenness",
    "log_betweenness",
    "cell_x",
    "cell_y",
}


def _feature_cols(df) -> List[str]:
    """从训练 df 推导特征列（排除全部元数据列）。"""
    return [c for c in df.columns if c not in AGENT_EXCLUDE_COLS]


def _build_block_ids(cell_x: np.ndarray, cell_y: np.ndarray,
                     n_blocks: int = 9) -> Optional[np.ndarray]:
    """
    等宽分箱空间分块（与 ml/train._build_spatial_block_ids 同算法，通用坐标版）。
    坐标 x/y 各分 n_side 箱，组合成 n_side² 个空间块；
    同一块的网格在空间块 CV 中始终同折，避免空间自相关泄漏。
    无空间差异（<2 唯一块）时返回 None。
    """
    if len(cell_x) == 0 or len(cell_y) != len(cell_x):
        return None
    n_side = max(2, int(np.ceil(np.sqrt(max(2, int(n_blocks))))))
    try:
        xbin = pd.Series(pd.cut(cell_x, bins=n_side, labels=False, include_lowest=True),
                         dtype="float64")
        ybin = pd.Series(pd.cut(cell_y, bins=n_side, labels=False, include_lowest=True),
                         dtype="float64")
    except Exception:
        return None
    if xbin.isna().all() or ybin.isna().all():
        return None
    xbin = xbin.fillna(0).astype(int)
    ybin = ybin.fillna(0).astype(int)
    block_id = (xbin * n_side + ybin).to_numpy(dtype=np.int64)
    if len(np.unique(block_id)) < 2:
        return None
    return block_id


def spatial_cv_evaluate(
    df,
    target_col: str = "log1p_betweenness",
    n_blocks: int = 9,
    n_splits: int = 5,
    xgb_params: Optional[Dict] = None,
) -> Dict:
    """
    空间分块交叉验证：按空间块分组留一（GroupKFold），
    消除随机划分中相邻网格自相关导致的 R² 虚高。
    返回各折指标 mean±std 及块数/折数；无法分块时返回 {} 并告警。

    ponytail: 折内不用早停（无内层验证集）；若空间 CV 明显劣化再考虑折内调参。
    """
    if not HAS_XGB:
        raise ImportError("需要 xgboost")
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    block_id = _build_block_ids(
        pd.to_numeric(df["cell_x"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df["cell_y"], errors="coerce").to_numpy(dtype=float),
        n_blocks,
    )
    if block_id is None:
        logger.warning("spatial_cv_evaluate: 无法构建空间块，跳过")
        return {}

    params = dict(AGENT_XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)

    feats = _feature_cols(df)
    X = df[feats].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    uniq = np.unique(block_id)
    n_use = int(min(max(2, int(n_splits)), len(uniq)))
    gkf = GroupKFold(n_splits=n_use)

    fold_metrics: List[Dict[str, float]] = []
    for tr_idx, te_idx in gkf.split(X, y, groups=block_id):
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[te_idx])
        fold_metrics.append({
            "r2": r2_score(y[te_idx], pred),
            "rmse": float(np.sqrt(mean_squared_error(y[te_idx], pred))),
            "mae": mean_absolute_error(y[te_idx], pred),
        })

    out: Dict = {"n_blocks_used": int(len(uniq)), "n_splits_used": n_use}
    for k in ("r2", "rmse", "mae"):
        vals = np.array([fm[k] for fm in fold_metrics], dtype=float)
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    logger.info("空间块 CV: R²=%.4f±%.4f (blocks=%d, splits=%d)",
                out["r2_mean"], out["r2_std"], out["n_blocks_used"], out["n_splits_used"])
    return out


def _lopo_split(unique_periods) -> List[Tuple[List, object]]:
    """留一期外推划分：返回 [(train_periods(list), test_period)]，互斥且覆盖全集。"""
    periods = list(unique_periods)
    return [
        ([p for p in periods if p != test_p], test_p)
        for test_p in periods
    ]


def ranking_metrics(
    y_true,
    y_pred,
    top_fractions: Tuple[float, ...] = (0.05, 0.10, 0.20),
) -> Dict:
    """计算全局秩相关与高值 Top-K 排序一致性。

    Top-K 并列值通过原始行号稳定打破，保证同一输入可复现。NDCG 使用
    非负真实相关性和标准对数折损；该函数不拟合模型，可独立单测。
    """
    from scipy.stats import kendalltau, spearmanr

    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(true) != len(pred) or len(true) < 2:
        raise ValueError("y_true/y_pred 必须等长且至少包含2个样本")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("排序指标不接受 NaN 或无穷值")

    spearman = float(spearmanr(true, pred).statistic)
    kendall = float(kendalltau(true, pred).statistic)
    degenerate = not np.isfinite(spearman) or not np.isfinite(kendall)
    if not np.isfinite(spearman):
        spearman = 0.0
    if not np.isfinite(kendall):
        kendall = 0.0

    indices = np.arange(len(true), dtype=int)
    true_order = np.lexsort((indices, -true))
    pred_order = np.lexsort((indices, -pred))
    relevance = true - float(np.min(true))

    out: Dict = {
        "status": "degenerate" if degenerate else "ok",
        "n_samples": int(len(true)),
        "spearman": spearman,
        "kendall": kendall,
    }
    for fraction in top_fractions:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError(f"非法 Top-K 比例: {fraction}")
        k = max(1, int(np.ceil(len(true) * float(fraction))))
        true_top = set(true_order[:k].tolist())
        pred_top = set(pred_order[:k].tolist())
        overlap = len(true_top & pred_top) / float(k)

        discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
        dcg = float(np.sum(relevance[pred_order[:k]] * discounts))
        ideal = float(np.sum(relevance[true_order[:k]] * discounts))
        ndcg = 1.0 if ideal <= 0.0 else dcg / ideal

        label = f"{int(round(float(fraction) * 100))}pct"
        out[f"top_{label}_k"] = int(k)
        out[f"top_{label}_overlap"] = float(overlap)
        out[f"ndcg_{label}"] = float(ndcg)
    return out


def leave_one_period_out_evaluate(
    df,
    target_col: str = "log1p_betweenness",
    xgb_params: Optional[Dict] = None,
) -> Dict:
    """
    Leave-One-Period-Out 跨期外推评估：
    每次留出一个完整时期作测试集、其余时期训练，
    度量「对未见断裂网络的泛化能力」。
    返回 per_period 逐期指标与聚合 r2_mean/r2_std；时期数 <2 时返回 {}。

    ponytail: 各期网格规模不同，逐期指标才是科学结论主体；均值仅作总览。
    """
    if not HAS_XGB:
        raise ImportError("需要 xgboost")
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    periods = sorted(df["period"].unique())
    if len(periods) < 2:
        logger.warning("leave_one_period_out_evaluate: 时期数 <2，跳过")
        return {}

    params = dict(AGENT_XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)

    feats = _feature_cols(df)
    per_period: Dict[str, Dict] = {}
    r2_list: List[float] = []
    rank_rows: List[Dict] = []
    for train_ps, test_p in _lopo_split(periods):
        tr = df[df["period"].isin(train_ps)]
        te = df[df["period"] == test_p]
        m = xgb.XGBRegressor(**params)
        m.fit(tr[feats].to_numpy(dtype=float), tr[target_col].to_numpy(dtype=float))
        pred = m.predict(te[feats].to_numpy(dtype=float))
        y_true = te[target_col].to_numpy(dtype=float)
        metrics = {
            "r2": float(r2_score(y_true, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "mae": float(mean_absolute_error(y_true, pred)),
            "n_test": int(len(te)),
            "ranking": ranking_metrics(y_true, pred),
        }
        per_period[str(test_p)] = metrics
        r2_list.append(metrics["r2"])
        rank_rows.append(metrics["ranking"])
        logger.info("LOPO %s: R²=%.4f RMSE=%.4f (n=%d)",
                    test_p, metrics["r2"], metrics["rmse"], metrics["n_test"])

    rank_keys = [
        "spearman", "kendall",
        "top_5pct_overlap", "top_10pct_overlap", "top_20pct_overlap",
        "ndcg_5pct", "ndcg_10pct", "ndcg_20pct",
    ]
    rank_aggregate = {
        f"{key}_mean": float(np.mean([row[key] for row in rank_rows]))
        for key in rank_keys
    }
    return {
        "per_period": per_period,
        "r2_mean": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list)),
        "ranking_aggregate": rank_aggregate,
    }


# ---------------------------------------------------------------------------
# 空间特征工程
# ---------------------------------------------------------------------------
def _compute_neighbor_features(
    gdf,
    X_mat: np.ndarray,
    cols: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    计算每个网格的邻域统计特征（4连通邻居）：
      - 每个拓扑属性的邻居均值、最大值
      - 非零邻居数量（局部密度）
    """
    n = len(gdf)
    # 获取网格坐标
    centroids = gdf.geometry.centroid
    xs = np.array([p.x for p in centroids])
    ys = np.array([p.y for p in centroids])

    # 构建坐标→索引映射（用于快速查找邻居）
    coord_to_idx = {}
    for i in range(n):
        key = (round(xs[i], 1), round(ys[i], 1))
        coord_to_idx[key] = i

    # 网格步长（复用 percolation.GRID_STEP，可经 config 覆盖）
    step = GRID_STEP
    tol = step * 0.5  # 容差

    # 计算邻居特征
    neighbor_mean = np.zeros((n, len(cols)))
    neighbor_max = np.zeros((n, len(cols)))
    neighbor_count = np.zeros(n)  # 非零邻居数量

    # 4连通方向偏移
    offsets = [(step, 0), (-step, 0), (0, step), (0, -step)]

    for i in range(n):
        xi, yi = xs[i], ys[i]
        neighbors = []
        for dx, dy in offsets:
            nx_coord = xi + dx
            ny_coord = yi + dy
            # 在容差范围内查找
            for check_x in [nx_coord - tol, nx_coord, nx_coord + tol]:
                for check_y in [ny_coord - tol, ny_coord, ny_coord + tol]:
                    key = (round(check_x, 1), round(check_y, 1))
                    if key in coord_to_idx:
                        j = coord_to_idx[key]
                        if j != i:
                            neighbors.append(j)

        if neighbors:
            neighbor_vals = X_mat[neighbors]
            neighbor_mean[i] = neighbor_vals.mean(axis=0)
            neighbor_max[i] = neighbor_vals.max(axis=0)
            neighbor_count[i] = sum(1 for j in neighbors if X_mat[j].sum() > 0)

    # 新特征列名
    new_cols = []
    for col in cols:
        new_cols.append(f"{col}_nbr_mean")
    for col in cols:
        new_cols.append(f"{col}_nbr_max")
    new_cols.append("nonzero_neighbor_count")

    features = np.column_stack([neighbor_mean, neighbor_max, neighbor_count])
    return features, new_cols


def _compute_spatial_features(gdf) -> Tuple[np.ndarray, List[str]]:
    """
    计算空间位置特征：
      - 归一化 x, y 坐标
      - 距中心距离
    """
    centroids = gdf.geometry.centroid
    xs = np.array([p.x for p in centroids])
    ys = np.array([p.y for p in centroids])

    # 归一化到 [0, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    x_norm = (xs - x_min) / x_range
    y_norm = (ys - y_min) / y_range

    # 距中心距离
    cx, cy = x_norm.mean(), y_norm.mean()
    dist_center = np.sqrt((x_norm - cx)**2 + (y_norm - cy)**2)

    features = np.column_stack([x_norm, y_norm, dist_center])
    cols = ["x_norm", "y_norm", "dist_to_center"]
    return features, cols


# ---------------------------------------------------------------------------
# 训练数据构建
# ---------------------------------------------------------------------------
def build_agent_training_data(
    edge_weight_col: str = "NC_A",
    use_spatial_features: bool = True,
    use_neighbor_features: bool = True,
    distance_transform: str = "inverse",
) -> pd.DataFrame:
    """
    构建代理模型训练数据 v2：
      特征：6个拓扑属性 + 空间位置 + 邻域统计
      标签：节点的 betweenness centrality（分位数变换后）
    """
    if not HAS_NX:
        raise ImportError("需要 networkx")

    period_gdfs = load_all_periods()
    all_rows = []

    for period_name, gdf in period_gdfs.items():
        logger.info("构建图并计算 betweenness: %s (%d 网格)", period_name, len(gdf))
        G = build_grid_graph(
            gdf,
            edge_weight_col=edge_weight_col,
            weight_mode="min",
            distance_transform=distance_transform,
        )

        # 计算 betweenness（精确计算，避免近似误差）
        n_nodes = G.number_of_nodes()
        bc_raw = nx.betweenness_centrality(G, weight="distance",
                                            normalized=False)
        logger.info("  精确 betweenness 计算完成: %d 节点", n_nodes)

        # 原始质心坐标（元数据，用于空间分块评估，不进特征）
        centroids = gdf.geometry.centroid
        cell_xs = np.array([p.x for p in centroids], dtype=float)
        cell_ys = np.array([p.y for p in centroids], dtype=float)

        # 提取拓扑特征
        _, X_mat, cols = get_topology_matrix(gdf)

        # 计算空间特征
        spatial_feat, spatial_cols = _compute_spatial_features(gdf)

        # 计算邻域特征
        if use_neighbor_features:
            nbr_feat, nbr_cols = _compute_neighbor_features(gdf, X_mat, cols)
        else:
            nbr_feat, nbr_cols = np.zeros((len(gdf), 0)), []

        for i in range(len(gdf)):
            bc_val = bc_raw.get(i, 0.0)
            # 包含所有网格（零值网格也有 betweenness，作为负样本）
            # 但排除完全孤立的零值网格（betweenness=0且拓扑全零）
            if X_mat[i].sum() <= 0 and bc_val < 1e-10:
                continue

            row = {col: float(X_mat[i, j]) for j, col in enumerate(cols)}

            # 添加空间特征
            if use_spatial_features:
                for j, col in enumerate(spatial_cols):
                    row[col] = float(spatial_feat[i, j])

            # 添加邻域特征
            if use_neighbor_features:
                for j, col in enumerate(nbr_cols):
                    row[col] = float(nbr_feat[i, j])

            row["period"] = period_name
            row["cell_x"] = float(cell_xs[i])
            row["cell_y"] = float(cell_ys[i])
            row["betweenness"] = bc_val
            row["log1p_betweenness"] = np.log1p(bc_val)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    logger.info("训练数据: %d 条, %d 列", len(df), len(df.columns))

    # 添加交互特征（在 DataFrame 上操作，避免逐行计算）
    df = _add_interaction_features(df, cols)

    return df


def _add_interaction_features(df: pd.DataFrame, topo_cols: List[str]) -> pd.DataFrame:
    """
    添加拓扑属性的交互特征，帮助模型学习非线性关系。
    """
    # 乘积特征（捕捉协同效应）
    df["NC_A_x_NC_NB"] = df["NC_A"] * df["NC_NB"]
    df["NC_A_x_NC_NL"] = df["NC_A"] * df["NC_NL"]
    df["NB_A_x_NC_NB"] = df["NB_A"] * df["NC_NB"]
    df["NC_NB_x_NC_NL"] = df["NC_NB"] * df["NC_NL"]

    # 比值特征（捕捉结构比例）
    eps = 1e-10  # 防止除零
    df["NC_A_div_NL_A"] = df["NC_A"] / (df["NL_A"] + eps)
    df["NB_A_div_NL_A"] = df["NB_A"] / (df["NL_A"] + eps)
    df["NC_NB_div_NC_NL"] = df["NC_NB"] / (df["NC_NL"] + eps)

    # 非线性变换（帮助模型拟合极端值）
    df["NC_A_sqrt"] = np.sqrt(df["NC_A"].clip(lower=0))
    df["NC_NB_log1p"] = np.log1p(df["NC_NB"].clip(lower=0))
    df["NC_A_sq"] = df["NC_A"] ** 2

    logger.info("添加交互特征: %d → %d 列", len(topo_cols) + 3, len(df.columns))
    return df


# ---------------------------------------------------------------------------
# 代理模型训练（改进版）
# ---------------------------------------------------------------------------
def train_agent_model(
    df: pd.DataFrame,
    target_col: str = "log1p_betweenness",
    test_size: float = 0.2,
) -> dict:
    """
    训练 XGBoost 代理模型 v2：
      - 使用所有可用特征（拓扑 + 空间 + 邻域）
      - 增大模型容量 + early stopping
      - 多维度评估（R²、MAE、分类准确率）
    """
    # 确定特征列（元数据列统一排除）
    feature_cols = _feature_cols(df)
    logger.info("使用特征: %d 个", len(feature_cols))
    logger.info("特征列表: %s", feature_cols)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    # 划分
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # 进一步划分验证集用于 early stopping
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    model = xgb.XGBRegressor(**AGENT_XGB_PARAMS)

    model.fit(
        X_train_sub, y_train_sub,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_test)

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # 分段 R²（基于目标值的分位数）
    quantiles = np.percentile(y_test, [33, 66])
    low_mask = y_test <= quantiles[0]
    mid_mask = (y_test > quantiles[0]) & (y_test <= quantiles[1])
    high_mask = y_test > quantiles[1]

    r2_low = r2_score(y_test[low_mask], y_pred[low_mask]) if low_mask.sum() > 5 else float("nan")
    r2_mid = r2_score(y_test[mid_mask], y_pred[mid_mask]) if mid_mask.sum() > 5 else float("nan")
    r2_high = r2_score(y_test[high_mask], y_pred[high_mask]) if high_mask.sum() > 5 else float("nan")

    # 分类准确率：将betweenness分为高/中/低三类，看预测是否能区分
    # 使用更宽松的阈值
    high_thresh = np.percentile(y_test, 75)
    low_thresh = np.percentile(y_test, 25)
    y_test_class = np.where(y_test >= high_thresh, 2, np.where(y_test >= low_thresh, 1, 0))
    y_pred_class = np.where(y_pred >= high_thresh, 2, np.where(y_pred >= low_thresh, 1, 0))
    class_acc = np.mean(y_test_class == y_pred_class)

    # 预测值在真实值±0.5范围内的比例
    within_05 = np.mean(np.abs(y_pred - y_test) < 0.5)
    within_10 = np.mean(np.abs(y_pred - y_test) < 1.0)

    metrics = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "r2_low": r2_low,
        "r2_mid": r2_mid,
        "r2_high": r2_high,
        "class_accuracy": class_acc,
        "within_0.5": within_05,
        "within_1.0": within_10,
    }
    logger.info("代理模型 R²=%.4f  RMSE=%.4f  MAE=%.4f", r2, rmse, mae)
    logger.info("分段 R²: 低=%.3f  中=%.3f  高=%.3f", r2_low, r2_mid, r2_high)
    logger.info("分类准确率: %.3f  ±0.5准确率: %.3f  ±1.0准确率: %.3f",
                class_acc, within_05, within_10)

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model": model,
        "feature_importance": imp_df,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# 可视化（改进版）
# ---------------------------------------------------------------------------
def plot_agent_results(result: dict, out_dir: str) -> Dict[str, str]:
    """生成代理模型结果图 v2。"""
    paths = {}
    if not HAS_MPL:
        return paths

    try:
        from utils.matplotlib_chinese import setup_matplotlib_chinese
        setup_matplotlib_chinese()
    except Exception:
        pass

    imp = result["feature_importance"]
    y_test = result["y_test"]
    y_pred = result["y_pred"]
    metrics = result["metrics"]

    # --- 特征重要性（Top 15）---
    fig, ax = plt.subplots(figsize=(10, 7))
    top_imp = imp.head(15)
    colors = ["#E63946" if "nbr" in f or "spatial" in f or "norm" in f or "dist" in f
              else "#457B9D" for f in top_imp["feature"]]
    ax.barh(range(len(top_imp)), top_imp["importance"].values,
            color=colors, alpha=0.85)
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(top_imp["feature"].values, fontsize=9)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"特征重要性 (R²={metrics['r2']:.4f})", fontsize=13)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # 图例
    legend_elements = [
        Patch(facecolor="#E63946", label="空间/邻域特征"),
        Patch(facecolor="#457B9D", label="拓扑属性"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    imp_path = os.path.join(out_dir, "agent_feature_importance.png")
    fig.savefig(imp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["importance"] = imp_path

    # --- 预测 vs 真实散点图 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：全范围
    ax = axes[0]
    ax.scatter(y_test, y_pred, s=10, alpha=0.4, c="#457B9D", edgecolors="none")
    lim_min = min(y_test.min(), y_pred.min())
    lim_max = max(y_test.max(), y_pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1.5, alpha=0.8,
            label="理想预测线")
    ax.set_xlabel("真实值 log(betweenness+1)", fontsize=11)
    ax.set_ylabel("预测值 log(betweenness+1)", fontsize=11)
    ax.set_title(f"预测 vs 真实 (R²={metrics['r2']:.4f})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # 右图：残差分布
    ax = axes[1]
    residuals = y_pred - y_test
    ax.scatter(y_test, residuals, s=10, alpha=0.4, c="#2A9D8F", edgecolors="none")
    ax.axhline(y=0, color="r", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("真实值 log(betweenness+1)", fontsize=11)
    ax.set_ylabel("残差 (预测 - 真实)", fontsize=11)
    ax.set_title("残差分析", fontsize=12)
    ax.grid(True, alpha=0.3)

    # 添加分段 R² 标注
    r2_text = (f"分段 R²:\n"
               f"  低值区: {metrics.get('r2_low', 0):.3f}\n"
               f"  中值区: {metrics.get('r2_mid', 0):.3f}\n"
               f"  高值区: {metrics.get('r2_high', 0):.3f}")
    ax.text(0.02, 0.98, r2_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    pred_path = os.path.join(out_dir, "agent_pred_vs_true.png")
    fig.savefig(pred_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["pred_vs_true"] = pred_path

    # --- 目标变量分布图 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(y_test, bins=50, color="#457B9D", alpha=0.7, edgecolor="white")
    ax.set_xlabel("log(betweenness+1)", fontsize=11)
    ax.set_ylabel("频次", fontsize=11)
    ax.set_title("目标变量分布", fontsize=12)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(y_test[y_test > 0], bins=50, color="#E63946", alpha=0.7, edgecolor="white")
    ax.set_xlabel("log(betweenness+1)", fontsize=11)
    ax.set_ylabel("频次", fontsize=11)
    ax.set_title("目标变量分布（仅非零值）", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    dist_path = os.path.join(out_dir, "agent_target_distribution.png")
    fig.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["target_distribution"] = dist_path

    # --- SHAP 可解释性分析 ---
    try:
        import shap
        logger.info("计算 SHAP 值...")
        # 使用测试集的一个子集计算 SHAP（避免内存问题）
        X_sample = result["X_test"][:min(500, len(result["X_test"]))]

        # 使用 KernelExplainer（通用方法，兼容性更好）
        # 先用 kmeans 聚类中心作为背景数据
        background = shap.kmeans(X_sample, 50)
        explainer = shap.KernelExplainer(result["model"].predict, background)
        shap_values = explainer.shap_values(X_sample[:100])  # 只取100个样本加速

        # SHAP Summary Plot（蜂群图）
        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_sample[:100],
            feature_names=result["feature_cols"],
            show=False, max_display=15
        )
        plt.title("SHAP 特征重要性（蜂群图）", fontsize=13)
        plt.tight_layout()
        shap_path = os.path.join(out_dir, "agent_shap_summary.png")
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        paths["shap_summary"] = shap_path

        # SHAP Bar Plot（平均绝对SHAP值）
        fig = plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values, X_sample[:100],
            feature_names=result["feature_cols"],
            plot_type="bar", show=False, max_display=15
        )
        plt.title("SHAP 平均特征重要性", fontsize=13)
        plt.tight_layout()
        shap_bar_path = os.path.join(out_dir, "agent_shap_bar.png")
        plt.savefig(shap_bar_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        paths["shap_bar"] = shap_bar_path

        logger.info("SHAP 分析完成: %d 张图", 2)
    except Exception as e:
        logger.warning("SHAP 分析失败: %s", e)
        import traceback
        traceback.print_exc()

    return paths


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_agent_pipeline(
    out_dir: Optional[str] = None,
    edge_weight_col: str = "NC_A",
) -> dict:
    """
    Stage 4：代理模型主流程 v2。
    1. 构建训练数据（拓扑 + 空间 + 邻域特征）
    2. 训练 XGBoost 模型
    3. 评估 + 可视化
    """
    if not HAS_XGB:
        raise ImportError("代理模型需要 xgboost: pip install xgboost")

    if out_dir is None:
        out_dir = os.path.join(_THIS_DIR, "data", "processed", "multiperiod")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=== Stage 4: Agent Model v2 ===")

    # 1) 构建训练数据
    df = build_agent_training_data(edge_weight_col=edge_weight_col)

    # 2) 训练
    result = train_agent_model(df)

    # 3) 诚实评估：随机划分 vs 空间块 CV vs 留一期外推
    spatial_cv = spatial_cv_evaluate(df)
    lopo = leave_one_period_out_evaluate(df)
    result["spatial_cv"] = spatial_cv
    result["lopo"] = lopo

    logger.info("=== 诚实评估对比 ===")
    logger.info("  随机划分 R²: %.4f（相邻网格泄漏，可能虚高）", result["metrics"]["r2"])
    if spatial_cv:
        logger.info("  空间块 CV R²: %.4f±%.4f", spatial_cv["r2_mean"], spatial_cv["r2_std"])
    if lopo:
        logger.info("  留一期外推 R²: %.4f±%.4f（跨期泛化）", lopo["r2_mean"], lopo["r2_std"])

    summary = {
        "random_split": dict(result["metrics"]),
        "spatial_cv": spatial_cv,
        "lopo": lopo,
    }
    eval_json = os.path.join(out_dir, "agent_eval_summary.json")
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("诚实评估摘要: %s", eval_json)

    # 4) 导出
    imp_csv = os.path.join(out_dir, "agent_feature_importance.csv")
    result["feature_importance"].to_csv(imp_csv, index=False, encoding="utf-8-sig")
    logger.info("特征重要性: %s", imp_csv)

    # 5) 可视化
    plot_paths = plot_agent_results(result, out_dir)
    for k, v in plot_paths.items():
        logger.info("  %s: %s", k, v)

    result["_plot_paths"] = plot_paths
    result["_out_dir"] = out_dir
    return result


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = run_agent_pipeline()
    print(f"\n=== Agent Model v2 ===")
    print(f"R² = {result['metrics']['r2']:.4f}")
    print(f"RMSE = {result['metrics']['rmse']:.4f}")
    print(f"分段 R²: 低={result['metrics']['r2_low']:.3f}  中={result['metrics']['r2_mid']:.3f}  高={result['metrics']['r2_high']:.3f}")
    print("\n特征重要性 (Top 10):")
    print(result["feature_importance"].head(10).to_string(index=False))
