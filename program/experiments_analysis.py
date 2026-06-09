# -*- coding: utf-8 -*-
"""
实验与分析模块（不影响主界面功能）。

包含：
- 3.1 消融实验（Ablation Study）：
  * 仅几何/属性特征
  * 仅拓扑特征
  * 几何 + 拓扑（融合）
  * 几何 + 拓扑 + 图嵌入（GNN）
- 3.2 稳健性 / 泛化性实验：
  * 空间留一交叉验证（spatial CV）
  * 噪声敏感性实验
- 3.3 算法族对比：
  * XGBoost、随机森林、简单 GNN / 规则融合基线

注意：
- 该文件仅做离线实验与绘图，不在主界面中调用，避免影响日常使用。
- 建议在 program 目录下运行：cd program && python experiments_analysis.py
"""

import os
import sys

# 保证 program 目录在 path 中，便于导入 feature_engineering、fusion_algorithm 等
_PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)

from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from feature_engineering import build_feature_matrix
from fusion_algorithm import weighted_fusion, build_grid_graph
from gnn_embeddings import gnn_embedding_graphsage
from ml.train import train_xgboost_regression
from evaluation import regression_metrics


def _split_geometry_topology_columns(
    df: pd.DataFrame,
    topo_keywords: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    简单按列名拆分「几何/属性特征」与「拓扑特征」。
    topo_keywords 命中则视为拓扑特征，其余视为几何/属性。
    """
    if topo_keywords is None:
        topo_keywords = [
            "Connections",
            "Connection",
            "Intensity",
            "Frequency",
            "Dimensionless",
            "Branch",
            "Trace",
            "Node",
        ]
    topo_cols: List[str] = []
    geom_cols: List[str] = []
    for c in df.columns:
        if any(k in c for k in topo_keywords):
            topo_cols.append(c)
        else:
            geom_cols.append(c)
    return geom_cols, topo_cols


# ---------- 3.1 消融实验 ----------


def run_ablation_experiments(
    csv_path: str,
    target_column: str,
    out_dir: Optional[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    3.1 消融实验：几何/拓扑/融合/融合+图嵌入 四种配置对比。

    返回：
        DataFrame，每行一个配置，包含 R2、MAE、RMSE 等指标。
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(csv_path), "data", "experiments")
    os.makedirs(out_dir, exist_ok=True)

    # 使用完整特征工程后再做列筛选，保证预处理一致
    r = build_feature_matrix(
        csv_path,
        target_column=target_column,
        out_processed_dir=None,
    )
    X_full = r["X"]
    y = r["y"]
    df_meta = r["df"]
    feature_names = r["feature_names"]

    # 基于原始 CSV 列名划分几何 vs 拓扑（再映射到选中特征）
    df_raw = pd.read_csv(csv_path)
    geom_all, topo_all = _split_geometry_topology_columns(df_raw)
    geom_cols = [c for c in feature_names if c in geom_all]
    topo_cols = [c for c in feature_names if c in topo_all]

    idx_geom = [feature_names.index(c) for c in geom_cols] if geom_cols else []
    idx_topo = [feature_names.index(c) for c in topo_cols] if topo_cols else []

    def _subset(X: np.ndarray, idx: List[int]) -> np.ndarray:
        return X[:, idx] if idx else np.zeros((len(X), 0))

    rows: List[Dict[str, Any]] = []

    # 1) 仅几何/属性特征
    if idx_geom:
        res_geom = train_xgboost_regression(_subset(X_full, idx_geom), y, random_state=random_state)
        rows.append({"config": "geom_only", **res_geom["cv_agg"], **{f"test_{k}": v for k, v in res_geom["test_metrics"].items()}})

    # 2) 仅拓扑特征
    if idx_topo:
        res_topo = train_xgboost_regression(_subset(X_full, idx_topo), y, random_state=random_state)
        rows.append({"config": "topo_only", **res_topo["cv_agg"], **{f"test_{k}": v for k, v in res_topo["test_metrics"].items()}})

    # 3) 几何 + 拓扑（融合）
    res_all = train_xgboost_regression(X_full, y, random_state=random_state)
    rows.append({"config": "geom_topo_fusion", **res_all["cv_agg"], **{f"test_{k}": v for k, v in res_all["test_metrics"].items()}})

    # 4) 几何 + 拓扑 + 图嵌入（GraphSAGE）
    try:
        edge_index, _ = build_grid_graph(n_nodes=X_full.shape[0])
        Z_node, g_graph, gnn_metrics = gnn_embedding_graphsage(X_full, edge_index)
        X_gnn = np.concatenate([X_full, Z_node], axis=1)
        res_gnn = train_xgboost_regression(X_gnn, y, random_state=random_state)
        rows.append(
            {
                "config": "geom_topo_gnn",
                **res_gnn["cv_agg"],
                **{f"test_{k}": v for k, v in res_gnn["test_metrics"].items()},
            }
        )
    except ImportError:
        pass

    df_res = pd.DataFrame(rows)
    df_res_path = os.path.join(out_dir, f"ablation_{target_column}.csv")
    df_res.to_csv(df_res_path, index=False)
    return df_res


# ---------- 3.2 稳健性 / 泛化性 ----------


def spatial_cv_split(
    df: pd.DataFrame,
    n_folds: int = 4,
    row_col_cols: Tuple[str, str] = ("row", "col"),
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    简单按网格块做 spatial CV：将行列网格划分为 n_folds×n_folds 个大块。

    返回：
        folds: 列表，每个元素是 (train_idx, test_idx)。
    """
    row_col_missing = [c for c in row_col_cols if c not in df.columns]
    if row_col_missing:
        raise ValueError(f"数据中缺少行列信息列: {row_col_missing}，无法做 spatial CV。")
    row, col = df[row_col_cols[0]].values, df[row_col_cols[1]].values
    # 归一化到 0-1，再等分
    r_norm = (row - row.min()) / max(1e-6, row.max() - row.min())
    c_norm = (col - col.min()) / max(1e-6, col.max() - col.min())
    r_bin = np.floor(r_norm * n_folds).astype(int)
    c_bin = np.floor(c_norm * n_folds).astype(int)
    r_bin = np.clip(r_bin, 0, n_folds - 1)
    c_bin = np.clip(c_bin, 0, n_folds - 1)
    block_id = r_bin * n_folds + c_bin
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for b in np.unique(block_id):
        test_idx = np.where(block_id == b)[0]
        train_idx = np.where(block_id != b)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        folds.append((train_idx, test_idx))
    return folds


def run_spatial_cv_experiment(
    csv_path: str,
    target_column: str,
    n_folds: int = 4,
    row_col_cols: Tuple[str, str] = ("row", "col"),
) -> pd.DataFrame:
    """
    空间留一交叉验证：按大网格块留一，评估跨区泛化能力。
    """
    r = build_feature_matrix(csv_path, target_column=target_column, out_processed_dir=None)
    X = r["X"]
    y = r["y"]
    df_meta = r["df"].copy()
    folds = spatial_cv_split(df_meta, n_folds=n_folds, row_col_cols=row_col_cols)
    rows: List[Dict[str, Any]] = []
    for i, (train_idx, test_idx) in enumerate(folds):
        from sklearn.ensemble import RandomForestRegressor

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        metrics = regression_metrics(y_te, y_pred)
        rows.append({"fold": i, **metrics, "n_train": len(train_idx), "n_test": len(test_idx)})
    return pd.DataFrame(rows)


def run_noise_sensitivity_experiment(
    csv_path: str,
    target_column: str,
    noise_cols_keywords: Optional[List[str]] = None,
    noise_levels: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    噪声敏感性实验：在指定列上加入不同强度的高斯噪声，观察指标变化。
    """
    if noise_cols_keywords is None:
        noise_cols_keywords = ["Intensity", "Frequency"]
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.2]

    r = build_feature_matrix(csv_path, target_column=target_column, out_processed_dir=None)
    X = r["X"]
    y = r["y"]
    feature_names = r["feature_names"]

    target_idx = [i for i, name in enumerate(feature_names) if any(k in name for k in noise_cols_keywords)]
    if not target_idx:
        raise ValueError(f"未在特征中找到包含关键字 {noise_cols_keywords} 的列。")

    rows: List[Dict[str, Any]] = []
    for sigma in noise_levels:
        X_noisy = X.copy()
        if sigma > 0:
            noise = np.random.normal(loc=0.0, scale=sigma, size=(X.shape[0], len(target_idx)))
            X_noisy[:, target_idx] = X_noisy[:, target_idx] * (1.0 + noise)
        res = train_xgboost_regression(X_noisy, y, random_state=42)
        rows.append(
            {
                "noise_sigma": sigma,
                **res["cv_agg"],
                **{f"test_{k}": v for k, v in res["test_metrics"].items()},
            }
        )
    return pd.DataFrame(rows)


# ---------- 3.3 算法族对比 ----------


def run_model_family_comparison(
    csv_path: str,
    target_column: str,
) -> pd.DataFrame:
    """
    3.3 算法族对比：XGBoost / 随机森林 / 简单规则基线 / GNN+XGBoost。
    """
    from sklearn.ensemble import RandomForestRegressor

    r = build_feature_matrix(csv_path, target_column=target_column, out_processed_dir=None)
    X = r["X"]
    y = r["y"]
    feature_names = r["feature_names"]

    results: List[Dict[str, Any]] = []

    # 1) 规则基线：加权融合分数直接作为预测（可看成一维模型）
    fusion_score = weighted_fusion(X, feature_names)
    metrics_rule = regression_metrics(y, fusion_score)
    results.append({"model": "rule_weighted_fusion", **metrics_rule})

    # 2) XGBoost
    res_xgb = train_xgboost_regression(X, y, random_state=42)
    results.append({"model": "xgboost", **res_xgb["cv_agg"], **{f"test_{k}": v for k, v in res_xgb["test_metrics"].items()}})

    # 3) 随机森林
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)
    metrics_rf = regression_metrics(y_te, y_pred_rf)
    results.append({"model": "random_forest", **metrics_rf})

    # 4) GNN + XGBoost
    try:
        edge_index, _ = build_grid_graph(n_nodes=X.shape[0])
        Z_node, g_graph, gnn_metrics = gnn_embedding_graphsage(X, edge_index)
        X_gnn = np.concatenate([X, Z_node], axis=1)
        res_xgb_gnn = train_xgboost_regression(X_gnn, y, random_state=42)
        results.append(
            {
                "model": "xgboost_gnn",
                **res_xgb_gnn["cv_agg"],
                **{f"test_{k}": v for k, v in res_xgb_gnn["test_metrics"].items()},
            }
        )
    except ImportError:
        pass

    return pd.DataFrame(results)


if __name__ == "__main__":
    """
    示例：命令行运行某个实验（按需修改）。
    例如：
        python experiments_analysis.py
    """
    csv_demo = "Yingmai 2 area in Tarim Basin.csv"
    if not os.path.isfile(csv_demo):
        print("请将 experiments_analysis.py 与示例 CSV 放在同一目录，或手动修改 csv_demo 路径。")
    else:
        print("运行示例消融实验...")
        df_ab = run_ablation_experiments(csv_demo, target_column="Porosity")
        print(df_ab)

