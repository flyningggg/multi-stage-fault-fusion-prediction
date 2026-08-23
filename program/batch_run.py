# -*- coding: utf-8 -*-
"""
多区块 CSV 批量流水线：特征工程 → 融合 → 训练。
用法：python batch_run.py [csv1.csv csv2.csv ...]  或  python batch_run.py  # 使用 config 中列表或默认英买2区
"""
import os
import sys
import glob
import argparse
from typing import List, Optional

_PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)

try:
    from utils.logging_utils import get_logger
    from utils.config_loader import load_config
    from utils.validation import check_path_exists
except ImportError:
    get_logger = lambda **kw: __import__("logging").getLogger("batch")
    load_config = lambda: {}
    check_path_exists = lambda p, **kw: p

from feature_engineering import build_feature_matrix, DEFAULT_FEATURE_COLUMNS
from fusion_algorithm import run_weighted_fusion_pipeline

logger = get_logger("batch_run")


def run_pipeline_for_one(
    csv_path: str,
    out_processed_dir: Optional[str] = None,
    out_fusion_csv: bool = True,
    high_value_weight: float = 1.5,
) -> dict:
    """
    对单个网格 CSV 执行：特征工程 → 加权融合。
    流程：加载 → IQR 异常值截断 → StandardScaler → 加权融合 → 写出 CSV。
    返回摘要 dict（样本数、特征数、融合得分均值）。
    """
    check_path_exists(csv_path, "CSV")
    cfg = load_config()
    fe_cfg = cfg.get("feature_engineering", {})
    if out_processed_dir is None:
        out_processed_dir = os.path.join(_PROGRAM_DIR, "data", "processed")
    os.makedirs(out_processed_dir, exist_ok=True)
    # 步骤 1：特征工程（异常值处理 + 归一化 + 特征筛选）
    logger.info("特征工程: %s", csv_path)
    r = build_feature_matrix(
        csv_path,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        normalize_method=fe_cfg.get("normalize_method", "standard"),
        outlier_method=fe_cfg.get("outlier_method", "iqr"),
        variance_threshold=fe_cfg.get("variance_threshold", 1e-6),
        n_select_mi=fe_cfg.get("n_select_mi"),
        out_processed_dir=out_processed_dir,
    )
    logger.info("特征矩阵形状: %s, 特征数: %s", r["X"].shape, r["n_features"])
    # 步骤 2：加权融合（连通性列高权重，其余列权重 1.0）
    logger.info("加权融合: %s", csv_path)
    df_fusion = run_weighted_fusion_pipeline(
        csv_path,
        high_value_weight=high_value_weight,
        out_dir=out_processed_dir,
    )
    if out_fusion_csv:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = os.path.join(out_processed_dir, f"{base}_fusion.csv")
        df_fusion.to_csv(out_path, index=False)
        logger.info("融合结果已写: %s", out_path)
    return {
        "csv": csv_path,
        "n_samples": r["n_samples"],
        "n_features": r["n_features"],
        "fusion_score_mean": float(df_fusion["weighted_fusion_score"].mean()),
    }


# =========================================================================
# Stage 1: 三期独立融合（PCA + 加权 + XGBoost 特征重要性）
# =========================================================================

import numpy as np
import pandas as pd
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from multiperiod_data import (
        DEFAULT_CSV_PATHS, PERIOD_NAMES, TOPOLOGY_ATTRIBUTES,
        load_period_csv, get_topology_matrix, get_topology_summary,
    )
    HAS_MULTIPERIOD = True
except ImportError:
    HAS_MULTIPERIOD = False

try:
    from topology_fusion import fuse_with_pca, cluster_labels, interpret_clusters
    HAS_TOPO_FUSION = True
except ImportError:
    HAS_TOPO_FUSION = False

try:
    from fusion_algorithm import weighted_fusion
    HAS_FUSION_ALGO = True
except ImportError:
    HAS_FUSION_ALGO = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


STAGE1_XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.05,
    "random_state": 42,
    "verbosity": 0,
}


def _build_spatial_block_ids(
    xs: np.ndarray,
    ys: np.ndarray,
    n_blocks: int = 9,
) -> Optional[np.ndarray]:
    """按坐标等宽分箱构造空间块；退化坐标返回 None。"""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.shape != ys.shape or xs.ndim != 1 or xs.size < 2:
        return None
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        return None
    if np.ptp(xs) <= 0 or np.ptp(ys) <= 0:
        return None

    n_side = max(2, int(np.ceil(np.sqrt(max(2, n_blocks)))))
    x_bin = pd.cut(xs, bins=n_side, labels=False, include_lowest=True, duplicates="drop")
    y_bin = pd.cut(ys, bins=n_side, labels=False, include_lowest=True, duplicates="drop")
    if pd.isna(x_bin).any() or pd.isna(y_bin).any():
        return None

    groups = np.asarray(y_bin, dtype=int) * n_side + np.asarray(x_bin, dtype=int)
    return groups if np.unique(groups).size >= 2 else None


def _spatial_cv_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    n_blocks: int = 9,
    n_splits: int = 5,
    model_params: Optional[dict] = None,
) -> dict:
    """按空间块执行 GroupKFold，返回均值、波动和明确状态。"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) < 4:
        return {"status": "invalid_input", "n_splits_used": 0}
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        return {"status": "non_finite_input", "n_splits_used": 0}
    if np.ptp(y) <= 1e-12:
        return {"status": "constant_target", "n_splits_used": 0}

    groups = _build_spatial_block_ids(xs, ys, n_blocks=n_blocks)
    if groups is None:
        return {"status": "insufficient_spatial_blocks", "n_splits_used": 0}

    unique_groups = np.unique(groups)
    splits = min(int(n_splits), int(unique_groups.size))
    if splits < 2:
        return {"status": "insufficient_spatial_blocks", "n_splits_used": 0}

    params = dict(STAGE1_XGB_PARAMS)
    if model_params:
        params.update(model_params)

    fold_metrics = []
    for train_idx, test_idx in GroupKFold(n_splits=splits).split(X, y, groups):
        if len(test_idx) < 2 or np.ptp(y[test_idx]) <= 1e-12:
            return {
                "status": "degenerate_test_fold",
                "n_blocks_used": int(unique_groups.size),
                "n_splits_used": splits,
            }
        model = xgb.XGBRegressor(**params)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        fold_metrics.append({
            "r2": float(r2_score(y[test_idx], pred)),
            "rmse": float(np.sqrt(mean_squared_error(y[test_idx], pred))),
            "mae": float(mean_absolute_error(y[test_idx], pred)),
        })

    result = {
        "status": "ok",
        "n_blocks_used": int(unique_groups.size),
        "n_splits_used": splits,
        "folds": fold_metrics,
    }
    for metric in ("r2", "rmse", "mae"):
        values = np.asarray([row[metric] for row in fold_metrics], dtype=float)
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values))
    return result


def _prepare_multiperiod_data():
    """加载三期数据并提取拓扑属性矩阵。"""
    from multiperiod_data import load_all_periods
    period_data = load_all_periods()
    results = {}
    for period_name, gdf in period_data.items():
        _, X, cols = get_topology_matrix(gdf)
        results[period_name] = {"gdf": gdf, "X": X, "cols": cols}
    return results


def _run_pca_fusion(data: dict, n_components: int = 2, n_clusters: int = 4) -> dict:
    """对每期数据执行 PCA 降维 + KMeans 聚类。"""
    from topology_fusion import build_cluster_name_map, compute_cluster_quality_metrics
    pca_results = {}
    for period_name, d in data.items():
        X = d["X"]
        X_pca, scaler, pca = fuse_with_pca(X, n_components=n_components, standardize=True)
        labels, kmeans = cluster_labels(X_pca, n_clusters=n_clusters)
        cluster_means = interpret_clusters(d["gdf"], d["cols"], labels, n_clusters)
        quality = compute_cluster_quality_metrics(X_pca, labels)
        pca_results[period_name] = {
            "X_pca": X_pca,
            "labels": labels,
            "scaler": scaler,
            "pca": pca,
            "kmeans": kmeans,
            "cluster_means": cluster_means,
            "quality": quality,
        }
    return pca_results


def _run_weighted_fusion(data: dict, high_value_weight: float = 1.5) -> dict:
    """对每期数据执行加权融合。"""
    high_attrs = ["NC_NB", "NC_NL", "NC_A"]
    weighted_results = {}
    for period_name, d in data.items():
        X = d["X"]
        cols = d["cols"]
        score = weighted_fusion(X, cols, high_value_weight=high_value_weight,
                                high_value_attrs=high_attrs)
        weighted_results[period_name] = score
    return weighted_results


def _run_xgboost_importance(data: dict, out_dir: str) -> dict:
    """训练全量模型提取重要性，并用空间块 CV 评估泛化。"""
    xgb_results = {}
    for period_name, d in data.items():
        gdf = d["gdf"]
        X = d["X"]
        cols = d["cols"]

        if "Fracture Intensity B21" not in gdf.columns:
            logger.warning("%s: 缺少 Fracture Intensity B21 列，跳过 XGBoost", period_name)
            xgb_results[period_name] = None
            continue

        y = pd.to_numeric(gdf["Fracture Intensity B21"], errors="coerce").fillna(0.0).values
        mask = ~np.isnan(y) & (X.sum(axis=1) >= 0)
        X_valid = X[mask]
        y_valid = y[mask]

        if len(X_valid) < 10:
            logger.warning("%s: 有效样本不足 (%d)，跳过 XGBoost", period_name, len(X_valid))
            xgb_results[period_name] = None
            continue

        model = xgb.XGBRegressor(**STAGE1_XGB_PARAMS)
        model.fit(X_valid, y_valid)
        importance = model.feature_importances_
        pred = model.predict(X)
        train_pred = model.predict(X_valid)
        from sklearn.metrics import r2_score
        train_r2 = float(r2_score(y_valid, train_pred))

        centroids = gdf.geometry.centroid
        xs = np.asarray([p.x for p in centroids], dtype=float)[mask]
        ys = np.asarray([p.y for p in centroids], dtype=float)[mask]
        spatial_cv = _spatial_cv_xgboost(X_valid, y_valid, xs, ys)

        imp_df = pd.DataFrame({"属性": cols, "importance": importance})
        imp_df.sort_values("importance", ascending=False, inplace=True)

        xgb_results[period_name] = {
            "model": model,
            "importance": imp_df,
            "r2": train_r2,  # 向后兼容；仅表示训练拟合，不作泛化主指标
            "train_r2": train_r2,
            "spatial_cv": spatial_cv,
            "predictions": pred,
        }
        if spatial_cv.get("status") == "ok":
            logger.info(
                "%s XGBoost 训练拟合R²=%.3f  空间CV R²=%.3f±%.3f  top1=%s(%.3f)",
                period_name,
                train_r2,
                spatial_cv["r2_mean"],
                spatial_cv["r2_std"],
                imp_df.iloc[0]["属性"],
                imp_df.iloc[0]["importance"],
            )
        else:
            logger.warning(
                "%s XGBoost 训练拟合R²=%.3f，空间CV不可用: %s",
                period_name,
                train_r2,
                spatial_cv.get("status"),
            )
    return xgb_results


def _plot_pca_comparison(pca_results: dict, out_dir: str) -> str:
    """绘制三期 PCA 聚类对比图（3个子图横向排列）。"""
    periods = list(pca_results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cmap = plt.cm.Set2
    for ax, period_name in zip(axes, periods):
        r = pca_results[period_name]
        labels = r["labels"]
        n_clusters = len(np.unique(labels))
        for c in range(n_clusters):
            mask = labels == c
            ax.scatter(r["X_pca"][mask, 0], r["X_pca"][mask, 1],
                       c=[cmap(c / max(n_clusters - 1, 1))], label=f"簇{c+1}",
                       s=10, alpha=0.7, edgecolors="none")
        ax.set_title(f"{period_name}\n(轮廓系数={r['quality'].get('silhouette_score', 0):.3f})",
                     fontsize=13)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=2, fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    plt.suptitle("三期 PCA 降维 + KMeans 聚类对比", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "multiperiod_pca_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_weighted_comparison(weighted_results: dict, out_dir: str) -> str:
    """绘制三期加权融合得分分布对比（直方图叠加）。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for (period_name, scores), color in zip(weighted_results.items(), colors):
        mask = scores > 1e-10
        ax.hist(scores[mask], bins=40, alpha=0.5, color=color, label=period_name,
                density=True, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("加权融合得分 (0-1)")
    ax.set_ylabel("密度")
    ax.set_title("三期加权融合得分分布对比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "multiperiod_weighted_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_xgboost_importance(xgb_results: dict, out_dir: str) -> str:
    """绘制三期 XGBoost 特征重要性对比（分组柱状图）。"""
    periods = [k for k, v in xgb_results.items() if v is not None]
    if len(periods) < 2:
        # 单期也用柱状图
        fig, ax = plt.subplots(figsize=(8, 5))
        for period_name in periods:
            imp = xgb_results[period_name]["importance"]
            ax.barh(imp["属性"], imp["importance"], color="#4C72B0", alpha=0.8)
            result = xgb_results[period_name]
            cv = result["spatial_cv"]
            metric_text = (
                f"空间CV R²={cv['r2_mean']:.3f}"
                if cv.get("status") == "ok"
                else f"训练拟合 R²={result['train_r2']:.3f}"
            )
            ax.set_title(f"{period_name} XGBoost 特征重要性 ({metric_text})")
        ax.set_xlabel("重要性")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, "multiperiod_xgboost_importance.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    all_attrs = list(TOPOLOGY_ATTRIBUTES)
    n_attrs = len(all_attrs)
    n_periods = len(periods)
    x = np.arange(n_attrs)
    width = 0.8 / n_periods
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, period_name in enumerate(periods):
        imp = xgb_results[period_name]["importance"]
        vals = [float(imp.loc[imp["属性"] == a, "importance"].iloc[0])
                if a in imp["属性"].values else 0.0 for a in all_attrs]
        result = xgb_results[period_name]
        cv = result["spatial_cv"]
        metric_text = (
            f"空间CV R²={cv['r2_mean']:.3f}"
            if cv.get("status") == "ok"
            else f"训练拟合 R²={result['train_r2']:.3f}"
        )
        ax.bar(x + i * width, vals, width, color=colors[i], alpha=0.85,
               label=f"{period_name} ({metric_text})")
    ax.set_xticks(x + width * (n_periods - 1) / 2)
    ax.set_xticklabels(all_attrs, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("特征重要性")
    ax.set_title("三期 XGBoost 特征重要性对比（6个拓扑属性 → 预测 B21）")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "multiperiod_xgboost_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run_multiperiod_fusion(
    out_dir: Optional[str] = None,
    n_components: int = 2,
    n_clusters: int = 4,
    high_value_weight: float = 1.5,
) -> dict:
    """
    Stage 1：三期独立融合对比。
    对海西期、印支燕山期、喜山期分别运行：
      - PCA 降维 + KMeans 聚类
      - 加权融合（NC_NB/NC_NL/NC_A 高权重）
      - XGBoost 特征重要性（6属性 → 预测 Fracture Intensity B21）
    产出三期对比图到 out_dir。
    返回完整结果 dict。
    """
    if not HAS_MULTIPERIOD:
        raise ImportError("需要 multiperiod_data 模块")
    if out_dir is None:
        out_dir = os.path.join(_PROGRAM_DIR, "data", "processed", "multiperiod")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=== Stage 1: 三期独立融合对比 ===")
    logger.info("输出目录: %s", out_dir)

    data = _prepare_multiperiod_data()
    for period_name, d in data.items():
        logger.info("  %s: %d 网格, %d 非零", period_name, len(d["gdf"]),
                     int((d["X"].sum(axis=1) > 0).sum()))

    # 1) PCA + KMeans
    logger.info("--- 1/3 PCA + KMeans 聚类 ---")
    pca_results = _run_pca_fusion(data, n_components=n_components, n_clusters=n_clusters)
    for pn, r in pca_results.items():
        sil = r["quality"].get("silhouette_score", float("nan"))
        logger.info("  %s: silhouette=%.3f  clusters=%d", pn, sil, n_clusters)

    # 2) 加权融合
    logger.info("--- 2/3 加权融合 ---")
    weighted_results = _run_weighted_fusion(data, high_value_weight=high_value_weight)
    for pn, scores in weighted_results.items():
        nonzero = (scores > 1e-10).sum()
        logger.info("  %s: 得分均值=%.3f  非零网格=%d", pn, scores.mean(), int(nonzero))

    # 3) XGBoost 特征重要性
    logger.info("--- 3/3 XGBoost 特征重要性 ---")
    xgb_results = _run_xgboost_importance(data, out_dir)

    # 导出汇总表
    summary_rows = []
    for period_name in data:
        d = data[period_name]
        gdf = d["gdf"]
        pca_r = pca_results[period_name]
        w_score = weighted_results[period_name]
        xgb_r = xgb_results.get(period_name)
        spatial_cv = xgb_r["spatial_cv"] if xgb_r else {}
        summary_rows.append({
            "时期": period_name,
            "网格数": len(gdf),
            "非零网格": int((d["X"].sum(axis=1) > 0).sum()),
            "PCA Silhouette": round(pca_r["quality"].get("silhouette_score", 0), 4),
            "加权得分均值": round(float(w_score.mean()), 4),
            "XGBoost训练拟合R²": round(xgb_r["train_r2"], 4) if xgb_r else None,
            "XGBoost空间CV_R²均值": (
                round(spatial_cv["r2_mean"], 4)
                if spatial_cv.get("status") == "ok" else None
            ),
            "XGBoost空间CV_R²标准差": (
                round(spatial_cv["r2_std"], 4)
                if spatial_cv.get("status") == "ok" else None
            ),
            "XGBoost空间CV折数": spatial_cv.get("n_splits_used", 0),
            "XGBoost空间CV状态": spatial_cv.get("status") if xgb_r else "not_run",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "multiperiod_fusion_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info("汇总表: %s", summary_path)

    # 生成图表
    plot_paths = {}
    if HAS_MPL:
        try:
            setup_matplotlib_fonts()
            logger.info("--- 生成对比图 ---")
            plot_paths["pca"] = _plot_pca_comparison(pca_results, out_dir)
            plot_paths["weighted"] = _plot_weighted_comparison(weighted_results, out_dir)
            if any(v is not None for v in xgb_results.values()):
                plot_paths["xgboost"] = _plot_xgboost_importance(xgb_results, out_dir)
            for k, v in plot_paths.items():
                logger.info("  %s: %s", k, v)
        except Exception as e:
            logger.warning("图表生成失败: %s", e)

    result = {
        "data": data,
        "pca": pca_results,
        "weighted": weighted_results,
        "xgboost": xgb_results,
        "summary_df": summary_df,
        "plot_paths": plot_paths,
        "out_dir": out_dir,
    }
    return result


def setup_matplotlib_fonts():
    """配置 matplotlib 中文字体支持。"""
    try:
        from utils.matplotlib_chinese import setup_matplotlib_chinese
        setup_matplotlib_chinese()
    except ImportError:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False


def main():
    parser = argparse.ArgumentParser(description="多区块批量：特征工程 → 融合")
    parser.add_argument("csv_list", nargs="*", help="CSV 路径列表，不传则用默认英买2区")
    parser.add_argument("--no-fusion-csv", action="store_true", help="不写融合结果 CSV")
    parser.add_argument("--multiperiod", action="store_true",
                        help="运行 Stage 1：三期独立融合对比")
    parser.add_argument("--n-clusters", type=int, default=4,
                        help="KMeans 聚类数（默认 4）")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="输出目录")
    args = parser.parse_args()

    if args.multiperiod:
        run_multiperiod_fusion(
            out_dir=args.out_dir,
            n_clusters=args.n_clusters,
        )
        return

    cfg = load_config()
    fusion_weight = (cfg.get("fusion") or {}).get("high_value_weight", 1.5)
    if args.csv_list:
        csv_paths = [p for p in args.csv_list if os.path.isfile(p)]
    else:
        default_csv = os.path.join(_PROGRAM_DIR, "Yingmai 2 area in Tarim Basin.csv")
        csv_paths = [default_csv] if os.path.isfile(default_csv) else []
    if not csv_paths:
        logger.warning("未找到任何 CSV，请传入路径或将 Yingmai 2 area in Tarim Basin.csv 放在 program 目录")
        return
    results = []
    for p in csv_paths:
        try:
            res = run_pipeline_for_one(
                p,
                out_fusion_csv=not args.no_fusion_csv,
                high_value_weight=fusion_weight,
            )
            results.append(res)
        except Exception as e:
            logger.exception("处理失败 %s: %s", p, e)
    logger.info("批量完成，共 %s 个文件", len(results))
    return results


if __name__ == "__main__":
    main()
