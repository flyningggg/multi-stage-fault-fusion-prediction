# -*- coding: utf-8 -*-
"""
训练逻辑：5 折交叉验证 + 独立测试集，训练 XGBoost / GAT，保存模型与效果报告。
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from sklearn.model_selection import KFold, GroupKFold, train_test_split

# 将 program 加入路径以便 import feature_engineering 等
_PROGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)

from evaluation import regression_metrics, classification_metrics
from utils.config_loader import load_config
from utils.export_utils import export_spatial_dataframe
from utils.logging_utils import get_logger


def _runtime_cfg():
    cfg = load_config()
    train_cfg = (cfg.get("train") or {}) if isinstance(cfg, dict) else {}
    log_cfg = (cfg.get("logging") or {}) if isinstance(cfg, dict) else {}
    logger = get_logger(
        "ml.train",
        level=log_cfg.get("level", "INFO"),
        log_file=os.path.join(_PROGRAM_DIR, log_cfg.get("file", "logs/pipeline.log")),
    )
    return train_cfg, logger


def _aggregate_metric_list(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """将多折 CV 的指标列表聚合为 mean ± std。"""
    keys = sorted({k for m in metrics_list for k in m.keys()})
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.asarray([m[k] for m in metrics_list if k in m], dtype=np.float64)
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=0))
    return out


def train_regression_baselines(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    训练线性回归和随机森林作为基线模型，与 XGBoost 对比。
    基线结果用于展示"从简单模型到复杂模型的逐步提升"。
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    train_cfg, logger = _runtime_cfg()
    rf_cfg = (train_cfg.get("random_forest") or {}) if isinstance(train_cfg, dict) else {}
    enabled = set(train_cfg.get("baseline_models") or ["linear_regression", "random_forest"])
    models = {}
    if "linear_regression" in enabled:
        models["linear_regression"] = LinearRegression()
    if "random_forest" in enabled:
        models["random_forest"] = RandomForestRegressor(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            max_depth=rf_cfg.get("max_depth"),
            random_state=random_state,
        )
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = regression_metrics(y_test, pred)
        logger.info("基线模型完成：%s test_R2=%.4f", name, results[name]["R2"])
    return results


def _compute_conformal_qhat(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
) -> float:
    """
    Split conformal 分位数 q_hat：基于校准集的绝对残差经验分位数。
    使用"有限样本有效"的分位数公式：ceil((n+1)*(1-alpha)) / n。
    返回对称预测区间的半宽度。
    """
    n = len(y_true)
    if n <= 0:
        return 0.0
    err = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))
    q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
    q_level = float(min(max(q_level, 0.0), 1.0))
    return float(np.quantile(err, q_level, method="higher"))


def _build_spatial_block_ids(df_meta: Optional[pd.DataFrame], n_blocks: int) -> Optional[np.ndarray]:
    """
    从网格顶点坐标构建空间 block ID（近似网格分块），用于空间交叉验证。
    计算每个网格的中心坐标 (x, y)，分别做等宽分箱，组合成 block ID。
    block ID 相同的样本在空间上相邻，在空间 CV 中会分到一起。
    需要 vertex1..4 坐标列；若缺失则返回 None。
    """
    if df_meta is None or len(df_meta) == 0:
        return None
    req = ("vertex1_x", "vertex2_x", "vertex3_x", "vertex4_x", "vertex1_y", "vertex2_y", "vertex3_y", "vertex4_y")
    if not all(c in df_meta.columns for c in req):
        return None
    n_side = max(2, int(np.ceil(np.sqrt(max(2, int(n_blocks))))))
    # 取四个顶点的平均值作为网格中心坐标
    x = (
        pd.to_numeric(df_meta["vertex1_x"], errors="coerce").to_numpy(dtype=np.float64)
        + pd.to_numeric(df_meta["vertex2_x"], errors="coerce").to_numpy(dtype=np.float64)
        + pd.to_numeric(df_meta["vertex3_x"], errors="coerce").to_numpy(dtype=np.float64)
        + pd.to_numeric(df_meta["vertex4_x"], errors="coerce").to_numpy(dtype=np.float64)
    ) / 4.0
    y = (
        pd.to_numeric(df_meta["vertex1_y"], errors="coerce").to_numpy(dtype=np.float64)
        + pd.to_numeric(df_meta["vertex2_y"], errors="coerce").to_numpy(dtype=np.float64)
        + pd.to_numeric(df_meta["vertex3_y"], errors="coerce").to_numpy(dtype=np.float64)
        + pd.to_numeric(df_meta["vertex4_y"], errors="coerce").to_numpy(dtype=np.float64)
    ) / 4.0
    # 等分箱：空间上相邻单元更容易落在同一 block
    try:
        xbin = pd.Series(pd.cut(x, bins=n_side, labels=False, include_lowest=True), dtype="float64")
        ybin = pd.Series(pd.cut(y, bins=n_side, labels=False, include_lowest=True), dtype="float64")
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


def _spatial_block_cv_regression(
    X: np.ndarray,
    y: np.ndarray,
    block_id: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
    xgb_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    空间 Block 交叉验证：按空间块分组留一，评估模型的跨区块泛化能力。
    使用 GroupKFold，同一空间 block 的样本始终在同一折中（全训练或全测试），
    避免空间自相关导致的过拟合假象。
    """
    import xgboost as xgb

    uniq = np.unique(block_id)
    if len(uniq) < 2:
        return {}
    n_cv = int(min(max(2, n_splits), len(uniq)))
    gkf = GroupKFold(n_splits=n_cv)
    metrics_list: List[Dict[str, float]] = []
    for tr_idx, va_idx in gkf.split(X, y, groups=block_id):
        model = xgb.XGBRegressor(random_state=random_state, **xgb_params)
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[va_idx])
        metrics_list.append(regression_metrics(y[va_idx], pred))
    out = _aggregate_metric_list(metrics_list)
    out["n_blocks_used"] = int(len(uniq))
    out["n_splits_used"] = int(n_cv)
    return out


def train_xgboost_regression(
    X: np.ndarray,
    y: np.ndarray,
    df_meta: Optional[pd.DataFrame] = None,
    n_splits: Optional[int] = None,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    **xgb_params,
) -> Dict[str, Any]:
    """
    XGBoost 回归完整训练流程：
    1. 划分训练集/测试集（默认 9:1）
    2. 训练集上 5 折交叉验证
    3. 全量训练集训练最终模型 → 测试集评估
    4. 训练线性回归/随机森林基线做对比
    5. 计算预测区间（Normal ± sigma×残差标准差 + Conformal q_hat）
    6. 可选空间 Block-CV（评估跨区泛化）
    7. 可选多随机种子稳定性测试
    """
    train_cfg, logger = _runtime_cfg()
    if n_splits is None:
        n_splits = int(train_cfg.get("n_splits", 5))
    if test_size is None:
        test_size = float(train_cfg.get("test_size", 0.1))
    if random_state is None:
        random_state = int(train_cfg.get("random_state", 42))
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("请安装 xgboost: pip install xgboost")
    
    # 清理参数，确保所有数值参数都是正确的类型
    def clean_param(value):
        if isinstance(value, str):
            # 移除括号和空格
            cleaned_value = value.strip().strip('[]')
            # 尝试转换为浮点数
            try:
                return float(cleaned_value)
            except:
                return value
        return value
    
    # 清理 xgb_params
    cleaned_xgb_params = {k: clean_param(v) for k, v in xgb_params.items()}
    # 清理 train_cfg 中的参数
    cleaned_train_cfg = {k: clean_param(v) for k, v in train_cfg.items()}
    
    logger.info("开始训练 XGBoost：samples=%s features=%s n_splits=%s test_size=%s", len(X), X.shape[1], n_splits, test_size)
    # 划分训练/测试集：测试集仅用于最终评估，不参与任何训练
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # 5 折交叉验证：在训练集上评估模型稳定性
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_metrics = []
    for train_idx, val_idx in kf.split(X_train):
        Xt, Xv = X_train[train_idx], X_train[val_idx]
        yt, yv = y_train[train_idx], y_train[val_idx]
        model = xgb.XGBRegressor(random_state=random_state, **cleaned_xgb_params)
        model.fit(Xt, yt)
        pred = model.predict(Xv)
        cv_metrics.append(regression_metrics(yv, pred))
    cv_agg = _aggregate_metric_list(cv_metrics)
    # 用全量训练集训练最终模型，测试集仅用于独立评估
    model = xgb.XGBRegressor(random_state=random_state, **cleaned_xgb_params)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_metrics = regression_metrics(y_test, test_pred)
    # 训练基线模型用于对比
    baseline_metrics = train_regression_baselines(X_train, X_test, y_train, y_test, random_state=random_state)
    # 预测区间：Normal 区间（sigma×残差标准差） + Conformal 区间（q_hat）
    sigma = float(cleaned_train_cfg.get("prediction_interval_sigma", 1.96))
    resid_std = float(np.std(y_test - test_pred, ddof=0)) if len(y_test) else 0.0
    # Conformal 预测区间：从训练集中再切出校准集，计算绝对残差分位数 q_hat
    conformal_alpha = float(cleaned_train_cfg.get("conformal_alpha", 0.1))
    conf_qhat = 0.0
    conf_cov_test = 0.0
    try:
        # 从训练集中再划分出 proper 训练集（~85%）和校准集（~15%）
        X_proper, X_cal, y_proper, y_cal = train_test_split(
            X_train, y_train, test_size=max(0.15, min(0.35, test_size)), random_state=random_state
        )
        model_cal = xgb.XGBRegressor(random_state=random_state, **xgb_params)
        model_cal.fit(X_proper, y_proper)
        pred_cal = model_cal.predict(X_cal)
        conf_qhat = _compute_conformal_qhat(y_cal, pred_cal, conformal_alpha)
        conf_cov_test = float(np.mean((y_test >= (test_pred - conf_qhat)) & (y_test <= (test_pred + conf_qhat))))
    except Exception:
        conf_qhat = 0.0
        conf_cov_test = 0.0
    interval = {
        "sigma": sigma,
        "residual_std": resid_std,
        "lower_mean_offset": -sigma * resid_std,
        "upper_mean_offset": sigma * resid_std,
        "conformal_alpha": conformal_alpha,
        "conformal_qhat": conf_qhat,
        "conformal_test_coverage": conf_cov_test,
        "conformal_lower_mean_offset": -conf_qhat,
        "conformal_upper_mean_offset": conf_qhat,
    }
    # 空间 Block-CV：评估模型在空间跨区场景下的泛化能力
    spatial_cv_agg: Optional[Dict[str, Any]] = None
    block_id = _build_spatial_block_ids(df_meta, int(train_cfg.get("spatial_cv_blocks", 9)))
    if block_id is not None and len(block_id) == len(X):
        try:
            spatial_cv_agg = _spatial_block_cv_regression(
                X,
                y,
                block_id,
                n_splits=int(train_cfg.get("n_splits", 5)),
                random_state=random_state,
                xgb_params=xgb_params,
            )
            logger.info(
                "空间 Block-CV 完成：R2_mean=%.4f R2_std=%.4f blocks=%s",
                spatial_cv_agg.get("R2_mean", float("nan")),
                spatial_cv_agg.get("R2_std", float("nan")),
                spatial_cv_agg.get("n_blocks_used"),
            )
        except Exception as e:
            logger.warning("空间 Block-CV 失败，已跳过：%s", e)
            spatial_cv_agg = None
    logger.info("XGBoost 训练完成：test_R2=%.4f cv_R2_mean=%.4f", test_metrics["R2"], cv_agg.get("R2_mean", float("nan")))
    # 多随机种子稳定性：用不同种子重跑训练/测试划分，评估 R² 波动幅度
    seed_stability = None
    raw_seeds = train_cfg.get("stability_seeds")
    if isinstance(raw_seeds, (list, tuple)) and len(raw_seeds) >= 2:
        r2_list: List[float] = []
        for sd in raw_seeds:
            sd = int(sd)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=sd)
            m = xgb.XGBRegressor(random_state=sd, **xgb_params)
            m.fit(X_tr, y_tr)
            pv = m.predict(X_te)
            r2_list.append(regression_metrics(y_te, pv)["R2"])
        r2_arr = np.asarray(r2_list, dtype=np.float64)
        seed_stability = {
            "test_R2_mean": float(np.mean(r2_arr)),
            "test_R2_std": float(np.std(r2_arr, ddof=0)),
            "seeds": [int(s) for s in raw_seeds],
        }
        logger.info(
            "多随机种子测试集 R²：mean=%.4f std=%.4f（seeds=%s）",
            seed_stability["test_R2_mean"],
            seed_stability["test_R2_std"],
            seed_stability["seeds"],
        )
    return {
        "model": model,
        "cv_metrics": cv_metrics,
        "cv_agg": cv_agg,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "prediction_interval": interval,
        "test_predictions": test_pred,
        "y_test": y_test,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "seed_stability": seed_stability,
        "spatial_cv_agg": spatial_cv_agg,
    }


def train_xgboost_classification(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    test_size: float = 0.1,
    random_state: int = 42,
    **xgb_params,
) -> Dict[str, Any]:
    """XGBoost 分类 + 5 折 CV + 留出测试集。"""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("请安装 xgboost: pip install xgboost")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_metrics = []
    for train_idx, val_idx in kf.split(X_train):
        Xt, Xv = X_train[train_idx], X_train[val_idx]
        yt, yv = y_train[train_idx], y_train[val_idx]
        model = xgb.XGBClassifier(random_state=random_state, **xgb_params)
        model.fit(Xt, yt)
        pred = model.predict(Xv)
        cv_metrics.append(classification_metrics(yv, pred))
    # 用划分出的训练集（全量数据的 1-test_size 部分）训练最终模型，测试集仅用于评估，不参与训练
    model = xgb.XGBClassifier(random_state=random_state, **xgb_params)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_metrics = classification_metrics(y_test, test_pred)
    cv_agg = {
        "accuracy_mean": np.mean([m["accuracy"] for m in cv_metrics]),
        "f1_mean": np.mean([m["f1"] for m in cv_metrics]),
    }
    return {
        "model": model,
        "cv_metrics": cv_metrics,
        "cv_agg": cv_agg,
        "test_metrics": test_metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def save_model_report(
    result: Dict[str, Any],
    model_dir: str,
    name: str = "xgboost",
    feature_names: Optional[list] = None,
) -> str:
    """保存模型与效果报告到 model_dir。"""
    os.makedirs(model_dir, exist_ok=True)
    model = result.get("model")
    if model is not None:
        path = os.path.join(model_dir, f"{name}.json")
        model.save_model(path)
    report = {k: v for k, v in result.items() if k != "model" and not callable(v)}

    def _serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(x) for x in obj]
        return obj

    report = _serialize(report)
    report_path = os.path.join(model_dir, f"{name}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if feature_names is not None:
        with open(os.path.join(model_dir, f"{name}_features.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(feature_names))
    return report_path


def export_prediction_results(
    df: pd.DataFrame,
    model_dir: str,
    stem: str = "xgboost_predictions",
    *,
    export_csv: bool = True,
    export_gpkg: bool = True,
) -> Dict[str, Optional[str]]:
    from utils.export_utils import enrich_predictions_for_gis, LAYER_PREDICTIONS_XGB

    df_e = enrich_predictions_for_gis(df, pred_col_preferred="prediction_xgboost")
    return export_spatial_dataframe(
        df_e,
        model_dir,
        stem,
        export_csv=export_csv,
        export_gpkg=export_gpkg,
        layer_name=LAYER_PREDICTIONS_XGB,
    )


if __name__ == "__main__":
    from feature_engineering import build_feature_matrix, DEFAULT_FEATURE_COLUMNS
    csv_path = os.path.join(_PROGRAM_DIR, "Yingmai 2 area in Tarim Basin.csv")
    if not os.path.isfile(csv_path):
        print("未找到 CSV，请指定网格 CSV 路径。")
        sys.exit(1)
    r = build_feature_matrix(csv_path, target_column=None, out_processed_dir=None)
    X, y = r["X"], r["y"]
    if y is None:
        y = r["X"][:, 0]  # 无标签时用第一列做演示
    res = train_xgboost_regression(X, y, n_estimators=50, max_depth=4)
    model_dir = os.path.join(_PROGRAM_DIR, "model")
    save_model_report(res, model_dir, name="xgboost_reg", feature_names=r.get("feature_names"))
    print("CV R2 mean:", res["cv_agg"]["R2_mean"])
    print("Test R2:", res["test_metrics"]["R2"])
    print("报告已保存至 model/")
