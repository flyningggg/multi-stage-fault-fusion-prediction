# -*- coding: utf-8 -*-
"""
超参数调优：贝叶斯优化（Optuna）调优 XGBoost / GAT 核心参数，保存调优日志。
"""
import os
import sys
import json
import numpy as np
from typing import Optional, Dict, Any, Callable

_PROGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)


def tune_xgboost_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 30,
    n_splits: int = 3,
    random_state: int = 42,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Optuna 贝叶斯优化 XGBoost 回归超参。"""
    try:
        import optuna
        import xgboost as xgb
    except ImportError:
        raise ImportError("请安装 optuna 与 xgboost: pip install optuna xgboost")
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer
    from sklearn.metrics import mean_squared_error

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": random_state,
        }
        model = xgb.XGBRegressor(**params)
        scores = -cross_val_score(
            model, X, y, cv=n_splits, scoring="neg_mean_squared_error"
        )
        return np.sqrt(scores.mean())  # RMSE

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    best_params = study.best_params
    best_value = study.best_value
    log = {
        "n_trials": n_trials,
        "best_rmse": best_value,
        "best_params": best_params,
        "n_completed": len(study.trials),
    }
    return {"study": study, "best_params": best_params, "best_rmse": best_value, "tune_log": log}


def tune_xgboost_classification(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 30,
    n_splits: int = 3,
    random_state: int = 42,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Optuna 贝叶斯优化 XGBoost 分类超参。"""
    try:
        import optuna
        import xgboost as xgb
    except ImportError:
        raise ImportError("请安装 optuna 与 xgboost: pip install optuna xgboost")
    from sklearn.model_selection import cross_val_score

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": random_state,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=n_splits, scoring="f1_weighted")
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    log = {
        "n_trials": n_trials,
        "best_params": study.best_params,
        "best_neg_f1": study.best_value,
        "n_completed": len(study.trials),
    }
    return {"study": study, "best_params": study.best_params, "tune_log": log}


def save_tune_log(result: Dict[str, Any], model_dir: str, name: str = "tune_xgb") -> str:
    """保存调优日志（best_params + tune_log）。"""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{name}_log.json")
    out = {k: v for k, v in result.items() if k != "study"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    from feature_engineering import build_feature_matrix
    csv_path = os.path.join(_PROGRAM_DIR, "Yingmai 2 area in Tarim Basin.csv")
    if not os.path.isfile(csv_path):
        print("未找到 CSV。")
        sys.exit(1)
    r = build_feature_matrix(csv_path, out_processed_dir=None)
    X, y = r["X"], r["y"]
    if y is None:
        y = r["X"][:, 0]
    res = tune_xgboost_regression(X, y, n_trials=10, n_splits=3)
    model_dir = os.path.join(_PROGRAM_DIR, "model")
    save_tune_log(res, model_dir, name="tune_xgb_reg")
    print("Best RMSE:", res["best_rmse"])
    print("Best params:", res["best_params"])
