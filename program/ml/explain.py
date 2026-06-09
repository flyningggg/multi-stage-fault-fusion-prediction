# -*- coding: utf-8 -*-
"""
可解释性：SHAP 特征重要性（如「连通率对渗透率预测的贡献占比 35%」），答辩时可直观讲清模型逻辑。
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

_PROGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)

# 绘图前配置中文字体
from utils.matplotlib_chinese import setup_matplotlib_chinese
from utils.config_loader import load_config
from utils.export_utils import export_table
from utils.logging_utils import get_logger
setup_matplotlib_chinese()


def _runtime_cfg():
    cfg = load_config()
    explain_cfg = (cfg.get("explain") or {}) if isinstance(cfg, dict) else {}
    log_cfg = (cfg.get("logging") or {}) if isinstance(cfg, dict) else {}
    logger = get_logger(
        "ml.explain",
        level=log_cfg.get("level", "INFO"),
        log_file=os.path.join(_PROGRAM_DIR, log_cfg.get("file", "logs/pipeline.log")),
    )
    return explain_cfg, logger


def _reorder_by_emphasis(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    emphasize_first: Optional[List[str]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """把 emphasize_first 中出现的特征列排到最前，便于 summary 图优先展示。"""
    if not emphasize_first:
        return shap_values, X, feature_names
    fn = list(feature_names)
    order: List[int] = []
    for name in emphasize_first:
        if name in fn:
            i = fn.index(name)
            if i not in order:
                order.append(i)
    for i in range(len(fn)):
        if i not in order:
            order.append(i)
    shap_values = shap_values[:, order]
    X = X[:, order]
    return shap_values, X, [fn[i] for i in order]


def connectivity_shap_breakdown(df_imp: pd.DataFrame):
    """
    从 SHAP 重要性表中筛出「连通性特征组」的行（按 importance 排序），
    并返回该组 contribution_pct 累计占比。
    """
    from feature_engineering import is_connectivity_feature

    if df_imp is None or df_imp.empty or "feature" not in df_imp.columns:
        return pd.DataFrame(), 0.0
    sub = df_imp[df_imp["feature"].map(lambda n: is_connectivity_feature(str(n)))].copy()
    if sub.empty:
        return sub, 0.0
    sub = sub.sort_values("importance", ascending=False)
    pct_col = "contribution_pct" if "contribution_pct" in sub.columns else None
    cum = float(sub[pct_col].sum()) if pct_col else 0.0
    return sub, cum


def shap_feature_importance(
    model,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    is_tree: bool = False,  # 强制使用 KernelExplainer 避免 TreeExplainer 的参数问题
    out_plot_path: Optional[str] = None,
    emphasize_first: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    计算 SHAP 特征重要性，返回每特征 mean(|SHAP|) 及占比。
    强制使用 KernelExplainer 避免 TreeExplainer 的参数问题。
    """
    try:
        import shap
    except ImportError:
        raise ImportError("请安装 shap: pip install shap")
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    
    # 强制使用 KernelExplainer 避免 TreeExplainer 的参数问题
    explainer = shap.KernelExplainer(model.predict, X[: min(100, len(X))])
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values, X, feature_names = _reorder_by_emphasis(
        shap_values, X, list(feature_names), emphasize_first
    )
    importance = np.abs(shap_values).mean(axis=0)
    total = importance.sum()
    pct = (importance / total * 100) if total > 0 else np.zeros_like(importance)
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "contribution_pct": pct,
    }).sort_values("importance", ascending=False)
    if out_plot_path:
        os.makedirs(os.path.dirname(out_plot_path) or ".", exist_ok=True)
        import matplotlib.pyplot as plt
        try:
            shap.summary_plot(
                shap_values, X, feature_names=feature_names, show=False, sort=False
            )
        except TypeError:
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.savefig(out_plot_path, bbox_inches="tight", dpi=150)
        plt.close()
    return df


def shap_dependence_plot(
    model,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_feature_index: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """SHAP 依赖图：关键特征与 SHAP 值的关系，答辩时展示单特征贡献。"""
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("请安装 shap: pip install shap")
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    
    # 使用 KernelExplainer 避免 TreeExplainer 的参数问题
    explainer = shap.KernelExplainer(model.predict, X[: min(100, len(X))])
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    plt.figure(figsize=(6, 4))
    shap.dependence_plot(
        top_feature_index,
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def explain_xgboost(
    model_path: str,
    csv_path: str,
    is_classifier: bool = False,
    feature_columns: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
    emphasize_first: Optional[List[str]] = None,
) -> pd.DataFrame:
    """加载 XGBoost 模型与数据，输出 SHAP 特征重要性表与可选图。"""
    explain_cfg, logger = _runtime_cfg()
    try:
        from .infer import load_xgboost_model
    except ImportError:
        from infer import load_xgboost_model
    from feature_engineering import build_feature_matrix, load_raw
    
    # 加载模型
    model = load_xgboost_model(model_path, is_classifier=is_classifier)
    
    # 修复模型参数中的字符串格式数值
    if hasattr(model, 'get_params'):
        params = model.get_params()
        for key, value in params.items():
            if isinstance(value, str):
                # 尝试将字符串转换为数值
                try:
                    # 移除括号和空格
                    cleaned_value = value.strip().strip('[]')
                    # 尝试转换为浮点数
                    params[key] = float(cleaned_value)
                except:
                    pass
        # 更新模型参数
        model.set_params(**params)
    
    # 加载原始数据
    df = load_raw(csv_path)
    
    # 尝试读取模型训练时保存的特征名称文件
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_dir = os.path.dirname(model_path)
    features_file = os.path.join(model_dir, f"{model_name}_features.txt")
    
    if os.path.exists(features_file):
        # 读取模型训练时使用的特征名称
        with open(features_file, 'r', encoding='utf-8') as f:
            model_features = [line.strip() for line in f if line.strip()]
        logger.info(f"从文件读取特征名称: {model_features}")
        # 确保所有特征都在数据中
        available = [c for c in model_features if c in df.columns]
        if not available:
            raise ValueError(f"CSV 中未找到任何模型训练时使用的特征列: {model_features}")
        if len(available) != len(model_features):
            missing = [c for c in model_features if c not in df.columns]
            logger.warning(f"部分特征在CSV中未找到: {missing}")
    else:
        # 如果没有特征文件，使用提供的特征列或默认特征列
        if feature_columns is None:
            from feature_engineering import DEFAULT_FEATURE_COLUMNS
            feature_columns = DEFAULT_FEATURE_COLUMNS
        available = [c for c in feature_columns if c in df.columns]
        if not available:
            raise ValueError(f"CSV 中未找到任何特征列: {feature_columns}")
    
    # 转换特征数据
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
    
    X_raw = df[available].copy()
    for col in X_raw.columns:
        X_raw[col] = X_raw[col].apply(convert_to_float)
    
    X = X_raw.values.astype(np.float64)
    names = available
    
    # 应用相同的预处理步骤
    from feature_engineering import handle_outliers, normalize
    X = handle_outliers(X, method="iqr")
    X_norm, _ = normalize(X, method="standard")
    
    logger.info(f"SHAP 分析使用的特征数量: {X_norm.shape[1]}")
    
    out_plot = os.path.join(out_dir, "shap_summary.png") if out_dir else None
    df_imp = shap_feature_importance(
        model,
        X_norm,
        feature_names=names,
        is_tree=False,  # 使用 KernelExplainer 避免参数问题
        out_plot_path=out_plot,
        emphasize_first=emphasize_first,
    )
    if out_dir and bool(explain_cfg.get("export_shap_csv", True)):
        csv_out = export_table(df_imp, out_dir, "shap_top_features")
        df_imp.attrs["csv_path"] = csv_out
        logger.info("SHAP 表已导出：%s", csv_out)
        if not df_imp.empty:
            top_name = str(df_imp.iloc[0]["feature"])
            dep_path = os.path.join(out_dir, f"shap_dependence_{top_name}.png")
            try:
                top_idx = list(names).index(top_name)
                shap_dependence_plot(
                    model,
                    X_norm,  # 使用归一化后的数据
                    feature_names=names,
                    top_feature_index=top_idx,
                    save_path=dep_path,
                )
                df_imp.attrs["dependence_plot"] = dep_path
                logger.info("SHAP Dependence 图已生成：%s", dep_path)
            except Exception as e:
                logger.warning("SHAP dependence 图生成失败：%s", e)
    return df_imp


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(_PROGRAM_DIR, "model")
    path = os.path.join(model_dir, "xgboost_reg.json")
    csv_path = os.path.join(_PROGRAM_DIR, "Yingmai 2 area in Tarim Basin.csv")
    if not os.path.isfile(path) or not os.path.isfile(csv_path):
        print("请先运行 train.py 并确保 CSV 存在。")
        sys.exit(1)
    df = explain_xgboost(path, csv_path, out_dir=model_dir)
    print("特征贡献占比（示例）：")
    print(df.head(10).to_string())
    if os.path.exists(os.path.join(model_dir, "shap_summary.png")):
        print("SHAP 图已保存至 model/shap_summary.png")
