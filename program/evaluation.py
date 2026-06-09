# -*- coding: utf-8 -*-
"""
效果量化：对比「单属性分析」「加权融合」「GAT 融合」与 ML 预测效果。
回归：MAE / RMSE / R²；分类：准确率 / 召回率 / F1 / 混淆矩阵；
行业指标：融合后分析效率提升率、勘探成本预估降低率。
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ---------- 回归 ----------


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """回归任务：MAE, RMSE, R²。"""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


# ---------- 分类 ----------


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> Dict[str, Any]:
    """分类任务：准确率、召回率、F1、混淆矩阵。"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ---------- 行业指标（可量化答辩） ----------


def efficiency_improvement_rate(
    time_baseline: float,
    time_fusion: float,
) -> float:
    """
    融合后分析效率提升率（对比传统人工/单属性分析）。
    time_baseline: 传统方式耗时（如小时）；time_fusion: 融合分析耗时。
    返回提升率，如 0.35 表示效率提升 35%。
    """
    if time_baseline <= 0:
        return 0.0
    return max(0.0, (time_baseline - time_fusion) / time_baseline)


def exploration_cost_reduction_rate(
    cost_baseline: float,
    cost_optimized: float,
) -> float:
    """
    勘探成本预估降低率（基于融合结果优化勘探点位后）。
    cost_baseline: 原方案预估成本；cost_optimized: 优化后预估成本。
    返回降低率，如 0.2 表示成本降低 20%。
    """
    if cost_baseline <= 0:
        return 0.0
    return max(0.0, (cost_baseline - cost_optimized) / cost_baseline)


def industry_metrics_report(
    time_baseline: float = 10.0,
    time_fusion: float = 4.0,
    cost_baseline: float = 1.0,
    cost_optimized: float = 0.8,
) -> Dict[str, float]:
    """汇总行业指标（示例数值可替换为实际测算）。"""
    return {
        "efficiency_improvement_rate": efficiency_improvement_rate(time_baseline, time_fusion),
        "exploration_cost_reduction_rate": exploration_cost_reduction_rate(cost_baseline, cost_optimized),
    }


# ---------- 可视化（评审材料加分） ----------


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "混淆矩阵",
) -> None:
    """混淆矩阵热力图，用于分类任务评估展示。"""
    import matplotlib.pyplot as plt
    from utils.matplotlib_chinese import setup_matplotlib_chinese
    setup_matplotlib_chinese()
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_title(title)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_cv_metrics(
    cv_metrics: List[Dict[str, float]],
    metric_names: List[str],
    save_path: Optional[str] = None,
    title: str = "各折交叉验证指标",
) -> None:
    """CV 各折指标折线图（如 MAE、R² 每折）。"""
    import matplotlib.pyplot as plt
    from utils.matplotlib_chinese import setup_matplotlib_chinese
    setup_matplotlib_chinese()
    folds = list(range(1, len(cv_metrics) + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in metric_names:
        vals = [m.get(name, np.nan) for m in cv_metrics]
        ax.plot(folds, vals, marker="o", label=name)
    ax.set_xlabel("折")
    ax.set_ylabel("指标")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_fusion_comparison_boxplot(
    scores_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    title: str = "加权融合 vs GAT 融合 得分分布",
) -> None:
    """加权融合与 GAT 融合得分分布箱线图，突出 GAT 优势。"""
    import matplotlib.pyplot as plt
    from utils.matplotlib_chinese import setup_matplotlib_chinese
    setup_matplotlib_chinese()
    labels = list(scores_dict.keys())
    data = [np.asarray(scores_dict[k]).ravel() for k in labels]
    n_max = max(len(d) for d in data) if data else 0
    # 样本多时须线外的「离群点」默认画成圆点；成千上万点重叠会像黑条，故大样本不画离群点
    show_fliers = n_max <= 300
    flier_kw = dict(marker="o", markersize=2, alpha=0.25, markeredgecolor="0.4", linestyle="none")
    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=show_fliers,
        flierprops=flier_kw if show_fliers else None,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.set_ylabel("融合得分")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


# ---------- 对比报告 ----------


def compare_fusion_methods(
    y_true: np.ndarray,
    pred_single: Optional[np.ndarray] = None,
    pred_weighted: Optional[np.ndarray] = None,
    pred_gat: Optional[np.ndarray] = None,
    task: str = "regression",
) -> pd.DataFrame:
    """
    对比单属性、加权融合、GAT 融合的预测效果。
    pred_* 为各方法预测值；task 为 'regression' 或 'classification'。
    返回汇总表。
    """
    rows = []
    if task == "regression":
        for name, pred in [("single", pred_single), ("weighted", pred_weighted), ("gat", pred_gat)]:
            if pred is not None and len(pred) == len(y_true):
                rows.append({"method": name, **regression_metrics(y_true, pred)})
    else:
        for name, pred in [("single", pred_single), ("weighted", pred_weighted), ("gat", pred_gat)]:
            if pred is not None and len(pred) == len(y_true):
                rows.append({"method": name, **classification_metrics(y_true, pred)})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


if __name__ == "__main__":
    # 示例
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p1 = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    p2 = np.array([1.05, 1.95, 3.05, 3.95, 5.05])
    print(compare_fusion_methods(y, pred_single=p1, pred_weighted=p2, pred_gat=p2))
    print(industry_metrics_report())
