# -*- coding: utf-8 -*-
"""
算法效果对比实验主脚本
======================
一键运行以下四类对比实验，并自动生成图表与汇总报告：

1. 融合方法对比：规则加权融合 vs GAT 融合（得分分布箱线图）
2. 消融实验：仅几何特征 / 仅拓扑特征 / 几何+拓扑融合 / 融合+图嵌入
3. 算法族对比：规则基线 / 随机森林 / XGBoost / XGBoost+GNN
4. 稳健性实验：不同噪声强度下指标变化折线图

运行方式：
    cd program
    python run_comparison_experiment.py                          # 使用默认 CSV
    python run_comparison_experiment.py --csv path/to/data.csv  # 指定 CSV
    python run_comparison_experiment.py --target "Fracture Intensity B21"  # 指定目标列
    python run_comparison_experiment.py --skip-gat              # 跳过 GAT（无 PyG 时）

输出目录：program/data/experiments/
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

_PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.matplotlib_chinese import setup_matplotlib_chinese

setup_matplotlib_chinese()

from feature_engineering import build_feature_matrix, DEFAULT_FEATURE_COLUMNS
from fusion_algorithm import weighted_fusion, build_grid_graph
from evaluation import (
    regression_metrics,
    plot_fusion_comparison_boxplot,
)
from experiments_analysis import (
    run_ablation_experiments,
    run_model_family_comparison,
    run_noise_sensitivity_experiment,
)

# ─────────────────────────── 默认配置 ───────────────────────────

DEFAULT_CSV = "Yingmai 2 area in Tarim Basin.csv"
DEFAULT_TARGET = "Fracture Intensity B21"
OUT_DIR = os.path.join(_PROGRAM_DIR, "data", "experiments")


# ─────────────────────────── 辅助绘图 ───────────────────────────


def _bar_comparison(
    df: pd.DataFrame,
    x_col: str,
    metric_cols: list,
    title: str,
    save_path: str,
    higher_is_better: dict | None = None,
) -> None:
    """通用柱状图对比：每个方法一组柱，每个指标一种颜色。"""
    if df.empty:
        return
    if higher_is_better is None:
        higher_is_better = {}

    n_groups = len(df)
    n_metrics = len(metric_cols)
    x = np.arange(n_groups)
    width = 0.8 / max(n_metrics, 1)

    colors = plt.cm.tab10(np.linspace(0, 0.9, n_metrics))
    fig, ax = plt.subplots(figsize=(max(7, n_groups * 1.5), 4.5))

    for i, col in enumerate(metric_cols):
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        bars = ax.bar(x + i * width, vals, width, label=col, color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * (abs(v) + 1e-9),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(df[x_col].tolist(), rotation=15, ha="right", fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [图表] 已保存: {save_path}")


def _line_noise(df: pd.DataFrame, metric_cols: list, title: str, save_path: str) -> None:
    """噪声敏感性折线图。"""
    if df.empty or "noise_sigma" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(metric_cols)))
    for col, c in zip(metric_cols, colors):
        if col not in df.columns:
            continue
        ax.plot(df["noise_sigma"], df[col], marker="o", label=col, color=c)
    ax.set_xlabel("噪声强度 σ")
    ax.set_ylabel("指标值")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [图表] 已保存: {save_path}")


def _radar_chart(
    df: pd.DataFrame,
    method_col: str,
    metric_cols: list,
    title: str,
    save_path: str,
) -> None:
    """雷达图：对比各方法在多个指标上的表现（指标需先归一化到 0-1）。"""
    if df.empty:
        return
    available = [c for c in metric_cols if c in df.columns]
    if len(available) < 3:
        return

    labels = available
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(df)))

    # 对每列做 min-max 归一化（越大越好统一处理）
    norm_df = df[available].copy()
    for col in available:
        col_min, col_max = norm_df[col].min(), norm_df[col].max()
        if col_max > col_min:
            norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
        else:
            norm_df[col] = 0.5

    for idx, (_, row) in enumerate(df.iterrows()):
        vals = norm_df.loc[row.name, available].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[idx], linewidth=1.5, label=str(row[method_col]))
        ax.fill(angles, vals, color=colors[idx], alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    ax.set_title(title, fontsize=11, pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [图表] 已保存: {save_path}")


def _summary_table(results: dict, save_path: str) -> None:
    """将所有实验结果汇总为一张 HTML 表格（方便展示/答辩）。"""
    lines = ["<html><head><meta charset='utf-8'>",
             "<style>table{border-collapse:collapse;font-family:sans-serif;font-size:13px}"
             "th,td{border:1px solid #ccc;padding:6px 12px}th{background:#4472C4;color:#fff}"
             "tr:nth-child(even){background:#f2f2f2}</style></head><body>"]
    for exp_name, df in results.items():
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            continue
        lines.append(f"<h3>{exp_name}</h3>")
        if isinstance(df, pd.DataFrame):
            lines.append(df.to_html(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("<br>")
    lines.append("</body></html>")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [报告] 汇总 HTML 已保存: {save_path}")


# ─────────────────────────── 实验主流程 ───────────────────────────


def run_all(csv_path: str, target_column: str, skip_gat: bool = False) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  算法效果对比实验")
    print(f"  数据: {os.path.basename(csv_path)}")
    print(f"  目标列: {target_column}")
    print(f"{'='*60}\n")

    all_results: dict = {}

    # ── 实验 1：融合方法对比（加权融合 vs GAT 融合）──────────────
    print("【实验 1】融合方法得分分布对比（加权融合 vs GAT 融合）")
    try:
        r = build_feature_matrix(csv_path, out_processed_dir=None)
        X, names = r["X"], r["feature_names"]
        n = X.shape[0]
        w_scores = weighted_fusion(X, names)

        gat_scores = None
        if not skip_gat:
            try:
                from fusion_algorithm import gat_fusion
                edge_index, _ = build_grid_graph(n_nodes=n)
                gat_raw, _, _ = gat_fusion(X, edge_index, epochs=80, out_channels=1)
                gat_scores = np.asarray(gat_raw).ravel()
            except ImportError:
                print("  [提示] 未安装 PyTorch Geometric，跳过 GAT 融合。")
            except Exception as e:
                print(f"  [警告] GAT 融合失败: {e}")

        scores_dict = {"加权融合": w_scores}
        if gat_scores is not None and np.ptp(gat_scores) > 1e-10:
            scores_dict["GAT 融合"] = gat_scores

        boxplot_path = os.path.join(OUT_DIR, "exp1_fusion_boxplot.png")
        plot_fusion_comparison_boxplot(
            scores_dict,
            save_path=boxplot_path,
            title="实验1：融合方法得分分布对比",
        )

        fusion_summary = pd.DataFrame({
            "方法": list(scores_dict.keys()),
            "均值": [np.mean(v) for v in scores_dict.values()],
            "标准差": [np.std(v) for v in scores_dict.values()],
            "最小值": [np.min(v) for v in scores_dict.values()],
            "最大值": [np.max(v) for v in scores_dict.values()],
        })
        print(fusion_summary.to_string(index=False))
        all_results["实验1_融合方法对比"] = fusion_summary
    except Exception as e:
        print(f"  [错误] 实验1 失败: {e}")

    # ── 实验 2：消融实验 ──────────────────────────────────────────
    print("\n【实验 2】消融实验（几何 / 拓扑 / 融合 / 融合+GNN）")
    try:
        df_ablation = run_ablation_experiments(csv_path, target_column=target_column, out_dir=OUT_DIR)
        print(df_ablation.to_string(index=False))
        all_results["实验2_消融实验"] = df_ablation

        metric_cols_ab = [c for c in ["test_R2", "test_MAE", "test_RMSE", "R2_mean", "MAE_mean"] if c in df_ablation.columns]
        _bar_comparison(
            df_ablation, "config", metric_cols_ab,
            "实验2：消融实验指标对比",
            os.path.join(OUT_DIR, "exp2_ablation_bar.png"),
        )
        _radar_chart(
            df_ablation, "config", metric_cols_ab,
            "实验2：消融实验雷达图",
            os.path.join(OUT_DIR, "exp2_ablation_radar.png"),
        )
    except Exception as e:
        print(f"  [错误] 实验2 失败: {e}")

    # ── 实验 3：算法族对比 ────────────────────────────────────────
    print("\n【实验 3】算法族对比（规则基线 / 随机森林 / XGBoost / XGBoost+GNN）")
    try:
        df_models = run_model_family_comparison(csv_path, target_column=target_column)
        print(df_models.to_string(index=False))
        all_results["实验3_算法族对比"] = df_models

        metric_cols_m = [c for c in ["test_R2", "R2", "test_MAE", "MAE", "test_RMSE", "RMSE", "R2_mean", "MAE_mean"] if c in df_models.columns]
        _bar_comparison(
            df_models, "model", metric_cols_m,
            "实验3：算法族对比",
            os.path.join(OUT_DIR, "exp3_model_bar.png"),
        )
        _radar_chart(
            df_models, "model", metric_cols_m,
            "实验3：算法族雷达图",
            os.path.join(OUT_DIR, "exp3_model_radar.png"),
        )
    except Exception as e:
        print(f"  [错误] 实验3 失败: {e}")

    # ── 实验 4：噪声稳健性 ────────────────────────────────────────
    print("\n【实验 4】噪声稳健性实验（不同噪声强度下指标变化）")
    try:
        df_noise = run_noise_sensitivity_experiment(
            csv_path,
            target_column=target_column,
            noise_levels=[0.0, 0.05, 0.10, 0.20, 0.30],
        )
        print(df_noise.to_string(index=False))
        all_results["实验4_噪声稳健性"] = df_noise

        noise_metrics = [c for c in ["R2_mean", "MAE_mean", "RMSE_mean", "test_R2", "test_MAE"] if c in df_noise.columns]
        _line_noise(
            df_noise, noise_metrics,
            "实验4：噪声稳健性（指标随噪声变化）",
            os.path.join(OUT_DIR, "exp4_noise_line.png"),
        )
    except Exception as e:
        print(f"  [错误] 实验4 失败: {e}")

    # ── 汇总 HTML 报告 ────────────────────────────────────────────
    _summary_table(all_results, os.path.join(OUT_DIR, "experiment_summary.html"))

    # ── 保存各实验 CSV ────────────────────────────────────────────
    for name, df in all_results.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            fname = name.replace(" ", "_").replace("/", "-") + ".csv"
            df.to_csv(os.path.join(OUT_DIR, fname), index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"  全部实验完成！结果已保存至: {OUT_DIR}")
    print(f"{'='*60}\n")


# ─────────────────────────── 入口 ───────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="算法效果对比实验")
    parser.add_argument("--csv", default=None, help="网格 CSV 路径（默认使用 MY 区域示例数据）")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="回归目标列名")
    parser.add_argument("--skip-gat", action="store_true", help="跳过 GAT 融合（无 PyTorch Geometric 时使用）")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    csv_path = args.csv
    if csv_path is None:
        # 自动查找示例 CSV
        candidates = [
            os.path.join(_PROGRAM_DIR, DEFAULT_CSV),
            os.path.join(_PROGRAM_DIR, "MY", DEFAULT_CSV),
        ]
        for c in candidates:
            if os.path.isfile(c):
                csv_path = c
                break
        if csv_path is None:
            print(
                f"[错误] 未找到默认 CSV 文件 '{DEFAULT_CSV}'。\n"
                "请先运行：python export_grid_csv.py MY\n"
                "或通过 --csv 参数指定 CSV 路径。"
            )
            sys.exit(1)

    run_all(csv_path, target_column=args.target, skip_gat=args.skip_gat)
