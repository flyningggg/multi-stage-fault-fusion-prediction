# -*- coding: utf-8 -*-
"""
主线 A：面向断裂网络的空间–拓扑特征融合学习框架

目标：
- 将 fractopo 的拓扑指标 + 空间统计特征（长度、密度、数量等）+ 地质约束
  统一到一个特征矩阵中；
- 在此之上使用加权融合 + GAT 图模型 + XGBoost 回归做预测与解释。

说明：
- 这里假定「空间统计特征 + fractopo 拓扑指标」已经通过前置脚本
  写入网格 CSV（例如 Yingmai 2 area in Tarim Basin.csv）；
- 本模块负责：特征工程 → 融合（加权 / GAT）→ XGBoost 训练 → SHAP 解释，
  形成一条完整的算法主线，便于在国奖说明书中展示。
"""

import os
import sys
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd


def _program_dir_this_module() -> str:
    """
    本模块所在目录（与打包后的 config.yaml 同层）。
    不使用模块级全局 _PROGRAM_DIR，避免 PyInstaller / 子线程导入时出现 NameError。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # onedir：config 通常在 _MEIPASS 根目录，与本模块同层
        if os.path.isfile(os.path.join(here, "config.yaml")):
            return here
        parent = os.path.dirname(here)
        if os.path.isfile(os.path.join(parent, "config.yaml")):
            return parent
    return here


from feature_engineering import (
    build_feature_matrix,
    DEFAULT_FEATURE_COLUMNS,
)
from fusion_algorithm import (
    weighted_fusion,
    build_grid_graph,
    gat_fusion,
    adaptive_weighted_fusion,
)
from gnn_embeddings import (
    gnn_embedding_graphsage,
)
from multiscale_features import build_multiscale_pyramid
from ml.train import train_xgboost_regression
from ml.explain import shap_feature_importance
from utils.config_loader import load_config
from utils.export_utils import (
    export_spatial_dataframe,
    enrich_predictions_for_gis,
    LAYER_PREDICTIONS_XGB,
    export_table,
    build_run_metadata,
    write_run_manifest,
)
from utils.logging_utils import get_logger


def run_spatial_topology_fusion_pipeline(
    csv_path: str,
    target_column: Optional[str],
    config_path: Optional[str] = None,
    use_gat: bool = True,
    use_gnn_embedding: bool = True,
    use_multiscale: bool = True,
    use_adaptive_weight: bool = True,
    compute_shap: bool = True,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    面向断裂网络的「空间–拓扑特征融合学习」完整流水线。

    步骤：
    1. 利用 feature_engineering.build_feature_matrix 构建特征矩阵（拓扑 + 空间统计）；
    2. 基于专家规则做加权融合，得到勘探价值得分；
    3. 在规则网格图上运行 GAT，自适应地学习拓扑结构加权（可选）；
    4. 用 XGBoost 对目标属性做回归预测，评估 CV 与测试集指标；
    5. 使用 SHAP 对 XGBoost 模型做特征贡献度解释（可选）。

    返回：
        {
            "df": 带有融合得分和预测结果的 DataFrame,
            "xgb_result": XGBoost 训练结果 dict,
            "shap_importance": SHAP 特征重要性 DataFrame（如 compute_shap=False 则为 None),
            "gat_metrics": GAT 训练指标（若 use_gat 且依赖满足）
        }
    """
    cfg = load_config(config_path)
    fe_cfg = cfg.get("feature_engineering", {}) if isinstance(cfg, dict) else {}
    fusion_cfg = cfg.get("fusion", {}) if isinstance(cfg, dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    explain_cfg = cfg.get("explain", {}) if isinstance(cfg, dict) else {}
    log_cfg = cfg.get("logging", {}) if isinstance(cfg, dict) else {}
    high_value_attrs: Optional[List[str]] = cfg.get("high_value_attrs")
    logger = get_logger(
        "spatial_topology_framework",
        level=log_cfg.get("level", "INFO"),
        log_file=os.path.join(os.path.dirname(os.path.abspath(csv_path)), log_cfg.get("file", "logs/pipeline.log")),
    )
    if not target_column:
        target_column = train_cfg.get("target_column")

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 特征工程：空间 + 拓扑特征矩阵
    #    对原始 CSV 做 IQR 异常值截断 → StandardScaler 标准化 → 方差筛选 → 互信息精选
    r = build_feature_matrix(
        csv_path,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        target_column=target_column,
        normalize_method=fe_cfg.get("normalize_method", "standard"),
        outlier_method=fe_cfg.get("outlier_method", "iqr"),
        variance_threshold=float(fe_cfg.get("variance_threshold", 1e-6)),
        n_select_mi=fe_cfg.get("n_select_mi"),
        out_processed_dir=None,
    )
    X = r["X"]
    y = r["y"]
    feature_names = r["feature_names"]
    df = r["df"].copy()

    if y is None:
        raise ValueError("run_spatial_topology_fusion_pipeline 需要提供有效的 target_column。")
    logger.info("空间-拓扑融合启动：csv=%s target=%s samples=%s", csv_path, target_column, len(df))

    # 网格行列数：集中计算一次，供多尺度金字塔和 build_grid_graph 共用，
    # 避免两处独立 sqrt 因非完全平方数产生不一致的 n_rows/n_cols。
    _n_samples = X.shape[0]
    _grid_rows = int(np.sqrt(_n_samples)) or 1
    _grid_cols = (_n_samples + _grid_rows - 1) // _grid_rows
    _grid_padded = _grid_rows * _grid_cols  # build_grid_graph 使用的节点总数

    # 可选：多尺度空间特征金字塔
    #   对原始特征在更粗尺度上做均值 pooling，与细网格特征拼接，
    #   增强模型对多尺度空间模式的捕捉能力（如 2×2、4×4 邻域聚合）
    if use_multiscale:
        # 允许网格略大于样本数（末尾不足的格子用零填充）
        if _grid_padded > _n_samples:
            X_pad = np.zeros((_grid_padded, X.shape[1]), dtype=X.dtype)
            X_pad[:_n_samples] = X
        else:
            X_pad = X
        try:
            X_ms, suffixes = build_multiscale_pyramid(
                X_pad,
                n_rows=_grid_rows,
                n_cols=_grid_cols,
                scales=fusion_cfg.get("multiscale_scales", [1, 2, 4]),
                mode=fusion_cfg.get("multiscale_mode", "mean"),
            )
            # 还原到实际样本数
            X_ms = X_ms[:_n_samples]
            new_feature_names = []
            for suf in suffixes:
                new_feature_names.extend([f"{fn}_{suf}" for fn in feature_names])
            X = X_ms
            feature_names = new_feature_names
        except Exception as _ms_err:
            import warnings as _w
            _w.warn(f"多尺度金字塔构建失败（已跳过）：{_ms_err}")

    # 2. 加权融合：凸显高价值拓扑属性（规则版 + 自适应版）
    #    规则版：对 Connections per Branch/Trace、Connection Frequency 赋 1.5 倍权重
    high_value_weight = float(fusion_cfg.get("high_value_weight", 1.5))
    fused_weighted = weighted_fusion(
        X,
        feature_names,
        high_value_weight=high_value_weight,
        high_value_attrs=high_value_attrs,
    )
    df["fusion_weighted_score"] = fused_weighted

    # 自适应可学习加权（Adaptive Feature Weighting）
    adaptive_scores = None
    adaptive_weights_mean = None
    if use_adaptive_weight:
        try:
            adaptive_scores, adaptive_weights_mean = adaptive_weighted_fusion(
                X,
                feature_names,
                context_features=None,
                hidden_dim=int(fusion_cfg.get("adaptive_hidden_dim", 32)),
                epochs=int(fusion_cfg.get("adaptive_epochs", 200)),
                lr=float(fusion_cfg.get("adaptive_lr", 1e-3)),
            )
        except ImportError:
            adaptive_scores = None
    if adaptive_scores is not None:
        df["fusion_adaptive_score"] = adaptive_scores

    # 3. GAT 图模型融合（可选）
    #    将网格视为图节点，4 邻域连接为边，GAT 通过自编码式训练学习空间邻域感知的融合权重
    gat_scores = None
    gat_metrics: Dict[str, Any] = {}
    edge_index, _ = build_grid_graph(n_rows=_grid_rows, n_cols=_grid_cols)
    if use_gat:
        try:
            gat_scores, _, gat_metrics = gat_fusion(
                X,
                edge_index,
                epochs=int(fusion_cfg.get("gat_epochs", 80)),
                out_channels=1,
            )
        except (ImportError, RuntimeError, OSError) as e:
            gat_scores = None
            gat_metrics = {"error": f"GAT 不可用（已跳过）：{e}"}
        except Exception as e:
            gat_scores = None
            gat_metrics = {"error": f"GAT 运行时失败（已跳过，继续加权融合与 XGBoost）：{e}"}
    if gat_scores is not None:
        df["fusion_gat_score"] = gat_scores

    # 3.1 图级表示学习（GraphSAGE 嵌入）
    #   用 GraphSAGE 对网格图做节点嵌入，将图结构信息编码为稠密向量，
    #   拼接到工程特征后面，形成 "工程特征 + 图嵌入" 的联合特征表示
    gnn_node_emb = None
    gnn_graph_emb = None
    gnn_metrics: Dict[str, Any] = {}
    if use_gnn_embedding:
        try:
            gnn_node_emb, gnn_graph_emb, gnn_metrics = gnn_embedding_graphsage(
                X,
                edge_index,
                hidden_dim=int(fusion_cfg.get("gnn_hidden_dim", 64)),
                out_dim=int(fusion_cfg.get("gnn_out_dim", 32)),
                epochs=int(fusion_cfg.get("gnn_epochs", 200)),
            )
        except ImportError:
            gnn_metrics = {"error": "PyTorch / PyTorch Geometric 未安装，GraphSAGE 未运行"}
    if gnn_node_emb is not None:
        # 节点级嵌入拼接到特征矩阵，形成「工程特征 + 图嵌入」联合表示
        X = np.concatenate([X, gnn_node_emb], axis=1)
        # 为嵌入生成列名
        emb_names = [f"gnn_sage_{i}" for i in range(gnn_node_emb.shape[1])]
        feature_names = feature_names + emb_names

    # 4. XGBoost 回归：在融合特征基础上做监督预测
    #   5 折交叉验证 + 留出测试集，评估模型泛化能力
    xgb_result = train_xgboost_regression(
        X,
        y,
        df_meta=df,
        n_splits=int(train_cfg.get("n_splits", 5)),
        test_size=float(train_cfg.get("test_size", 0.1)),
        random_state=int(train_cfg.get("random_state", 42)),
    )
    model = xgb_result["model"]
    import matplotlib.pyplot as plt
    xgb_fig = plt.gcf()
    # 全量样本预测：计算每个网格的预测值及其 ±sigma 区间，供 GIS 可视化
    try:
        import numpy as _np

        y_pred_all = model.predict(_np.asarray(X))
    except Exception:
        y_pred_all = None
    if y_pred_all is not None and len(y_pred_all) == len(df):
        df["xgb_pred"] = y_pred_all
        sigma = float((xgb_result.get("prediction_interval") or {}).get("sigma", 1.96))
        resid_std = float((xgb_result.get("prediction_interval") or {}).get("residual_std", 0.0))
        df["xgb_pred_lower"] = df["xgb_pred"] - sigma * resid_std
        df["xgb_pred_upper"] = df["xgb_pred"] + sigma * resid_std

    # 5. SHAP 解释：特征贡献度
    #    用 KernelExplainer 计算每个特征对预测的平均贡献，输出贡献占比表
    shap_df = None
    if compute_shap:
        try:
            shap_df = shap_feature_importance(
                model,
                X,
                feature_names=feature_names,
                is_tree=True,
                out_plot_path=os.path.join(out_dir, "shap_summary.png"),
            )
            if bool(explain_cfg.get("export_shap_csv", True)):
                shap_csv = export_table(shap_df, out_dir, "shap_top_features")
                shap_df.attrs["csv_path"] = shap_csv
        except Exception as e:
            shap_df = pd.DataFrame(
                {"feature": feature_names, "importance": 0.0, "contribution_pct": 0.0}
            )
            shap_df.attrs["error"] = f"SHAP 计算失败: {e}"

    _cfg_dir = _program_dir_this_module()
    _cfg_yaml = os.path.join(_cfg_dir, "config.yaml")
    run_meta = build_run_metadata(
        config_path=_cfg_yaml,
        extra={"target_column": target_column},
    )
    df_export = enrich_predictions_for_gis(df, pred_col_preferred="xgb_pred")
    df_export["processing_run_id"] = run_meta.get("processing_run_id", "")
    df_export["run_timestamp_utc"] = run_meta.get("run_timestamp_utc", "")
    df_export["config_hash_sha256"] = run_meta.get("config_hash_sha256", "")
    export_paths = export_spatial_dataframe(
        df_export,
        out_dir,
        "spatial_topology_pipeline_results",
        export_csv=bool(train_cfg.get("export_predictions_csv", True)),
        export_gpkg=bool(train_cfg.get("export_predictions_gpkg", True)),
        layer_name=LAYER_PREDICTIONS_XGB,
    )
    manifest_path = write_run_manifest(
        out_dir,
        run_id=str(run_meta.get("processing_run_id", "")) or None,
        kind="pipeline",
        config_path=_cfg_yaml,
        artifacts={"csv": export_paths.get("csv"), "gpkg": export_paths.get("gpkg"), "shap_csv": shap_df.attrs.get("csv_path") if shap_df is not None else None},
        extra={"target_column": target_column},
    )
    export_paths["manifest"] = manifest_path
    logger.info("空间-拓扑融合结束：csv=%s gpkg=%s", export_paths.get("csv"), export_paths.get("gpkg"))

    return {
        "df": df,
        "xgb_result": xgb_result,
        "xgb_fig": xgb_fig,
        "shap_importance": shap_df,
        "gat_metrics": gat_metrics,
        "gnn_metrics": gnn_metrics,
        "adaptive_weights_mean": adaptive_weights_mean,
        "export_paths": export_paths,
    }


if __name__ == "__main__":
    """
    示例用法（命令行）：

    python spatial_topology_framework.py \
        "Yingmai 2 area in Tarim Basin.csv" \
        --target "Porosity"
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="面向断裂网络的空间–拓扑特征融合学习框架（示例运行入口）"
    )
    parser.add_argument("csv_path", help="网格级特征 CSV，包含 fractopo 拓扑指标与空间统计特征")
    parser.add_argument(
        "--target", dest="target_column", required=True, help="监督学习目标列名（如 Porosity）"
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="可选 config.yaml 路径，未提供时使用 program/config.yaml",
    )
    parser.add_argument(
        "--no-gat", dest="use_gat", action="store_false", help="关闭 GAT 融合，仅使用加权融合与 XGBoost"
    )
    parser.add_argument(
        "--no-shap", dest="compute_shap", action="store_false", help="关闭 SHAP 解释计算"
    )
    args = parser.parse_args()

    result = run_spatial_topology_fusion_pipeline(
        csv_path=args.csv_path,
        target_column=args.target_column,
        config_path=args.config_path,
        use_gat=args.use_gat,
        compute_shap=args.compute_shap,
    )
    print("融合框架运行完成。")
    print("XGBoost CV 指标聚合：", result["xgb_result"]["cv_agg"])
    print("XGBoost 测试集指标：", result["xgb_result"]["test_metrics"])
    if result["shap_importance"] is not None:
        print("Top5 特征贡献：")
        print(result["shap_importance"].head(5).to_string(index=False))

