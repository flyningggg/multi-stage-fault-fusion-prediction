# -*- coding: utf-8 -*-
"""多期断裂精确分析与候选勘探有利区统一正式主流程。"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from candidate_targeting import (
    add_period_scores,
    cluster_candidate_cells,
    match_period_nodes,
    select_candidate_cells,
)
from artifact_paths import portable_artifact_path
from external_validation import validate_external_points
from multiperiod_data import load_all_periods
from percolation import (
    _capacity_attr,
    apply_distance_transform,
    build_grid_graph,
    simulate_percolation,
)
from screening_contracts import ScreeningRunResult
from utils.config_loader import load_config
from utils.config_validation import validate_config
from utils.export_utils import config_file_hash


ProgressCallback = Optional[Callable[[str], None]]


def _emit(callback: ProgressCallback, message: str) -> None:
    if callback is not None:
        callback(str(message))


def _period_metric_config_hash(cfg: Dict) -> str:
    """仅对影响精确时期指标的配置生成缓存兼容哈希。"""
    screening = cfg.get("screening") or {}
    grid = cfg.get("grid") or {}
    payload = {
        "edge_weight_column": screening.get("edge_weight_column", "NC_A"),
        "distance_transforms": screening.get("distance_transforms"),
        "top_betweenness_fraction": screening.get("top_betweenness_fraction"),
        "network_criticality_weights": screening.get("network_criticality_weights"),
        "grid_step_m": grid.get("step_m"),
        "edge_dist_tolerance_m": grid.get("edge_dist_tolerance_m"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_top_nodes(values: Dict[int, float], fraction: float) -> set:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("Top-K比例必须在(0,1]内")
    ordered = sorted(values.items(), key=lambda item: (-float(item[1]), int(item[0])))
    k = max(1, int(np.ceil(len(ordered) * float(fraction))))
    return {int(node) for node, _ in ordered[:k]}


def compute_node_removal_impact(graph) -> Dict[int, float]:
    """精确计算割点移除导致的额外最大连通分量损失，非割点为0。"""
    import networkx as nx

    impacts = {int(node): 0.0 for node in graph.nodes}
    if graph.number_of_nodes() < 3 or graph.number_of_edges() == 0:
        return impacts
    largest_nodes = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_nodes).copy()
    n = subgraph.number_of_nodes()
    if n < 3:
        return impacts
    for node in nx.articulation_points(subgraph):
        reduced = subgraph.copy()
        reduced.remove_node(node)
        largest_after = max((len(c) for c in nx.connected_components(reduced)), default=0)
        impacts[int(node)] = float(max(0.0, 1.0 - largest_after / float(n - 1)))
    return impacts


def compute_period_node_metrics(
    gdf,
    period_name: str,
    edge_weight_col: str,
    distance_transforms: List[str],
    top_fraction: float,
    grid_step: float,
    edge_tolerance: float,
    criticality_weights: Optional[Dict[str, float]] = None,
    progress_callback: ProgressCallback = None,
) -> tuple[pd.DataFrame, Dict]:
    """计算单期精确中心性、稳定率、节点移除影响和渗流摘要。"""
    import networkx as nx

    if not distance_transforms:
        raise ValueError("distance_transforms 不能为空")
    graph = build_grid_graph(
        gdf,
        edge_weight_col=edge_weight_col,
        weight_mode="min",
        grid_step=float(grid_step),
        edge_dist_tolerance=float(edge_tolerance),
        distance_transform=distance_transforms[0],
    )
    _emit(progress_callback, f"{period_name}: 图构建完成，开始精确中心性")

    bc_by_transform: Dict[str, Dict[int, float]] = {}
    top_sets: Dict[str, set] = {}
    for transform in distance_transforms:
        apply_distance_transform(graph, transform)
        bc = nx.betweenness_centrality(graph, weight="distance", normalized=True)
        bc_by_transform[transform] = {int(k): float(v) for k, v in bc.items()}
        top_sets[transform] = _stable_top_nodes(bc_by_transform[transform], top_fraction)
        _emit(progress_callback, f"{period_name}: {transform} 精确中心性完成")

    baseline_transform = distance_transforms[0]
    baseline_bc = bc_by_transform[baseline_transform]
    apply_distance_transform(graph, baseline_transform)
    pagerank = nx.pagerank(graph, weight=_capacity_attr(graph), alpha=0.85)
    removal = compute_node_removal_impact(graph)
    fractions, sizes, threshold = simulate_percolation(graph, n_steps=100)

    centroids = gdf.geometry.centroid
    rows = []
    for node in graph.nodes:
        node = int(node)
        stable_count = sum(node in top_sets[name] for name in distance_transforms)
        rows.append({
            "period": str(period_name),
            "node_idx": node,
            "pos_x": float(centroids.iloc[node].x),
            "pos_y": float(centroids.iloc[node].y),
            "exact_betweenness": float(baseline_bc.get(node, 0.0)),
            "pagerank": float(pagerank.get(node, 0.0)),
            "degree": int(graph.degree(node)),
            "removal_impact": float(removal.get(node, 0.0)),
            "stability_rate": float(stable_count / len(distance_transforms)),
            "is_top20": bool(node in top_sets[baseline_transform]),
            "geometry": gdf.geometry.iloc[node],
        })
    frame = add_period_scores(pd.DataFrame(rows), criticality_weights=criticality_weights)
    summary = {
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "n_isolated": int(sum(1 for _, degree in graph.degree() if degree == 0)),
        "n_articulation_nodes": int(sum(value > 0 for value in removal.values())),
        "percolation_threshold_pc50": float(threshold),
        "distance_transforms": list(distance_transforms),
        "top_fraction": float(top_fraction),
        "top_counts": {name: int(len(nodes)) for name, nodes in top_sets.items()},
        "percolation_curve_points": int(len(fractions)),
        "percolation_final_lcc_fraction": float(sizes[-1]),
    }
    return frame, summary


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _export_frame_csv(frame: pd.DataFrame, path: Path) -> str:
    exported = frame.copy()
    if "geometry" in exported.columns:
        exported["geometry_wkt"] = exported["geometry"].map(
            lambda geometry: geometry.wkt if geometry is not None else ""
        )
        exported.drop(columns=["geometry"], inplace=True)
    exported.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def _load_cached_period_frame(path: Path) -> pd.DataFrame:
    """加载本流程导出的精确时期指标，并恢复空间几何。"""
    from shapely import wkt

    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "geometry_wkt" not in frame.columns:
        raise ValueError(f"缓存缺少 geometry_wkt: {path}")
    frame["geometry"] = frame["geometry_wkt"].map(
        lambda value: wkt.loads(value) if isinstance(value, str) and value else None
    )
    frame.drop(columns=["geometry_wkt"], inplace=True)
    return frame


def _export_targets_gpkg(targets: pd.DataFrame, stable_targets: pd.DataFrame, crs, path: Path) -> str:
    """导出可直接进入GIS的全部与稳定候选图层。"""
    import geopandas as gpd

    def _as_geodataframe(frame: pd.DataFrame):
        exported = frame.copy()
        for column in exported.columns:
            if column == "geometry":
                continue
            exported[column] = exported[column].map(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict)) else value
            )
        return gpd.GeoDataFrame(exported, geometry="geometry", crs=crs)

    if targets.empty:
        return ""
    _as_geodataframe(targets).to_file(
        path, layer="all_candidate_targets", driver="GPKG", mode="w"
    )
    if not stable_targets.empty:
        _as_geodataframe(stable_targets).to_file(
            path, layer="stable_candidate_targets", driver="GPKG", mode="a"
        )
    return str(path)


def _plot_candidate_map(period_gdfs, candidate_cells, targets, out_path: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.matplotlib_chinese import setup_matplotlib_chinese

    setup_matplotlib_chinese()

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#7C9A92", "#A8B7A5", "#9AA6B2"]
    for (period, gdf), color in zip(period_gdfs.items(), colors):
        gdf.boundary.plot(ax=ax, color=color, linewidth=0.25, alpha=0.20, label=period)
    if candidate_cells is not None and not candidate_cells.empty:
        points = ax.scatter(
            candidate_cells["centroid_x"], candidate_cells["centroid_y"],
            c=candidate_cells["total_score"], cmap="YlOrRd", s=18,
            edgecolors="none", alpha=0.75, label="候选单元",
        )
        colorbar = fig.colorbar(points, ax=ax, fraction=0.035, pad=0.02)
        colorbar.set_label("综合评分")
    if targets is not None and not targets.empty:
        for _, row in targets.iterrows():
            geometry = row.get("geometry")
            is_unstable = row.get("target_level") == "不稳定候选"
            if geometry is not None:
                try:
                    import geopandas as gpd
                    gpd.GeoSeries([geometry]).boundary.plot(
                        ax=ax,
                        color="#8B1E3F" if not is_unstable else "#7A7A7A",
                        linewidth=2.2 if not is_unstable else 0.8,
                        alpha=1.0 if not is_unstable else 0.45,
                        linestyle="--" if is_unstable else "-",
                    )
                except Exception:
                    pass
            if not is_unstable:
                ax.text(
                    row.get("representative_x", row["centroid_x"]),
                    row.get("representative_y", row["centroid_y"]),
                    f"{row['target_id']}\n{row.get('target_level', '')}",
                    fontsize=8, weight="bold", ha="center", va="center",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0},
                )
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], color="#8B1E3F", lw=2.2, label="稳定候选有利区"),
            Line2D([0], [0], color="#7A7A7A", lw=0.8, ls="--", label="不稳定候选（仅供复核）"),
        ], loc="upper right", frameon=False)
    ax.set_title("多期断裂网络候选勘探有利区")
    ax.set_xlabel("X 坐标（米）")
    ax.set_ylabel("Y 坐标（米）")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def run_target_screening(
    output_dir: str,
    config_path: Optional[str] = None,
    external_points: Optional[pd.DataFrame] = None,
    progress_callback: ProgressCallback = None,
    reuse_period_metrics_from: Optional[str] = None,
    period_gdfs_override: Optional[Mapping[str, Any]] = None,
    input_role: str = "operational_processed_grid",
) -> Dict:
    """正式候选靶区管线；可注入合成数据做受控验证，不调用代理模型。"""
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    period_dir = output / "period_metrics"
    map_dir = output / "maps"
    period_dir.mkdir(exist_ok=True)
    map_dir.mkdir(exist_ok=True)
    run_id = f"screening-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    warnings: List[str] = []
    limitations = [
        "候选靶区由内部拓扑证据生成，不等同于油气发现概率。",
        "未提供井位、储层或专家盲评时，外部有效性保持未验证。",
        "代理模型未参与正式评分或靶区决策。",
    ]
    if input_role not in {"operational_processed_grid", "synthetic_controlled"}:
        raise ValueError(f"未知输入角色: {input_role}")
    if period_gdfs_override is not None and input_role != "synthetic_controlled":
        raise ValueError("注入时期数据时必须显式声明 input_role='synthetic_controlled'")
    if input_role == "synthetic_controlled":
        limitations.append(
            "本次输入为受控合成网络，只验证算法恢复与失败边界，不构成工区或油气有效性证据。"
        )

    cfg = load_config(config_path)
    errors = validate_config(cfg)
    if errors:
        raise ValueError("配置校验失败: " + "; ".join(errors))
    screening = cfg.get("screening") or {}
    grid = cfg.get("grid") or {}
    score_weights = screening.get("score_weights") or None
    criticality_weights = screening.get("network_criticality_weights") or None
    transforms = list(screening.get("distance_transforms") or ["inverse", "inverse_sqrt", "neglog"])
    top_fraction = float(screening.get("top_betweenness_fraction", 0.20))

    _emit(progress_callback, "加载三期数据")
    period_gdfs = (
        dict(period_gdfs_override)
        if period_gdfs_override is not None
        else load_all_periods()
    )
    if len(period_gdfs) < 2:
        raise ValueError("候选筛选至少需要两个时期的数据")
    period_frames: Dict[str, pd.DataFrame] = {}
    period_summaries = {}
    artifact_paths: Dict[str, str] = {}
    reuse_source = Path(reuse_period_metrics_from).resolve() if reuse_period_metrics_from else None
    cached_result = None
    if reuse_source is not None:
        cached_result_path = reuse_source / "result.json"
        cached_manifest_path = reuse_source / "manifest.json"
        if not cached_result_path.is_file() or not cached_manifest_path.is_file():
            raise ValueError("复用目录必须包含 result.json 与 manifest.json")
        cached_result = json.loads(cached_result_path.read_text(encoding="utf-8"))
        cached_manifest = json.loads(cached_manifest_path.read_text(encoding="utf-8"))
        cached_snapshot = reuse_source / "config_snapshot.yaml"
        if not cached_snapshot.is_file():
            raise ValueError("复用目录缺少 config_snapshot.yaml")
        cached_cfg = load_config(str(cached_snapshot))
        if _period_metric_config_hash(cached_cfg) != _period_metric_config_hash(cfg):
            raise ValueError("缓存精确时期指标的相关配置与当前配置不一致，拒绝复用")
        _emit(progress_callback, f"复用精确时期指标: {reuse_source}")
    for period_name, gdf in period_gdfs.items():
        if reuse_source is not None:
            source_csv = reuse_source / "period_metrics" / f"{period_name}_nodes.csv"
            if not source_csv.is_file():
                raise ValueError(f"复用目录缺少时期指标: {source_csv}")
            frame = _load_cached_period_frame(source_csv)
            try:
                summary = cached_result["period_results"][period_name]
            except KeyError as exc:
                raise ValueError(f"缓存结果缺少时期摘要: {period_name}") from exc
        else:
            _emit(progress_callback, f"开始分析 {period_name}")
            frame, summary = compute_period_node_metrics(
                gdf,
                period_name=period_name,
                edge_weight_col=str(screening.get("edge_weight_column", "NC_A")),
                distance_transforms=transforms,
                top_fraction=top_fraction,
                grid_step=float(grid.get("step_m", 3000.0)),
                edge_tolerance=float(grid.get("edge_dist_tolerance_m", 150.0)),
                criticality_weights=criticality_weights,
                progress_callback=progress_callback,
            )
        period_frames[period_name] = frame
        period_summaries[period_name] = summary
        artifact_paths[f"period_{period_name}_csv"] = _export_frame_csv(
            frame, period_dir / f"{period_name}_nodes.csv"
        )

    if all(summary["n_articulation_nodes"] == 0 for summary in period_summaries.values()):
        message = "三期网络均无割点，节点移除连通分量影响项无区分力，本轮综合分上限相应降低。"
        warnings.append(message)
        limitations.append(message)

    _emit(progress_callback, "多期节点空间匹配与透明评分")
    matched = match_period_nodes(
        period_frames,
        tolerance_m=float(grid.get("centroid_match_tolerance_m", 1500.0)),
        score_weights=score_weights,
    )
    candidate_cells, candidate_contract = select_candidate_cells(
        matched,
        score_quantile=float(screening.get("candidate_cell_quantile", 0.80)),
        min_periods=int(screening.get("min_supporting_periods", 2)),
    )
    clustered_cells, targets, evidence_cards = cluster_candidate_cells(
        candidate_cells,
        eps_m=float((screening.get("target_clustering") or {}).get(
            "eps_m", grid.get("target_eps_m", 4500.0)
        )),
        min_samples=int((screening.get("target_clustering") or {}).get(
            "min_samples", grid.get("target_min_cluster_size", 3)
        )),
        max_diameter_m=float((screening.get("target_clustering") or {}).get(
            "max_diameter_m", 18000.0
        )),
    )
    external = validate_external_points(
        targets,
        external_points,
        buffer_m=float((screening.get("external_validation") or {}).get("buffer_m", 0.0)),
    )

    artifact_paths["matched_cells_csv"] = _export_frame_csv(
        matched, output / "matched_cells.csv"
    )
    artifact_paths["candidate_cells_csv"] = _export_frame_csv(
        clustered_cells, output / "candidate_cells.csv"
    )
    artifact_paths["candidate_targets_csv"] = _export_frame_csv(
        targets, output / "candidate_targets.csv"
    )
    stable_targets = (
        targets[targets["evidence_status"] == "internal_supported"].copy()
        if not targets.empty and "evidence_status" in targets.columns
        else targets.copy()
    )
    artifact_paths["stable_targets_csv"] = _export_frame_csv(
        stable_targets, output / "stable_candidate_targets.csv"
    )
    source_crs = next(iter(period_gdfs.values())).crs
    if source_crs is None:
        message = "源CSV未声明坐标参考系，GeoPackage保留原始米制坐标但不写入EPSG；对接GIS前需人工确认CRS。"
        warnings.append(message)
        limitations.append(message)
    try:
        gpkg_path = _export_targets_gpkg(
            targets,
            stable_targets,
            source_crs,
            output / "candidate_targets.gpkg",
        )
        if gpkg_path:
            artifact_paths["candidate_targets_gpkg"] = gpkg_path
    except Exception as exc:
        warnings.append(f"GeoPackage导出失败: {exc}")
    evidence_path = output / "evidence_cards.json"
    evidence_path.write_text(
        json.dumps(_jsonable(evidence_cards), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    artifact_paths["evidence_cards_json"] = str(evidence_path)
    external_path = output / "external_validation.json"
    external_path.write_text(
        json.dumps(_jsonable(external), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    artifact_paths["external_validation_json"] = str(external_path)
    map_path = map_dir / "candidate_targets.png"
    artifact_paths["candidate_targets_png"] = _plot_candidate_map(
        period_gdfs, clustered_cells, targets, map_path
    )

    target_records = []
    if not targets.empty:
        export_columns = [column for column in targets.columns if column != "geometry"]
        target_records = targets[export_columns].to_dict("records")
    stability_summary = {
        "distance_transforms": transforms,
        "stable_target_count": int(sum(
            float(row.get("parameter_stability", 0.0)) >= 0.80
            for row in target_records
        )),
        "target_count": int(len(target_records)),
        "definition": "节点进入各距离转换Top20%的比例，按靶区候选单元取均值。",
    }
    input_summary = {
        "input_role": input_role,
        "period_count": int(len(period_gdfs)),
        "period_grid_counts": {name: int(len(gdf)) for name, gdf in period_gdfs.items()},
        "matched_cell_count": int(len(matched)),
        **candidate_contract,
        "candidate_target_count": int(len(target_records)),
        "stable_target_count": int(sum(
            row.get("evidence_status") == "internal_supported" for row in target_records
        )),
        "unstable_target_count": int(sum(
            row.get("evidence_status") != "internal_supported" for row in target_records
        )),
    }
    elapsed = time.perf_counter() - started
    status = "completed" if target_records else "completed_no_targets"
    result = ScreeningRunResult(
        run_id=run_id,
        status=status,
        input_summary=input_summary,
        period_results=period_summaries,
        candidate_targets=_jsonable(target_records),
        stability_summary=stability_summary,
        external_validation=external,
        artifact_paths=artifact_paths,
        limitations=limitations,
        warnings=warnings,
    ).to_dict()
    result["elapsed_seconds"] = float(elapsed)
    result["evaluation_summary"] = {
        "outcome": status,
        "claim_update": "supports_internal_candidate_screening_only",
        "baseline_relation": "extends_validation_v2_exact_topology_path",
        "comparability": "exact_graph_semantics_preserved",
        "failure_mode": None,
        "next_action": "接入真实井位、储层或专家盲评进行独立外部验证",
    }

    config_used = Path(config_path).resolve() if config_path else Path(__file__).with_name("config.yaml")
    if config_used.exists():
        shutil.copy2(config_used, output / "config_snapshot.yaml")
        artifact_paths["config_snapshot"] = str(output / "config_snapshot.yaml")
    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": portable_artifact_path(config_used),
        "config_hash_sha256": config_file_hash(str(config_used)) if config_used.exists() else None,
        "period_metric_config_sha256": _period_metric_config_hash(cfg),
        "distance_transforms": transforms,
        "top_fraction": top_fraction,
        "score_weights": score_weights,
        "elapsed_seconds": elapsed,
        "status": status,
        "reused_period_metrics_from": portable_artifact_path(reuse_source),
        "input_role": input_role,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    artifact_paths["manifest_json"] = str(manifest_path)
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    artifact_paths["result_json"] = str(result_path)
    report = [
        "# 多期断裂候选勘探有利区分析报告",
        "",
        f"- 运行ID：`{run_id}`",
        f"- 状态：`{status}`",
        f"- 三期有效网格：{sum(input_summary['period_grid_counts'].values())}",
        f"- 多期匹配单元：{len(matched)}",
        f"- 候选单元：{len(candidate_cells)}",
        f"- 候选靶区：{len(target_records)}",
        f"- 稳定候选靶区：{input_summary['stable_target_count']}",
        f"- 不稳定候选（仅供复核）：{input_summary['unstable_target_count']}",
        f"- 外部验证：`{external['status']}`",
        f"- 运行耗时：{elapsed:.1f}秒",
        "",
        "## 证据边界",
        "",
        *[f"- {item}" for item in limitations],
    ]
    report_path = output / "report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    artifact_paths["report_md"] = str(report_path)

    # 产物路径补写后刷新结果文件。
    artifact_paths = {
        key: portable_artifact_path(value) for key, value in artifact_paths.items()
    }
    result["artifact_paths"] = artifact_paths
    result_path.write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _emit(progress_callback, f"完成：{len(target_records)}个候选靶区")
    return result
