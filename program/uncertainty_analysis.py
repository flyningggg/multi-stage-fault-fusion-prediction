# -*- coding: utf-8 -*-
"""P2 参数不确定性：预注册场景、跨场景靶区匹配与稳定性汇总。"""
from __future__ import annotations

import copy
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

from candidate_targeting import (
    add_period_scores,
    cluster_candidate_cells,
    match_period_nodes,
    select_candidate_cells,
)
from multiperiod_data import load_all_periods
from percolation import _capacity_attr, apply_distance_transform, build_grid_graph
from screening_pipeline import (
    _export_frame_csv,
    _load_cached_period_frame,
    _period_metric_config_hash,
    _stable_top_nodes,
    run_target_screening,
)
from utils.config_loader import load_config


ProgressCallback = Optional[Callable[[str], None]]


@dataclass(frozen=True)
class UncertaintyScenario:
    scenario_id: str
    label: str
    factor: str
    direction: str
    overrides: Dict[str, object]


def _emit(callback: ProgressCallback, message: str) -> None:
    if callback is not None:
        callback(str(message))


def _set_nested(mapping: Dict, dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    cursor = mapping
    for key in parts[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[parts[-1]] = copy.deepcopy(value)


def apply_overrides(config: Dict, overrides: Dict[str, object]) -> Dict:
    out = copy.deepcopy(config)
    for key, value in overrides.items():
        _set_nested(out, key, value)
    return out


def _emphasize_weight(weights: Dict[str, float], key: str, delta: float = 0.10) -> Dict[str, float]:
    values = {name: float(value) for name, value in weights.items()}
    if key not in values:
        raise ValueError(f"未知评分权重: {key}")
    new_key = min(0.90, values[key] + float(delta))
    remaining_old = 1.0 - values[key]
    remaining_new = 1.0 - new_key
    if remaining_old <= 0:
        raise ValueError("无法增强已占满全部权重的分项")
    for name in values:
        if name != key:
            values[name] = values[name] * remaining_new / remaining_old
    values[key] = new_key
    return values


def build_default_scenarios(config: Dict) -> List[UncertaintyScenario]:
    """构建固定的单因素扰动集合；不根据运行结果追加场景。"""
    grid = config.get("grid") or {}
    screening = config.get("screening") or {}
    clustering = screening.get("target_clustering") or {}
    weights = screening.get("score_weights") or {}
    transforms = list(screening.get("distance_transforms") or [])

    step = float(grid.get("step_m", 3000.0))
    edge_tol = float(grid.get("edge_dist_tolerance_m", 150.0))
    match_tol = float(grid.get("centroid_match_tolerance_m", 1500.0))
    quantile = float(screening.get("candidate_cell_quantile", 0.80))
    eps = float(clustering.get("eps_m", 4500.0))
    diameter = float(clustering.get("max_diameter_m", 18000.0))

    scenarios = [
        UncertaintyScenario("grid_step_low", "网格步长 -10%", "grid_step_m", "low", {"grid.step_m": step * 0.90}),
        UncertaintyScenario("grid_step_high", "网格步长 +10%", "grid_step_m", "high", {"grid.step_m": step * 1.10}),
        UncertaintyScenario("adjacency_tight", "邻接容差收紧", "edge_tolerance_m", "low", {"grid.edge_dist_tolerance_m": edge_tol * 0.50}),
        UncertaintyScenario("adjacency_loose", "邻接容差放宽", "edge_tolerance_m", "high", {"grid.edge_dist_tolerance_m": edge_tol * 2.00}),
        UncertaintyScenario("matching_tight", "跨期匹配容差收紧", "match_tolerance_m", "low", {"grid.centroid_match_tolerance_m": match_tol * 0.80}),
        UncertaintyScenario("matching_loose", "跨期匹配容差放宽", "match_tolerance_m", "high", {"grid.centroid_match_tolerance_m": match_tol * 1.20}),
        UncertaintyScenario("quantile_low", "候选分位数降低", "candidate_quantile", "low", {"screening.candidate_cell_quantile": max(0.05, quantile - 0.05)}),
        UncertaintyScenario("quantile_high", "候选分位数提高", "candidate_quantile", "high", {"screening.candidate_cell_quantile": min(0.95, quantile + 0.05)}),
        UncertaintyScenario("cluster_eps_low", "聚类半径 -1/6", "cluster_eps_m", "low", {"screening.target_clustering.eps_m": eps * 5.0 / 6.0}),
        UncertaintyScenario("cluster_eps_high", "聚类半径 +1/6", "cluster_eps_m", "high", {"screening.target_clustering.eps_m": eps * 7.0 / 6.0}),
        UncertaintyScenario("diameter_low", "最大直径 -1/6", "max_diameter_m", "low", {"screening.target_clustering.max_diameter_m": diameter * 5.0 / 6.0}),
        UncertaintyScenario("diameter_high", "最大直径 +1/6", "max_diameter_m", "high", {"screening.target_clustering.max_diameter_m": diameter * 7.0 / 6.0}),
    ]
    if weights:
        scenarios.extend([
            UncertaintyScenario(
                "weight_topology", "增强网络关键性权重", "score_weights", "topology",
                {"screening.score_weights": _emphasize_weight(weights, "network_criticality")},
            ),
            UncertaintyScenario(
                "weight_persistence", "增强多期持续性权重", "score_weights", "persistence",
                {"screening.score_weights": _emphasize_weight(weights, "period_persistence")},
            ),
        ])
    for transform in ("inverse_sqrt", "neglog"):
        if transform in transforms and transforms and transforms[0] != transform:
            reordered = [transform] + [name for name in transforms if name != transform]
            scenarios.append(UncertaintyScenario(
                f"distance_{transform}", f"以 {transform} 为基准距离变换",
                "distance_transform", transform,
                {"screening.distance_transforms": reordered},
            ))
    return scenarios


def load_target_table(run_dir: str | Path) -> pd.DataFrame:
    path = Path(run_dir) / "candidate_targets.csv"
    if not path.is_file():
        raise FileNotFoundError(f"候选靶区表不存在: {path}")
    frame = pd.read_csv(path)
    required = {"target_id", "centroid_x", "centroid_y", "total_score", "evidence_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"候选靶区表缺少列: {sorted(missing)}")
    return frame


def match_target_tables(
    baseline: pd.DataFrame,
    scenario: pd.DataFrame,
    max_distance_m: float,
) -> pd.DataFrame:
    """用带虚拟未匹配列的全局最小距离分配，确定性匹配跨场景靶区。"""
    if max_distance_m <= 0:
        raise ValueError("靶区匹配半径必须为正数")
    columns = ["baseline_index", "scenario_index", "distance_m"]
    if baseline.empty or scenario.empty:
        return pd.DataFrame(columns=columns)
    base_xy = baseline[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    scen_xy = scenario[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    if not np.isfinite(base_xy).all() or not np.isfinite(scen_xy).all():
        raise ValueError("靶区质心包含 NaN 或无穷值")
    distance = np.linalg.norm(base_xy[:, None, :] - scen_xy[None, :, :], axis=2)
    n_base, n_scenario = distance.shape
    infeasible = float(max_distance_m) * 1e6
    real_cost = np.where(distance <= float(max_distance_m), distance, infeasible)
    dummy_cost = np.full((n_base, n_base), float(max_distance_m) + 1e-6)
    cost = np.concatenate([real_cost, dummy_cost], axis=1)
    rows, cols = linear_sum_assignment(cost)
    records = []
    for row, col in zip(rows, cols):
        if col < n_scenario and distance[row, col] <= float(max_distance_m):
            records.append({
                "baseline_index": int(row),
                "scenario_index": int(col),
                "distance_m": float(distance[row, col]),
            })
    return pd.DataFrame(records, columns=columns)


def _scenario_observations(
    baseline: pd.DataFrame,
    scenario_frames: Dict[str, pd.DataFrame],
    max_distance_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations: List[dict] = []
    summaries: List[dict] = []
    n_base = len(baseline)
    for scenario_id, frame in scenario_frames.items():
        ranked = frame.copy().reset_index(drop=True)
        if not ranked.empty:
            ranked["scenario_rank"] = ranked["total_score"].rank(
                method="first", ascending=False
            ).astype(int)
        matches = match_target_tables(baseline, ranked, max_distance_m)
        match_lookup = {
            int(row.baseline_index): (int(row.scenario_index), float(row.distance_m))
            for row in matches.itertuples(index=False)
        }
        scenario_union = None
        stable_union = None
        if "geometry_wkt" in ranked.columns:
            from shapely import wkt
            from shapely.ops import unary_union

            geometries = [
                wkt.loads(value) for value in ranked["geometry_wkt"].dropna()
                if str(value).strip()
            ]
            stable_values = ranked.loc[
                ranked["evidence_status"].astype(str) == "internal_supported",
                "geometry_wkt",
            ].dropna()
            stable_geometries = [
                wkt.loads(value) for value in stable_values if str(value).strip()
            ]
            scenario_union = unary_union(geometries) if geometries else None
            stable_union = unary_union(stable_geometries) if stable_geometries else None
        summaries.append({
            "scenario_id": scenario_id,
            "target_count": int(len(ranked)),
            "stable_target_count": int(
                (ranked.get("evidence_status", pd.Series(dtype=str)) == "internal_supported").sum()
            ),
            "matched_baseline_count": int(len(matches)),
            "baseline_target_recall": float(len(matches) / n_base) if n_base else float("nan"),
            "scenario_target_precision": float(len(matches) / len(ranked)) if len(ranked) else float("nan"),
        })
        for baseline_index, base_row in baseline.reset_index(drop=True).iterrows():
            record = {
                "scenario_id": scenario_id,
                "baseline_target_id": str(base_row["target_id"]),
                "candidate_present": False,
                "internally_stable": False,
                "scenario_target_id": None,
                "total_score": float("nan"),
                "rank": float("nan"),
                "rank_percentile": float("nan"),
                "centroid_shift_m": float("nan"),
                "geometry_coverage": float("nan"),
                "stable_geometry_coverage": float("nan"),
                "geometry_present_25": False,
                "geometry_present_50": False,
                "stable_geometry_present_25": False,
            }
            if "geometry_wkt" in base_row and pd.notna(base_row["geometry_wkt"]):
                from shapely import wkt

                base_geometry = wkt.loads(str(base_row["geometry_wkt"]))
                base_area = float(base_geometry.area)
                if base_area > 0:
                    coverage = (
                        float(base_geometry.intersection(scenario_union).area / base_area)
                        if scenario_union is not None else 0.0
                    )
                    stable_coverage = (
                        float(base_geometry.intersection(stable_union).area / base_area)
                        if stable_union is not None else 0.0
                    )
                    record.update({
                        "geometry_coverage": coverage,
                        "stable_geometry_coverage": stable_coverage,
                        "geometry_present_25": coverage >= 0.25,
                        "geometry_present_50": coverage >= 0.50,
                        "stable_geometry_present_25": stable_coverage >= 0.25,
                    })
            if baseline_index in match_lookup:
                scenario_index, distance_m = match_lookup[baseline_index]
                row = ranked.iloc[scenario_index]
                rank = int(row["scenario_rank"])
                rank_pct = 1.0 if len(ranked) <= 1 else 1.0 - (rank - 1) / float(len(ranked) - 1)
                record.update({
                    "candidate_present": True,
                    "internally_stable": str(row["evidence_status"]) == "internal_supported",
                    "scenario_target_id": str(row["target_id"]),
                    "total_score": float(row["total_score"]),
                    "rank": rank,
                    "rank_percentile": float(rank_pct),
                    "centroid_shift_m": float(distance_m),
                })
            observations.append(record)
    return pd.DataFrame(observations), pd.DataFrame(summaries)


def aggregate_target_uncertainty(
    baseline: pd.DataFrame,
    observations: pd.DataFrame,
    robust_threshold: float = 0.80,
    conditional_threshold: float = 0.50,
) -> pd.DataFrame:
    """把逐场景观测汇总为每个基准靶区的出现频率和区间。"""
    if not 0.0 < float(conditional_threshold) <= float(robust_threshold) <= 1.0:
        raise ValueError("出现频率门槛必须满足 0 < conditional <= robust <= 1")
    output = []
    for _, base_row in baseline.reset_index(drop=True).iterrows():
        target_id = str(base_row["target_id"])
        part = observations[observations["baseline_target_id"] == target_id]
        present = part[part["candidate_present"].astype(bool)]
        centroid_occurrence = float(part["candidate_present"].mean()) if len(part) else 0.0
        stable_centroid_occurrence = float(part["internally_stable"].mean()) if len(part) else 0.0
        geometry_available = bool(
            "geometry_coverage" in part and part["geometry_coverage"].notna().any()
        )
        geometry_occurrence_25 = (
            float(part["geometry_present_25"].mean()) if geometry_available else float("nan")
        )
        geometry_occurrence_50 = (
            float(part["geometry_present_50"].mean()) if geometry_available else float("nan")
        )
        stable_geometry_occurrence_25 = (
            float(part["stable_geometry_present_25"].mean())
            if geometry_available else float("nan")
        )
        occurrence = geometry_occurrence_25 if geometry_available else centroid_occurrence
        stable_decision_occurrence = (
            stable_geometry_occurrence_25
            if geometry_available else stable_centroid_occurrence
        )

        def q(column: str, quantile: float) -> float:
            values = pd.to_numeric(present[column], errors="coerce").dropna()
            return float(values.quantile(quantile)) if len(values) else float("nan")

        def q_all(column: str, quantile: float) -> float:
            values = pd.to_numeric(part[column], errors="coerce").dropna()
            return float(values.quantile(quantile)) if len(values) else float("nan")

        if occurrence >= float(robust_threshold):
            robustness = "robust_occurrence"
        elif occurrence >= float(conditional_threshold):
            robustness = "conditional_occurrence"
        else:
            robustness = "fragile_occurrence"
        baseline_supported = str(base_row["evidence_status"]) == "internal_supported"
        if (
            baseline_supported
            and occurrence >= float(robust_threshold)
            and stable_decision_occurrence >= float(robust_threshold)
        ):
            decision_tier = "high_confidence_internal"
        elif baseline_supported and occurrence >= float(robust_threshold):
            decision_tier = "recurring_but_grade_sensitive"
        elif baseline_supported:
            decision_tier = "conditional_internal_candidate"
        elif occurrence >= float(robust_threshold):
            decision_tier = "recurring_unstable_candidate"
        else:
            decision_tier = "fragile_unstable_candidate"
        output.append({
            "baseline_target_id": target_id,
            "centroid_x": float(base_row["centroid_x"]),
            "centroid_y": float(base_row["centroid_y"]),
            "baseline_score": float(base_row["total_score"]),
            "baseline_evidence_status": str(base_row["evidence_status"]),
            "scenario_count": int(len(part)),
            "matched_scenario_count": int(len(present)),
            "candidate_occurrence_frequency": occurrence,
            "centroid_match_occurrence_frequency": centroid_occurrence,
            "geometry_occurrence_frequency_25": geometry_occurrence_25,
            "geometry_occurrence_frequency_50": geometry_occurrence_50,
            "internally_stable_occurrence_frequency": stable_centroid_occurrence,
            "internally_stable_geometry_occurrence_frequency_25": stable_geometry_occurrence_25,
            "geometry_coverage_mean": q_all("geometry_coverage", 0.50),
            "geometry_coverage_q05": q_all("geometry_coverage", 0.05),
            "score_q05": q("total_score", 0.05),
            "score_median": q("total_score", 0.50),
            "score_q95": q("total_score", 0.95),
            "rank_q05": q("rank", 0.05),
            "rank_median": q("rank", 0.50),
            "rank_q95": q("rank", 0.95),
            "rank_percentile_q05": q("rank_percentile", 0.05),
            "rank_percentile_median": q("rank_percentile", 0.50),
            "rank_percentile_q95": q("rank_percentile", 0.95),
            "centroid_shift_q50_m": q("centroid_shift_m", 0.50),
            "centroid_shift_q95_m": q("centroid_shift_m", 0.95),
            "robustness_tier": robustness,
            "decision_tier": decision_tier,
        })
    return pd.DataFrame(output)


def _plot_uncertainty(targets: pd.DataFrame, scenarios: pd.DataFrame, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    if targets.empty:
        axes[0].text(0.5, 0.5, "No baseline targets", ha="center", va="center")
        axes[0].set_axis_off()
    else:
        origin_x = float(targets["centroid_x"].min())
        origin_y = float(targets["centroid_y"].min())
        plot_x = (targets["centroid_x"] - origin_x) / 1000.0
        plot_y = (targets["centroid_y"] - origin_y) / 1000.0
        scatter = axes[0].scatter(
            plot_x, plot_y,
            c=targets["candidate_occurrence_frequency"],
            s=35 + 100 * targets["candidate_occurrence_frequency"],
            cmap="RdYlGn", vmin=0, vmax=1, edgecolors="#4F5B62", linewidths=0.35,
        )
        axes[0].set_title("Candidate occurrence frequency")
        axes[0].set_aspect("equal", adjustable="datalim")
        axes[0].set_xlabel(f"local x (km; origin {origin_x:.0f} m)")
        axes[0].set_ylabel(f"local y (km; origin {origin_y:.0f} m)")
        axes[0].grid(alpha=0.15)
        if "decision_tier" in targets:
            high = targets["decision_tier"] == "high_confidence_internal"
            conditional = targets["decision_tier"] == "conditional_internal_candidate"
            axes[0].scatter(
                plot_x[high], plot_y[high], marker="*", s=190,
                facecolors="none", edgecolors="#24323A", linewidths=1.1,
                label="high-confidence internal",
            )
            axes[0].scatter(
                plot_x[conditional], plot_y[conditional], marker="X", s=90,
                color="#A96D6D", edgecolors="#24323A", linewidths=0.5,
                label="conditional internal",
            )
            if high.any() or conditional.any():
                axes[0].legend(loc="best", fontsize=8)
        fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04)
    if scenarios.empty:
        axes[1].set_axis_off()
    else:
        plot = scenarios.dropna(subset=["target_count"]).sort_values(
            "target_count", ascending=True
        )
        baseline_values = plot.loc[plot["scenario_id"] == "baseline", "target_count"]
        baseline_count = float(baseline_values.iloc[0]) if len(baseline_values) else float(plot["target_count"].median())
        colors = [
            "#A96D6D" if abs(float(value) - baseline_count) >= 10
            else ("#A7A9A6" if str(name) == "baseline" else "#8FA58A")
            for name, value in zip(plot["scenario_id"], plot["target_count"])
        ]
        bars = axes[1].barh(plot["scenario_id"], plot["target_count"], color=colors)
        axes[1].bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
        axes[1].set_title("Target count by scenario")
        axes[1].set_xlabel("count")
        axes[1].grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#F3EEE8")
    plt.close(fig)
    return str(path)


def _config_signature(config: Dict) -> str:
    payload = {"grid": config.get("grid") or {}, "screening": config.get("screening") or {}}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _existing_baseline_is_compatible(run_dir: Path, config: Dict) -> bool:
    snapshot = run_dir / "config_snapshot.yaml"
    required = [run_dir / "result.json", run_dir / "manifest.json", run_dir / "candidate_targets.csv"]
    if not snapshot.is_file() or not all(path.is_file() for path in required):
        return False
    return _config_signature(load_config(str(snapshot))) == _config_signature(config)


def _load_period_metric_frames(run_dir: Path) -> Dict[str, pd.DataFrame]:
    frames = {}
    for period_name in load_all_periods():
        path = run_dir / "period_metrics" / f"{period_name}_nodes.csv"
        if not path.is_file():
            raise FileNotFoundError(f"缺少基准时期指标: {path}")
        frames[period_name] = _load_cached_period_frame(path)
    return frames


def _graphs_equivalent_for_period_metrics(
    baseline_config: Dict,
    scenario_config: Dict,
    period_gdfs: Dict,
) -> tuple[bool, Dict[str, dict]]:
    """精确比较节点、边及容量；仅边数相同不视为等价。"""
    details = {}
    base_grid = baseline_config.get("grid") or {}
    scen_grid = scenario_config.get("grid") or {}
    screening = baseline_config.get("screening") or {}
    transform = list(screening.get("distance_transforms") or ["inverse"])[0]
    edge_weight = str(screening.get("edge_weight_column", "NC_A"))
    all_equal = True
    for period_name, gdf in period_gdfs.items():
        graphs = []
        for grid in (base_grid, scen_grid):
            graphs.append(build_grid_graph(
                gdf,
                edge_weight_col=edge_weight,
                weight_mode="min",
                grid_step=float(grid.get("step_m", 3000.0)),
                edge_dist_tolerance=float(grid.get("edge_dist_tolerance_m", 150.0)),
                distance_transform=transform,
            ))
        left, right = graphs
        capacity_left = _capacity_attr(left)
        capacity_right = _capacity_attr(right)
        left_edges = {
            (min(int(u), int(v)), max(int(u), int(v))): float(data[capacity_left])
            for u, v, data in left.edges(data=True)
        }
        right_edges = {
            (min(int(u), int(v)), max(int(u), int(v))): float(data[capacity_right])
            for u, v, data in right.edges(data=True)
        }
        same_keys = left_edges.keys() == right_edges.keys()
        same_values = same_keys and all(
            np.isclose(left_edges[key], right_edges[key], rtol=0, atol=1e-12)
            for key in left_edges
        )
        equal = bool(
            left.nodes == right.nodes and same_keys and same_values
        )
        all_equal = all_equal and equal
        details[period_name] = {
            "equivalent": equal,
            "baseline_edges": int(left.number_of_edges()),
            "scenario_edges": int(right.number_of_edges()),
        }
    return all_equal, details


def _compute_alternate_distance_frames(
    config: Dict,
    baseline_frames: Dict[str, pd.DataFrame],
    period_gdfs: Dict,
    transforms: Sequence[str],
    progress_callback: ProgressCallback,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """每期构图一次，只计算需要的替代距离介数，再共享给多个 P2 场景。"""
    import networkx as nx

    screening = config.get("screening") or {}
    grid = config.get("grid") or {}
    edge_weight = str(screening.get("edge_weight_column", "NC_A"))
    top_fraction = float(screening.get("top_betweenness_fraction", 0.20))
    criticality_weights = screening.get("network_criticality_weights") or None
    output = {transform: {} for transform in transforms}
    for period_name, gdf in period_gdfs.items():
        graph = build_grid_graph(
            gdf,
            edge_weight_col=edge_weight,
            weight_mode="min",
            grid_step=float(grid.get("step_m", 3000.0)),
            edge_dist_tolerance=float(grid.get("edge_dist_tolerance_m", 150.0)),
            distance_transform=list(screening.get("distance_transforms") or ["inverse"])[0],
        )
        for transform in transforms:
            _emit(progress_callback, f"P2 距离共享计算 {period_name}: {transform}")
            apply_distance_transform(graph, transform)
            bc = {
                int(node): float(value)
                for node, value in nx.betweenness_centrality(
                    graph, weight="distance", normalized=True
                ).items()
            }
            top_nodes = _stable_top_nodes(bc, top_fraction)
            frame = baseline_frames[period_name].copy()
            frame["exact_betweenness"] = frame["node_idx"].map(bc).fillna(0.0)
            frame["is_top20"] = frame["node_idx"].astype(int).isin(top_nodes)
            output[transform][period_name] = add_period_scores(
                frame, criticality_weights=criticality_weights
            )
    return output


def _run_postmetric_scenario(
    period_frames: Dict[str, pd.DataFrame],
    config: Dict,
    output_dir: Path,
) -> tuple[pd.DataFrame, int]:
    """复用正式候选评分/聚类函数，避免重复时期中心性计算。"""
    screening = config.get("screening") or {}
    grid = config.get("grid") or {}
    matched = match_period_nodes(
        period_frames,
        tolerance_m=float(grid.get("centroid_match_tolerance_m", 1500.0)),
        score_weights=screening.get("score_weights") or None,
    )
    candidate_cells, contract = select_candidate_cells(
        matched,
        score_quantile=float(screening.get("candidate_cell_quantile", 0.80)),
        min_periods=int(screening.get("min_supporting_periods", 2)),
    )
    clustered, targets, _ = cluster_candidate_cells(
        candidate_cells,
        eps_m=float((screening.get("target_clustering") or {}).get("eps_m", 4500.0)),
        min_samples=int((screening.get("target_clustering") or {}).get("min_samples", 3)),
        max_diameter_m=float((screening.get("target_clustering") or {}).get("max_diameter_m", 18000.0)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _export_frame_csv(matched, output_dir / "matched_cells.csv")
    _export_frame_csv(clustered, output_dir / "candidate_cells.csv")
    _export_frame_csv(targets, output_dir / "candidate_targets.csv")
    minimal_result = {
        "status": "completed" if len(targets) else "completed_no_targets",
        "execution_mode": "postmetric_exact_reuse",
        "input_summary": {
            "candidate_cell_count": int(contract["candidate_cell_count"]),
            "candidate_target_count": int(len(targets)),
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(minimal_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return targets, int(contract["candidate_cell_count"])


def run_uncertainty_analysis(
    output_dir: str,
    config_path: str,
    baseline_screening_dir: Optional[str] = None,
    scenario_ids: Optional[Iterable[str]] = None,
    progress_callback: ProgressCallback = None,
) -> Dict:
    """执行 P2 预注册参数集合并保存可复核结果。"""
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs_dir = output / "configs"
    runs_dir = output / "scenario_runs"
    configs_dir.mkdir(exist_ok=True)
    runs_dir.mkdir(exist_ok=True)
    config = load_config(config_path)
    (output / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    all_scenarios = build_default_scenarios(config)
    requested = set(scenario_ids or [])
    scenarios = [s for s in all_scenarios if not requested or s.scenario_id in requested]
    unknown = requested - {s.scenario_id for s in all_scenarios}
    if unknown:
        raise ValueError(f"未知 P2 场景: {sorted(unknown)}")

    baseline_dir = Path(baseline_screening_dir).resolve() if baseline_screening_dir else runs_dir / "baseline"
    if not _existing_baseline_is_compatible(baseline_dir, config):
        _emit(progress_callback, "P2 baseline: 运行正式筛选")
        run_target_screening(str(baseline_dir), config_path, progress_callback=progress_callback)
    else:
        _emit(progress_callback, f"P2 baseline: 复用 {baseline_dir}")
    baseline = load_target_table(baseline_dir)
    baseline_result = json.loads((baseline_dir / "result.json").read_text(encoding="utf-8"))
    uncertainty_cfg = config.get("uncertainty") or {}
    match_radius = float(uncertainty_cfg.get("target_match_radius_m", 9000.0))

    scenario_frames: Dict[str, pd.DataFrame] = {"baseline": baseline}
    scenario_meta = {
        "baseline": {
            "scenario_id": "baseline", "label": "基准配置", "factor": "baseline",
            "direction": "baseline", "overrides": {}, "status": "completed",
            "period_metrics_reused": True, "output_dir": str(baseline_dir),
            "candidate_cell_count": int(baseline_result["input_summary"].get("candidate_cell_count", 0)),
        }
    }
    failures = []
    exclusions = []
    period_gdfs = None
    baseline_period_frames = None
    alternate_distance_frames = None
    for index, scenario in enumerate(scenarios, start=1):
        _emit(progress_callback, f"P2 {index}/{len(scenarios)}: {scenario.label}")
        scenario_config = apply_overrides(config, scenario.overrides)
        scenario_config_path = configs_dir / f"{scenario.scenario_id}.yaml"
        scenario_config_path.write_text(
            yaml.safe_dump(scenario_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        scenario_output = runs_dir / scenario.scenario_id
        can_reuse = (
            _period_metric_config_hash(scenario_config)
            == _period_metric_config_hash(config)
        )
        meta = asdict(scenario)
        meta.update({
            "period_metrics_reused": bool(can_reuse),
            "output_dir": str(scenario_output),
        })
        if scenario.factor == "grid_step_m":
            reason = (
                "现有输入是固定网格统计 CSV；仅修改 step_m 会破坏邻接几何。"
                "网格尺度敏感性必须从同坐标系原始断裂线重新网格化。"
            )
            scenario_output.mkdir(parents=True, exist_ok=True)
            exclusion = {
                "scenario_id": scenario.scenario_id,
                "status": "excluded_requires_raw_regridding",
                "reason": reason,
            }
            (scenario_output / "excluded.json").write_text(
                json.dumps(exclusion, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            meta.update(exclusion)
            exclusions.append(exclusion)
            scenario_meta[scenario.scenario_id] = meta
            continue
        if scenario.factor == "edge_tolerance_m":
            if period_gdfs is None:
                period_gdfs = load_all_periods()
            equivalent, details = _graphs_equivalent_for_period_metrics(
                config, scenario_config, period_gdfs
            )
            if equivalent:
                scenario_output.mkdir(parents=True, exist_ok=True)
                equivalence = {
                    "scenario_id": scenario.scenario_id,
                    "status": "equivalent_to_baseline",
                    "comparison": details,
                }
                (scenario_output / "equivalence.json").write_text(
                    json.dumps(equivalence, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                scenario_frames[scenario.scenario_id] = baseline.copy()
                meta.update({
                    "status": "equivalent_to_baseline",
                    "period_metrics_reused": True,
                    "candidate_cell_count": int(
                        baseline_result["input_summary"].get("candidate_cell_count", 0)
                    ),
                    "equivalence": details,
                })
                scenario_meta[scenario.scenario_id] = meta
                continue
        existing_targets = scenario_output / "candidate_targets.csv"
        existing_result = scenario_output / "result.json"
        if existing_targets.is_file() and existing_result.is_file():
            existing_payload = json.loads(existing_result.read_text(encoding="utf-8"))
            scenario_frames[scenario.scenario_id] = load_target_table(scenario_output)
            meta.update({
                "status": "reused_existing_scenario_output",
                "execution_mode": "existing_exact_output",
                "candidate_cell_count": int(
                    (existing_payload.get("input_summary") or {}).get(
                        "candidate_cell_count", 0
                    )
                ),
            })
            scenario_meta[scenario.scenario_id] = meta
            continue
        if scenario.factor == "distance_transform":
            if period_gdfs is None:
                period_gdfs = load_all_periods()
            if baseline_period_frames is None:
                baseline_period_frames = _load_period_metric_frames(baseline_dir)
            if alternate_distance_frames is None:
                needed = [
                    str(item.direction) for item in scenarios
                    if item.factor == "distance_transform"
                ]
                alternate_distance_frames = _compute_alternate_distance_frames(
                    config, baseline_period_frames, period_gdfs, needed, progress_callback
                )
            targets, candidate_count = _run_postmetric_scenario(
                alternate_distance_frames[str(scenario.direction)],
                scenario_config,
                scenario_output,
            )
            scenario_frames[scenario.scenario_id] = targets
            meta.update({
                "status": "completed",
                "period_metrics_reused": False,
                "execution_mode": "shared_exact_distance_metrics",
                "candidate_cell_count": int(candidate_count),
            })
            scenario_meta[scenario.scenario_id] = meta
            continue
        try:
            result = run_target_screening(
                str(scenario_output), str(scenario_config_path),
                progress_callback=progress_callback,
                reuse_period_metrics_from=str(baseline_dir) if can_reuse else None,
            )
            scenario_frames[scenario.scenario_id] = load_target_table(scenario_output)
            meta.update({
                "status": str(result["status"]),
                "candidate_cell_count": int(result["input_summary"].get("candidate_cell_count", 0)),
            })
        except Exception as exc:  # 逐场景失败需保留，其余场景继续。
            meta.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            failures.append({"scenario_id": scenario.scenario_id, "error": meta["error"]})
        scenario_meta[scenario.scenario_id] = meta

    observations, scenario_summary = _scenario_observations(
        baseline, scenario_frames, match_radius
    )
    meta_frame = pd.DataFrame(scenario_meta.values())
    scenario_summary = scenario_summary.merge(
        meta_frame.drop(columns=["overrides"], errors="ignore"), on="scenario_id", how="outer"
    )
    target_summary = aggregate_target_uncertainty(
        baseline,
        observations,
        robust_threshold=float(uncertainty_cfg.get("robust_occurrence_threshold", 0.80)),
        conditional_threshold=float(
            uncertainty_cfg.get("conditional_occurrence_threshold", 0.50)
        ),
    )

    observations.to_csv(output / "target_scenario_observations.csv", index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(output / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    target_summary.to_csv(output / "target_uncertainty.csv", index=False, encoding="utf-8-sig")
    figure_path = _plot_uncertainty(target_summary, scenario_summary, output / "maps" / "p2_uncertainty.png")

    tier_counts = (
        target_summary["robustness_tier"].value_counts().sort_index().to_dict()
        if not target_summary.empty else {}
    )
    decision_tier_counts = (
        target_summary["decision_tier"].value_counts().sort_index().to_dict()
        if not target_summary.empty else {}
    )
    status = "completed" if not failures else "partial"
    result = {
        "run_id": f"p2-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "status": status,
        "claim_scope": "internal_parameter_robustness_only",
        "baseline_screening_dir": str(baseline_dir),
        "target_match_radius_m": match_radius,
        "occurrence_definition": "baseline_geometry_coverage_at_least_25_percent",
        "secondary_occurrence_definition": "baseline_geometry_coverage_at_least_50_percent",
        "planned_scenario_count": int(len(scenarios) + 1),
        "statistical_scenario_count": int(len(scenario_frames)),
        "successful_scenario_count": int(len(scenario_frames)),
        "failed_scenarios": failures,
        "excluded_scenarios": exclusions,
        "baseline_target_count": int(len(baseline)),
        "robustness_tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "decision_tier_counts": {
            str(k): int(v) for k, v in decision_tier_counts.items()
        },
        "median_candidate_occurrence_frequency": (
            float(target_summary["candidate_occurrence_frequency"].median())
            if not target_summary.empty else float("nan")
        ),
        "limitations": [
            "P2 只评估内部参数稳健性，不构成油气发现概率验证。",
            "主出现频率按基准靶区面积覆盖计算；25% 阈值是项目工程门槛。",
            "质心一对一匹配仅用于分数、排名与漂移区间，分裂/合并时这些区间可能缺失。",
            "场景范围是项目预注册工程范围，不代表所有地质解释不确定性。",
        ],
        "artifact_paths": {
            "result_json": str(output / "result.json"),
            "target_uncertainty_csv": str(output / "target_uncertainty.csv"),
            "scenario_summary_csv": str(output / "scenario_summary.csv"),
            "observations_csv": str(output / "target_scenario_observations.csv"),
            "uncertainty_png": figure_path,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_path": str(Path(config_path).resolve()),
        "scenario_contract": [asdict(s) for s in scenarios],
        "successful_scenario_ids": list(scenario_frames),
        "status": status,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# P2 参数不确定性报告", "",
        f"- 状态：`{status}`",
        f"- 成功场景：{len(scenario_frames)}/{len(scenarios) + 1}",
        f"- 排除场景：{len(exclusions)}（固定网格 CSV 不能仅改步长）",
        f"- 基准候选靶区：{len(baseline)}",
        f"- 稳健性分层：`{tier_counts}`",
        f"- 决策分层：`{decision_tier_counts}`",
        f"- 候选出现频率中位数：{result['median_candidate_occurrence_frequency']:.3f}",
        "", "本结果只支持内部参数稳健性判断，不支持油气预测有效性主张。",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result
