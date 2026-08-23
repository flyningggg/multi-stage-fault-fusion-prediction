# -*- coding: utf-8 -*-
"""透明的多期候选单元评分、空间聚合、分级与证据卡。"""
from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from screening_contracts import CandidateTargetRecord


DEFAULT_SCORE_WEIGHTS = {
    "network_criticality": 0.35,
    "removal_impact": 0.25,
    "period_persistence": 0.25,
    "parameter_stability": 0.15,
}
DEFAULT_CRITICALITY_WEIGHTS = {"betweenness": 0.70, "pagerank": 0.30}


def validate_weights(weights: Dict[str, float], required: Iterable[str]) -> None:
    required_set = set(required)
    if set(weights) != required_set:
        raise ValueError(f"权重键必须为 {sorted(required_set)}，实际为 {sorted(weights)}")
    values = np.asarray([weights[key] for key in required_set], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("权重必须为有限非负数")
    if not np.isclose(float(values.sum()), 1.0, atol=1e-8):
        raise ValueError(f"权重之和必须为1，实际为 {values.sum():.8f}")


def percentile_rank(values) -> np.ndarray:
    """返回[0,1]百分位；常数列无区分力，统一记为0。"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy()
    if not np.isfinite(arr).all():
        raise ValueError("百分位输入包含NaN或无穷值")
    if float(np.max(arr) - np.min(arr)) <= 1e-15:
        return np.zeros(arr.shape, dtype=float)
    return pd.Series(arr).rank(method="average", pct=True).to_numpy(dtype=float)


def add_period_scores(
    frame: pd.DataFrame,
    criticality_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """给单期节点表添加可解释百分位与网络关键性分项。"""
    weights = dict(criticality_weights or DEFAULT_CRITICALITY_WEIGHTS)
    validate_weights(weights, DEFAULT_CRITICALITY_WEIGHTS)
    required = {"exact_betweenness", "pagerank", "removal_impact"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"单期节点表缺少列: {sorted(missing)}")

    out = frame.copy()
    out["betweenness_percentile"] = percentile_rank(out["exact_betweenness"])
    out["pagerank_percentile"] = percentile_rank(out["pagerank"])
    out["removal_impact_percentile"] = percentile_rank(out["removal_impact"])
    out["network_criticality"] = (
        weights["betweenness"] * out["betweenness_percentile"]
        + weights["pagerank"] * out["pagerank_percentile"]
    )
    return out


def match_period_nodes(
    period_frames: Dict[str, pd.DataFrame],
    tolerance_m: float,
    score_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """按质心确定性匹配两期/三期节点，并计算透明综合分。

    采用带约束的一对一凝聚匹配：每个匹配组中同一时期至多一个节点，且组内
    任意两点距离均不超过容差。这样可避免DBSCAN经相邻点链式扩张后把大片网格
    错并为一个多期单元。
    """
    from scipy.spatial import cKDTree

    if tolerance_m <= 0:
        raise ValueError("匹配容差必须为正数")
    weights = dict(score_weights or DEFAULT_SCORE_WEIGHTS)
    validate_weights(weights, DEFAULT_SCORE_WEIGHTS)
    if len(period_frames) < 2:
        raise ValueError("多期匹配至少需要两个时期")

    rows: List[pd.DataFrame] = []
    required = {
        "period", "node_idx", "pos_x", "pos_y", "network_criticality",
        "removal_impact_percentile", "stability_rate", "is_top20",
    }
    for period, frame in period_frames.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{period} 节点表缺少列: {sorted(missing)}")
        part = frame.copy()
        part["period"] = str(period)
        rows.append(part)
    combined = pd.concat(rows, ignore_index=True)
    coords = combined[["pos_x", "pos_y"]].to_numpy(dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("节点坐标包含NaN或无穷值")

    n_rows = len(combined)
    parent = np.arange(n_rows, dtype=int)
    members = [{index} for index in range(n_rows)]
    component_periods = [{str(combined.iloc[index]["period"])} for index in range(n_rows)]

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)

    period_names = sorted(combined["period"].astype(str).unique())
    period_indices = {
        name: combined.index[combined["period"].astype(str) == name].to_numpy(dtype=int)
        for name in period_names
    }
    candidate_pairs = []
    for left_pos, left_name in enumerate(period_names):
        left_indices = period_indices[left_name]
        left_tree = cKDTree(coords[left_indices])
        for right_name in period_names[left_pos + 1:]:
            right_indices = period_indices[right_name]
            right_tree = cKDTree(coords[right_indices])
            neighbours = left_tree.query_ball_tree(right_tree, r=float(tolerance_m))
            for left_local, right_locals in enumerate(neighbours):
                left_index = int(left_indices[left_local])
                for right_local in right_locals:
                    right_index = int(right_indices[right_local])
                    distance = float(np.linalg.norm(coords[left_index] - coords[right_index]))
                    candidate_pairs.append((distance, left_index, right_index))

    # 距离优先、行号打破并列，保证同一输入得到稳定的一对一匹配。
    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    for _, left_index, right_index in candidate_pairs:
        left_root = _find(left_index)
        right_root = _find(right_index)
        if left_root == right_root:
            continue
        if component_periods[left_root] & component_periods[right_root]:
            continue
        merged_members = members[left_root] | members[right_root]
        merged_coords = coords[sorted(merged_members)]
        pairwise = merged_coords[:, None, :] - merged_coords[None, :, :]
        max_distance = float(np.sqrt(np.sum(pairwise * pairwise, axis=2)).max())
        if max_distance > float(tolerance_m) + 1e-9:
            continue
        keep_root, drop_root = sorted((left_root, right_root))
        parent[drop_root] = keep_root
        members[keep_root] = merged_members
        component_periods[keep_root] |= component_periods[drop_root]

    roots = [_find(index) for index in range(n_rows)]
    root_to_cluster = {
        root: cluster_id for cluster_id, root in enumerate(sorted(set(roots)))
    }
    combined["_cluster"] = [root_to_cluster[root] for root in roots]

    groups: List[dict] = []
    total_periods = len(period_frames)
    for _, raw_group in combined.groupby("_cluster", sort=False):
        cx = float(raw_group["pos_x"].mean())
        cy = float(raw_group["pos_y"].mean())
        selected_rows = []
        ambiguous_count = 0
        for _, same_period in raw_group.groupby("period", sort=True):
            if len(same_period) > 1:  # 防御性保护；约束匹配正常不会进入此分支。
                ambiguous_count += int(len(same_period) - 1)
            dist2 = (same_period["pos_x"] - cx) ** 2 + (same_period["pos_y"] - cy) ** 2
            selected_rows.append(same_period.loc[dist2.idxmin()])
        selected = pd.DataFrame(selected_rows)
        periods = sorted(selected["period"].astype(str).tolist())
        period_count = len(periods)
        persistence = (
            (period_count - 1) / float(total_periods - 1)
            if total_periods > 1 else 0.0
        )
        network_score = float(selected["network_criticality"].mean())
        removal_score = float(selected["removal_impact_percentile"].mean())
        stability = float(selected["stability_rate"].mean())
        total_score = (
            weights["network_criticality"] * network_score
            + weights["removal_impact"] * removal_score
            + weights["period_persistence"] * persistence
            + weights["parameter_stability"] * stability
        )
        node_ids = {
            str(row["period"]): int(row["node_idx"])
            for _, row in selected.iterrows()
        }
        geometry = selected.iloc[0].get("geometry", None)
        groups.append({
            "centroid_x": float(selected["pos_x"].mean()),
            "centroid_y": float(selected["pos_y"].mean()),
            "period_count": int(period_count),
            "supporting_periods": "|".join(periods),
            "node_ids_json": json.dumps(node_ids, ensure_ascii=False, sort_keys=True),
            "network_criticality": network_score,
            "removal_impact": removal_score,
            "period_persistence": float(persistence),
            "parameter_stability": stability,
            "total_score": float(total_score),
            "any_top20": bool(selected["is_top20"].astype(bool).any()),
            "ambiguous_match_count": int(ambiguous_count),
            "geometry": geometry,
        })

    groups.sort(key=lambda row: (-row["centroid_y"], row["centroid_x"]))
    for index, row in enumerate(groups, start=1):
        row["match_id"] = f"M{index:05d}"
    return pd.DataFrame(groups)


def select_candidate_cells(
    matched: pd.DataFrame,
    score_quantile: float = 0.80,
    min_periods: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not 0.0 < score_quantile < 1.0:
        raise ValueError("候选分位阈值必须在(0,1)内")
    eligible = matched[matched["period_count"] >= int(min_periods)].copy()
    if eligible.empty:
        return eligible, {
            "score_threshold": float("nan"),
            "eligible_count": 0,
            "candidate_cell_count": 0,
        }
    threshold = float(eligible["total_score"].quantile(score_quantile))
    candidates = eligible[
        (eligible["total_score"] >= threshold) & eligible["any_top20"].astype(bool)
    ].copy()
    return candidates, {
        "score_threshold": threshold,
        "eligible_count": int(len(eligible)),
        "candidate_cell_count": int(len(candidates)),
    }


def cluster_candidate_cells(
    candidate_cells: pd.DataFrame,
    eps_m: float = 4500.0,
    min_samples: int = 3,
    max_diameter_m: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    """将候选单元聚合为稳定编号靶区，并生成证据卡。

    DBSCAN先识别连续区；若连续区的空间跨度过大，再用完全链接聚类按最大直径
    拆分，避免相邻网格链式扩张成一个失去决策意义的超大靶区。
    """
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
    from shapely.ops import unary_union

    if candidate_cells.empty:
        empty_targets = pd.DataFrame(columns=[
            "target_id", "target_level", "centroid_x", "centroid_y", "area",
            "total_score", "network_criticality", "removal_impact",
            "period_persistence", "parameter_stability", "supporting_periods",
            "cell_count", "evidence_status", "recommendation_reason", "geometry",
        ])
        return candidate_cells.copy(), empty_targets, []
    coords = candidate_cells[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    labels = DBSCAN(eps=float(eps_m), min_samples=int(min_samples)).fit_predict(coords)
    if max_diameter_m is not None:
        if float(max_diameter_m) <= 0:
            raise ValueError("靶区最大直径必须为正数")
        bounded_labels = np.full(len(labels), -1, dtype=int)
        next_label = 0
        for base_label in sorted(set(labels) - {-1}):
            indices = np.flatnonzero(labels == base_label)
            part_coords = coords[indices]
            deltas = part_coords[:, None, :] - part_coords[None, :, :]
            diameter = float(np.sqrt(np.sum(deltas * deltas, axis=2)).max())
            if diameter <= float(max_diameter_m) + 1e-9:
                sublabels = np.zeros(len(indices), dtype=int)
            else:
                sublabels = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=float(max_diameter_m),
                    linkage="complete",
                    compute_full_tree=True,
                ).fit_predict(part_coords)
            subgroups = []
            for sublabel in sorted(set(sublabels)):
                subgroup = indices[sublabels == sublabel]
                if len(subgroup) >= int(min_samples):
                    center = coords[subgroup].mean(axis=0)
                    subgroups.append((float(center[1]), float(center[0]), subgroup))
            for _, _, subgroup in sorted(subgroups, key=lambda item: (-item[0], item[1])):
                bounded_labels[subgroup] = next_label
                next_label += 1
        labels = bounded_labels
    cells = candidate_cells.copy()
    cells["target_cluster"] = labels

    summaries: List[dict] = []
    for cluster_id in sorted(set(labels) - {-1}):
        part = cells[cells["target_cluster"] == cluster_id]
        period_names = sorted({
            period
            for value in part["supporting_periods"]
            for period in str(value).split("|") if period
        })
        geometries = [geom for geom in part.get("geometry", []) if geom is not None]
        geometry = unary_union(geometries) if geometries else None
        summaries.append({
            "_cluster": int(cluster_id),
            "centroid_x": float(part["centroid_x"].mean()),
            "centroid_y": float(part["centroid_y"].mean()),
            "area": float(getattr(geometry, "area", 0.0)) if geometry is not None else 0.0,
            "total_score": float(part["total_score"].mean()),
            "peak_score": float(part["total_score"].max()),
            "network_criticality": float(part["network_criticality"].mean()),
            "removal_impact": float(part["removal_impact"].mean()),
            "period_persistence": float(part["period_persistence"].mean()),
            "parameter_stability": float(part["parameter_stability"].mean()),
            "supporting_periods": "|".join(period_names),
            "cell_count": int(len(part)),
            "geometry": geometry,
        })
    if not summaries:
        return cells, pd.DataFrame(), []

    summaries.sort(key=lambda row: (-row["centroid_y"], row["centroid_x"]))
    scores = np.asarray([row["total_score"] for row in summaries], dtype=float)
    q90 = float(np.quantile(scores, 0.90))
    q70 = float(np.quantile(scores, 0.70))
    target_rows = []
    evidence_cards = []
    for index, row in enumerate(summaries, start=1):
        target_id = f"T{index:03d}"
        is_stable = row["parameter_stability"] >= 0.80
        if not is_stable:
            level = "不稳定候选"
        elif row["total_score"] >= q90:
            level = "一级"
        elif row["total_score"] >= q70:
            level = "二级"
        else:
            level = "三级"
        evidence_status = "internal_supported" if is_stable else "internal_partial"
        reason = (
            f"由{row['cell_count']}个高值网格组成；支持时期为"
            f"{row['supporting_periods']}；参数稳定率{row['parameter_stability']:.2f}。"
        )
        record = CandidateTargetRecord(
            target_id=target_id,
            target_level=level,
            centroid_x=row["centroid_x"],
            centroid_y=row["centroid_y"],
            area=row["area"],
            total_score=row["total_score"],
            network_criticality=row["network_criticality"],
            removal_impact=row["removal_impact"],
            period_persistence=row["period_persistence"],
            parameter_stability=row["parameter_stability"],
            supporting_periods=row["supporting_periods"].split("|"),
            cell_count=row["cell_count"],
            evidence_status=evidence_status,
            recommendation_reason=reason,
            limitations=["尚未接入井位、储层或专家盲评外部证据"],
        )
        exported = record.to_dict()
        exported["supporting_periods"] = "|".join(exported["supporting_periods"])
        exported["geometry"] = row["geometry"]
        target_rows.append(exported)
        card = record.to_dict()
        card["external_validation_status"] = "not_validated"
        evidence_cards.append(card)
        cells.loc[cells["target_cluster"] == row["_cluster"], "target_id"] = target_id

    return cells, pd.DataFrame(target_rows), evidence_cards
