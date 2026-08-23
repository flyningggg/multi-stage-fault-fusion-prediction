from __future__ import annotations

from typing import Dict, List, Any


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float))


def validate_config(cfg: Dict[str, Any]) -> List[str]:
    """
    返回配置错误列表（空列表表示通过）。
    只做关键约束，避免运行中途才报错。
    """
    errors: List[str] = []
    if not isinstance(cfg, dict):
        return ["config.yaml 顶层必须是映射（dict）。"]

    train = cfg.get("train") or {}
    clustering = cfg.get("clustering") or {}
    export_grid = cfg.get("export_grid") or {}
    screening = cfg.get("screening") or {}

    if not _is_num(export_grid.get("cell_width", 0)) or float(export_grid.get("cell_width", 0)) <= 0:
        errors.append("export_grid.cell_width 必须为正数。")

    if not isinstance(clustering.get("n_clusters", 0), int) or int(clustering.get("n_clusters", 0)) < 2:
        errors.append("clustering.n_clusters 必须为整数且 >= 2。")

    k_min = clustering.get("k_search_min", 2)
    k_max = clustering.get("k_search_max", 12)
    if not isinstance(k_min, int) or not isinstance(k_max, int) or k_min < 2 or k_max <= k_min:
        errors.append("clustering.k_search_min / k_search_max 配置无效（要求 k_max > k_min >= 2）。")

    ts = float(train.get("test_size", 0.1)) if _is_num(train.get("test_size", 0.1)) else -1
    if ts <= 0 or ts >= 0.5:
        errors.append("train.test_size 建议在 (0, 0.5) 范围内。")

    alpha = float(train.get("conformal_alpha", 0.1)) if _is_num(train.get("conformal_alpha", 0.1)) else -1
    if alpha <= 0 or alpha >= 1:
        errors.append("train.conformal_alpha 必须在 (0, 1) 区间。")

    blocks = train.get("spatial_cv_blocks", 9)
    if not isinstance(blocks, int) or blocks < 4:
        errors.append("train.spatial_cv_blocks 建议为整数且 >= 4。")

    seeds = train.get("stability_seeds")
    if seeds is not None:
        if not isinstance(seeds, list) or not all(isinstance(s, int) for s in seeds):
            errors.append("train.stability_seeds 必须为整数列表。")

    target = train.get("target_column")
    if target is not None and not isinstance(target, str):
        errors.append("train.target_column 必须为字符串。")

    if screening:
        score_weights = screening.get("score_weights") or {}
        required_weights = {
            "network_criticality", "removal_impact",
            "period_persistence", "parameter_stability",
        }
        if set(score_weights) != required_weights:
            errors.append(f"screening.score_weights 必须包含且仅包含 {sorted(required_weights)}。")
        elif not all(_is_num(value) and float(value) >= 0 for value in score_weights.values()):
            errors.append("screening.score_weights 必须为非负数。")
        elif abs(sum(float(value) for value in score_weights.values()) - 1.0) > 1e-8:
            errors.append("screening.score_weights 之和必须为1。")

        criticality_weights = screening.get("network_criticality_weights") or {}
        required_criticality = {"betweenness", "pagerank"}
        if set(criticality_weights) != required_criticality:
            errors.append(
                "screening.network_criticality_weights 必须包含且仅包含 "
                f"{sorted(required_criticality)}。"
            )
        elif not all(_is_num(value) and float(value) >= 0 for value in criticality_weights.values()):
            errors.append("screening.network_criticality_weights 必须为非负数。")
        elif abs(sum(float(value) for value in criticality_weights.values()) - 1.0) > 1e-8:
            errors.append("screening.network_criticality_weights 之和必须为1。")

        transforms = screening.get("distance_transforms") or []
        allowed_transforms = {"inverse", "inverse_sqrt", "neglog"}
        if not isinstance(transforms, list) or not transforms:
            errors.append("screening.distance_transforms 必须为非空列表。")
        elif not all(isinstance(value, str) and value in allowed_transforms for value in transforms):
            errors.append(f"screening.distance_transforms 仅支持 {sorted(allowed_transforms)}。")

        top_fraction = screening.get("top_betweenness_fraction", 0.20)
        if not _is_num(top_fraction) or not 0 < float(top_fraction) <= 1:
            errors.append("screening.top_betweenness_fraction 必须在(0,1]内。")
        candidate_quantile = screening.get("candidate_cell_quantile", 0.80)
        if not _is_num(candidate_quantile) or not 0 < float(candidate_quantile) < 1:
            errors.append("screening.candidate_cell_quantile 必须在(0,1)内。")
        min_periods = screening.get("min_supporting_periods", 2)
        if not isinstance(min_periods, int) or min_periods < 2:
            errors.append("screening.min_supporting_periods 必须为整数且 >= 2。")

        target_clustering = screening.get("target_clustering") or {}
        eps_m = target_clustering.get("eps_m", 4500.0)
        min_samples = target_clustering.get("min_samples", 3)
        max_diameter_m = target_clustering.get("max_diameter_m", 18000.0)
        if not _is_num(eps_m) or float(eps_m) <= 0:
            errors.append("screening.target_clustering.eps_m 必须为正数。")
        if not isinstance(min_samples, int) or min_samples < 1:
            errors.append("screening.target_clustering.min_samples 必须为整数且 >= 1。")
        if not _is_num(max_diameter_m) or float(max_diameter_m) <= 0:
            errors.append("screening.target_clustering.max_diameter_m 必须为正数。")

    return errors

