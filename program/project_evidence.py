# -*- coding: utf-8 -*-
"""统一汇总项目证据与外部数据就绪状态。

本模块只读取已有产物和用户提供的数据，不重新计算筛选分数，也不把
PorePy 方法演示解释为塔里木候选区物理验证。GUI 和 CLI 共用这一真值源。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


PERIOD_KEYS = ("海西期", "喜山期", "印支燕山期")
RAW_NAME_ALIASES = {
    "海西期": ("海西", "海西期", "haixi"),
    "喜山期": ("喜山", "喜山期", "xishan", "himalayan"),
    "印支燕山期": ("印支燕山", "印支燕山期", "yinzhiyanshan", "indosinian_yanshanian"),
}
SUPPORTED_VECTOR_SUFFIXES = (".geojson", ".json", ".gpkg", ".shp")


def _crs_value_from_geojson(crs: Any) -> Any:
    if not isinstance(crs, dict):
        return crs
    properties = crs.get("properties")
    if isinstance(properties, dict):
        return properties.get("name") or properties.get("href") or crs
    return crs


def _is_projected_metric_crs(crs: Any) -> bool:
    """只接受适合米制距离/面积计算的投影坐标系。"""

    if crs is None:
        return False
    try:
        from pyproj import CRS

        parsed = CRS.from_user_input(crs)
        unit_names = {
            str(axis.unit_name).strip().lower()
            for axis in parsed.axis_info
            if axis.unit_name
        }
        return bool(parsed.is_projected) and any(
            unit in {"metre", "meter", "metres", "meters"} for unit in unit_names
        )
    except Exception:
        return False


def _repo_root(repo_root: Optional[Path | str] = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return data


def _discover_raw_sources(raw_dir: Path) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    if not raw_dir.is_dir():
        return discovered
    files = [p for p in raw_dir.rglob("*") if p.is_file()]
    for period, aliases in RAW_NAME_ALIASES.items():
        for path in files:
            stem = path.stem.lower().replace("-", "_").replace(" ", "_")
            if path.suffix.lower() not in SUPPORTED_VECTOR_SUFFIXES:
                continue
            if any(alias.lower().replace(" ", "_") in stem for alias in aliases):
                discovered[period] = path
                break
    return discovered


def _inspect_geojson(path: Path) -> Dict[str, Any]:
    data = _read_json(path)
    features = data.get("features", [])
    if not isinstance(features, list):
        features = []
    geometry_types = {
        str((feature.get("geometry") or {}).get("type", ""))
        for feature in features
        if isinstance(feature, dict)
    }
    geometry_types.discard("")
    invalid_types = sorted(geometry_types.difference({"LineString", "MultiLineString"}))
    crs = data.get("crs")
    crs_value = _crs_value_from_geojson(crs)
    crs_text = json.dumps(crs, ensure_ascii=False) if crs else None
    return {
        "feature_count": len(features),
        "geometry_types": sorted(geometry_types),
        "line_geometry_only": bool(features) and not invalid_types,
        "crs": crs_text,
        "crs_present": bool(crs_text),
        "crs_projected_metric": _is_projected_metric_crs(crs_value),
        "read_error": None,
    }


def _inspect_vector(path: Path) -> Dict[str, Any]:
    base = {
        "path": str(path),
        "exists": path.is_file(),
        "supported_format": path.suffix.lower() in SUPPORTED_VECTOR_SUFFIXES,
        "feature_count": 0,
        "geometry_types": [],
        "line_geometry_only": False,
        "crs": None,
        "crs_present": False,
        "crs_projected_metric": False,
        "read_error": None,
    }
    if not base["exists"] or not base["supported_format"]:
        return base
    try:
        if path.suffix.lower() in {".geojson", ".json"}:
            base.update(_inspect_geojson(path))
        else:
            import geopandas as gpd

            gdf = gpd.read_file(path)
            types = sorted({str(value) for value in gdf.geometry.geom_type.dropna().unique()})
            base.update(
                {
                    "feature_count": int(len(gdf)),
                    "geometry_types": types,
                    "line_geometry_only": bool(len(gdf))
                    and set(types).issubset({"LineString", "MultiLineString"}),
                    "crs": str(gdf.crs) if gdf.crs else None,
                    "crs_present": gdf.crs is not None,
                    "crs_projected_metric": _is_projected_metric_crs(gdf.crs),
                }
            )
    except Exception as exc:  # 数据检查必须返回可解释状态，而不是让 GUI 崩溃
        base["read_error"] = str(exc)
    return base


def assess_data_readiness(
    repo_root: Optional[Path | str] = None,
    raw_sources: Optional[Mapping[str, Path | str]] = None,
    raw_dir: Optional[Path | str] = None,
    physical_properties_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """检查当前数据能支撑的证据层级。

    ``source_regridding_ready`` 要求三期原始线均可读、只有线几何、使用一致的
    投影米制 CRS。``real_physics_validation_ready`` 还要求提供物性文件。
    """

    root = _repo_root(repo_root)
    program_dir = root / "program"
    processed_paths = {
        "海西期": program_dir / "海西.csv",
        "喜山期": program_dir / "喜山.csv",
        "印支燕山期": program_dir / "印支燕山.csv",
    }
    processed = {
        period: {"path": str(path), "exists": path.is_file()}
        for period, path in processed_paths.items()
    }

    candidate_raw_dir = Path(raw_dir) if raw_dir else program_dir / "data" / "raw_faults"
    supplied = {
        str(period): Path(path)
        for period, path in (raw_sources or {}).items()
        if path is not None and str(path).strip()
    }
    discovered = _discover_raw_sources(candidate_raw_dir)
    source_paths = {**discovered, **supplied}
    raw = {
        period: _inspect_vector(source_paths[period])
        if period in source_paths
        else {
            "path": None,
            "exists": False,
            "supported_format": False,
            "feature_count": 0,
            "geometry_types": [],
            "line_geometry_only": False,
            "crs": None,
            "crs_present": False,
            "crs_projected_metric": False,
            "read_error": None,
        }
        for period in PERIOD_KEYS
    }

    contract_valid_raw = [
        period
        for period, item in raw.items()
        if item["exists"]
        and item["supported_format"]
        and item["line_geometry_only"]
        and item["crs_present"]
        and not item["read_error"]
    ]
    metric_valid_raw = [
        period for period in contract_valid_raw if raw[period]["crs_projected_metric"]
    ]
    crs_values = {
        raw[period]["crs"] for period in contract_valid_raw if raw[period]["crs"]
    }
    common_crs = (
        len(contract_valid_raw) == len(PERIOD_KEYS) and len(crs_values) == 1
    )

    property_candidates = []
    if physical_properties_path:
        property_candidates.append(Path(physical_properties_path))
    property_candidates.extend(
        candidate_raw_dir / name
        for name in (
            "physical_properties.yaml",
            "physical_properties.yml",
            "physical_properties.json",
            "physical_properties.csv",
        )
    )
    property_path = next((p for p in property_candidates if p.is_file()), None)

    grid_ready = all(item["exists"] for item in processed.values())
    regrid_ready = len(metric_valid_raw) == len(PERIOD_KEYS) and common_crs
    physics_ready = regrid_ready and property_path is not None
    missing = []
    for period in PERIOD_KEYS:
        item = raw[period]
        if not item["exists"]:
            missing.append(f"{period}同坐标系原始断裂线")
        elif item["read_error"]:
            missing.append(f"{period}原始断裂线可读性")
        elif not item["line_geometry_only"]:
            missing.append(f"{period}线几何合同")
        elif not item["crs_present"]:
            missing.append(f"{period}CRS")
        elif not item["crs_projected_metric"]:
            missing.append(f"{period}投影米制CRS")
    if len(contract_valid_raw) == len(PERIOD_KEYS) and not common_crs:
        missing.append("三期统一CRS")
    if property_path is None:
        missing.append("实测或有来源依据的物性参数")

    return {
        "status": "ready_for_colocated_validation" if physics_ready else "external_data_pending",
        "raw_data_directory": str(candidate_raw_dir),
        "processed_grid_csv": processed,
        "raw_fault_sources": raw,
        "physical_properties_path": str(property_path) if property_path else None,
        "grid_screening_ready": grid_ready,
        "source_regridding_ready": regrid_ready,
        "real_physics_validation_ready": physics_ready,
        "common_crs": common_crs,
        "missing_items": missing,
    }


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if str(item).strip()))


def build_project_evidence(
    repo_root: Optional[Path | str] = None,
    raw_sources: Optional[Mapping[str, Path | str]] = None,
    raw_dir: Optional[Path | str] = None,
    physical_properties_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """从 P2/P3 固化产物和数据检查结果生成统一证据对象。"""

    root = _repo_root(repo_root)
    artifact_root = root / "artifacts" / "experiment" / "p2-p3-v1"
    p2_path = artifact_root / "p2" / "result.json"
    p3_path = artifact_root / "p3" / "result.json"
    p2 = _read_json(p2_path) if p2_path.is_file() else {}
    p3 = _read_json(p3_path) if p3_path.is_file() else {}
    readiness = assess_data_readiness(
        root,
        raw_sources=raw_sources,
        raw_dir=raw_dir,
        physical_properties_path=physical_properties_path,
    )

    p2_counts = p2.get("decision_tier_counts", {})
    high_confidence = int(p2_counts.get("high_confidence_internal", 0))
    p3_success = int(p3.get("successful_scenario_count", 0))
    p3_planned = int(p3.get("planned_scenario_count", 0))
    limitations = _unique(
        list(p2.get("limitations", []))
        + list(p3.get("limitations", []))
        + [
            "缺少同位原始断裂线时，网格尺度敏感性和候选区物理验证均保持待验证状态。"
        ]
    )
    figures = [
        artifact_root / "p2" / "maps" / "p2_uncertainty.png",
        artifact_root / "p3" / "maps" / "p3_scenario_response.png",
    ]
    existing_figures = [str(path) for path in figures if path.is_file()]
    return {
        "status": "evidence_baseline_ready" if p2 and p3 else "evidence_incomplete",
        "claim_scope": "internal_screening_and_method_pilot_only",
        "headline": (
            f"高置信内部候选 {high_confidence} 个 · "
            f"P3 方法情景 {p3_success}/{p3_planned} · "
            f"真实同位数据{'已就绪' if readiness['real_physics_validation_ready'] else '待补'}"
        ),
        "p2": {
            "available": bool(p2),
            "run_id": p2.get("run_id"),
            "statistical_scenario_count": int(p2.get("statistical_scenario_count", 0)),
            "baseline_target_count": int(p2.get("baseline_target_count", 0)),
            "high_confidence_internal_count": high_confidence,
            "decision_tier_counts": p2_counts,
            "excluded_scenario_count": len(p2.get("excluded_scenarios", [])),
        },
        "p3": {
            "available": bool(p3),
            "run_id": p3.get("run_id"),
            "successful_scenario_count": p3_success,
            "planned_scenario_count": p3_planned,
            "solver_residual_pass": bool(p3.get("solver_residual_pass", False)),
            "mapping_to_screening_targets": p3.get("mapping_to_screening_targets"),
            "data_role": (p3.get("geometry_summary") or {}).get("data_role"),
        },
        "data_readiness": readiness,
        "limitations": limitations,
        "figure_paths": existing_figures,
        "artifact_paths": {
            "p2_result": str(p2_path) if p2_path.is_file() else None,
            "p3_result": str(p3_path) if p3_path.is_file() else None,
            "summary": str(artifact_root / "summary.md"),
            "claim_validation": str(artifact_root / "claim_validation.md"),
        },
    }


def format_evidence_card(evidence: Mapping[str, Any], compact: bool = False) -> str:
    """把统一证据对象格式化为 GUI/终端可读中文。"""

    p2 = evidence.get("p2", {})
    p3 = evidence.get("p3", {})
    readiness = evidence.get("data_readiness", {})
    lines = [
        "【项目证据与数据状态】",
        str(evidence.get("headline", "证据状态未知")),
        "",
        (
            f"P2 参数稳健性：{p2.get('statistical_scenario_count', 0)} 个有效场景；"
            f"{p2.get('high_confidence_internal_count', 0)} 个高置信内部候选。"
        ),
        (
            f"P3 流动方法试验：{p3.get('successful_scenario_count', 0)}/"
            f"{p3.get('planned_scenario_count', 0)} 完成；"
            f"全局残差{'通过' if p3.get('solver_residual_pass') else '未通过或缺失'}。"
        ),
        "证据边界：P3 使用 KB11 方法演示几何，尚未映射到塔里木候选区。",
        "",
        f"三期网格筛选：{'就绪' if readiness.get('grid_screening_ready') else '未就绪'}",
        f"原始断裂线重网格化：{'就绪' if readiness.get('source_regridding_ready') else '待数据'}",
        f"真实同位物理验证：{'就绪' if readiness.get('real_physics_validation_ready') else '待数据'}",
    ]
    missing = list(readiness.get("missing_items", []))
    if missing:
        shown = missing[:3] if compact else missing
        lines.extend(["", "下一步需要："])
        lines.extend(f"- {item}" for item in shown)
        if compact and len(missing) > len(shown):
            lines.append(f"- 其余 {len(missing) - len(shown)} 项请点击“证据与数据状态”查看")
    if not compact:
        lines.extend(["", "当前结论：可用于内部辅助筛选与方法演示，不构成油气发现概率验证。"])
    return "\n".join(lines)
