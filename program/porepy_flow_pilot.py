# -*- coding: utf-8 -*-
"""P3 PorePy 二维单相流方法试验。

该模块刻意与候选区评分解耦。仓库当前缺少与三期网格同坐标系的原始断裂线
及实测物性，因此 KB11 只用于验证几何转换、网格化和求解链路。
"""
import importlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, shape
from shapely.ops import unary_union

from utils.config_loader import load_config


ProgressCallback = Optional[Callable[[str], None]]


def _emit(callback: ProgressCallback, message: str) -> None:
    if callback is not None:
        callback(str(message))


def porepy_package_status() -> Dict[str, object]:
    try:
        pp = importlib.import_module("porepy")
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "version": str(getattr(pp, "__version__", "unknown")),
        "error": None,
    }


def lines_to_segments(
    geometries: Iterable,
    simplify_tolerance_m: float,
    minimum_segment_length_m: float,
) -> List[List[float]]:
    """把折线简化并拆成唯一的直线段，返回 [x1, y1, x2, y2]。"""
    if simplify_tolerance_m < 0 or minimum_segment_length_m <= 0:
        raise ValueError("简化容差必须非负，最小线段长度必须为正")
    segments: List[List[float]] = []
    seen = set()
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        simplified = geometry.simplify(float(simplify_tolerance_m), preserve_topology=True)
        parts = list(simplified.geoms) if isinstance(simplified, MultiLineString) else [simplified]
        for part in parts:
            if not isinstance(part, LineString):
                continue
            coords = list(part.coords)
            for left, right in zip(coords[:-1], coords[1:]):
                x1, y1 = float(left[0]), float(left[1])
                x2, y2 = float(right[0]), float(right[1])
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length < float(minimum_segment_length_m):
                    continue
                a = (round(x1, 8), round(y1, 8))
                b = (round(x2, 8), round(y2, 8))
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                seen.add(key)
                segments.append([x1, y1, x2, y2])
    return segments


def _read_geojson(path: Path) -> tuple[List, Optional[str]]:
    """轻量读取 GeoJSON，避免 PorePy 隔离环境依赖 GDAL/Fiona。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection":
        raise ValueError(f"仅支持 GeoJSON FeatureCollection: {path}")
    geometries = [
        shape(feature["geometry"])
        for feature in payload.get("features", [])
        if feature.get("geometry") is not None
    ]
    crs = payload.get("crs") or {}
    crs_name = (crs.get("properties") or {}).get("name")
    return geometries, str(crs_name) if crs_name else None


def _crs_is_geographic(crs_name: Optional[str]) -> bool:
    if crs_name is None:
        return False
    normalized = crs_name.upper()
    return "EPSG::4326" in normalized or normalized.endswith("EPSG:4326")


def prepare_trace_geometry(
    traces_path: str,
    area_path: str,
    top_trace_count: int = 12,
    simplify_tolerance_m: float = 0.5,
    minimum_segment_length_m: float = 0.2,
    domain_padding_m: float = 1.0,
) -> Dict[str, object]:
    if int(top_trace_count) < 1:
        raise ValueError("top_trace_count 必须至少为1")
    if float(domain_padding_m) < 0:
        raise ValueError("domain_padding_m 不能为负")
    traces_file = Path(traces_path).resolve()
    area_file = Path(area_path).resolve()
    if not traces_file.is_file() or not area_file.is_file():
        raise FileNotFoundError(f"P3 几何文件不存在: {traces_file}, {area_file}")
    trace_geometries, trace_crs = _read_geojson(traces_file)
    area_geometries, area_crs = _read_geojson(area_file)
    if trace_crs != area_crs:
        raise ValueError(f"裂缝线与区域 CRS 不一致: {trace_crs} vs {area_crs}")
    if _crs_is_geographic(trace_crs):
        raise ValueError("P3 需要米制投影坐标，不能直接使用经纬度")
    polygon = unary_union(area_geometries)
    clipped_geometries = []
    for geometry in trace_geometries:
        if geometry is None or geometry.is_empty:
            continue
        clipped = geometry.intersection(polygon)
        if clipped.is_empty:
            continue
        if isinstance(clipped, MultiLineString):
            clipped_geometries.extend([part for part in clipped.geoms if part.length > 0])
        elif isinstance(clipped, LineString) and clipped.length > 0:
            clipped_geometries.append(clipped)
    clipped_geometries.sort(key=lambda geometry: (-float(geometry.length), geometry.wkt))
    selected = clipped_geometries[: int(top_trace_count)]
    segments = lines_to_segments(
        selected,
        simplify_tolerance_m=float(simplify_tolerance_m),
        minimum_segment_length_m=float(minimum_segment_length_m),
    )
    if not segments:
        raise ValueError("P3 几何清洗后没有可用线段")
    xmin, ymin, xmax, ymax = map(float, polygon.bounds)
    padding = float(domain_padding_m)
    origin_x, origin_y = xmin - padding, ymin - padding
    normalized = [
        [x1 - origin_x, y1 - origin_y, x2 - origin_x, y2 - origin_y]
        for x1, y1, x2, y2 in segments
    ]
    return {
        "source_traces_path": str(traces_file),
        "source_area_path": str(area_file),
        "source_crs": trace_crs,
        "source_trace_count": int(len(trace_geometries)),
        "clipped_trace_count": int(len(clipped_geometries)),
        "selected_trace_count": int(len(selected)),
        "fracture_segment_count": int(len(normalized)),
        "simplify_tolerance_m": float(simplify_tolerance_m),
        "minimum_segment_length_m": float(minimum_segment_length_m),
        "origin_xy": [origin_x, origin_y],
        "domain_extent_m": [float(xmax - xmin + 2 * padding), float(ymax - ymin + 2 * padding)],
        "segments_m": normalized,
        "data_role": "method_demonstration_geometry_not_tarim_validation",
    }


def _build_porepy_model(pp):
    from dataclasses import dataclass as dc
    from typing import ClassVar

    from porepy.compositional.materials import SolidConstants
    from porepy.models.constitutive_laws import DimensionDependentPermeability

    @dc(kw_only=True)
    class FractureSolidConstants(SolidConstants):
        SI_units: ClassVar[dict[str, str]] = dict(**SolidConstants.SI_units)
        SI_units.update({"fracture_permeability": "m^2"})
        fracture_permeability: pp.number = 1.0

    class Geometry(pp.PorePyModel):
        def set_domain(self) -> None:
            width, height = [float(value) for value in self.params["domain_extent_m"]]
            self._domain = pp.Domain({
                "xmin": 0.0,
                "xmax": self.units.convert_units(width, "m"),
                "ymin": 0.0,
                "ymax": self.units.convert_units(height, "m"),
            })

        def set_fractures(self) -> None:
            fractures = []
            for x1, y1, x2, y2 in self.params["fracture_segments_m"]:
                points = np.asarray([[x1, x2], [y1, y2]], dtype=float)
                fractures.append(pp.LineFracture(self.units.convert_units(points, "m")))
            self._fractures = fractures

        def grid_type(self) -> str:
            return "simplex"

        def meshing_arguments(self) -> Dict[str, float]:
            cell_size = self.units.convert_units(float(self.params["cell_size_m"]), "m")
            return {
                "cell_size": cell_size,
                "cell_size_fracture": cell_size,
                "cell_size_boundary": cell_size,
                "cell_size_min": cell_size * 0.35,
            }

    class BoundaryConditions(pp.PorePyModel):
        def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
            sides = self.domain_boundary_sides(sd)
            if self.params["flow_direction"] == "x":
                faces = sides.west | sides.east
            else:
                faces = sides.south | sides.north
            return pp.BoundaryCondition(sd, faces, "dir")

        def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
            sides = self.domain_boundary_sides(bg)
            values = np.zeros(bg.num_cells)
            pressure_drop = self.units.convert_units(
                float(self.params["pressure_drop_pa"]), "Pa"
            )
            if self.params["flow_direction"] == "x":
                values[sides.west] = pressure_drop
                values[sides.east] = 0.0
            else:
                values[sides.south] = pressure_drop
                values[sides.north] = 0.0
            return values

    class Permeability(DimensionDependentPermeability):
        solid: FractureSolidConstants

        def fracture_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            size = sum(sd.num_cells for sd in subdomains)
            permeability = pp.wrap_as_dense_ad_array(
                self.solid.fracture_permeability,
                size,
                name="fracture permeability",
            )
            return self.isotropic_second_order_tensor(subdomains, permeability)

    class PilotModel(Geometry, BoundaryConditions, Permeability, pp.SinglePhaseFlow):
        pass

    return PilotModel, FractureSolidConstants


def _plot_pressure(model, pressure: np.ndarray, geometry: Dict, out_path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = model.mdg.subdomains(dim=2)[0]
    centers = matrix.cell_centers
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    scatter = ax.scatter(
        centers[0], centers[1], c=pressure, s=10, cmap="viridis", linewidths=0
    )
    for x1, y1, x2, y2 in geometry["segments_m"]:
        ax.plot([x1, x2], [y1, y2], color="#A96D6D", linewidth=0.8, alpha=0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("PorePy matrix pressure")
    ax.set_xlabel("x (m, local)")
    ax.set_ylabel("y (m, local)")
    fig.colorbar(scatter, ax=ax, label="pressure (Pa)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#F3EEE8")
    plt.close(fig)
    return str(out_path)


def run_porepy_scenario(
    geometry: Dict[str, object],
    direction: str,
    fracture_permeability_ratio: float,
    matrix_permeability_m2: float,
    residual_aperture_m: float,
    pressure_drop_pa: float,
    cell_size_m: float,
    output_dir: str | Path,
) -> Dict[str, object]:
    if direction not in {"x", "y"}:
        raise ValueError("flow direction 仅支持 x 或 y")
    if min(fracture_permeability_ratio, matrix_permeability_m2, residual_aperture_m, pressure_drop_pa, cell_size_m) <= 0:
        raise ValueError("P3 物理情景参数必须为正数")
    pp = importlib.import_module("porepy")
    Model, FractureSolidConstants = _build_porepy_model(pp)
    fracture_permeability = float(matrix_permeability_m2) * float(fracture_permeability_ratio)
    solid = FractureSolidConstants(
        residual_aperture=float(residual_aperture_m),
        permeability=float(matrix_permeability_m2),
        normal_permeability=fracture_permeability,
        fracture_permeability=fracture_permeability,
    )
    fluid = pp.FluidComponent(viscosity=1.0e-3, density=1000.0)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    params = {
        "fracture_segments_m": geometry["segments_m"],
        "domain_extent_m": geometry["domain_extent_m"],
        "flow_direction": direction,
        "pressure_drop_pa": float(pressure_drop_pa),
        "cell_size_m": float(cell_size_m),
        "material_constants": {"solid": solid, "fluid": fluid},
        "folder_name": str(out / "visualization"),
        "file_name": "p3",
        "times_to_export": [],
    }
    started = time.perf_counter()
    model = Model(params)
    pp.run_time_dependent_model(model, {"progressbars": False})
    matrix = model.mdg.subdomains(dim=2)[0]
    pressure = np.asarray(
        model.equation_system.evaluate(model.pressure([matrix])), dtype=float
    )
    darcy_flux = np.asarray(
        model.equation_system.evaluate(model.darcy_flux([matrix])), dtype=float
    )
    equation_residual = np.asarray(
        model.equation_system.assemble(evaluate_jacobian=False), dtype=float
    )
    sides = model.domain_boundary_sides(matrix)
    if direction == "x":
        inlet_mask, outlet_mask = sides.west, sides.east
        flow_length, cross_length = map(float, geometry["domain_extent_m"])
    else:
        inlet_mask, outlet_mask = sides.south, sides.north
        cross_length, flow_length = map(float, geometry["domain_extent_m"])
    q_in = float(abs(darcy_flux[inlet_mask].sum()))
    q_out = float(abs(darcy_flux[outlet_mask].sum()))
    mean_q = 0.5 * (q_in + q_out)
    balance = float(abs(q_in - q_out) / max(mean_q, np.finfo(float).tiny))
    conductance = float(mean_q / float(pressure_drop_pa))
    # PorePy 的 darcy_flux 运算符不包含流体 mobility；换算 Darcy 定律中的
    # 等效渗透率时 mobility 与粘度相消，因此这里不再额外乘粘度。
    effective_permeability = float(
        conductance * flow_length / max(cross_length, np.finfo(float).tiny)
    )
    finite = bool(np.isfinite(pressure).all() and np.isfinite(darcy_flux).all())
    scenario_id = f"{direction}_ratio_{fracture_permeability_ratio:g}"
    pressure_path = _plot_pressure(
        model, pressure, geometry, out / "pressure.png"
    )
    return {
        "scenario_id": scenario_id,
        "status": "completed" if finite else "failed_nonfinite",
        "flow_direction": direction,
        "fracture_permeability_ratio": float(fracture_permeability_ratio),
        "matrix_permeability_m2": float(matrix_permeability_m2),
        "fracture_permeability_m2": fracture_permeability,
        "pressure_drop_pa": float(pressure_drop_pa),
        "cell_size_m": float(cell_size_m),
        "matrix_cell_count": int(matrix.num_cells),
        "subdomain_cell_count": int(sum(sd.num_cells for sd in model.mdg.subdomains())),
        "interface_cell_count": int(sum(intf.num_cells for intf in model.mdg.interfaces())),
        "pressure_min_pa": float(np.min(pressure)),
        "pressure_max_pa": float(np.max(pressure)),
        "inlet_flux_abs_m2_s": q_in,
        "outlet_flux_abs_m2_s": q_out,
        "inlet_flux_signed": float(darcy_flux[inlet_mask].sum()),
        "outlet_flux_signed": float(darcy_flux[outlet_mask].sum()),
        "mass_balance_relative_error": balance,
        "matrix_boundary_flux_relative_mismatch": balance,
        "equation_residual_linf": float(np.max(np.abs(equation_residual))),
        "equation_residual_l2": float(np.linalg.norm(equation_residual)),
        "hydraulic_conductance_m2_pa_s": conductance,
        "effective_permeability_m2": effective_permeability,
        "finite_solution": finite,
        "runtime_seconds": float(time.perf_counter() - started),
        "pressure_plot": pressure_path,
    }


def _plot_scenario_summary(frame: pd.DataFrame, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for direction, part in frame.groupby("flow_direction"):
        part = part.sort_values("fracture_permeability_ratio")
        ax.plot(
            part["fracture_permeability_ratio"], part["effective_permeability_m2"],
            marker="o", label=f"{direction}-direction",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fracture / matrix permeability ratio")
    ax.set_ylabel("apparent effective permeability (m²)")
    ax.set_title("P3 PorePy scenario response")
    ax.grid(alpha=0.2, which="both")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#F3EEE8")
    plt.close(fig)
    return str(path)


def run_porepy_pilot(
    output_dir: str,
    config_path: str,
    progress_callback: ProgressCallback = None,
) -> Dict[str, object]:
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    cfg = config.get("physics_pilot") or {}
    root = Path(config_path).resolve().parents[1]

    def resolve_data_path(value: str) -> str:
        path = Path(value)
        return str(path.resolve() if path.is_absolute() else (root / path).resolve())

    geometry = prepare_trace_geometry(
        resolve_data_path(str(cfg.get("traces_path", "program/KB11/KB11_traces_100.geojson"))),
        resolve_data_path(str(cfg.get("area_path", "program/KB11/KB11_area.geojson"))),
        top_trace_count=int(cfg.get("top_trace_count", 12)),
        simplify_tolerance_m=float(cfg.get("simplify_tolerance_m", 0.5)),
        minimum_segment_length_m=float(cfg.get("minimum_segment_length_m", 0.2)),
        domain_padding_m=float(cfg.get("domain_padding_m", 1.0)),
    )
    geometry_path = output / "prepared_geometry.json"
    geometry_path.write_text(json.dumps(geometry, ensure_ascii=False, indent=2), encoding="utf-8")
    package = porepy_package_status()
    if not package["available"]:
        result = {
            "run_id": f"p3-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "status": "blocked",
            "failure_type": "external_dependency_blocked",
            "package_check": package,
            "geometry": geometry,
            "claim_scope": "no_physics_result",
            "limitations": ["PorePy 未安装，未生成任何物理计算结果。"],
            "artifact_paths": {"prepared_geometry_json": str(geometry_path)},
            "runtime_seconds": float(time.perf_counter() - started),
        }
        (output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    directions = list(cfg.get("flow_directions") or ["x", "y"])
    ratios = [float(value) for value in cfg.get(
        "fracture_permeability_ratios", [100.0, 10000.0, 1000000.0]
    )]
    rows = []
    failures = []
    total = len(directions) * len(ratios)
    index = 0
    for direction in directions:
        for ratio in ratios:
            index += 1
            scenario_id = f"{direction}_ratio_{ratio:g}"
            _emit(progress_callback, f"P3 {index}/{total}: {scenario_id}")
            try:
                rows.append(run_porepy_scenario(
                    geometry=geometry,
                    direction=str(direction),
                    fracture_permeability_ratio=ratio,
                    matrix_permeability_m2=float(cfg.get("matrix_permeability_m2", 1e-14)),
                    residual_aperture_m=float(cfg.get("residual_aperture_m", 1e-3)),
                    pressure_drop_pa=float(cfg.get("pressure_drop_pa", 1.0)),
                    cell_size_m=float(cfg.get("cell_size_m", 4.0)),
                    output_dir=output / "scenarios" / scenario_id,
                ))
            except Exception as exc:
                failures.append({
                    "scenario_id": scenario_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "scenario_metrics.csv", index=False, encoding="utf-8-sig")
    summary_plot = None
    if not frame.empty:
        summary_plot = _plot_scenario_summary(frame, output / "maps" / "p3_scenario_response.png")
    direction_summary = {}
    if not frame.empty:
        for ratio, part in frame.groupby("fracture_permeability_ratio"):
            values = part.set_index("flow_direction")["effective_permeability_m2"]
            if "x" in values and "y" in values and float(values["y"]) > 0:
                direction_summary[str(ratio)] = float(values["x"] / values["y"])
    monotonic = {}
    for direction, part in frame.groupby("flow_direction") if not frame.empty else []:
        ordered = part.sort_values("fracture_permeability_ratio")["effective_permeability_m2"].to_numpy(float)
        monotonic[str(direction)] = bool(np.all(np.diff(ordered) >= -1e-20))
    residual_tolerance = float(cfg.get("equation_residual_linf_tolerance", 1e-10))
    boundary_warning_threshold = float(
        cfg.get("matrix_boundary_flux_warning_threshold", 5e-3)
    )
    solver_residual_pass = bool(
        not frame.empty
        and np.isfinite(frame["equation_residual_linf"]).all()
        and (frame["equation_residual_linf"] <= residual_tolerance).all()
    )
    boundary_flux_warnings = (
        frame.loc[
            frame["matrix_boundary_flux_relative_mismatch"] > boundary_warning_threshold,
            "scenario_id",
        ].astype(str).tolist()
        if not frame.empty else []
    )
    status = "completed" if len(frame) == total and solver_residual_pass else "partial"
    result = {
        "run_id": f"p3-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "status": status,
        "package_check": package,
        "claim_scope": "porepy_method_pilot_only",
        "mapping_to_screening_targets": "not_validated_missing_colocated_raw_traces",
        "planned_scenario_count": total,
        "successful_scenario_count": int(len(frame)),
        "failed_scenarios": failures,
        "equation_residual_linf_threshold": residual_tolerance,
        "solver_residual_pass": solver_residual_pass,
        "matrix_boundary_flux_warning_threshold": boundary_warning_threshold,
        "matrix_boundary_flux_warning_scenarios": boundary_flux_warnings,
        "monotonic_response_by_direction": monotonic,
        "x_to_y_effective_permeability_ratio": direction_summary,
        "geometry_summary": {
            key: value for key, value in geometry.items() if key != "segments_m"
        },
        "limitations": [
            "KB11 为方法演示几何，不是塔里木工区验证数据。",
            "渗透率、孔隙度、压差和孔隙流体参数为情景假设，不是实测标定值。",
            "P3 结果不进入候选区综合分，也不证明油气预测有效。",
        ],
        "artifact_paths": {
            "result_json": str(output / "result.json"),
            "prepared_geometry_json": str(geometry_path),
            "scenario_metrics_csv": str(output / "scenario_metrics.csv"),
            "scenario_response_png": summary_plot,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "porepy": package,
        "config_path": str(Path(config_path).resolve()),
        "status": status,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# P3 PorePy 二维单相流方法试验", "",
        f"- 状态：`{status}`",
        f"- PorePy：`{package['version']}`",
        f"- 成功情景：{len(frame)}/{total}",
        f"- 全局方程残差检查：`{'pass' if solver_residual_pass else 'fail'}`",
        f"- 二维边界通量诊断警告：`{boundary_flux_warnings}`",
        f"- x/y 表观渗透率比：`{direction_summary}`",
        "", "该结果只证明方法链路和假设情景响应，不是现有候选区的物理验证。",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result
