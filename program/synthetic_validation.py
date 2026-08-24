# -*- coding: utf-8 -*-
"""受控合成网络上的真值恢复与鲁棒性验证。

本模块复用正式筛选管线，只替换输入数据。所有输出都明确标记为合成内部
验证，不能用于声称真实工区、储层或油气发现有效性。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from artifact_paths import portable_artifact_path
from screening_pipeline import run_target_screening


PERIOD_NAMES = ("海西期", "喜山期", "印支燕山期")
SCENARIOS: tuple[Dict[str, Any], ...] = (
    {"scenario_id": "baseline", "seed": 42, "noise": 0.03},
    {"scenario_id": "baseline_repeat", "seed": 42, "noise": 0.03},
    {"scenario_id": "weight_noise_15", "seed": 43, "noise": 0.15},
    {"scenario_id": "weight_noise_30", "seed": 44, "noise": 0.30},
    {
        "scenario_id": "one_period_attenuated",
        "seed": 45,
        "noise": 0.08,
        "attenuated_period": 2,
    },
    {
        "scenario_id": "one_period_shifted",
        "seed": 46,
        "noise": 0.08,
        "shifted_period": 2,
    },
    {
        "scenario_id": "single_period_decoy",
        "seed": 47,
        "noise": 0.08,
        "decoy_period": 0,
    },
    {
        "scenario_id": "combined_stress",
        "seed": 48,
        "noise": 0.30,
        "shifted_period": 2,
        "weaken_fraction": 0.15,
    },
)


def _scenario_spec(scenario_id: str) -> Dict[str, Any]:
    for spec in SCENARIOS:
        if spec["scenario_id"] == scenario_id:
            return dict(spec)
    raise ValueError(f"未知合成场景: {scenario_id}")


def generate_synthetic_periods(
    scenario_id: str,
    *,
    n_side: int = 13,
    grid_step_m: float = 3000.0,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """生成三期规则网格、中心 3×3 真值区和右上角 3×3 单期诱饵区。"""
    import geopandas as gpd
    from shapely.geometry import box

    if n_side < 9 or n_side % 2 == 0:
        raise ValueError("n_side 必须是大于等于9的奇数")
    if grid_step_m <= 0:
        raise ValueError("grid_step_m 必须为正数")

    spec = _scenario_spec(scenario_id)
    center = n_side // 2
    truth_cells = {
        (row, col)
        for row in range(center - 1, center + 2)
        for col in range(center - 1, center + 2)
    }
    decoy_center = (2, n_side - 3)
    decoy_cells = {
        (row, col)
        for row in range(decoy_center[0] - 1, decoy_center[0] + 2)
        for col in range(decoy_center[1] - 1, decoy_center[1] + 2)
    }
    periods: Dict[str, Any] = {}
    half = float(grid_step_m) * 0.46

    for period_index, period_name in enumerate(PERIOD_NAMES):
        rng = np.random.default_rng(int(spec["seed"]) + period_index * 1009)
        local_center = (center, center)
        if spec.get("shifted_period") == period_index:
            local_center = (center, center + 1)
        local_truth = {
            (row, col)
            for row in range(local_center[0] - 1, local_center[0] + 2)
            for col in range(local_center[1] - 1, local_center[1] + 2)
        }
        boost = 8.0
        if spec.get("attenuated_period") == period_index:
            boost = 1.25

        records = []
        for row in range(n_side):
            for col in range(n_side):
                cx = float(col * grid_step_m)
                cy = float(row * grid_step_m)
                smooth_background = (
                    1.0
                    + 0.10 * math.sin((row + 1) * 0.73)
                    + 0.08 * math.cos((col + 1) * 0.61)
                )
                weight = max(0.15, smooth_background)
                chebyshev = max(abs(row - local_center[0]), abs(col - local_center[1]))
                if (row, col) in local_truth:
                    weight += boost
                elif chebyshev == 2:
                    weight += boost * 0.22
                if spec.get("decoy_period") == period_index and (row, col) in decoy_cells:
                    weight += 10.0
                weight *= max(0.05, 1.0 + rng.normal(0.0, float(spec["noise"])))
                records.append(
                    {
                        "row": row,
                        "col": col,
                        "vertex1_x": cx,
                        "vertex1_y": cy,
                        "NC_A": float(weight),
                        "is_truth_cell": (row, col) in truth_cells,
                        "is_decoy_cell": (row, col) in decoy_cells,
                        "geometry": box(cx - half, cy - half, cx + half, cy + half),
                    }
                )
        frame = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:32645")
        weaken_fraction = float(spec.get("weaken_fraction", 0.0))
        if weaken_fraction > 0:
            count = max(1, int(round(len(frame) * weaken_fraction)))
            weakened = rng.choice(len(frame), size=count, replace=False)
            frame.loc[weakened, "NC_A"] *= 0.20
        periods[period_name] = frame

    metadata = {
        "scenario_id": scenario_id,
        "seed": int(spec["seed"]),
        "n_side": int(n_side),
        "grid_step_m": float(grid_step_m),
        "truth_center_x": float(center * grid_step_m),
        "truth_center_y": float(center * grid_step_m),
        "truth_cell_count": len(truth_cells),
        "truth_cell_coordinates": sorted(
            [[float(col * grid_step_m), float(row * grid_step_m)] for row, col in truth_cells]
        ),
        "decoy_center_x": float(decoy_center[1] * grid_step_m),
        "decoy_center_y": float(decoy_center[0] * grid_step_m),
        "decoy_cell_count": len(decoy_cells),
        "intervention": {
            key: value for key, value in spec.items() if key not in {"scenario_id", "seed"}
        },
    }
    return periods, metadata


def _target_signature(targets: Iterable[Mapping[str, Any]]) -> str:
    stable_rows = []
    for row in targets:
        if row.get("evidence_status") != "internal_supported":
            continue
        stable_rows.append(
            {
                "target_id": row.get("target_id"),
                "centroid_x": round(float(row.get("centroid_x", 0.0)), 6),
                "centroid_y": round(float(row.get("centroid_y", 0.0)), 6),
                "representative_x": round(float(row.get("representative_x", 0.0)), 6),
                "representative_y": round(float(row.get("representative_y", 0.0)), 6),
                "total_score": round(float(row.get("total_score", 0.0)), 12),
                "cell_count": int(row.get("cell_count", 0)),
            }
        )
    encoded = json.dumps(stable_rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_synthetic_run(
    result: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """依据预注册空间门槛评价一次正式管线输出。"""
    grid_step = float(metadata["grid_step_m"])
    truth_center = np.asarray(
        [metadata["truth_center_x"], metadata["truth_center_y"]], dtype=float
    )
    decoy_center = np.asarray(
        [metadata["decoy_center_x"], metadata["decoy_center_y"]], dtype=float
    )
    truth_coords = {
        (round(float(x), 6), round(float(y), 6))
        for x, y in metadata["truth_cell_coordinates"]
    }
    targets = list(result.get("candidate_targets", []))
    stable = [row for row in targets if row.get("evidence_status") == "internal_supported"]
    ranked = sorted(stable, key=lambda row: -float(row.get("total_score", 0.0)))

    nearest_distance = None
    nearest_rank = None
    representative_distance = None
    representative_rank = None
    if ranked:
        distances = [
            float(
                np.linalg.norm(
                    np.asarray([row["centroid_x"], row["centroid_y"]], dtype=float)
                    - truth_center
                )
            )
            for row in ranked
        ]
        nearest_index = int(np.argmin(distances))
        nearest_distance = distances[nearest_index]
        nearest_rank = nearest_index + 1
        representative_distances = [
            float(
                np.linalg.norm(
                    np.asarray(
                        [
                            row.get("representative_x", row["centroid_x"]),
                            row.get("representative_y", row["centroid_y"]),
                        ],
                        dtype=float,
                    )
                    - truth_center
                )
            )
            for row in ranked
        ]
        representative_index = int(np.argmin(representative_distances))
        representative_distance = representative_distances[representative_index]
        representative_rank = representative_index + 1
    hit_radius = 1.5 * grid_step
    stable_target_hit = nearest_distance is not None and nearest_distance <= hit_radius
    stable_representative_hit = (
        representative_distance is not None and representative_distance <= hit_radius
    )

    decoy_nearby = False
    for row in stable:
        point = np.asarray([row["centroid_x"], row["centroid_y"]], dtype=float)
        if float(np.linalg.norm(point - decoy_center)) <= hit_radius:
            decoy_nearby = True
            break

    candidate_path = (result.get("artifact_paths") or {}).get("candidate_cells_csv")
    predicted_coords: set[tuple[float, float]] = set()
    if candidate_path and Path(candidate_path).is_file():
        candidate_cells = pd.read_csv(candidate_path, encoding="utf-8-sig")
        if "target_id" in candidate_cells.columns:
            candidate_cells = candidate_cells[candidate_cells["target_id"].notna()]
        predicted_coords = {
            (round(float(row.centroid_x), 6), round(float(row.centroid_y), 6))
            for row in candidate_cells.itertuples()
        }
    true_positive = len(predicted_coords & truth_coords)
    precision = true_positive / len(predicted_coords) if predicted_coords else 0.0
    recall = true_positive / len(truth_coords) if truth_coords else 0.0

    return {
        "scenario_id": metadata["scenario_id"],
        "status": result.get("status"),
        "stable_target_count": len(stable),
        "stable_target_hit": bool(stable_target_hit),
        "localization_error_m": nearest_distance,
        "nearest_target_rank": nearest_rank,
        "stable_representative_hit": bool(stable_representative_hit),
        "representative_localization_error_m": representative_distance,
        "nearest_representative_rank": representative_rank,
        "candidate_cell_precision": float(precision),
        "candidate_cell_recall": float(recall),
        "truth_core_recovered": bool(recall >= 0.80),
        "predicted_clustered_cell_count": len(predicted_coords),
        "truth_cell_count": len(truth_coords),
        "decoy_rejected": not decoy_nearby,
        "target_signature_sha256": _target_signature(targets),
        "intervention": metadata.get("intervention", {}),
    }


def _plot_campaign(
    output: Path,
    metrics: pd.DataFrame,
    baseline_periods: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
    baseline_metadata: Mapping[str, Any],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from utils.matplotlib_chinese import setup_matplotlib_chinese

    setup_matplotlib_chinese()
    figure_path = output / "maps" / "synthetic_recovery.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], hspace=0.30, wspace=0.24)
    ax_input = fig.add_subplot(grid[0, 0])
    ax_recovery = fig.add_subplot(grid[0, 1])
    ax_rates = fig.add_subplot(grid[1, 0])
    ax_error = fig.add_subplot(grid[1, 1])

    period_frame = next(iter(baseline_periods.values()))
    points = ax_input.scatter(
        period_frame.geometry.centroid.x,
        period_frame.geometry.centroid.y,
        c=period_frame["NC_A"],
        cmap="YlGnBu",
        marker="s",
        s=48,
        linewidths=0,
    )
    fig.colorbar(points, ax=ax_input, fraction=0.040, pad=0.02, label="合成连通能力")
    center_x = float(baseline_metadata["truth_center_x"])
    center_y = float(baseline_metadata["truth_center_y"])
    step = float(baseline_metadata["grid_step_m"])
    ax_input.add_patch(
        Rectangle(
            (center_x - 1.5 * step, center_y - 1.5 * step),
            3 * step,
            3 * step,
            fill=False,
            edgecolor="#9b3d4d",
            linewidth=2.2,
            label="预置真值区",
        )
    )
    ax_input.set_title("A  受控合成输入与预置真值")
    ax_input.legend(frameon=False, loc="upper left")

    ax_recovery.scatter(
        period_frame.geometry.centroid.x,
        period_frame.geometry.centroid.y,
        color="#dfe7e5",
        marker="s",
        s=38,
        linewidths=0,
    )
    candidate_path = baseline_result["artifact_paths"].get("candidate_cells_csv")
    if candidate_path and Path(candidate_path).is_file():
        candidate = pd.read_csv(candidate_path, encoding="utf-8-sig")
        if "target_id" in candidate.columns:
            candidate = candidate[candidate["target_id"].notna()]
        ax_recovery.scatter(
            candidate["centroid_x"],
            candidate["centroid_y"],
            facecolors="none",
            edgecolors="#356f70",
            marker="s",
            s=68,
            linewidths=1.4,
            label="聚类候选单元",
        )
    stable = [
        row
        for row in baseline_result.get("candidate_targets", [])
        if row.get("evidence_status") == "internal_supported"
    ]
    if stable:
        ax_recovery.scatter(
            [row.get("representative_x", row["centroid_x"]) for row in stable],
            [row.get("representative_y", row["centroid_y"]) for row in stable],
            color="#9b3d4d",
            edgecolors="white",
            marker="*",
            s=190,
            linewidths=0.8,
            label="稳定靶区最高分代表点",
        )
        ax_recovery.scatter(
            [row["centroid_x"] for row in stable],
            [row["centroid_y"] for row in stable],
            color="#596663",
            marker="x",
            s=52,
            linewidths=1.3,
            label="稳定靶区几何质心",
        )
    ax_recovery.add_patch(
        Rectangle(
            (center_x - 1.5 * step, center_y - 1.5 * step),
            3 * step,
            3 * step,
            fill=False,
            edgecolor="#d69b36",
            linewidth=2.2,
            label="预置真值区",
        )
    )
    ax_recovery.set_title("B  基准场景恢复结果")
    ax_recovery.legend(frameon=False, fontsize=9, loc="upper left")

    labels = metrics["scenario_id"].str.replace("_", "\n", regex=False)
    positions = np.arange(len(metrics))
    ax_rates.bar(
        positions - 0.18,
        metrics["candidate_cell_precision"],
        width=0.36,
        color="#6f918b",
        label="候选单元精确率",
    )
    ax_rates.bar(
        positions + 0.18,
        metrics["candidate_cell_recall"],
        width=0.36,
        color="#c29a78",
        label="候选单元召回率",
    )
    ax_rates.scatter(
        positions,
        np.where(metrics["stable_target_hit"], 1.04, 0.04),
        marker="o",
        color=np.where(metrics["stable_target_hit"], "#356f70", "#9b3d4d"),
        s=32,
        label="靶区命中",
        zorder=5,
    )
    ax_rates.set_xticks(positions, labels, fontsize=8)
    ax_rates.set_ylim(0, 1.12)
    ax_rates.set_ylabel("比例")
    ax_rates.set_title("C  各扰动场景的单元恢复")
    ax_rates.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")

    errors = metrics["localization_error_m"].astype(float) / 1000.0
    representative_errors = (
        metrics["representative_localization_error_m"].astype(float) / 1000.0
    )
    ax_error.barh(
        positions - 0.18,
        errors,
        height=0.34,
        color="#9aa6a2",
        alpha=0.90,
        label="几何质心",
    )
    ax_error.barh(
        positions + 0.18,
        representative_errors,
        height=0.34,
        color="#6f918b",
        alpha=0.95,
        label="最高分代表点",
    )
    ax_error.axvline(1.5 * step / 1000.0, color="#d69b36", ls="--", lw=1.5, label="命中门槛")
    ax_error.set_yticks(positions, labels, fontsize=8)
    ax_error.invert_yaxis()
    ax_error.set_xlabel("最近稳定靶区定位误差（km）")
    ax_error.set_title("D  定位误差与失败边界")
    ax_error.legend(frameon=False, fontsize=8, ncol=3)
    ax_error.grid(axis="x", alpha=0.18)

    for ax in (ax_input, ax_recovery):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X（m）")
        ax.set_ylabel("Y（m）")
        ax.grid(alpha=0.12)
    fig.suptitle("正式筛选流程的合成真值恢复与鲁棒性验证", fontsize=16, y=0.98)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(figure_path)


def run_synthetic_recovery_campaign(
    output_dir: str | Path,
    *,
    config_path: Optional[str] = None,
    n_side: int = 13,
    scenario_ids: Optional[Iterable[str]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """运行预注册合成场景并固化逐场景证据。"""
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = list(scenario_ids) if scenario_ids is not None else [
        spec["scenario_id"] for spec in SCENARIOS
    ]
    unknown = sorted(set(selected) - {spec["scenario_id"] for spec in SCENARIOS})
    if unknown:
        raise ValueError(f"未知场景: {unknown}")

    metrics = []
    baseline_periods = None
    baseline_result = None
    baseline_metadata = None
    for scenario_id in selected:
        if progress_callback:
            progress_callback(f"合成验证场景：{scenario_id}")
        periods, metadata = generate_synthetic_periods(
            scenario_id, n_side=n_side
        )
        scenario_result = run_target_screening(
            str(output / "runs" / scenario_id),
            config_path=config_path,
            progress_callback=None,
            period_gdfs_override=periods,
            input_role="synthetic_controlled",
        )
        scenario_metrics = evaluate_synthetic_run(scenario_result, metadata)
        metrics.append(scenario_metrics)
        if scenario_id == "baseline":
            baseline_periods = periods
            baseline_result = scenario_result
            baseline_metadata = metadata

    metrics_frame = pd.DataFrame(metrics)
    metrics_path = output / "scenario_metrics.csv"
    export_frame = metrics_frame.copy()
    export_frame["intervention"] = export_frame["intervention"].map(
        lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    export_frame.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    by_id = {row["scenario_id"]: row for row in metrics}
    baseline_hit = bool(by_id.get("baseline", {}).get("stable_target_hit", False))
    deterministic = bool(
        by_id.get("baseline", {}).get("target_signature_sha256")
        and by_id.get("baseline", {}).get("target_signature_sha256")
        == by_id.get("baseline_repeat", {}).get("target_signature_sha256")
    )
    robustness_ids = [
        "baseline",
        "weight_noise_15",
        "weight_noise_30",
        "one_period_attenuated",
        "one_period_shifted",
    ]
    available_robustness = [scenario_id for scenario_id in robustness_ids if scenario_id in by_id]
    robust_hit_count = sum(
        bool(by_id[scenario_id]["stable_target_hit"])
        for scenario_id in available_robustness
    )
    robustness_required = min(4, len(available_robustness))
    robustness_pass = (
        bool(available_robustness) and robust_hit_count >= robustness_required
    )
    decoy_rejected = bool(
        by_id.get("single_period_decoy", {}).get("decoy_rejected", False)
    )
    overall_pass = baseline_hit and deterministic and robustness_pass and decoy_rejected
    representative_hit_count = sum(
        bool(by_id[scenario_id]["stable_representative_hit"])
        for scenario_id in available_robustness
    )
    truth_core_recovery_count = sum(
        bool(by_id[scenario_id]["truth_core_recovered"])
        for scenario_id in available_robustness
    )
    gates = {
        "baseline_hit": baseline_hit,
        "deterministic_repeat": deterministic,
        "robustness_hit_count": robust_hit_count,
        "robustness_scenario_count": len(available_robustness),
        "robustness_required": robustness_required,
        "robustness_pass": robustness_pass,
        "single_period_decoy_rejected": decoy_rejected,
        "diagnostic_representative_hit_count": representative_hit_count,
        "diagnostic_truth_core_recovery_count": truth_core_recovery_count,
        "overall_pass": overall_pass,
    }

    figure_path = None
    if baseline_periods is not None and baseline_result is not None and baseline_metadata is not None:
        figure_path = _plot_campaign(
            output,
            metrics_frame,
            baseline_periods,
            baseline_result,
            baseline_metadata,
        )
    limitations = [
        "本轮数据和真值均为规则网格上的人工构造，只检验算法可恢复性。",
        "合成噪声与扰动不覆盖真实构造解释误差、测线偏差和储层非唯一性。",
        "结果不能替代井位、储层、产量、专家盲评或同位物理模型验证。",
    ]
    claim_update = (
        "supports_controlled_synthetic_recovery_under_preregistered_perturbations"
        if overall_pass
        else "synthetic_recovery_contract_not_fully_met"
    )
    campaign_result = {
        "run_id": "synthetic-recovery-v1",
        "status": "completed",
        "claim_scope": "controlled_synthetic_internal_validation_only",
        "planned_scenario_count": len(selected),
        "completed_scenario_count": len(metrics),
        "scenario_metrics": metrics,
        "acceptance_gates": gates,
        "claim_update": claim_update,
        "comparability": "same_pipeline_config_and_grid_contract; declared_input_perturbations_only",
        "limitations": limitations,
        "artifact_paths": {
            "scenario_metrics_csv": portable_artifact_path(metrics_path),
            "figure_png": portable_artifact_path(figure_path),
            "plan_md": portable_artifact_path(output / "PLAN.md"),
            "report_md": portable_artifact_path(output / "report.md"),
            "result_json": portable_artifact_path(output / "result.json"),
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "next_action": (
            "retain_as_internal_algorithm_evidence_and_wait_for_external_data"
            if overall_pass
            else "inspect_failed_synthetic_slices_before_strengthening_claim"
        ),
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(campaign_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# 合成真值恢复与鲁棒性验证报告",
        "",
        f"- 场景完成：{len(metrics)}/{len(selected)}",
        f"- 基准真值命中：{'通过' if baseline_hit else '未通过'}",
        f"- 确定性复现：{'通过' if deterministic else '未通过'}",
        f"- 扰动场景命中：{robust_hit_count}/{len(available_robustness)}（门槛 {robustness_required}）",
        f"- 单期诱饵拒绝：{'通过' if decoy_rejected else '未通过'}",
        f"- 诊断性最高分代表点命中：{representative_hit_count}/{len(available_robustness)}",
        f"- 诊断性真值核心单元恢复：{truth_core_recovery_count}/{len(available_robustness)}",
        f"- 总门槛：{'通过' if overall_pass else '未通过'}",
        "",
        "## 逐场景结果",
        "",
        "| 场景 | 质心命中 | 质心误差(m) | 代表点误差(m) | 单元精确率 | 单元召回率 | 诱饵拒绝 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        error = row["localization_error_m"]
        error_text = "NA" if error is None else f"{error:.1f}"
        representative_error = row["representative_localization_error_m"]
        representative_error_text = (
            "NA" if representative_error is None else f"{representative_error:.1f}"
        )
        report_lines.append(
            f"| {row['scenario_id']} | {row['stable_target_hit']} | {error_text} | "
            f"{representative_error_text} | "
            f"{row['candidate_cell_precision']:.3f} | {row['candidate_cell_recall']:.3f} | "
            f"{row['decoy_rejected']} |"
        )
    report_lines.extend(["", "## 证据边界", ""])
    report_lines.extend(f"- {item}" for item in limitations)
    report_lines.extend(["", f"结论代码：`{claim_update}`", ""])
    (output / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return campaign_result
