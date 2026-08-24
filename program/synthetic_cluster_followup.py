# -*- coding: utf-8 -*-
"""合成恢复失败后的靶区最大直径单因素跟进分析。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from artifact_paths import portable_artifact_path
from synthetic_validation import run_synthetic_recovery_campaign
from utils.config_loader import load_config


DIAMETERS_M = (18000.0, 15000.0, 12000.0, 9000.0)
FOLLOWUP_SCENARIOS = ("baseline", "weight_noise_30", "one_period_shifted")


def _write_config(base_config_path: str, output: Path, diameter_m: float) -> Path:
    cfg = copy.deepcopy(load_config(base_config_path))
    cfg.setdefault("screening", {}).setdefault("target_clustering", {})[
        "max_diameter_m"
    ] = float(diameter_m)
    path = output / "configs" / f"diameter_{int(diameter_m)}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def _load_p2_diameter_context(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / "artifacts" / "experiment" / "p2-p3-v1" / "p2" / "scenario_summary.csv"
    if not path.is_file():
        return {"available": False, "path": portable_artifact_path(path)}
    frame = pd.read_csv(path, encoding="utf-8-sig")
    rows = frame[frame["scenario_id"].isin(["baseline", "diameter_low"])]
    lookup = {str(row.scenario_id): row for row in rows.itertuples()}
    if "baseline" not in lookup or "diameter_low" not in lookup:
        return {"available": False, "path": portable_artifact_path(path)}
    baseline_count = int(float(lookup["baseline"].target_count))
    low_count = int(float(lookup["diameter_low"].target_count))
    return {
        "available": True,
        "path": portable_artifact_path(path),
        "baseline_target_count": baseline_count,
        "diameter_15000_target_count": low_count,
        "target_count_ratio": float(low_count / baseline_count),
        "baseline_target_recall": float(lookup["diameter_low"].baseline_target_recall),
    }


def _plot_followup(frame: pd.DataFrame, output: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.matplotlib_chinese import setup_matplotlib_chinese

    setup_matplotlib_chinese()
    figure_path = output / "maps" / "cluster_diameter_followup.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {
        "baseline": "#356f70",
        "weight_noise_30": "#9b3d4d",
        "one_period_shifted": "#c28d52",
    }
    for scenario_id, part in frame.groupby("scenario_id", sort=False):
        part = part.sort_values("diameter_m")
        axes[0].plot(
            part["diameter_m"] / 1000.0,
            part["localization_error_m"] / 1000.0,
            marker="o",
            linewidth=2,
            color=colors.get(scenario_id, "#596663"),
            label=scenario_id,
        )
        axes[1].plot(
            part["diameter_m"] / 1000.0,
            part["stable_target_count"],
            marker="o",
            linewidth=2,
            color=colors.get(scenario_id, "#596663"),
            label=scenario_id,
        )
    axes[0].axhline(4.5, color="#9b3d4d", ls="--", lw=1.3, label="4.5 km 命中门槛")
    axes[0].set_ylabel("最近稳定靶区质心误差（km）")
    axes[1].set_ylabel("稳定候选靶区数量")
    for ax in axes:
        ax.set_xlabel("靶区最大直径（km）")
        ax.set_xticks([9, 12, 15, 18])
        ax.grid(alpha=0.20)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("A  定位偏移")
    axes[1].set_title("B  目标碎片化与稳定性")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("靶区最大直径单因素失败分析", fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(figure_path)


def run_cluster_diameter_followup(
    output_dir: str | Path,
    *,
    config_path: str,
    repo_root: Optional[str | Path] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for diameter_m in DIAMETERS_M:
        if progress_callback:
            progress_callback(f"聚类直径跟进：{diameter_m / 1000:.0f} km")
        scenario_config = _write_config(config_path, output, diameter_m)
        result = run_synthetic_recovery_campaign(
            output / "runs" / f"diameter_{int(diameter_m)}",
            config_path=str(scenario_config),
            scenario_ids=FOLLOWUP_SCENARIOS,
        )
        for metric in result["scenario_metrics"]:
            rows.append(
                {
                    "diameter_m": diameter_m,
                    "scenario_id": metric["scenario_id"],
                    "stable_target_hit": metric["stable_target_hit"],
                    "localization_error_m": metric["localization_error_m"],
                    "representative_localization_error_m": metric[
                        "representative_localization_error_m"
                    ],
                    "stable_target_count": metric["stable_target_count"],
                    "candidate_cell_precision": metric["candidate_cell_precision"],
                    "candidate_cell_recall": metric["candidate_cell_recall"],
                }
            )
    frame = pd.DataFrame(rows)
    metrics_path = output / "cluster_diameter_metrics.csv"
    frame.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    repo = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[1]
    p2_context = _load_p2_diameter_context(repo)
    lookup = {
        (float(row.diameter_m), str(row.scenario_id)): row
        for row in frame.itertuples()
    }
    baseline_15 = lookup[(15000.0, "baseline")]
    hits_18 = int(frame.loc[frame["diameter_m"] == 18000.0, "stable_target_hit"].sum())
    hits_15 = int(frame.loc[frame["diameter_m"] == 15000.0, "stable_target_hit"].sum())
    gates = {
        "baseline_15000_hit": bool(baseline_15.stable_target_hit),
        "pressure_hits_not_worse": hits_15 >= hits_18,
        "p2_baseline_recall_one": bool(
            p2_context.get("available")
            and np.isclose(p2_context["baseline_target_recall"], 1.0)
        ),
        "p2_target_inflation_within_50_percent": bool(
            p2_context.get("available") and p2_context["target_count_ratio"] <= 1.50
        ),
    }
    promote_15000 = all(gates.values())
    figure_path = _plot_followup(frame, output)
    result = {
        "run_id": "synthetic-cluster-diameter-followup-v1",
        "status": "completed",
        "parent": "synthetic-recovery-v1",
        "claim_scope": "synthetic_failure_analysis_only",
        "diameters_m": list(DIAMETERS_M),
        "scenario_ids": list(FOLLOWUP_SCENARIOS),
        "acceptance_gates": gates,
        "promote_15000_as_default": promote_15000,
        "p2_operational_context": p2_context,
        "claim_update": (
            "supports_reducing_target_max_diameter_to_15000m"
            if promote_15000
            else "retain_18000m_and_record_localization_limit"
        ),
        "limitations": [
            "直径跟进由合成失败触发，只能用于定位聚合偏移原因。",
            "真实三期上下文来自既有 P2 参数场景，不是外部油气有效性证据。",
            "12 km 与 9 km 仅作为碎片化边界，不参与默认值推广门槛。",
        ],
        "artifact_paths": {
            "metrics_csv": portable_artifact_path(metrics_path),
            "figure_png": portable_artifact_path(figure_path),
            "plan_md": portable_artifact_path(output / "FOLLOWUP_PLAN.md"),
            "result_json": portable_artifact_path(output / "cluster_diameter_result.json"),
            "report_md": portable_artifact_path(output / "cluster_diameter_report.md"),
        },
    }
    (output / "cluster_diameter_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# 靶区最大直径失败分析报告",
        "",
        f"- 18 km 压力场景命中：{hits_18}/{len(FOLLOWUP_SCENARIOS)}",
        f"- 15 km 压力场景命中：{hits_15}/{len(FOLLOWUP_SCENARIOS)}",
        f"- P2 真实三期候选数量比：{p2_context.get('target_count_ratio', float('nan')):.3f}",
        f"- 是否推广 15 km：{'是' if promote_15000 else '否'}",
        "",
        f"结论代码：`{result['claim_update']}`",
        "",
        "该结论只涉及内部靶区聚合尺度，不支持油气预测有效性主张。",
    ]
    (output / "cluster_diameter_report.md").write_text("\n".join(report), encoding="utf-8")
    return result
