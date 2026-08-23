# -*- coding: utf-8 -*-
"""validation-v2：LOPO高值排序与容量-距离定义敏感性验证。"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "program"
RUN_DIR = ROOT / "artifacts" / "experiment" / "validation-v2"
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _version(module_name: str) -> str:
    module = __import__(module_name)
    return str(getattr(module, "__version__", "unknown"))


def _log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (RUN_DIR / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _flatten_ranking(period: str, metrics: dict) -> dict:
    return {
        "period": period,
        "r2": metrics["r2"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        **metrics["ranking"],
    }


def _proxy_decision(per_period: dict) -> tuple[str, str]:
    rows = [metrics["ranking"] for metrics in per_period.values()]
    if all(
        row["spearman"] >= 0.70 and row["top_20pct_overlap"] >= 0.70
        for row in rows
    ):
        return "go", "三期均达到项目级高值排序门槛"
    if all(
        row["spearman"] >= 0.50 and row["top_20pct_overlap"] >= 0.50
        for row in rows
    ):
        return "conditional", "仅允许代理粗筛，候选区必须精确重算"
    return "stop_proxy", "至少一个时期未达到最低排序门槛"


def _plot_ranking(rows: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    periods = [row["period"] for row in rows]
    x = np.arange(len(periods), dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width, [row["spearman"] for row in rows], width, label="Spearman")
    ax.bar(x, [row["top_10pct_overlap"] for row in rows], width, label="Top 10% overlap")
    ax.bar(x + width, [row["top_20pct_overlap"] for row in rows], width, label="Top 20% overlap")
    ax.axhline(0.70, color="#2a9d8f", linestyle="--", linewidth=1.2, label="go threshold")
    ax.axhline(0.50, color="#e76f51", linestyle=":", linewidth=1.2, label="minimum threshold")
    ax.set_xticks(x, periods)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("score")
    ax.set_title("LOPO high-value ranking consistency")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_sensitivity(rows: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = ["inverse_sqrt", "neglog"]
    periods = sorted({row["period"] for row in rows})
    x = np.arange(len(periods), dtype=float)
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, variant in zip((-width / 2, width / 2), variants):
        vals = [
            next(
                row["top_20pct_overlap"]
                for row in rows
                if row["period"] == period and row["transform"] == variant
            )
            for period in periods
        ]
        ax.bar(x + offset, vals, width, label=variant)
    ax.axhline(0.70, color="#e76f51", linestyle="--", linewidth=1.2, label="robust threshold")
    ax.set_xticks(x, periods)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Top 20% overlap vs inverse")
    ax.set_title("Exact-label sensitivity to distance transform")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    from agent_model import (
        build_agent_training_data,
        leave_one_period_out_evaluate,
        ranking_metrics,
    )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "run.log").write_text("", encoding="utf-8")
    start = time.perf_counter()

    environment = {
        "run_id": "validation-v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).splitlines(),
        "versions": {
            name: _version(name)
            for name in ("numpy", "pandas", "scipy", "sklearn", "networkx", "xgboost", "matplotlib")
        },
        "random_state": 42,
        "command": f"{sys.executable} scripts/run_validation_v2.py",
        "fixed_conditions": {
            "periods": 3,
            "edge_weight": "NC_A",
            "weight_mode": "min",
            "grid_step_m": 3000.0,
            "model": "AGENT_XGB_PARAMS",
            "split": "leave_one_period_out",
        },
    }
    (RUN_DIR / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _log("构建 inverse 精确标签和代理特征")
    baseline_df = build_agent_training_data(distance_transform="inverse")
    _log(f"基准训练数据完成: n={len(baseline_df)}")

    _log("运行三折 LOPO 并计算高值排序指标")
    lopo = leave_one_period_out_evaluate(baseline_df)
    ranking_rows = [
        _flatten_ranking(period, metrics)
        for period, metrics in lopo["per_period"].items()
    ]
    _write_csv(RUN_DIR / "ranking_metrics.csv", ranking_rows)
    _plot_ranking(ranking_rows, RUN_DIR / "ranking_stability.png")
    proxy_status, proxy_reason = _proxy_decision(lopo["per_period"])
    _log(f"代理排序判定: {proxy_status}; {proxy_reason}")

    _log("运行精确标签距离转换敏感性: inverse_sqrt, neglog")
    keys = ["period", "cell_x", "cell_y"]
    base_labels = baseline_df[keys + ["log1p_betweenness"]].rename(
        columns={"log1p_betweenness": "baseline"}
    )
    sensitivity_rows: list[dict] = []
    for transform in ("inverse_sqrt", "neglog"):
        variant_df = build_agent_training_data(
            use_spatial_features=False,
            use_neighbor_features=False,
            distance_transform=transform,
        )
        variant_labels = variant_df[keys + ["log1p_betweenness"]].rename(
            columns={"log1p_betweenness": "variant"}
        )
        merged = base_labels.merge(variant_labels, on=keys, how="inner", validate="one_to_one")
        if len(merged) != len(base_labels) or len(merged) != len(variant_labels):
            raise RuntimeError(
                f"{transform} 标签坐标未完整对齐: baseline={len(base_labels)}, "
                f"variant={len(variant_labels)}, merged={len(merged)}"
            )
        for period, part in merged.groupby("period", sort=True):
            rank = ranking_metrics(part["baseline"], part["variant"])
            sensitivity_rows.append({
                "period": str(period),
                "transform": transform,
                **rank,
            })
        _log(f"距离转换完成: {transform}")

    _write_csv(RUN_DIR / "distance_sensitivity.csv", sensitivity_rows)
    _plot_sensitivity(sensitivity_rows, RUN_DIR / "distance_sensitivity.png")
    label_robust = all(row["top_20pct_overlap"] >= 0.70 for row in sensitivity_rows)
    sensitivity_status = "robust" if label_robust else "fragile"
    _log(f"精确标签敏感性判定: {sensitivity_status}")

    if proxy_status == "go" and label_robust:
        next_action = "进入外部地质盲评；代理模型可继续作为拓扑高值初筛工具"
        claim_update = "supported_internal_only"
    elif proxy_status == "conditional" and label_robust:
        next_action = "采用代理粗筛+候选区精确重算，并进入外部地质盲评"
        claim_update = "narrowed"
    else:
        next_action = "停止跨期代理主线，保留精确中心性并优先获取外部地质标签"
        claim_update = "refuted_or_fragile"

    elapsed = time.perf_counter() - start
    result = {
        "run_id": "validation-v2",
        "parent_run": "correctness-v1",
        "parent_commit": "d69b7a3",
        "research_question": "代理模型能否跨期稳定筛出精确betweenness高值节点，且标签对合理距离转换是否稳健？",
        "lopo": lopo,
        "proxy_decision": {"status": proxy_status, "reason": proxy_reason},
        "distance_sensitivity": {
            "baseline": "inverse",
            "variants": ["inverse_sqrt", "neglog"],
            "status": sensitivity_status,
            "criterion": "all period/variant Top20 overlap >= 0.70",
            "rows": sensitivity_rows,
        },
        "evaluation_summary": {
            "outcome": proxy_status,
            "claim_update": claim_update,
            "baseline_relation": "same_data_same_lopo_contract",
            "comparability": "high_for_proxy_ranking; transform_slice_changes_label_definition_by_design",
            "failure_mode": None if proxy_status == "go" and label_robust else "direction_underperforming",
            "next_action": next_action,
        },
        "limitations": [
            "betweenness是内部拓扑代理标签，不是油气发现标签",
            "尚无井位、产量、储层或专家盲评外部证据",
            "本轮未改变网格尺度、邻接容差或聚类参数",
        ],
        "elapsed_seconds": elapsed,
    }
    (RUN_DIR / "result.json").write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(f"完成: {elapsed:.1f}s; next_action={next_action}")
    print(f"RESULT_JSON={RUN_DIR / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
