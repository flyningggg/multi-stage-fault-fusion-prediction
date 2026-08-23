# -*- coding: utf-8 -*-
"""重跑 Stage 1/3/4，并将可比较结果写入独立实验目录。"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "program"
RUN_DIR = ROOT / "artifacts" / "experiment" / "correctness-v1"
OUTPUT_DIR = RUN_DIR / "outputs"
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


def _version(module_name: str) -> str:
    module = __import__(module_name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "versions": {
            name: _version(name)
            for name in ("numpy", "pandas", "sklearn", "networkx", "xgboost")
        },
        "random_state": 42,
        "config": str(PROGRAM / "config.yaml"),
    }
    (RUN_DIR / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from batch_run import run_multiperiod_fusion
    from percolation import run_percolation_pipeline
    from agent_model import run_agent_pipeline

    print("[1/3] Stage 1: fusion + honest spatial CV", flush=True)
    stage1_dir = OUTPUT_DIR / "stage1"
    stage1 = run_multiperiod_fusion(out_dir=str(stage1_dir))

    print("[2/3] Stage 3: percolation capacity semantics regression", flush=True)
    stage3_dir = OUTPUT_DIR / "stage3"
    stage3 = run_percolation_pipeline(out_dir=str(stage3_dir))

    print("[3/3] Stage 4: corrected resistance-label agent evaluation", flush=True)
    stage4_dir = OUTPUT_DIR / "stage4"
    stage4 = run_agent_pipeline(out_dir=str(stage4_dir))

    stage1_metrics = {}
    for period, result in stage1["xgboost"].items():
        if result is None:
            stage1_metrics[period] = {"status": "not_run"}
        else:
            stage1_metrics[period] = {
                "train_r2": result["train_r2"],
                "spatial_cv": result["spatial_cv"],
                "top_feature": result["importance"].iloc[0].to_dict(),
            }

    stage3_metrics = {
        period: {
            "threshold_pc50": result["threshold"],
            "threshold_steepest": result["threshold_steepest"],
            "key_node_indices": result["key_nodes"]["node_idx"].astype(int).tolist(),
            "key_node_metric": "pagerank_capacity",
        }
        for period, result in stage3.items()
        if not str(period).startswith("_")
    }

    result = {
        "run_id": "correctness-v1",
        "baseline_commit": "d74e669c63e1f3cde21eafbb039023fb53ff0ef5",
        "stage1": stage1_metrics,
        "stage3": stage3_metrics,
        "stage4": {
            "random_split": stage4["metrics"],
            "spatial_cv": stage4["spatial_cv"],
            "lopo": stage4["lopo"],
            "target": "log1p_betweenness_with_inverse_capacity_distance",
        },
    }
    (RUN_DIR / "result.json").write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_JSON={RUN_DIR / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
