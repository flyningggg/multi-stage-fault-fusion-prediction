# -*- coding: utf-8 -*-
"""命令行运行正式多期候选靶区筛选。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "program"
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))


def main() -> int:
    parser = argparse.ArgumentParser(description="多期断裂候选勘探有利区精确筛选")
    parser.add_argument("--config", default=str(PROGRAM / "config.yaml"))
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "screening" / "latest"),
    )
    parser.add_argument(
        "--reuse-period-metrics-from",
        default=None,
        help="复用该既有运行的精确时期指标；配置哈希不一致时拒绝复用",
    )
    args = parser.parse_args()

    from screening_pipeline import run_target_screening

    result = run_target_screening(
        output_dir=args.output,
        config_path=args.config,
        progress_callback=lambda message: print(f"[screening] {message}", flush=True),
        reuse_period_metrics_from=args.reuse_period_metrics_from,
    )
    print(json.dumps({
        "run_id": result["run_id"],
        "status": result["status"],
        "candidate_target_count": result["input_summary"]["candidate_target_count"],
        "stable_target_count": result["input_summary"]["stable_target_count"],
        "unstable_target_count": result["input_summary"]["unstable_target_count"],
        "external_validation": result["external_validation"]["status"],
        "result_json": result["artifact_paths"]["result_json"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
