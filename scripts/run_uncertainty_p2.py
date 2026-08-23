# -*- coding: utf-8 -*-
"""运行 P2 参数不确定性情景集合。"""
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
    parser = argparse.ArgumentParser(description="P2 候选靶区参数不确定性分析")
    parser.add_argument("--config", default=str(PROGRAM / "config.yaml"))
    parser.add_argument(
        "--output", default=str(ROOT / "artifacts" / "experiment" / "p2-p3-v1" / "p2")
    )
    parser.add_argument(
        "--baseline-screening-dir",
        default=str(ROOT / "artifacts" / "experiment" / "target-screening-mvp-v1" / "final"),
    )
    parser.add_argument(
        "--scenario", action="append", dest="scenarios",
        help="只运行指定场景，可重复；默认运行全部预注册场景",
    )
    args = parser.parse_args()

    from uncertainty_analysis import run_uncertainty_analysis

    result = run_uncertainty_analysis(
        output_dir=args.output,
        config_path=args.config,
        baseline_screening_dir=args.baseline_screening_dir,
        scenario_ids=args.scenarios,
        progress_callback=lambda message: print(f"[p2] {message}", flush=True),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
