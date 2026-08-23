# -*- coding: utf-8 -*-
"""运行 P3 PorePy 独立物理方法试验。"""
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
    parser = argparse.ArgumentParser(description="P3 PorePy 二维裂缝单相流方法试验")
    parser.add_argument("--config", default=str(PROGRAM / "config.yaml"))
    parser.add_argument(
        "--output", default=str(ROOT / "artifacts" / "experiment" / "p2-p3-v1" / "p3")
    )
    args = parser.parse_args()

    from porepy_flow_pilot import run_porepy_pilot

    result = run_porepy_pilot(
        args.output,
        args.config,
        progress_callback=lambda message: print(f"[p3] {message}", flush=True),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
