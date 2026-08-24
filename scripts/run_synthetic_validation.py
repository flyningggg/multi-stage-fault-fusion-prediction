#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行不依赖外部数据的合成真值恢复与鲁棒性验证。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "program"
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))

from synthetic_validation import run_synthetic_recovery_campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "experiment" / "synthetic-recovery-v1"),
    )
    parser.add_argument("--config", default=str(PROGRAM / "config.yaml"))
    parser.add_argument("--n-side", type=int, default=13)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    args = parser.parse_args()

    result = run_synthetic_recovery_campaign(
        args.output,
        config_path=args.config,
        n_side=args.n_side,
        scenario_ids=args.scenarios,
        progress_callback=print,
    )
    print(json.dumps(result["acceptance_gates"], ensure_ascii=False, indent=2))
    return 0 if result["acceptance_gates"]["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
