#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行靶区最大直径的合成失败跟进分析。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "program"
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))

from synthetic_cluster_followup import run_cluster_diameter_followup


def main() -> int:
    output = ROOT / "artifacts" / "experiment" / "synthetic-recovery-v1"
    result = run_cluster_diameter_followup(
        output,
        config_path=str(PROGRAM / "config.yaml"),
        repo_root=ROOT,
        progress_callback=print,
    )
    print(json.dumps(result["acceptance_gates"], ensure_ascii=False, indent=2))
    print(f"promote_15000_as_default={result['promote_15000_as_default']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
