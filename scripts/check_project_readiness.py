#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""打印项目证据卡和原始数据就绪状态。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "program"
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))

from project_evidence import build_project_evidence, format_evidence_card


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="三期原始断裂线目录；默认检查 program/data/raw_faults/",
    )
    parser.add_argument("--json", action="store_true", help="输出结构化 JSON")
    args = parser.parse_args()

    evidence = build_project_evidence(ROOT, raw_dir=args.raw_dir)
    if args.json:
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
    else:
        print(format_evidence_card(evidence))
        print(f"\n建议数据目录：{evidence['data_readiness']['raw_data_directory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
