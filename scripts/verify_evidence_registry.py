#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""审计当前活动证据链并保存结构化哈希清单。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "program"
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))

from evidence_audit import audit_evidence_registry


def main() -> int:
    result = audit_evidence_registry(ROOT)
    output = ROOT / "artifacts" / "experiment" / "evidence_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"evidence_audit={result['status']} "
        f"checks={result.get('checks_passed', 0)}/{result.get('checks_total', len(result['checks']))}"
    )
    for check in result["checks"]:
        print(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['check_id']}: {check['detail']}")
    print(f"output={output}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
