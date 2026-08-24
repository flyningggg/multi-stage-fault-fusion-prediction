# -*- coding: utf-8 -*-
"""活动证据登记表的一致性、边界与哈希审计。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from artifact_paths import portable_artifact_path


REQUIRED_ACTIVE_KEYS = (
    "target_screening_result",
    "p2_result",
    "p3_result",
    "synthetic_recovery_result",
    "cluster_followup_result",
)


def _read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return data


def _resolve(root: Path, value: str) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_evidence_registry(
    repo_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    registry_path = root / "artifacts" / "experiment" / "evidence_registry.json"
    checks = []

    def record(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "passed": bool(passed), "detail": detail})

    if not registry_path.is_file():
        record(
            "registry_exists",
            False,
            str(portable_artifact_path(registry_path, repo_root=root)),
        )
        return {
            "status": "failed",
            "registry_path": portable_artifact_path(registry_path, repo_root=root),
            "checks": checks,
            "artifact_hashes_sha256": {},
        }
    registry = _read_json(registry_path)
    record(
        "registry_exists",
        True,
        str(portable_artifact_path(registry_path, repo_root=root)),
    )
    record(
        "registry_schema",
        registry.get("schema_version") == 1,
        f"schema_version={registry.get('schema_version')}",
    )
    active = registry.get("active") if isinstance(registry.get("active"), dict) else {}
    resolved = {
        key: _resolve(root, active[key])
        for key in REQUIRED_ACTIVE_KEYS
        if key in active
    }
    missing_keys = sorted(set(REQUIRED_ACTIVE_KEYS) - set(resolved))
    record("active_keys_complete", not missing_keys, f"missing={missing_keys}")
    for key in REQUIRED_ACTIVE_KEYS:
        path = resolved.get(key)
        record(
            f"active_file_{key}",
            bool(path and path.is_file()),
            str(portable_artifact_path(path, repo_root=root)),
        )

    if any(not item["passed"] for item in checks):
        return {
            "status": "failed",
            "registry_path": portable_artifact_path(registry_path, repo_root=root),
            "checks": checks,
            "artifact_hashes_sha256": {
                key: _sha256(path) for key, path in resolved.items() if path.is_file()
            },
        }

    screening = _read_json(resolved["target_screening_result"])
    p2 = _read_json(resolved["p2_result"])
    p3 = _read_json(resolved["p3_result"])
    synthetic = _read_json(resolved["synthetic_recovery_result"])
    followup = _read_json(resolved["cluster_followup_result"])
    screening_summary = screening.get("input_summary") or {}
    stored_p2_baseline = str(p2.get("baseline_screening_dir", "")).replace("\\", "/").rstrip("/")
    active_baseline_suffix = str(
        Path(str(active["target_screening_result"])).parent
    ).replace("\\", "/").rstrip("/")

    config_snapshot = resolved["target_screening_result"].parent / "config_snapshot.yaml"
    diameter = None
    if config_snapshot.is_file():
        cfg = yaml.safe_load(config_snapshot.read_text(encoding="utf-8")) or {}
        diameter = (
            ((cfg.get("screening") or {}).get("target_clustering") or {}).get(
                "max_diameter_m"
            )
        )
    record("active_config_is_15000m", float(diameter or 0.0) == 15000.0, f"value={diameter}")
    record(
        "screening_is_operational_internal_only",
        screening_summary.get("input_role") == "operational_processed_grid"
        and (screening.get("external_validation") or {}).get("status") == "not_validated",
        (
            f"input_role={screening_summary.get('input_role')}; "
            f"external={(screening.get('external_validation') or {}).get('status')}"
        ),
    )
    record(
        "p2_matches_active_baseline",
        int(p2.get("baseline_target_count", -1))
        == int(screening_summary.get("candidate_target_count", -2))
        and stored_p2_baseline.endswith(active_baseline_suffix),
        (
            f"p2_count={p2.get('baseline_target_count')}; "
            f"screening_count={screening_summary.get('candidate_target_count')}; "
            f"baseline_suffix={active_baseline_suffix}"
        ),
    )
    record(
        "p2_claim_boundary",
        p2.get("claim_scope") == "internal_parameter_robustness_only"
        and int(p2.get("statistical_scenario_count", 0)) == 15,
        f"scope={p2.get('claim_scope')}; scenarios={p2.get('statistical_scenario_count')}",
    )
    record(
        "p3_method_demo_boundary",
        p3.get("claim_scope") == "porepy_method_pilot_only"
        and p3.get("mapping_to_screening_targets")
        == "not_validated_missing_colocated_raw_traces"
        and (p3.get("geometry_summary") or {}).get("data_role")
        == "method_demonstration_geometry_not_tarim_validation",
        (
            f"scope={p3.get('claim_scope')}; "
            f"mapping={p3.get('mapping_to_screening_targets')}"
        ),
    )
    synthetic_gates = synthetic.get("acceptance_gates") or {}
    record(
        "synthetic_failure_remains_visible",
        synthetic.get("claim_update") == "synthetic_recovery_contract_not_fully_met"
        and synthetic_gates.get("overall_pass") is False,
        (
            f"claim_update={synthetic.get('claim_update')}; "
            f"overall_pass={synthetic_gates.get('overall_pass')}"
        ),
    )
    followup_gates = followup.get("acceptance_gates") or {}
    record(
        "cluster_followup_supports_15000m",
        followup.get("promote_15000_as_default") is True
        and bool(followup_gates)
        and all(bool(value) for value in followup_gates.values()),
        f"gates={followup_gates}",
    )

    status = "passed" if all(item["passed"] for item in checks) else "failed"
    return {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_path": portable_artifact_path(registry_path, repo_root=root),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "artifact_hashes_sha256": {
            key: _sha256(path) for key, path in resolved.items()
        },
    }
