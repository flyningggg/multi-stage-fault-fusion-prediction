from pathlib import Path

from evidence_audit import audit_evidence_registry


ROOT = Path(__file__).resolve().parents[1]


def test_active_evidence_registry_is_consistent_and_keeps_failure_visible():
    result = audit_evidence_registry(ROOT)
    assert result["status"] == "passed"
    assert result["checks_passed"] == result["checks_total"]
    by_id = {item["check_id"]: item for item in result["checks"]}
    assert by_id["active_config_is_15000m"]["passed"]
    assert by_id["p2_matches_active_baseline"]["passed"]
    assert by_id["p3_method_demo_boundary"]["passed"]
    assert by_id["synthetic_failure_remains_visible"]["passed"]
    assert len(result["artifact_hashes_sha256"]) == 5
