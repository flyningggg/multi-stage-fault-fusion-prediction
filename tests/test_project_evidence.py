import json
from pathlib import Path

from project_evidence import (
    assess_data_readiness,
    build_project_evidence,
    format_evidence_card,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_geojson(
    path: Path,
    geometry_type: str = "LineString",
    crs: bool = True,
    crs_name: str = "EPSG:32645",
):
    geometry = (
        {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]}
        if geometry_type == "LineString"
        else {"type": geometry_type, "coordinates": [0.0, 0.0]}
    )
    data = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"trace_id": "F1"}, "geometry": geometry}
        ],
    }
    if crs:
        data["crs"] = {"type": "name", "properties": {"name": crs_name}}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _make_minimal_repo(tmp_path: Path) -> Path:
    program = tmp_path / "program"
    program.mkdir()
    for name in ("海西.csv", "喜山.csv", "印支燕山.csv"):
        (program / name).write_text("x,y\n0,0\n", encoding="utf-8")
    return tmp_path


def test_current_evidence_card_preserves_claim_boundary():
    evidence = build_project_evidence(ROOT)
    assert evidence["p2"]["high_confidence_internal_count"] == 7
    assert evidence["p2"]["statistical_scenario_count"] == 15
    assert evidence["p3"]["successful_scenario_count"] == 6
    assert evidence["p3"]["planned_scenario_count"] == 6
    assert not evidence["data_readiness"]["real_physics_validation_ready"]
    card = format_evidence_card(evidence)
    assert "KB11 方法演示几何" in card
    assert "不构成油气发现概率验证" in card


def test_grid_csv_does_not_imply_source_or_physics_readiness(tmp_path):
    root = _make_minimal_repo(tmp_path)
    readiness = assess_data_readiness(root)
    assert readiness["grid_screening_ready"]
    assert not readiness["source_regridding_ready"]
    assert not readiness["real_physics_validation_ready"]
    assert "海西期同坐标系原始断裂线" in readiness["missing_items"]


def test_three_line_sources_with_common_crs_and_properties_are_ready(tmp_path):
    root = _make_minimal_repo(tmp_path)
    raw_dir = root / "raw"
    raw_dir.mkdir()
    for filename in ("海西.geojson", "喜山.geojson", "印支燕山.geojson"):
        _write_geojson(raw_dir / filename)
    properties = raw_dir / "physical_properties.yaml"
    properties.write_text("matrix_permeability_m2: 1.0e-14\n", encoding="utf-8")

    readiness = assess_data_readiness(root, raw_dir=raw_dir)
    assert readiness["source_regridding_ready"]
    assert readiness["real_physics_validation_ready"]
    assert readiness["common_crs"]
    assert all(
        item["crs_projected_metric"]
        for item in readiness["raw_fault_sources"].values()
    )
    assert readiness["missing_items"] == []


def test_non_line_geometry_or_missing_crs_is_rejected(tmp_path):
    root = _make_minimal_repo(tmp_path)
    raw_dir = root / "raw"
    raw_dir.mkdir()
    _write_geojson(raw_dir / "海西.geojson", geometry_type="Point")
    _write_geojson(raw_dir / "喜山.geojson", crs=False)
    _write_geojson(raw_dir / "印支燕山.geojson")

    readiness = assess_data_readiness(root, raw_dir=raw_dir)
    assert not readiness["source_regridding_ready"]
    assert "海西期线几何合同" in readiness["missing_items"]
    assert "喜山期CRS" in readiness["missing_items"]


def test_geographic_crs_is_not_accepted_for_metric_regridding(tmp_path):
    root = _make_minimal_repo(tmp_path)
    raw_dir = root / "raw"
    raw_dir.mkdir()
    for filename in ("海西.geojson", "喜山.geojson", "印支燕山.geojson"):
        _write_geojson(raw_dir / filename, crs_name="EPSG:4326")

    readiness = assess_data_readiness(root, raw_dir=raw_dir)
    assert readiness["common_crs"]
    assert not readiness["source_regridding_ready"]
    assert "海西期投影米制CRS" in readiness["missing_items"]
