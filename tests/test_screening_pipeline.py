# -*- coding: utf-8 -*-
import json
from pathlib import Path

import networkx as nx
import pytest

import screening_pipeline
from screening_pipeline import compute_node_removal_impact, run_target_screening
from tests.conftest import make_grid_gdf


def test_node_removal_impact_detects_articulation_only():
    graph = nx.path_graph(4)
    out = compute_node_removal_impact(graph)
    assert out[1] > 0 and out[2] > 0
    assert out[0] == 0 and out[3] == 0


def test_formal_pipeline_has_no_agent_model_dependency():
    source = Path(screening_pipeline.__file__).read_text(encoding="utf-8")
    assert "agent_model" not in source


def test_tiny_three_period_pipeline_exports_durable_contract(tmp_path, monkeypatch):
    period_gdfs = {
        "A期": make_grid_gdf(3, 3, weights={4: 10.0}),
        "B期": make_grid_gdf(3, 3, weights={4: 9.0}),
        "C期": make_grid_gdf(3, 3, weights={4: 8.0}),
    }
    monkeypatch.setattr(screening_pipeline, "load_all_periods", lambda: period_gdfs)
    config_path = Path(screening_pipeline.__file__).with_name("config.yaml")
    output = tmp_path / "screening"
    result = run_target_screening(str(output), str(config_path))

    assert result["status"] in {"completed", "completed_no_targets"}
    assert result["input_summary"]["period_count"] == 3
    assert result["external_validation"]["status"] == "not_validated"
    for name in ["manifest.json", "result.json", "report.md", "matched_cells.csv"]:
        assert (output / name).is_file(), name
    assert (output / "stable_candidate_targets.csv").is_file()
    assert (
        result["input_summary"]["stable_target_count"]
        + result["input_summary"]["unstable_target_count"]
        == result["input_summary"]["candidate_target_count"]
    )
    assert (output / "maps" / "candidate_targets.png").is_file()

    def _should_not_recompute(*args, **kwargs):
        raise AssertionError("复用模式不应重新计算精确时期指标")

    monkeypatch.setattr(screening_pipeline, "compute_period_node_metrics", _should_not_recompute)
    reused_output = tmp_path / "screening_reused"
    reused = run_target_screening(
        str(reused_output),
        str(config_path),
        reuse_period_metrics_from=str(output),
    )
    assert reused["input_summary"] == result["input_summary"]
    manifest = json.loads((reused_output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reused_period_metrics_from"] == str(output.resolve())


def test_injected_period_data_requires_explicit_synthetic_role(tmp_path):
    period_gdfs = {
        "A期": make_grid_gdf(3, 3),
        "B期": make_grid_gdf(3, 3),
    }
    config_path = Path(screening_pipeline.__file__).with_name("config.yaml")
    with pytest.raises(ValueError, match="synthetic_controlled"):
        run_target_screening(
            str(tmp_path / "rejected"),
            str(config_path),
            period_gdfs_override=period_gdfs,
        )
