# -*- coding: utf-8 -*-
import json

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from candidate_targeting import (
    add_period_scores,
    cluster_candidate_cells,
    match_period_nodes,
    percentile_rank,
    select_candidate_cells,
    validate_weights,
)


def _period_frame(period, offset_x=0.0, scale=1.0):
    rows = []
    for idx, (x, y) in enumerate([(0, 0), (3000, 0), (6000, 0), (9000, 0)]):
        rows.append({
            "period": period,
            "node_idx": idx,
            "pos_x": x + offset_x,
            "pos_y": y,
            "exact_betweenness": scale * float(idx + 1),
            "pagerank": scale * float(idx + 1),
            "removal_impact": 1.0 if idx == 3 else 0.0,
            "stability_rate": 1.0 if idx >= 2 else 1.0 / 3.0,
            "is_top20": idx == 3,
            "geometry": box(x - 100, y - 100, x + 100, y + 100),
        })
    return add_period_scores(pd.DataFrame(rows))


def test_percentile_rank_constant_has_no_artificial_signal():
    assert np.array_equal(percentile_rank([5.0, 5.0, 5.0]), np.zeros(3))


def test_validate_weights_rejects_wrong_sum_and_keys():
    with pytest.raises(ValueError):
        validate_weights({"a": 0.6, "b": 0.6}, ["a", "b"])
    with pytest.raises(ValueError):
        validate_weights({"a": 1.0}, ["a", "b"])


def test_period_scores_are_monotone_for_betweenness_and_pagerank():
    frame = _period_frame("A")
    assert frame["network_criticality"].is_monotonic_increasing
    assert frame.iloc[-1]["network_criticality"] > frame.iloc[0]["network_criticality"]


def test_match_period_nodes_counts_support_and_is_deterministic():
    frames = {
        "A": _period_frame("A"),
        "B": _period_frame("B", offset_x=100.0),
        "C": _period_frame("C", offset_x=-100.0),
    }
    first = match_period_nodes(frames, tolerance_m=500.0)
    second = match_period_nodes(frames, tolerance_m=500.0)
    assert first["match_id"].tolist() == second["match_id"].tolist()
    assert set(first["period_count"]) == {3}
    assert first["supporting_periods"].str.count(r"\|").eq(2).all()
    assert first["node_ids_json"].map(json.loads).map(len).eq(3).all()
    assert first["ambiguous_match_count"].eq(0).all()


def test_match_period_nodes_prevents_transitive_chain_merging():
    frames = {
        "A": _period_frame("A").iloc[[0]].copy(),
        "B": _period_frame("B", offset_x=900.0).iloc[[0]].copy(),
        "C": _period_frame("C", offset_x=1800.0).iloc[[0]].copy(),
    }
    matched = match_period_nodes(frames, tolerance_m=1000.0)
    assert sorted(matched["period_count"].tolist()) == [1, 2]
    assert matched["ambiguous_match_count"].eq(0).all()
    for node_ids in matched["node_ids_json"].map(json.loads):
        assert len(node_ids) == len(set(node_ids))


def test_candidate_selection_requires_multiperiod_and_top20():
    frames = {"A": _period_frame("A"), "B": _period_frame("B", offset_x=50.0)}
    matched = match_period_nodes(frames, tolerance_m=500.0)
    candidates, contract = select_candidate_cells(matched, score_quantile=0.50, min_periods=2)
    assert contract["eligible_count"] == 4
    assert len(candidates) == 1
    assert candidates.iloc[0]["any_top20"]


def test_cluster_candidate_cells_produces_stable_target_and_evidence_card():
    rows = []
    for idx, x in enumerate([0.0, 3000.0, 6000.0]):
        rows.append({
            "match_id": f"M{idx}", "centroid_x": x, "centroid_y": 0.0,
            "period_count": 3, "supporting_periods": "A|B|C",
            "network_criticality": 0.9, "removal_impact": 0.6,
            "period_persistence": 1.0, "parameter_stability": 1.0,
            "total_score": 0.9, "any_top20": True,
            "geometry": box(x - 100, -100, x + 100, 100),
        })
    cells, targets, cards = cluster_candidate_cells(
        pd.DataFrame(rows), eps_m=3500.0, min_samples=2
    )
    assert set(cells["target_id"].dropna()) == {"T001"}
    assert len(targets) == len(cards) == 1
    assert targets.iloc[0]["target_level"] == "一级"
    assert targets.iloc[0]["representative_x"] == 0.0
    assert targets.iloc[0]["representative_y"] == 0.0
    assert targets.iloc[0]["diameter_m"] == 6000.0
    assert targets.iloc[0]["stable_cell_fraction"] == 1.0
    assert cards[0]["external_validation_status"] == "not_validated"


def test_cluster_candidate_cells_allows_empty_result():
    cells, targets, cards = cluster_candidate_cells(pd.DataFrame())
    assert cells.empty and targets.empty and cards == []


def test_unstable_target_is_not_packaged_as_a_grade():
    rows = []
    for idx, x in enumerate([0.0, 3000.0, 6000.0]):
        rows.append({
            "match_id": f"M{idx}", "centroid_x": x, "centroid_y": 0.0,
            "period_count": 3, "supporting_periods": "A|B|C",
            "network_criticality": 0.9, "removal_impact": 0.0,
            "period_persistence": 1.0, "parameter_stability": 0.5,
            "total_score": 0.7, "any_top20": True,
            "geometry": box(x - 100, -100, x + 100, 100),
        })
    _, targets, cards = cluster_candidate_cells(
        pd.DataFrame(rows), eps_m=3500.0, min_samples=2
    )
    assert targets.iloc[0]["target_level"] == "不稳定候选"
    assert cards[0]["evidence_status"] == "internal_partial"


def test_target_clustering_splits_long_density_chain():
    rows = []
    for idx in range(20):
        x = idx * 3000.0
        rows.append({
            "match_id": f"M{idx}", "centroid_x": x, "centroid_y": 0.0,
            "period_count": 2, "supporting_periods": "A|B",
            "network_criticality": 0.8, "removal_impact": 0.0,
            "period_persistence": 0.5, "parameter_stability": 1.0,
            "total_score": 0.6, "any_top20": True,
            "geometry": box(x - 100, -100, x + 100, 100),
        })
    _, targets, _ = cluster_candidate_cells(
        pd.DataFrame(rows), eps_m=3500.0, min_samples=2, max_diameter_m=9000.0
    )
    assert len(targets) > 1
    assert targets["cell_count"].max() < 20
