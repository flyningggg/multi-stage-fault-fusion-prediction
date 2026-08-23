# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from uncertainty_analysis import (
    aggregate_target_uncertainty,
    apply_overrides,
    build_default_scenarios,
    match_target_tables,
    _scenario_observations,
)


def _targets(rows):
    return pd.DataFrame(rows, columns=[
        "target_id", "centroid_x", "centroid_y", "total_score", "evidence_status"
    ])


def test_overrides_do_not_mutate_baseline_config():
    config = {"grid": {"step_m": 3000.0}}
    changed = apply_overrides(config, {"grid.step_m": 2700.0})
    assert config["grid"]["step_m"] == 3000.0
    assert changed["grid"]["step_m"] == 2700.0


def test_default_scenarios_cover_required_factors():
    config = {
        "grid": {"step_m": 3000, "edge_dist_tolerance_m": 150, "centroid_match_tolerance_m": 1500},
        "screening": {
            "candidate_cell_quantile": 0.8,
            "distance_transforms": ["inverse", "inverse_sqrt", "neglog"],
            "score_weights": {
                "network_criticality": 0.35, "removal_impact": 0.25,
                "period_persistence": 0.25, "parameter_stability": 0.15,
            },
            "target_clustering": {"eps_m": 4500, "max_diameter_m": 18000},
        },
    }
    scenarios = build_default_scenarios(config)
    factors = {scenario.factor for scenario in scenarios}
    assert {
        "grid_step_m", "edge_tolerance_m", "match_tolerance_m",
        "candidate_quantile", "cluster_eps_m", "max_diameter_m",
        "score_weights", "distance_transform",
    } <= factors
    assert len({scenario.scenario_id for scenario in scenarios}) == len(scenarios)


def test_grid_step_scenarios_remain_registered_for_explicit_exclusion():
    config = {
        "grid": {"step_m": 3000, "edge_dist_tolerance_m": 150, "centroid_match_tolerance_m": 1500},
        "screening": {
            "candidate_cell_quantile": 0.8,
            "distance_transforms": ["inverse"],
            "score_weights": {
                "network_criticality": 0.35, "removal_impact": 0.25,
                "period_persistence": 0.25, "parameter_stability": 0.15,
            },
            "target_clustering": {"eps_m": 4500, "max_diameter_m": 18000},
        },
    }
    grid_scenarios = [s for s in build_default_scenarios(config) if s.factor == "grid_step_m"]
    assert {s.scenario_id for s in grid_scenarios} == {"grid_step_low", "grid_step_high"}


def test_target_matching_uses_one_to_one_radius_constraint():
    baseline = _targets([
        ("T1", 0.0, 0.0, 0.9, "internal_supported"),
        ("T2", 10.0, 0.0, 0.8, "internal_partial"),
    ])
    scenario = _targets([
        ("A", 1.0, 0.0, 0.7, "internal_supported"),
        ("B", 2.0, 0.0, 0.6, "internal_partial"),
    ])
    matches = match_target_tables(baseline, scenario, max_distance_m=3.0)
    assert len(matches) == 1
    assert matches.iloc[0]["baseline_index"] == 0
    assert matches.iloc[0]["scenario_index"] == 0


def test_uncertainty_summary_counts_missing_scenario_as_absence():
    baseline = _targets([
        ("T1", 0.0, 0.0, 0.9, "internal_supported"),
        ("T2", 100.0, 0.0, 0.8, "internal_partial"),
    ])
    scenario_frames = {
        "baseline": baseline,
        "shifted": _targets([
            ("S1", 2.0, 0.0, 0.85, "internal_supported"),
        ]),
    }
    observations, _ = _scenario_observations(baseline, scenario_frames, max_distance_m=10.0)
    summary = aggregate_target_uncertainty(baseline, observations).set_index("baseline_target_id")
    assert summary.loc["T1", "candidate_occurrence_frequency"] == 1.0
    assert summary.loc["T2", "candidate_occurrence_frequency"] == 0.5
    assert summary.loc["T1", "robustness_tier"] == "robust_occurrence"
    assert summary.loc["T2", "robustness_tier"] == "conditional_occurrence"
    assert summary.loc["T1", "decision_tier"] == "high_confidence_internal"
    assert summary.loc["T2", "decision_tier"] == "fragile_unstable_candidate"
    assert np.isfinite(summary.loc["T1", "score_q05"])
