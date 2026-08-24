import numpy as np
import pandas as pd
import pytest

from synthetic_validation import (
    evaluate_synthetic_run,
    generate_synthetic_periods,
)


def test_synthetic_generator_is_deterministic_and_has_nine_truth_cells():
    first, first_meta = generate_synthetic_periods("baseline", n_side=9)
    second, second_meta = generate_synthetic_periods("baseline", n_side=9)
    assert first_meta == second_meta
    assert first_meta["truth_cell_count"] == 9
    for period in first:
        assert np.allclose(first[period]["NC_A"], second[period]["NC_A"])
        assert int(first[period]["is_truth_cell"].sum()) == 9
        truth_mean = first[period].loc[first[period]["is_truth_cell"], "NC_A"].mean()
        background_mean = first[period].loc[~first[period]["is_truth_cell"], "NC_A"].mean()
        assert truth_mean > background_mean * 4.0


def test_single_period_decoy_is_only_injected_into_first_period():
    periods, _ = generate_synthetic_periods("single_period_decoy", n_side=9)
    decoy_means = [
        float(frame.loc[frame["is_decoy_cell"], "NC_A"].mean())
        for frame in periods.values()
    ]
    assert decoy_means[0] > decoy_means[1] * 4.0
    assert decoy_means[0] > decoy_means[2] * 4.0


def test_evaluation_distinguishes_geometry_centroid_from_representative(tmp_path):
    candidate_path = tmp_path / "candidate_cells.csv"
    pd.DataFrame(
        {
            "centroid_x": [12000.0, 15000.0, 18000.0],
            "centroid_y": [18000.0, 18000.0, 18000.0],
            "target_id": ["T001", "T001", "T001"],
        }
    ).to_csv(candidate_path, index=False, encoding="utf-8-sig")
    _, metadata = generate_synthetic_periods("baseline", n_side=13)
    result = {
        "status": "completed",
        "candidate_targets": [
            {
                "target_id": "T001",
                "evidence_status": "internal_supported",
                "centroid_x": 24000.0,
                "centroid_y": 24000.0,
                "representative_x": 18000.0,
                "representative_y": 18000.0,
                "total_score": 0.8,
                "cell_count": 3,
            }
        ],
        "artifact_paths": {"candidate_cells_csv": str(candidate_path)},
    }
    metrics = evaluate_synthetic_run(result, metadata)
    assert not metrics["stable_target_hit"]
    assert metrics["stable_representative_hit"]
    assert metrics["representative_localization_error_m"] == 0.0
    assert metrics["candidate_cell_recall"] == pytest.approx(2.0 / 9.0)
