# -*- coding: utf-8 -*-
import pandas as pd
import pytest
from shapely.geometry import box

from external_validation import validate_external_points


def test_external_validation_missing_data_is_explicit():
    targets = pd.DataFrame({"geometry": [box(0, 0, 10, 10)]})
    out = validate_external_points(targets, None)
    assert out["status"] == "not_validated"
    assert out["hit_rate"] is None


def test_external_points_do_not_change_targets_and_report_hits():
    targets = pd.DataFrame({"target_id": ["T001"], "geometry": [box(0, 0, 10, 10)]})
    original = targets.copy(deep=True)
    points = pd.DataFrame({
        "x": [5.0, 20.0], "y": [5.0, 20.0], "outcome": ["productive", "failed"]
    })
    out = validate_external_points(targets, points)
    assert out["status"] == "evaluated"
    assert out["n_hits"] == 1
    assert out["hit_rate"] == pytest.approx(0.5)
    assert targets.equals(original)


def test_external_validation_rejects_missing_coordinates():
    targets = pd.DataFrame({"geometry": [box(0, 0, 10, 10)]})
    with pytest.raises(ValueError):
        validate_external_points(targets, pd.DataFrame({"name": ["well"]}))
