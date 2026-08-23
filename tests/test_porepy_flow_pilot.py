# -*- coding: utf-8 -*-
import pytest
from shapely.geometry import LineString
from copy import deepcopy
from pathlib import Path

from porepy_flow_pilot import lines_to_segments, porepy_package_status
from utils.config_loader import load_config
from utils.config_validation import validate_config


def test_lines_to_segments_simplifies_filters_and_deduplicates():
    line = LineString([(0, 0), (0.1, 0.0), (2.0, 0.0)])
    reverse = LineString([(2.0, 0.0), (0.0, 0.0)])
    segments = lines_to_segments(
        [line, reverse], simplify_tolerance_m=0.2, minimum_segment_length_m=0.5
    )
    assert len(segments) == 1
    assert segments[0] == pytest.approx([0.0, 0.0, 2.0, 0.0])


def test_lines_to_segments_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        lines_to_segments([], simplify_tolerance_m=-1, minimum_segment_length_m=1)


def test_porepy_package_check_has_stable_contract():
    status = porepy_package_status()
    assert set(status) == {"available", "version", "error"}
    assert isinstance(status["available"], bool)


def test_p2_p3_default_config_validates_and_rejects_bad_ratio():
    config = load_config(str(Path(__file__).parents[1] / "program" / "config.yaml"))
    assert validate_config(config) == []
    bad = deepcopy(config)
    bad["physics_pilot"]["fracture_permeability_ratios"] = [100.0, 0.0]
    assert any("fracture_permeability_ratios" in error for error in validate_config(bad))
