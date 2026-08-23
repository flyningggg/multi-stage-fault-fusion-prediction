# -*- coding: utf-8 -*-
"""高值排序评估的确定性与边界测试。"""
import numpy as np
import pytest

from agent_model import ranking_metrics


def test_ranking_metrics_perfect_order_is_one():
    y = np.arange(1.0, 21.0)
    out = ranking_metrics(y, y.copy())
    assert out["status"] == "ok"
    for key in [
        "spearman", "kendall", "top_5pct_overlap", "top_10pct_overlap",
        "top_20pct_overlap", "ndcg_5pct", "ndcg_10pct", "ndcg_20pct",
    ]:
        assert out[key] == pytest.approx(1.0)


def test_ranking_metrics_reversed_order_exposes_failure():
    y = np.arange(1.0, 21.0)
    out = ranking_metrics(y, y[::-1])
    assert out["spearman"] == pytest.approx(-1.0)
    assert out["kendall"] == pytest.approx(-1.0)
    assert out["top_20pct_overlap"] == pytest.approx(0.0)
    assert out["ndcg_20pct"] < 0.2


def test_ranking_metrics_top_k_sizes_and_ties_are_deterministic():
    y_true = np.array([3.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    y_pred = np.array([3.0, 2.9, 2.1, 1.0, 0.0, 0.0, 0.0])
    first = ranking_metrics(y_true, y_pred)
    second = ranking_metrics(y_true, y_pred)
    assert first == second
    assert first["top_5pct_k"] == 1
    assert first["top_10pct_k"] == 1
    assert first["top_20pct_k"] == 2


@pytest.mark.parametrize(
    "y_true,y_pred",
    [([1.0], [1.0]), ([1.0, 2.0], [1.0]), ([1.0, np.nan], [1.0, 2.0])],
)
def test_ranking_metrics_rejects_invalid_input(y_true, y_pred):
    with pytest.raises(ValueError):
        ranking_metrics(y_true, y_pred)
