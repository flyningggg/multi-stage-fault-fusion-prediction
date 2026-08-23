# -*- coding: utf-8 -*-
"""Stage 1 XGBoost 空间分块诚实评估合同。"""
import numpy as np
import pandas as pd

from batch_run import _build_spatial_block_ids, _spatial_cv_xgboost


def test_build_spatial_block_ids_is_deterministic_for_grid():
    xs = np.tile(np.arange(3, dtype=float), 3)
    ys = np.repeat(np.arange(3, dtype=float), 3)

    first = _build_spatial_block_ids(xs, ys, n_blocks=9)
    second = _build_spatial_block_ids(xs, ys, n_blocks=9)

    assert first is not None
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 9


def test_build_spatial_block_ids_rejects_degenerate_coordinates():
    groups = _build_spatial_block_ids(np.zeros(12), np.zeros(12), n_blocks=9)
    assert groups is None


def test_spatial_cv_reports_complete_finite_metrics():
    rng = np.random.default_rng(42)
    rows, cols = np.indices((6, 6))
    xs = cols.ravel().astype(float) * 3000.0
    ys = rows.ravel().astype(float) * 3000.0
    X = np.column_stack([
        xs / 3000.0,
        ys / 3000.0,
        rng.normal(size=xs.size),
    ])
    y = 0.7 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * X[:, 2]

    result = _spatial_cv_xgboost(
        X,
        y,
        xs,
        ys,
        n_blocks=9,
        n_splits=3,
        model_params={
            "n_estimators": 20,
            "max_depth": 2,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": 1,
        },
    )

    assert result["status"] == "ok"
    assert result["n_splits_used"] == 3
    for key in (
        "r2_mean",
        "r2_std",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
    ):
        assert np.isfinite(result[key]), key


def test_spatial_cv_returns_explicit_status_for_degenerate_coordinates():
    X = np.arange(60, dtype=float).reshape(20, 3)
    y = np.arange(20, dtype=float)
    result = _spatial_cv_xgboost(X, y, np.zeros(20), np.zeros(20))

    assert result["status"] == "insufficient_spatial_blocks"
    assert result["n_splits_used"] == 0


def test_spatial_cv_returns_explicit_status_for_constant_target():
    rows, cols = np.indices((4, 4))
    xs = cols.ravel().astype(float)
    ys = rows.ravel().astype(float)
    X = np.column_stack([xs, ys])
    result = _spatial_cv_xgboost(X, np.ones(len(X)), xs, ys)

    assert result["status"] == "constant_target"
    assert result["n_splits_used"] == 0
