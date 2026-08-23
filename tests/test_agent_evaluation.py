# -*- coding: utf-8 -*-
"""代理模型诚实评估测试：空间块 CV 与留一期外推的纯逻辑验证。
合成数据 + 小树参数，不依赖真实数据与真实 betweenness 计算。"""
import numpy as np
import pandas as pd
import pytest

from agent_model import (
    AGENT_EXCLUDE_COLS,
    _build_block_ids,
    _feature_cols,
    _lopo_split,
    leave_one_period_out_evaluate,
    spatial_cv_evaluate,
)

FAST_XGB = {"n_estimators": 10, "max_depth": 2, "verbosity": 0}


def make_synth_df(n_per_period=40, periods=("A", "B", "C"), seed=0):
    """合成训练 df：3 期 × n 网格，特征 f1/f2 + 元数据列。
    y 与 f1 相关且各期基线不同（LOPO 有信号可学）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for pi, p in enumerate(periods):
        for _ in range(n_per_period):
            x = float(rng.integers(0, 5)) * 3000.0
            y = float(rng.integers(0, 5)) * 3000.0
            f1 = rng.normal(loc=pi)
            rows.append({
                "f1": f1,
                "f2": rng.normal(),
                "cell_x": x,
                "cell_y": y,
                "period": p,
                "betweenness": abs(f1) + 1.0,
                "log1p_betweenness": float(np.log1p(abs(f1) + 1.0)),
            })
    return pd.DataFrame(rows)


def test_feature_cols_excludes_metadata():
    assert {"cell_x", "cell_y", "period", "betweenness",
            "log1p_betweenness", "log_betweenness"} <= set(AGENT_EXCLUDE_COLS)
    df = make_synth_df()
    feats = _feature_cols(df)
    assert feats == ["f1", "f2"]


def _grid_xy(n_side=3, step=3000.0):
    xs = np.repeat(np.arange(n_side) * step, n_side)
    ys = np.tile(np.arange(n_side) * step, n_side)
    return xs, ys


def test_build_block_ids_grid_layout():
    xs, ys = _grid_xy(3)
    ids = _build_block_ids(xs, ys, n_blocks=9)
    assert ids is not None
    assert len(ids) == 9
    assert len(set(ids.tolist())) == 9          # 每格一块
    assert set(ids.tolist()) <= set(range(9))   # id 在 [0, n_side²)


def test_build_block_ids_deterministic():
    xs, ys = _grid_xy(3)
    assert np.array_equal(_build_block_ids(xs, ys, 9), _build_block_ids(xs, ys, 9))


def test_build_block_ids_degenerate_returns_none():
    same_x = np.full(6, 100.0)
    same_y = np.full(6, 200.0)
    assert _build_block_ids(same_x, same_y, 9) is None  # 无空间差异 → 无法分块


def test_spatial_cv_returns_aggregates():
    df = make_synth_df()
    out = spatial_cv_evaluate(df, n_blocks=9, n_splits=3, xgb_params=FAST_XGB)
    for key in ["r2_mean", "r2_std", "rmse_mean", "rmse_std",
                "mae_mean", "mae_std", "n_blocks_used", "n_splits_used"]:
        assert key in out, f"缺少 {key}"
        assert np.isfinite(out[key]), f"{key} 非有限值: {out[key]}"
    assert out["n_blocks_used"] >= 2
    assert 2 <= out["n_splits_used"] <= 3
    assert -1.0 <= out["r2_mean"] <= 1.0


def test_lopo_split_disjoint_and_covers():
    splits = _lopo_split(["A", "B", "C"])
    assert len(splits) == 3
    for train_ps, test_p in splits:
        assert test_p not in train_ps
        assert sorted(train_ps + [test_p]) == ["A", "B", "C"]
    # 每个时期恰好被留出一次
    assert sorted(t for _, t in splits) == ["A", "B", "C"]


def test_lopo_evaluate_covers_all_periods():
    df = make_synth_df()
    out = leave_one_period_out_evaluate(df, xgb_params=FAST_XGB)
    assert set(out["per_period"].keys()) == {"A", "B", "C"}
    for m in out["per_period"].values():
        for key in ["r2", "rmse", "mae", "n_test"]:
            assert key in m and np.isfinite(m[key])
        assert m["n_test"] == 40
        assert set(["spearman", "kendall", "top_20pct_overlap"]) <= set(m["ranking"])
    assert np.isfinite(out["r2_mean"]) and np.isfinite(out["r2_std"])
    assert np.isfinite(out["ranking_aggregate"]["spearman_mean"])
