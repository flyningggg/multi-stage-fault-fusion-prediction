# -*- coding: utf-8 -*-
"""代理模型诚实评估测试：空间块 CV 与留一期外推的纯逻辑验证。
合成数据 + 小树参数，不依赖真实数据与真实 betweenness 计算。"""
import numpy as np
import pandas as pd
import pytest

from agent_model import (
    AGENT_EXCLUDE_COLS,
    _feature_cols,
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
