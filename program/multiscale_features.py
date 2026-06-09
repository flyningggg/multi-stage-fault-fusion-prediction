# -*- coding: utf-8 -*-
"""
多尺度空间特征（金字塔式融合）。

核心思想：
- 在细网格层面已有每个网格单元的空间–拓扑特征；
- 通过在更粗的尺度上做 pooling（平均 / 最大值等），构建多尺度特征金字塔：
  - 层 1：细网格（原始分辨率）
  - 层 2：中尺度（例如 2x2 邻域平均）
  - 层 3：大尺度（例如 4x4 邻域平均）
- 最终把各尺度特征拼接（concat）形成多尺度融合特征。

假设：
- 数据对应一个规则网格，有 n_rows * n_cols 个网格；
- 输入为特征矩阵 X (n_samples, n_features)，样本顺序为行优先（row-major）：
  第 r 行第 c 列的索引为 idx = r * n_cols + c。
"""

from typing import List, Tuple

import numpy as np


def _pool_level(
    X: np.ndarray,
    n_rows: int,
    n_cols: int,
    block_size: int,
    mode: str = "mean",
) -> np.ndarray:
    """
    对给定尺度 block_size 进行 pooling，并将结果映射回原始网格尺寸。
    """
    n_samples, n_feat = X.shape
    if n_samples != n_rows * n_cols:
        raise ValueError("样本数必须等于 n_rows * n_cols。")
    X_grid = X.reshape(n_rows, n_cols, n_feat)
    pooled = np.zeros_like(X_grid)
    for r in range(0, n_rows, block_size):
        for c in range(0, n_cols, block_size):
            r_end = min(r + block_size, n_rows)
            c_end = min(c + block_size, n_cols)
            block = X_grid[r:r_end, c:c_end, :]
            if mode == "max":
                val = block.max(axis=(0, 1))
            else:
                val = block.mean(axis=(0, 1))
            pooled[r:r_end, c:c_end, :] = val
    return pooled.reshape(n_samples, n_feat)


def build_multiscale_pyramid(
    X: np.ndarray,
    n_rows: int,
    n_cols: int,
    scales: List[int],
    mode: str = "mean",
) -> Tuple[np.ndarray, List[str]]:
    """
    构建多尺度特征金字塔，并按尺度拼接。

    参数：
        X: 原始细网格特征 (n_samples, n_features)
        n_rows, n_cols: 网格行列数
        scales: 如 [1, 2, 4]，1 表示原始尺度，2 表示 2x2 pooling，4 表示 4x4 pooling
        mode: mean | max

    返回：
        X_ms: 拼接后的多尺度特征 (n_samples, n_features * len(scales))
        suffixes: 对应每个尺度的后缀（便于生成列名）
    """
    scales = sorted(set(scales))
    feats = []
    suffixes = []
    for s in scales:
        if s <= 1:
            feats.append(X)
            suffixes.append("s1")
        else:
            feats.append(_pool_level(X, n_rows, n_cols, block_size=s, mode=mode))
            suffixes.append(f"s{s}")
    X_ms = np.concatenate(feats, axis=1)
    return X_ms, suffixes

