# -*- coding: utf-8 -*-
"""测试合成数据 fixture。
生成规则网格 GeoDataFrame，供 percolation 与 overlay 测试复用。
不依赖任何真实数据（KB11/THK/MY）。"""
import pytest
import geopandas as gpd
from shapely.geometry import box


def make_grid_gdf(n_cols=3, n_rows=3, step=3000.0,
                  weight_col="NC_A", weights=None, crs="EPSG:32645",
                  include_vertex_cols=True):
    """规则网格 GeoDataFrame。

    - centroid 在 (col*step, row*step)，row-major 排序
      （行索引 i = row*n_cols + col，与 build_grid_graph 的节点 id 一致）
    - 每个单元是 centroid 周围的方形（半边长 100m，远小于 step，几何不重叠）
    - 包含 weight_col（percolation 用）
    - include_vertex_cols=True 时附加 vertex1_x/vertex1_y（overlay 用）

    参数：
      weights: 可选 list/dict。list 时按行索引赋权；dict 时 {行索引: 权重}。
               None 时权重全为 1.0。
    """
    half = 100.0  # 半边长，远小于 step
    rows = []
    for row in range(n_rows):
        for col in range(n_cols):
            i = row * n_cols + col
            cx, cy = col * step, row * step
            geom = box(cx - half, cy - half, cx + half, cy + half)
            if weights is None:
                w = 1.0
            elif isinstance(weights, dict):
                w = float(weights.get(i, 1.0))
            else:
                w = float(weights[i])
            rec = {"geometry": geom, weight_col: w}
            if include_vertex_cols:
                rec["vertex1_x"] = cx
                rec["vertex1_y"] = cy
            rows.append(rec)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return gdf


@pytest.fixture(params=[(3, 3), (4, 2), (2, 2)])
def grid_gdf(request):
    """参数化规则网格 fixture：3x3 / 4x2 / 2x2。"""
    n_cols, n_rows = request.param
    return make_grid_gdf(n_cols=n_cols, n_rows=n_rows)
