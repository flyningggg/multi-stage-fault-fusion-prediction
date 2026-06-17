# -*- coding: utf-8 -*-
"""S3 守护：build_grid_graph 节点-索引不变量。
钉住「节点 id == gdf 行索引」假设，及 4 邻接、边权 min/mean 语义。"""
import pandas as pd
import pytest
from percolation import build_grid_graph
from tests.conftest import make_grid_gdf


def test_node_count():
    gdf = make_grid_gdf(3, 3)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    assert G.number_of_nodes() == 9


def test_edge_count_4adj():
    # 3x3 网格 4 邻接：水平 3*2 + 垂直 2*3 = 12 条边（无对角）
    gdf = make_grid_gdf(3, 3)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    assert G.number_of_edges() == 12


def test_node_id_matches_row_index():
    # S3 核心：节点 i 的 pos 必须等于 gdf 第 i 行的 centroid
    gdf = make_grid_gdf(3, 3)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    centroids = gdf.geometry.centroid
    for i in range(len(gdf)):
        assert i in G.nodes, f"节点 {i} 不在图中"
        pos = G.nodes[i]["pos"]
        c = centroids.iloc[i]
        assert pos[0] == pytest.approx(c.x, abs=1e-6), f"节点 {i} 的 pos_x 不匹配"
        assert pos[1] == pytest.approx(c.y, abs=1e-6), f"节点 {i} 的 pos_y 不匹配"


def test_no_diagonal_edges():
    # 节点 0（左上角）的邻居只能是 {1（右）, 3（下）}，不含对角 4
    gdf = make_grid_gdf(3, 3)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    neighbors = set(G.neighbors(0))
    assert neighbors == {1, 3}, f"节点 0 邻居 {neighbors} 含对角或缺失"


def test_degree_sequence():
    # 3x3：角(0,2,6,8)度=2，边(1,3,5,7)度=3，中心(4)度=4
    gdf = make_grid_gdf(3, 3)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    deg = dict(G.degree())
    for corner in [0, 2, 6, 8]:
        assert deg[corner] == 2, f"角节点 {corner} 度应为 2，实际 {deg[corner]}"
    for edge in [1, 3, 5, 7]:
        assert deg[edge] == 3, f"边节点 {edge} 度应为 3，实际 {deg[edge]}"
    assert deg[4] == 4, f"中心节点 4 度应为 4，实际 {deg[4]}"


def test_weight_min_mode():
    # 节点 0 权=0.1，节点 1 权=0.9 → 边(0,1) 权=min=0.1
    gdf = make_grid_gdf(3, 3, weights={0: 0.1, 1: 0.9})
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    assert G[0][1]["weight"] == pytest.approx(0.1)


def test_weight_mean_mode():
    # 同上但 mean → 边(0,1) 权=(0.1+0.9)/2=0.5
    gdf = make_grid_gdf(3, 3, weights={0: 0.1, 1: 0.9})
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="mean")
    assert G[0][1]["weight"] == pytest.approx(0.5)


def test_isolated_node_exists():
    import geopandas as gpd
    from shapely.geometry import box
    base = make_grid_gdf(2, 2)  # 4 格在 (0..3000, 0..3000)
    far_box = box(50000 - 100, 50000 - 100, 50000 + 100, 50000 + 100)
    far = gpd.GeoDataFrame([{"geometry": far_box, "NC_A": 1.0}], crs="EPSG:32645")
    gdf = gpd.GeoDataFrame(pd.concat([base, far], ignore_index=True), crs="EPSG:32645")
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    assert G.number_of_nodes() == 5
    # 远离格（最后一行，索引 4）度=0
    assert dict(G.degree())[4] == 0
