# -*- coding: utf-8 -*-
"""S3 第二层守护：identify_key_nodes 返回的 node_idx 合法性。
只测结构不变量，不测 centrality 数值（networkx 版本敏感）。"""
import pandas as pd
import pytest
import networkx as nx
from percolation import build_grid_graph, identify_key_nodes
from tests.conftest import make_grid_gdf


def test_returns_dataframe_with_expected_columns():
    gdf = make_grid_gdf(3, 3)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    result = identify_key_nodes(G, top_n=5)
    assert isinstance(result, pd.DataFrame)
    for col in ["node_idx", "centrality", "pos_x", "pos_y", "weight"]:
        assert col in result.columns, f"缺少列 {col}"


def test_node_idx_valid_row_index():
    # S3 第二层：所有 node_idx 必须是合法的 gdf 行索引
    gdf = make_grid_gdf(3, 3)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    result = identify_key_nodes(G, top_n=10)
    n = len(gdf)
    for idx in result["node_idx"]:
        assert 0 <= idx < n, f"node_idx {idx} 越界（应在 [0, {n})）"


def test_pos_matches_gdf_centroid():
    # S3 第二层：返回行的 pos 必须等于对应 gdf 行的 centroid
    gdf = make_grid_gdf(3, 3)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    result = identify_key_nodes(G, top_n=9)
    centroids = gdf.geometry.centroid
    for _, row in result.iterrows():
        c = centroids.iloc[int(row["node_idx"])]
        assert row["pos_x"] == pytest.approx(c.x, abs=1e-6)
        assert row["pos_y"] == pytest.approx(c.y, abs=1e-6)


def test_top_n_limit():
    gdf = make_grid_gdf(3, 3)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    result = identify_key_nodes(G, top_n=3)
    assert len(result) <= 3


def test_exclude_boundary_zeroes_centrality():
    # exclude_boundary=True 时，边界节点 centrality 被置 0；
    # 用 5x5 网格（内部有 9 个非边界节点），pagerank 模式（值稳定非零），
    # top_n=5 应只返回内部非边界节点。
    # 5x5 内部节点 = 行1-3 × 列1-3 的索引：6,7,8,11,12,13,16,17,18
    gdf = make_grid_gdf(5, 5)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    result = identify_key_nodes(G, top_n=5, exclude_boundary=True, gdf=gdf,
                                use_pagerank=True)
    # 5x5 网格边界节点 = x∈{0,12000} 或 y∈{0,12000} 的节点（外围一圈）
    # 内部节点索引：行1-3(r=1,2,3) × 列1-3(c=1,2,3) → i=r*5+c = 6,7,8,11,12,13,16,17,18
    interior = {6, 7, 8, 11, 12, 13, 16, 17, 18}
    returned = set(result["node_idx"])
    # 返回的应全是内部节点
    assert returned.issubset(interior), f"含边界节点: {returned - interior}"


def test_empty_graph_returns_empty_df():
    G = nx.Graph()
    G.add_node(0)  # 无边
    result = identify_key_nodes(G, top_n=5)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_pagerank_vs_betweenness_both_run():
    gdf = make_grid_gdf(3, 3)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    pr = identify_key_nodes(G, top_n=5, use_pagerank=True)
    bt = identify_key_nodes(G, top_n=5, use_pagerank=False)
    # 两种模式都返回合法 DataFrame（不比较数值）
    assert isinstance(pr, pd.DataFrame) and len(pr) > 0
    assert isinstance(bt, pd.DataFrame) and len(bt) > 0
