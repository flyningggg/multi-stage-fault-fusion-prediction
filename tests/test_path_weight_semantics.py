# -*- coding: utf-8 -*-
"""容量与最短路径阻抗的语义守护测试。"""
import networkx as nx
import numpy as np
import pytest

from percolation import (
    build_grid_graph,
    capacity_to_distance,
    identify_key_nodes,
    simulate_percolation,
)
from tests.conftest import make_grid_gdf


def test_capacity_to_distance_is_positive_and_monotone():
    capacities = np.array([0.0, 0.1, 1.0, 10.0])
    distances = capacity_to_distance(capacities)

    assert np.isfinite(distances).all()
    assert (distances > 0).all()
    assert np.all(np.diff(distances) < 0), "连通能力越大，路径阻抗应越小"


def test_grid_edges_store_capacity_and_distance_without_breaking_weight_alias():
    gdf = make_grid_gdf(2, 1, weights={0: 0.2, 1: 0.8})
    graph = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    edge = graph[0][1]

    assert edge["capacity"] == pytest.approx(0.2)
    assert edge["weight"] == pytest.approx(edge["capacity"])
    assert edge["distance"] == pytest.approx(1.0 / (0.2 + 1e-12))


def test_betweenness_prefers_high_capacity_low_resistance_route():
    graph = nx.Graph()
    for node in range(4):
        graph.add_node(node, pos=(float(node), 0.0), weight=1.0)

    # 0→3 有两条二跳路径：经1的容量高，经2的容量低。
    # 正确的最短路径语义应选择节点1。
    for u, v, capacity in [
        (0, 1, 10.0),
        (1, 3, 10.0),
        (0, 2, 1.0),
        (2, 3, 1.0),
    ]:
        graph.add_edge(
            u,
            v,
            weight=capacity,
            capacity=capacity,
            distance=float(capacity_to_distance(capacity)),
        )

    result = identify_key_nodes(graph, top_n=4, k_sample=4, use_pagerank=False)
    centrality = result.set_index("node_idx")["centrality"].to_dict()
    assert centrality[1] > centrality[2]


def test_percolation_accepts_explicit_capacity_without_distance_confusion():
    graph = nx.Graph()
    graph.add_nodes_from(range(3))
    graph.add_edge(0, 1, capacity=0.1, distance=10.0)
    graph.add_edge(1, 2, capacity=1.0, distance=1.0)

    fractions, sizes, threshold = simulate_percolation(graph, n_steps=10)
    assert len(fractions) == len(sizes) == 11
    assert np.isfinite(sizes).all()
    assert 0.0 <= threshold <= 1.0
