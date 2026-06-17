# -*- coding: utf-8 -*-
"""S4 守护：simulate_percolation 渗流曲线数学性质。
钉住逆序 Union-Find 的 sizes_reverse/sizes_forward 对齐正确性。"""
import numpy as np
import networkx as nx
import pytest
from percolation import simulate_percolation


def _full_graph(n):
    """全连通图：n 节点两两相连，所有边权相同。"""
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=1.0)
    return G


def _line_graph(n):
    """线图 0-1-2-...-(n-1)，边权随 index 递增（保证排序稳定）。"""
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n - 1):
        G.add_edge(i, i + 1, weight=float(i + 1))
    return G


def test_array_lengths():
    G = _full_graph(9)
    fractions, sizes, threshold = simulate_percolation(G, n_steps=100)
    assert len(fractions) == 101
    assert len(sizes) == 101


def test_boundary_f0_full_graph():
    # f=0（无删边）：最大分量 = 全部节点 → 占比 1.0
    G = _full_graph(9)
    fractions, sizes, threshold = simulate_percolation(G, n_steps=100)
    assert sizes[0] == pytest.approx(1.0)


def test_boundary_f1_isolated():
    # f=1（全删边）：最大分量 = 1 个节点 → 占比 1/N
    G = _full_graph(9)
    fractions, sizes, threshold = simulate_percolation(G, n_steps=100)
    assert sizes[-1] == pytest.approx(1.0 / 9)


def test_monotonic_non_increasing():
    # S4 核心：渗流曲线必须单调非增
    G = _full_graph(9)
    fractions, sizes, threshold = simulate_percolation(G, n_steps=100)
    diffs = np.diff(sizes)
    assert np.all(diffs <= 1e-9), f"曲线非单调非增: max diff = {diffs.max()}"


def test_threshold_in_unit_interval():
    G = _full_graph(9)
    fractions, sizes, threshold = simulate_percolation(G, n_steps=100)
    assert 0.0 <= threshold <= 1.0


def test_line_graph_exact_curve():
    # 手算：线图 0-1-2-3（3 条边），n_steps=3 → x_interp == x_exact
    # sizes_forward = [4/4, 3/4, 2/4, 1/4] = [1.0, 0.75, 0.5, 0.25]
    G = _line_graph(4)  # 3 条边
    fractions, sizes, threshold = simulate_percolation(G, n_steps=3)
    assert fractions == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert sizes == pytest.approx([1.0, 0.75, 0.5, 0.25])


def test_empty_graph():
    # 0 边图：返回常数 1/N 曲线，threshold=0.0，不抛异常
    G = nx.Graph()
    for i in range(5):
        G.add_node(i)  # 无边
    fractions, sizes, threshold = simulate_percolation(G, n_steps=100)
    assert len(fractions) == 101
    assert len(sizes) == 101
    assert threshold == 0.0
    # 全孤立，最大分量恒为 1/N
    assert sizes[0] == pytest.approx(1.0 / 5)
    assert sizes[-1] == pytest.approx(1.0 / 5)
