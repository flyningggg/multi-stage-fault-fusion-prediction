# -*- coding: utf-8 -*-
"""_identify_boundary_nodes 边界识别测试。
钉住：默认行为不变；grid_step 显式传参与常量一致；参数真实生效。"""
from percolation import GRID_STEP, _identify_boundary_nodes, build_grid_graph
from tests.conftest import make_grid_gdf


def test_default_finds_border_cells():
    gdf = make_grid_gdf(3, 3)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    b = _identify_boundary_nodes(G, gdf)
    assert b == {0, 1, 2, 3, 5, 6, 7, 8}   # 除中心 4 外全是边界


def test_explicit_grid_step_matches_constant_default():
    gdf = make_grid_gdf(3, 3)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    b_const = _identify_boundary_nodes(G, gdf)
    b_explicit = _identify_boundary_nodes(G, gdf, grid_step=GRID_STEP)
    assert b_const == b_explicit == {0, 1, 2, 3, 5, 6, 7, 8}


def test_larger_step_widens_boundary_band():
    # 5x5 网格：grid_step 放大 3 倍 → tol=step/2 放大 → 更靠内的节点也算边界
    gdf = make_grid_gdf(5, 5)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    tight = _identify_boundary_nodes(G, gdf, grid_step=3000.0)   # tol=1500
    wide = _identify_boundary_nodes(G, gdf, grid_step=9000.0)    # tol=4500
    assert len(tight) == 16          # 仅最外圈
    assert 12 not in tight           # 正中心不是边界
    assert len(wide) > len(tight)
    assert 12 not in wide            # 中心仍不是边界
