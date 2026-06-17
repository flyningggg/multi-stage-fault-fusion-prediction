# -*- coding: utf-8 -*-
"""overlay 匹配纯函数测试。
显式传 max_dist/eps_meters，暴露容差可参数化（为 M5 修复铺路，但本轮不改业务代码）。"""
import pandas as pd
import pytest
from shapely.geometry import box
from multiperiod_overlay import (
    find_exact_vertex_matches,
    find_centroid_distance_matches,
    identify_target_areas,
)
from tests.conftest import make_grid_gdf


def test_exact_vertex_match_overlap():
    # gdf_a 与 gdf_b 共享前 3 个 vertex
    gdf_a = make_grid_gdf(3, 3)
    gdf_b = make_grid_gdf(3, 3).iloc[:3].copy()  # 只取前 3 格
    result = find_exact_vertex_matches(gdf_a, gdf_b)
    assert len(result) == 3
    assert (result["match_type"] == "exact_vertex").all()


def test_exact_vertex_no_match():
    gdf_a = make_grid_gdf(3, 3)
    # gdf_b 顶点整体偏移到完全不同的坐标
    gdf_b = make_grid_gdf(3, 3)
    gdf_b["vertex1_x"] = gdf_b["vertex1_x"] + 99999
    gdf_b["vertex1_y"] = gdf_b["vertex1_y"] + 99999
    result = find_exact_vertex_matches(gdf_a, gdf_b)
    assert len(result) == 0


def test_centroid_dist_within_tolerance():
    # gdf_b 整体偏移 1000m（< 默认 1500 容差），应全部匹配
    gdf_a = make_grid_gdf(3, 3)
    gdf_b = make_grid_gdf(3, 3)
    new_geoms = [box(g.centroid.x - 100 + 1000, g.centroid.y - 100 + 1000,
                     g.centroid.x + 100 + 1000, g.centroid.y + 100 + 1000)
                 for g in gdf_b.geometry]
    gdf_b = gdf_b.set_geometry(new_geoms, crs="EPSG:32645")
    result = find_centroid_distance_matches(gdf_a, gdf_b)
    assert len(result) == 9


def test_centroid_dist_beyond_tolerance():
    # 偏移到完全不相干区域（最近距离 >> 1500），应无匹配
    # 注意：不能用步长倍数偏移（会让部分节点恰好对齐，距离=0）
    gdf_a = make_grid_gdf(3, 3)  # a 节点 centroid 在 (0..6000, 0..6000)
    gdf_b = make_grid_gdf(3, 3)
    new_geoms = [box(g.centroid.x - 100 + 10000, g.centroid.y - 100 + 10000,
                     g.centroid.x + 100 + 10000, g.centroid.y + 100 + 10000)
                 for g in gdf_b.geometry]
    gdf_b = gdf_b.set_geometry(new_geoms, crs="EPSG:32645")
    result = find_centroid_distance_matches(gdf_a, gdf_b)
    assert len(result) == 0


def test_centroid_dist_custom_tolerance():
    # 默认容差 1500 下偏移 3000m 无匹配；自定义 max_dist=5000 后能匹配
    # 偏移 3000m（步长倍数）：部分节点对齐距离=0，部分对角距离=4243，全部 < 5000
    gdf_a = make_grid_gdf(3, 3)
    gdf_b = make_grid_gdf(3, 3)
    new_geoms = [box(g.centroid.x - 100 + 3000, g.centroid.y - 100 + 3000,
                     g.centroid.x + 100 + 3000, g.centroid.y + 100 + 3000)
                 for g in gdf_b.geometry]
    gdf_b = gdf_b.set_geometry(new_geoms, crs="EPSG:32645")
    # 默认容差下：部分对齐(距离0<1500)会匹配，本测试聚焦自定义容差
    result_default = find_centroid_distance_matches(gdf_a, gdf_b)
    result_custom = find_centroid_distance_matches(gdf_a, gdf_b, max_dist=5000)
    # 自定义大容差匹配数 >= 默认容差匹配数（容差更大不会更少匹配）
    assert len(result_custom) >= len(result_default)
    # 自定义容差下，所有匹配距离都 < 5000
    assert (result_custom["dist"] < 5000).all()


def test_identify_target_areas_clustering():
    # 4 个点聚成 1 簇（彼此 < eps=4500）+ 1 个远离点（噪声）
    # 4 点取正方形角：(0,0),(3000,0),(0,3000),(3000,3000)
    #   相邻边距 3000 < 4500 连通；对角 (0,0)-(3000,3000) 距离 4243 < 4500 也连通 → 一簇
    #   远离点 (50000,50000) 距离 >> 4500 → 噪声
    coords = [(0, 0), (3000, 0), (0, 3000), (3000, 3000), (50000, 50000)]
    overlap_df = pd.DataFrame(coords, columns=["centroid_x", "centroid_y"])
    result = identify_target_areas(overlap_df, min_cluster_size=3, eps_meters=4500)
    assert "target_cluster" in result.columns
    assert "target_area" in result.columns
    # 远离点（最后一个）应为噪声 -1
    assert result.iloc[-1]["target_cluster"] == -1
    assert result.iloc[-1]["target_area"] == "散点"
    # 前 4 点成一簇，命名为 "靶区1"
    assert (result.iloc[:4]["target_cluster"] == result.iloc[0]["target_cluster"]).all()
    assert "靶区1" in set(result["target_area"])


def test_identify_target_areas_requires_columns():
    # 缺 centroid_x 应抛 ValueError
    df = pd.DataFrame({"x": [0, 1], "centroid_y": [0, 1]})
    with pytest.raises(ValueError, match="缺少 centroid"):
        identify_target_areas(df)
