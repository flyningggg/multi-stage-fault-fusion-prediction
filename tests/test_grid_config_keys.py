# -*- coding: utf-8 -*-
"""grid 配置段键契约测试：钉住 config.yaml 与代码消费端的字段名契约。"""
import os
import yaml

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "program", "config.yaml",
)

# 五个键及当前行为等价的默认值——代码侧回退值必须与此一致
EXPECTED_KEYS = {
    "step_m": 3000.0,
    "edge_dist_tolerance_m": 150.0,
    "centroid_match_tolerance_m": 1500.0,
    "target_eps_m": 4500.0,
    "target_min_cluster_size": 3,
}


def _grid_cfg():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    assert isinstance(cfg.get("grid"), dict), "config.yaml 缺少 grid 段"
    return cfg["grid"]


def test_grid_section_has_all_keys():
    grid = _grid_cfg()
    for key, value in EXPECTED_KEYS.items():
        assert key in grid, f"grid 段缺少键 {key}"
        assert float(grid[key]) == float(value), f"grid.{key} 默认值应为 {value}"


def test_grid_values_are_numbers():
    grid = _grid_cfg()
    for key in EXPECTED_KEYS:
        assert isinstance(grid[key], (int, float)), f"grid.{key} 必须是数值"
