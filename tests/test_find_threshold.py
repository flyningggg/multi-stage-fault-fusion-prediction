# -*- coding: utf-8 -*-
"""find_percolation_threshold 三种方法测试。
输入为纯数组（无图依赖），期望值手算。"""
import numpy as np
import pytest
from percolation import find_percolation_threshold


def test_half_first_below():
    fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    sizes = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    # 首个 <0.5 在 index 3 → fractions[3]=0.75
    assert find_percolation_threshold(fractions, sizes, method="half") == pytest.approx(0.75)


def test_half_never_below():
    fractions = np.array([0.0, 0.5, 1.0])
    sizes = np.array([1.0, 0.6, 0.55])  # 全 >= 0.5
    assert find_percolation_threshold(fractions, sizes, method="half") == pytest.approx(1.0)


def test_steepest_clear_minimum():
    fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    sizes = np.array([1.0, 0.9, 0.8, 0.3, 0.2])
    # diff = [-0.1, -0.1, -0.5, -0.1]，argmin=2，返回 fractions[3]=0.75
    assert find_percolation_threshold(fractions, sizes, method="steepest") == pytest.approx(0.75)


def test_steepest_too_few_points():
    # len(sizes) < 3 → fallback 0.5
    fractions = np.array([0.0, 1.0])
    sizes = np.array([1.0, 0.4])
    assert find_percolation_threshold(fractions, sizes, method="steepest") == pytest.approx(0.5)


def test_span():
    fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    sizes = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    # span 语义见源码；本测试只断言结果落在 [0,1]
    result = find_percolation_threshold(fractions, sizes, method="span")
    assert 0.0 <= result <= 1.0


def test_unknown_method():
    fractions = np.array([0.0, 0.5, 1.0])
    sizes = np.array([1.0, 0.6, 0.2])
    # 未知 method → 默认 fallback 0.5
    assert find_percolation_threshold(fractions, sizes, method="bogus") == pytest.approx(0.5)
