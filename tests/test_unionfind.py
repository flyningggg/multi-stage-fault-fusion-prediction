# -*- coding: utf-8 -*-
"""percolation._UnionFind 并查集正确性测试。"""
from percolation import _UnionFind


def test_init_all_singletons():
    uf = _UnionFind(5)
    assert uf.max_component_size() == 1
    for i in range(5):
        assert uf.find(i) == i


def test_union_grows_component():
    uf = _UnionFind(5)
    uf.union(0, 1)
    assert uf.max_component_size() == 2
    uf.union(1, 2)
    assert uf.max_component_size() == 3


def test_union_idempotent():
    uf = _UnionFind(5)
    uf.union(0, 1)
    assert uf.max_component_size() == 2
    uf.union(0, 1)  # 再次 union 同一对，不应增长
    uf.union(1, 0)
    assert uf.max_component_size() == 2


def test_path_compression():
    uf = _UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(2, 3)
    # 3 的根应与 0 的根相同
    assert uf.find(3) == uf.find(0)
    # 触发 find 后再查应稳定
    assert uf.find(3) == uf.find(0)
    assert uf.max_component_size() == 4


def test_disjoint_groups():
    uf = _UnionFind(5)
    uf.union(0, 1)
    uf.union(2, 3)
    # 两个独立分量，最大为 2，不是 4
    assert uf.max_component_size() == 2
    assert uf.find(0) != uf.find(2)


def test_empty():
    uf = _UnionFind(0)
    assert uf.max_component_size() == 0
