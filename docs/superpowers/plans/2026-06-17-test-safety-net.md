# 测试安全网建立 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `percolation.py` 与 `multiperiod_overlay.py` 的纯函数建立 pytest 测试网（合成数据、无 IO），钉住 S3/S4 的关键不变量，为后续修复提供回归守护。不修任何业务代码。

**Architecture:** 根目录建立 pytest 基础设施（`conftest.py` 注入 sys.path、`pyproject.toml` 配置 pytest、`requirements-dev.txt` 隔离测试依赖）。`tests/` 目录下 6 个测试模块 + 1 个合成数据 fixture。所有测试用合成 GeoDataFrame（规则网格、米制 CRS），不依赖 KB11/THK/MY 真实数据。

**Tech Stack:** Python 3.12（`C:\Users\（-1，0）\AppData\Local\Programs\Python\Python312\python.exe`）、pytest ≥8.0、pytest-cov ≥4.1、networkx 3.6.1、geopandas 1.1.3、scipy 1.17.1、scikit-learn（DBSCAN）。

**对应 Spec:** `docs/superpowers/specs/2026-06-17-test-safety-net-design.md`

---

## ⚠️ 全局约束（每个任务都必须遵守）

1. **解释器**：所有 `python` / `pytest` 命令一律用完整路径
   `C:\Users\（-1，0）\AppData\Local\Programs\Python\Python312\python.exe`
   下文记作 `PY312`。**不要用默认 `python`（那是 3.13，无项目依赖）。**
2. **不碰 `program/`**：任何任务都不允许修改 `program/` 下任何文件。`git diff --stat program/` 在整个计划结束时必须为空（AC7）。
3. **TDD 调整**：被测代码已存在，所以每个测试任务的步骤是「写测试 → 跑测试验证**通过** → 提交」（不是先红后绿）。唯一例外是 Task 8 的 AC5/AC6 验证，那里故意引入 bug 看测试变红。
4. **工作目录**：所有命令在仓库根 `D:\PycharmProjects\multi-stage-fault-fusion-prediction` 执行。
5. **未提交改动**：会话开始时 `program/main.py` 和 `program/demo.py` 已有未提交改动，**不要把它们纳入本计划的任何 commit**。每个 `git add` 都只 add 本任务新建的文件。

---

## File Structure

| 文件 | 职责 | 创建/修改 |
|---|---|---|
| `requirements-dev.txt` | 测试依赖（pytest, pytest-cov） | 创建 |
| `pyproject.toml` | pytest 配置 | 创建 |
| `conftest.py`（根） | 注入 `program/` 到 sys.path | 创建 |
| `tests/__init__.py` | 标记 tests 为包 | 创建 |
| `tests/conftest.py` | `make_grid_gdf` 合成数据 fixture | 创建 |
| `tests/test_unionfind.py` | `_UnionFind` 正确性（6 用例） | 创建 |
| `tests/test_simulate_percolation.py` | S4 守护：渗流曲线数学性质（7 用例） | 创建 |
| `tests/test_find_threshold.py` | 三种阈值方法（6 用例） | 创建 |
| `tests/test_build_grid_graph.py` | S3 守护：节点-索引不变量（8 用例） | 创建 |
| `tests/test_identify_key_nodes.py` | S3 第二层守护：node_idx 合法性（7 用例） | 创建 |
| `tests/test_overlay_matching.py` | overlay 匹配纯函数（7 用例） | 创建 |

---

## Task 1: pytest 基础设施

**Files:**
- Create: `requirements-dev.txt`
- Create: `pyproject.toml`
- Create: `conftest.py`

- [ ] **Step 1: 创建 `requirements-dev.txt`**

文件内容（完整）：
```
pytest>=8.0
pytest-cov>=4.1
```

- [ ] **Step 2: 创建 `pyproject.toml`**

文件内容（完整）：
```toml
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-ra --strict-markers"
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::UserWarning",
]
```

- [ ] **Step 3: 创建根 `conftest.py`**

文件内容（完整）：
```python
# -*- coding: utf-8 -*-
"""根 conftest：把 program/ 注入 sys.path，使 `import percolation` 等可用。
不修改任何业务模块。"""
import os
import sys

_PROG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "program")
if _PROG not in sys.path:
    sys.path.insert(0, _PROG)
```

- [ ] **Step 4: 安装测试依赖**

Run:
```cmd
PY312 -m pip install -r requirements-dev.txt
```
Expected: 成功安装 pytest 和 pytest-cov。

- [ ] **Step 5: 验证 pytest 可运行（AC1）**

Run:
```cmd
PY312 -m pytest --version
```
Expected: 输出形如 `pytest 8.x.x`，无 ImportError。

- [ ] **Step 6: 验证安装未破坏现有依赖**

Run:
```cmd
PY312 -c "import networkx, geopandas, pandas, numpy, scipy; print('deps OK')"
```
Expected: 输出 `deps OK`。若失败说明安装破坏了环境，需排查。

- [ ] **Step 7: 提交**

```cmd
git add requirements-dev.txt pyproject.toml conftest.py
git commit -m "test: 添加 pytest 基础设施（requirements-dev/pyproject/conftest）"
```

---

## Task 2: 合成数据 fixture

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: 创建 `tests/__init__.py`**

文件内容（完整，空文件用于标记包）：
```python
```

- [ ] **Step 2: 创建 `tests/conftest.py`**

文件内容（完整）：
```python
# -*- coding: utf-8 -*-
"""测试合成数据 fixture。
生成规则网格 GeoDataFrame，供 percolation 与 overlay 测试复用。
不依赖任何真实数据（KB11/THK/MY）。"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import pytest


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
```

- [ ] **Step 3: 验证 fixture 可用**

Run:
```cmd
PY312 -c "import sys; sys.path.insert(0,'tests'); from conftest import make_grid_gdf; g=make_grid_gdf(3,3); print('rows', len(g)); print('cols', list(g.columns)); print('centroid0', g.geometry.centroid.iloc[0].x, g.geometry.centroid.iloc[0].y)"
```
Expected: 输出
```
rows 9
cols ['geometry', 'NC_A', 'vertex1_x', 'vertex1_y']
centroid0 0.0 0.0
```
（第一个单元 centroid 在 (0,0)，因为 row=0,col=0。）

- [ ] **Step 4: 提交**

```cmd
git add tests/__init__.py tests/conftest.py
git commit -m "test: 添加合成网格数据 fixture（make_grid_gdf）"
```

---

## Task 3: UnionFind 测试

**Files:**
- Create: `tests/test_unionfind.py`
- 被测：`program/percolation.py` 的 `_UnionFind` 类（line 135-159）

- [ ] **Step 1: 创建 `tests/test_unionfind.py`**

文件内容（完整）：
```python
# -*- coding: utf-8 -*-
"""percolation._UnionFind 并查集正确性测试。"""
import pytest
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
```

- [ ] **Step 2: 跑测试验证通过**

Run:
```cmd
PY312 -m pytest tests/test_unionfind.py -v
```
Expected: `6 passed`。

- [ ] **Step 3: 提交**

```cmd
git add tests/test_unionfind.py
git commit -m "test: 添加 _UnionFind 并查集正确性测试（6 用例）"
```

---

## Task 4: S4 守护 — simulate_percolation 测试

**Files:**
- Create: `tests/test_simulate_percolation.py`
- 被测：`program/percolation.py` 的 `simulate_percolation`（line 165-217）

- [ ] **Step 1: 创建 `tests/test_simulate_percolation.py`**

文件内容（完整）：
```python
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
    assert fractions == pytest.approx([0.0, 1/3, 2/3, 1.0])
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
```

- [ ] **Step 2: 跑测试验证通过**

Run:
```cmd
PY312 -m pytest tests/test_simulate_percolation.py -v
```
Expected: `7 passed`。

**若 `test_line_graph_exact_curve` 失败**：说明手算期望值与实际不符。先记录实际输出（`print(fractions, sizes)`），若实际 `[1.0, 0.75, 0.5, 0.25]` 则期望正确、说明是别的问题；若实际不同，需重新推演 `sizes_reverse` 对齐——这本身就是 S4 脆弱性的证据，记录在测试注释里，但**不改 percolation.py**。

- [ ] **Step 3: 提交**

```cmd
git add tests/test_simulate_percolation.py
git commit -m "test: 添加 simulate_percolation S4 守护测试（7 用例）"
```

---

## Task 5: find_percolation_threshold 测试

**Files:**
- Create: `tests/test_find_threshold.py`
- 被测：`program/percolation.py` 的 `find_percolation_threshold`（line 220-254）

- [ ] **Step 1: 创建 `tests/test_find_threshold.py`**

文件内容（完整）：
```python
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
    # sizes[0]=1.0 → N=1/1.0=100... 注：span 用 1/sizes[0] 估 N
    # span_threshold = 1/sqrt(N)
    fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    sizes = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    # N = 1/sizes[0] = 1/1.0 = 1 → span = 1/sqrt(1) = 1.0
    # 首个 <1.0 在 index 1 → fractions[1]=0.25
    result = find_percolation_threshold(fractions, sizes, method="span")
    assert 0.0 <= result <= 1.0


def test_unknown_method():
    fractions = np.array([0.0, 0.5, 1.0])
    sizes = np.array([1.0, 0.6, 0.2])
    # 未知 method → 默认 fallback 0.5
    assert find_percolation_threshold(fractions, sizes, method="bogus") == pytest.approx(0.5)
```

- [ ] **Step 2: 跑测试验证通过**

Run:
```cmd
PY312 -m pytest tests/test_find_threshold.py -v
```
Expected: `6 passed`。

**若 `test_span` 失败**：spec 里 span 的期望只要求 `0<=result<=1.0`（不锁具体值），若失败说明断言写错或代码行为异常，记录实际值后判断。

- [ ] **Step 3: 提交**

```cmd
git add tests/test_find_threshold.py
git commit -m "test: 添加 find_percolation_threshold 三方法测试（6 用例）"
```

---

## Task 6: S3 守护 — build_grid_graph 测试

**Files:**
- Create: `tests/test_build_grid_graph.py`
- 被测：`program/percolation.py` 的 `build_grid_graph`（line 51-129）
- 依赖：`tests/conftest.py` 的 `make_grid_gdf`

- [ ] **Step 1: 创建 `tests/test_build_grid_graph.py`**

文件内容（完整）：
```python
# -*- coding: utf-8 -*-
"""S3 守护：build_grid_graph 节点-索引不变量。
钉住「节点 id == gdf 行索引」假设，及 4 邻接、边权 min/mean 语义。"""
import numpy as np
import pytest
from percolation import build_grid_graph, GRID_STEP
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
    # 一个远离格（centroid 远超 GRID_STEP*1.15）应 degree=0 但仍作为节点存在
    import geopandas as gpd
    from shapely.geometry import box
    base = make_grid_gdf(2, 2)  # 4 格在 (0..3000, 0..3000)
    far_box = box(50000 - 100, 50000 - 100, 50000 + 100, 50000 + 100)
    far = gpd.GeoDataFrame([{"geometry": far_box, "NC_A": 1.0}], crs="EPSG:32645")
    gdf = gpd.GeoDataFrame(pd_concat(base, far), crs="EPSG:32645")
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    assert G.number_of_nodes() == 5
    # 远离格（最后一行）度=0
    assert dict(G.degree())[4] == 0
```

> 注：上面 `test_isolated_node_exists` 用到 `pd_concat` 辅助。为避免引入未定义符号，本步骤实际代码用更直接的拼接，见 Step 1b。

- [ ] **Step 1b: 修正 `test_isolated_node_exists`，避免 `pd_concat` 占位符**

把 Task 6 Step 1 文件里的 `test_isolated_node_exists` 函数替换为以下**完整可运行**版本：

```python
def test_isolated_node_exists():
    import pandas as pd
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
```

> 实施时：Step 1 的完整文件应**直接包含** Step 1b 的修正版（即 `test_isolated_node_exists` 用 `pd.concat`），而非先写 `pd_concat` 占位再改。本计划分两步展示是为说明设计意图；落地只写一次。

- [ ] **Step 2: 跑测试验证通过**

Run:
```cmd
PY312 -m pytest tests/test_build_grid_graph.py -v
```
Expected: `8 passed`。

**若 `test_no_diagonal_edges` 或 `test_degree_sequence` 失败**：检查 `make_grid_gdf` 的方形半边长（100m）是否干扰了 cKDTree 的 `query_ball_point(centroids[i], GRID_STEP*1.15=3450)`。centroid 间距正好 3000 < 3450，应能查到邻居；对角节点距离 3000*sqrt(2)≈4243 > 3450，查不到，符合预期。若仍失败，记录实际邻居集合。

- [ ] **Step 3: 提交**

```cmd
git add tests/test_build_grid_graph.py
git commit -m "test: 添加 build_grid_graph S3 守护测试（8 用例）"
```

---

## Task 7: S3 第二层守护 — identify_key_nodes 测试

**Files:**
- Create: `tests/test_identify_key_nodes.py`
- 被测：`program/percolation.py` 的 `identify_key_nodes`（line 292-347）

- [ ] **Step 1: 创建 `tests/test_identify_key_nodes.py`**

文件内容（完整）：
```python
# -*- coding: utf-8 -*-
"""S3 第二层守护：identify_key_nodes 返回的 node_idx 合法性。
只测结构不变量，不测 centrality 数值（networkx 版本敏感）。"""
import pandas as pd
import pytest
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
    # exclude_boundary=True 时，返回行不应含 3x3 的角节点 (0,2,6,8)
    gdf = make_grid_gdf(3, 3)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    result = identify_key_nodes(G, top_n=9, exclude_boundary=True, gdf=gdf)
    corners = {0, 2, 6, 8}
    returned = set(result["node_idx"])
    # 角节点被置 0，不应出现在 top（除非全部都是角，但 3x3 有非角节点）
    assert not (returned & corners), f"边界节点未被排除: {returned & corners}"


def test_empty_graph_returns_empty_df():
    import networkx as nx
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
```

- [ ] **Step 2: 跑测试验证通过**

Run:
```cmd
PY312 -m pytest tests/test_identify_key_nodes.py -v
```
Expected: `7 passed`。

**若 `test_exclude_boundary_zeroes_centrality` 失败**：`_identify_boundary_nodes` 用 `step=3000, tol=1500` 判定边界。3x3 网格的角节点 (0,2,6,8) 的 centroid 坐标为 (0,0),(6000,0),(0,6000),(6000,6000)，都在 x_min/x_max/y_min/y_max 上，应被识别为边界。若 pagerank 模式下角节点 centrality 恰好为 0 已被排除，但 top_n=9 可能仍包含它们（因为 9 个节点全取）。注意：`identify_key_nodes` 取 top_n 后排序，边界节点 centrality 被设 0 会排到末尾，top_n=9 会包含所有节点。**需复核**：若失败，把 `top_n` 改小（如 5），确保只取非边界高 centrality 节点。先记录实际输出再决定。

- [ ] **Step 3: 提交**

```cmd
git add tests/test_identify_key_nodes.py
git commit -m "test: 添加 identify_key_nodes S3 第二层守护测试（7 用例）"
```

---

## Task 8: overlay matching 测试

**Files:**
- Create: `tests/test_overlay_matching.py`
- 被测：`program/multiperiod_overlay.py` 的 `find_exact_vertex_matches`(44-65)、`find_centroid_distance_matches`(71-115)、`identify_target_areas`(224-267)

- [ ] **Step 1: 创建 `tests/test_overlay_matching.py`**

文件内容（完整）：
```python
# -*- coding: utf-8 -*-
"""overlay 匹配纯函数测试。
显式传 max_dist/eps_meters，暴露容差可参数化（为 M5 修复铺路，但本轮不改业务代码）。"""
import numpy as np
import pandas as pd
import pytest
from multiperiod_overlay import (
    find_exact_vertex_matches,
    find_centroid_distance_matches,
    identify_target_areas,
    CENTROID_MATCH_TOLERANCE_M,
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
    from shapely.geometry import box
    new_geoms = [box(g.centroid.x - 100 + 1000, g.centroid.y - 100 + 1000,
                     g.centroid.x + 100 + 1000, g.centroid.y + 100 + 1000)
                 for g in gdf_b.geometry]
    gdf_b = gdf_b.set_geometry(new_geoms, crs="EPSG:32645")
    result = find_centroid_distance_matches(gdf_a, gdf_b)
    assert len(result) == 9


def test_centroid_dist_beyond_tolerance():
    # 偏移 3000m（> 默认 1500），应无匹配
    gdf_a = make_grid_gdf(3, 3)
    gdf_b = make_grid_gdf(3, 3)
    from shapely.geometry import box
    new_geoms = [box(g.centroid.x - 100 + 3000, g.centroid.y - 100 + 3000,
                     g.centroid.x + 100 + 3000, g.centroid.y + 100 + 3000)
                 for g in gdf_b.geometry]
    gdf_b = gdf_b.set_geometry(new_geoms, crs="EPSG:32645")
    result = find_centroid_distance_matches(gdf_a, gdf_b)
    assert len(result) == 0


def test_centroid_dist_custom_tolerance():
    # 自定义 max_dist=5000，偏移 3000m 也能匹配
    gdf_a = make_grid_gdf(3, 3)
    gdf_b = make_grid_gdf(3, 3)
    from shapely.geometry import box
    new_geoms = [box(g.centroid.x - 100 + 3000, g.centroid.y - 100 + 3000,
                     g.centroid.x + 100 + 3000, g.centroid.y + 100 + 3000)
                 for g in gdf_b.geometry]
    gdf_b = gdf_b.set_geometry(new_geoms, crs="EPSG:32645")
    result = find_centroid_distance_matches(gdf_a, gdf_b, max_dist=5000)
    assert len(result) == 9


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
```

- [ ] **Step 2: 跑测试验证通过**

Run:
```cmd
PY312 -m pytest tests/test_overlay_matching.py -v
```
Expected: `7 passed`。

**若 `test_identify_target_areas_clustering` 失败**：先 `print(result)` 看实际簇划分。4 点 `(0,0),(3000,0),(0,3000),(3000,3000)` 在 eps=4500 下：相邻 3000<4500，对角 4243<4500，确定全连通成一簇。若仍分簇，检查 sklearn DBSCAN 的 `min_samples` 语义（本测试传 `min_cluster_size=3`）。

- [ ] **Step 3: 提交**

```cmd
git add tests/test_overlay_matching.py
git commit -m "test: 添加 overlay 匹配纯函数测试（7 用例）"
```

---

## Task 9: 全量回归 + 覆盖率（AC2/AC3/AC4）

**Files:** 无新建（验证任务）

- [ ] **Step 1: 全量跑测试（AC2）**

Run:
```cmd
PY312 -m pytest tests/ -v
```
Expected: `41 passed`，0 failed。记录实际数量。

- [ ] **Step 2: 记录运行时间（AC3）**

Run:
```cmd
PY312 -m pytest tests/ --durations=10
```
Expected: 总 wall time < 10s。记录实际时间。

- [ ] **Step 3: 覆盖率（AC4）**

Run:
```cmd
PY312 -m pytest tests/ --cov=program/percolation --cov=program/multiperiod_overlay --cov-report=term-missing
```
Expected: 以下函数行覆盖 = 100%（在 term-missing 报告里这些函数的行号范围无 Missing 标记）：
- `_UnionFind`（135-159）
- `simulate_percolation`（165-217）
- `find_percolation_threshold`（220-254）
- `build_grid_graph`（51-129）
- `identify_key_nodes`（292-347）
- `find_exact_vertex_matches`（44-65）
- `find_centroid_distance_matches`（71-115）
- `identify_target_areas`（224-267）

percolation.py 整体覆盖率会因 plot/pipeline 函数未测而较低（属预期）。

- [ ] **Step 4: 无需提交**（本任务是验证，无文件改动）

---

## Task 10: S4 守护有效性验证（AC5）

**Files:** 临时修改 `program/percolation.py`（验证后**必须还原**）

- [ ] **Step 1: 记录 percolation.py 当前状态**

Run:
```cmd
git stash list
PY312 -c "import percolation, inspect; src=inspect.getsource(percolation.simulate_percolation); open('tests/_s4_backup.txt','w',encoding='utf-8').write(src)"
```
备份 `simulate_percolation` 源码到临时文件（验证后删除）。

- [ ] **Step 2: 临时改坏 sizes_reverse 对齐**

编辑 `program/percolation.py` 第 203 行附近。找到：
```python
    sizes_forward = np.array(list(reversed(sizes_reverse)), dtype=np.float64)
```
改为（去掉 reversed，故意破坏对齐）：
```python
    sizes_forward = np.array(sizes_reverse, dtype=np.float64)
```

- [ ] **Step 3: 跑 S4 测试，验证变红**

Run:
```cmd
PY312 -m pytest tests/test_simulate_percolation.py -v
```
Expected: `test_monotonic_non_increasing`、`test_boundary_f0_full_graph`、`test_line_graph_exact_curve` 等**变红**。记录失败的测试名。这证明测试网对 S4 类 bug 有效。

- [ ] **Step 4: 还原 percolation.py**

Run:
```cmd
git checkout -- program/percolation.py
```
确认还原：
```cmd
git diff --stat program/percolation.py
```
Expected: 空（无改动）。

- [ ] **Step 5: 复跑确认全绿**

Run:
```cmd
PY312 -m pytest tests/ -v
```
Expected: `41 passed`。

- [ ] **Step 6: 删除临时备份**

Run:
```cmd
del tests\_s4_backup.txt
```

- [ ] **Step 7: 记录 AC5 结果**

在提交信息或后续验证报告里记录：「临时去掉 `reversed(sizes_reverse)` 后，N 个 S4 测试变红；还原后 41 passed」。

- [ ] **Step 8: 无需提交**（验证任务，percolation.py 已还原，无净改动）

---

## Task 11: S3 守护有效性验证（AC6）

**Files:** 临时修改 `program/percolation.py`（验证后**必须还原**）

- [ ] **Step 1: 备份 build_grid_graph 源码**

Run:
```cmd
PY312 -c "import percolation, inspect; src=inspect.getsource(percolation.build_grid_graph); open('tests/_s3_backup.txt','w',encoding='utf-8').write(src)"
```

- [ ] **Step 2: 临时改坏节点 id（用哈希而非行索引）**

编辑 `program/percolation.py` `build_grid_graph` 内。找到第 93-94 行附近：
```python
    for i in range(len(gdf)):
        G.add_node(i, pos=(centroids[i, 0], centroids[i, 1]), weight=float(weights[i]))
```
改为（用 `id(gdf.iloc[i])` 这种非顺序值，模拟「gdf 重排后 node id 错位」）：
```python
    for i in range(len(gdf)):
        G.add_node(f"node_{i*7+3}", pos=(centroids[i, 0], centroids[i, 1]), weight=float(weights[i]))
```
（用 `i*7+3` 造出非 0..N-1 的 id，模拟重排/过滤后的错位）

同时第 122 行的 `G.add_edge(i, j, ...)` 也要同步改，否则 KeyError。改为：
```python
                G.add_edge(f"node_{i*7+3}", f"node_{j*7+3}", weight=w, frac_deleted=None)
```

- [ ] **Step 3: 跑 S3 测试，验证变红**

Run:
```cmd
PY312 -m pytest tests/test_build_grid_graph.py tests/test_identify_key_nodes.py -v
```
Expected: `test_node_id_matches_row_index`（断言 `i in G.nodes` 失败）、`test_node_idx_valid_row_index`、`test_no_diagonal_edges` 等**变红**。记录失败测试名。

- [ ] **Step 4: 还原 percolation.py**

Run:
```cmd
git checkout -- program/percolation.py
git diff --stat program/percolation.py
```
Expected: 空。

- [ ] **Step 5: 复跑确认全绿**

Run:
```cmd
PY312 -m pytest tests/ -v
```
Expected: `41 passed`。

- [ ] **Step 6: 删除临时备份**

Run:
```cmd
del tests\_s3_backup.txt
```

- [ ] **Step 7: 记录 AC6 结果**

记录：「临时用非顺序 node id 后，S3 测试变红；还原后 41 passed」。

- [ ] **Step 8: 无需提交**

---

## Task 12: 最终验收（AC7）+ 收尾

**Files:** 无

- [ ] **Step 1: 确认 program/ 零改动（AC7）**

Run:
```cmd
git status
git diff --stat program/
```
Expected:
- `git status` 显示的新文件只有 `tests/`、`conftest.py`、`pyproject.toml`、`requirements-dev.txt`、`docs/superpowers/` 下的内容。
- `git diff --stat program/` 为空（会话开始前 main.py/demo.py 的改动若仍在，那是历史改动，不属于本计划）。
- **重要**：确认 `program/percolation.py`、`program/multiperiod_overlay.py` 等被测文件**无任何改动**（Task 10/11 的临时改动已还原）。

- [ ] **Step 2: 最终全量测试**

Run:
```cmd
PY312 -m pytest tests/ -v --cov=program/percolation --cov=program/multiperiod_overlay --cov-report=term-missing
```
Expected: `41 passed`，覆盖率符合 AC4。

- [ ] **Step 3: 确认所有提交**

Run:
```cmd
git log --oneline -10
```
Expected: 看到 Task 1-8 各自的提交（共约 8 个 test/docs 提交）。

- [ ] **Step 4: 验收总结**

确认以下全部为真（逐条核对实际输出）：
- [ ] AC1: `pytest --version` 成功
- [ ] AC2: 41 passed
- [ ] AC3: wall time < 10s
- [ ] AC4: 被测函数行覆盖 100%
- [ ] AC5: S4 守护曾变红、已还原、复绿
- [ ] AC6: S3 守护曾变红、已还原、复绿
- [ ] AC7: `program/` 零改动

- [ ] **Step 5: 无需提交**（收尾验证）

---

## Self-Review 记录

计划编写后已对照 spec 自审：

**1. Spec 覆盖：**
- spec 1.2 范围 8 个被测函数 → Task 3-8 全覆盖 ✓
- spec 2.2 三个基础设施文件 → Task 1 ✓
- spec 3.1 fixture → Task 2 ✓
- spec 3.2 S4 用例 7 个 → Task 4 ✓
- spec 3.3 S3 用例 8 个 → Task 6 ✓
- spec 4.1-4.4 其余模块 → Task 3/5/7/8 ✓
- spec 5.1 AC1-AC7 → Task 1(AC1)/9(AC2-4)/10(AC5)/11(AC6)/12(AC7) ✓

**2. 占位符扫描：** Task 6 Step 1 的 `pd_concat` 是设计说明，Step 1b 给出真实 `pd.concat` 版本——已注明落地只写一次。无其他 TBD/TODO。

**3. 类型一致性：** `make_grid_gdf` 签名在 Task 2 定义，Task 6/7/8 引用一致；`identify_key_nodes` 参数 `top_n/exclude_boundary/gdf/use_pagerank` 与源码 line 292-298 一致；`find_centroid_distance_matches` 的 `max_dist` 参数与源码 line 73 一致。

**4. 已知风险点（在对应 Task 标注）：**
- Task 6 `test_no_diagonal_edges`：依赖 cKDTree 半径判定，需确认对角距离 4243 > 3450
- Task 7 `test_exclude_boundary_zeroes_centrality`：top_n=9 可能包含被置 0 的边界节点，需调小 top_n
- Task 8 `test_identify_target_areas_clustering`：4 点正方形坐标 `(0,0),(3000,0),(0,3000),(3000,3000)` 在 eps=4500 下对角 4243<4500 确定全连通；若分簇则检查 sklearn `min_samples` 语义
