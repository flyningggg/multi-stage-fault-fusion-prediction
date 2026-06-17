# 测试安全网建立 — 设计文档

- **日期**: 2026-06-17
- **状态**: 已批准（设计阶段，待 spec 复审）
- **范围**: 只建立测试安全网，**不修任何业务代码**
- **背景**: 针对 `percolation.py` / `multiperiod_overlay.py` / `agent_model.py` / `main.py` 的代码审查发现 4 个严重（S1-S4）、6 个中等（M1-M6）、5 个轻微（L1-L5）问题。用户选择「先加测试网再决定修复顺序」，即先为后续修复建立回归守护，再单独决定每项修复。

---

## 1. 目标与范围

### 1.1 目标

**只建立测试安全网，不修任何业务代码。** 为后续修复 S3/S4（及可选的 S2/L4/M2/M5/M6）提供「改完即跑」的验证回路。测试网就位后，再按 brainstorming 单独决定修复顺序。

### 1.2 范围内（in scope）

1. **测试基础设施**：安装 `pytest`、`pytest-cov`；新增 `pyproject.toml` 的 `[tool.pytest.ini_options]`；新增根 `conftest.py`。
2. **纯函数单测**（合成数据，无 IO、无真实数据）：
   - `percolation._UnionFind`
   - `percolation.simulate_percolation` ← 钉住 S4
   - `percolation.find_percolation_threshold`
   - `percolation.build_grid_graph` ← 钉住 S3
   - `percolation.identify_key_nodes`（节点 id 假设）
   - `multiperiod_overlay.find_exact_vertex_matches`
   - `multiperiod_overlay.find_centroid_distance_matches`
   - `multiperiod_overlay.identify_target_areas`（DBSCAN 容差）
3. **合成数据 fixture**：手造规则网格 `GeoDataFrame`，不依赖 KB11/THK/MY 真实数据。

### 1.3 范围外（out of scope，本轮明确不做）

- ❌ 修复 S1-S4、M1-M6、L1-L5 中的任何业务逻辑
- ❌ GUI 层（`main.py` MainWindow）测试
- ❌ `run_percolation_pipeline` / `run_overlay_pipeline` 等带文件 IO 的编排函数集成测试（留待方案 B 阶段）
- ❌ 真实数据驱动测试
- ❌ CI 配置（仓库当前无 CI，本轮先让本地可跑）

---

## 2. 测试基础设施

### 2.1 运行环境

**关键约束：测试必须用 Python 3.12 解释器运行。** 默认 `python`（3.13）没有项目依赖。

- 解释器路径：`C:\Users\（-1，0）\AppData\Local\Programs\Python\Python312\python.exe`
- 依赖现状：networkx 3.6.1 / geopandas 1.1.3 / pandas 2.3.3 / numpy 1.26.4 / scipy 1.17.1
- 需新装：`pytest>=8.0`、`pytest-cov>=4.1`

### 2.2 新增文件

#### `requirements-dev.txt`（新建，根目录）
```
pytest>=8.0
pytest-cov>=4.1
```
单独成文件，不污染 `requirements.txt`（生产依赖）。

#### `pyproject.toml`（新建，根目录）
仓库当前无此文件。只放 pytest 配置，不引入构建系统：
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
`--strict-markers` 防止拼错的 marker 静默失效。

#### `conftest.py`（新建，根目录）
单一职责：把 `program/` 注入 `sys.path`，让 `import percolation` 直接可用，无需改任何业务模块。
```python
import sys, os
_PROG = os.path.join(os.path.dirname(__file__), "program")
if _PROG not in sys.path:
    sys.path.insert(0, _PROG)
```

### 2.3 目录结构
```
multi-stage-fault-fusion-prediction/
├── conftest.py                          ← 新建
├── pyproject.toml                       ← 新建
├── requirements-dev.txt                 ← 新建
├── program/                             ← 不改
└── tests/                               ← 新建
    ├── __init__.py
    ├── conftest.py                      ← 合成数据 fixture
    ├── test_unionfind.py
    ├── test_simulate_percolation.py     ← 钉住 S4
    ├── test_find_threshold.py
    ├── test_build_grid_graph.py         ← 钉住 S3
    ├── test_identify_key_nodes.py
    └── test_overlay_matching.py
```

### 2.4 运行方式

安装与运行都用 Python 3.12：
```cmd
C:\...\Python312\python.exe -m pip install -r requirements-dev.txt
C:\...\Python312\python.exe -m pytest tests/ -v
C:\...\Python312\python.exe -m pytest tests/ --cov=program/percolation --cov=program/multiperiod_overlay --cov-report=term-missing
```
**不引入 CI**（仓库当前无 CI，本轮先保证本地可跑；CI 留待后续）。

### 2.5 关键决策

- **不改 `requirements.txt`**：测试依赖放 `requirements-dev.txt`，与生产依赖隔离。
- **不用 `setup.py`/可安装包**：用 `conftest.py` 的 `sys.path` 注入，零侵入，符合现有「脚本式」项目结构。
- **测试放根 `tests/` 不放 `program/tests/`**：与源码隔离，避免 `program/__pycache__` 混乱，且 `testpaths=["tests"]` 明确。

---

## 3. 合成 fixture + S4/S3 测试用例

### 3.1 合成网格 fixture（`tests/conftest.py`）

单一工厂函数，生成规则网格 GeoDataFrame，覆盖 percolation 和 overlay 两类测试的需求：

```python
def make_grid_gdf(n_cols=3, n_rows=3, step=3000.0,
                  weight_col="NC_A", weights=None, crs="EPSG:32645"):
    """规则网格 gdf：centroid 在 (col*step, row*step)，row-major 排序。
    包含 vertex1_x/vertex1_y（overlay 用）和 weight_col（percolation 用）。"""
    # 行 row、列 col → 行索引 i = row*n_cols + col（与 build_grid_graph 的节点 id 一致）
    # 每个单元是 centroid 周围的方形（半边长 100m，远小于 step=3000，几何不重叠）
```

- **为什么用 metric CRS（EPSG:32645）**：`build_grid_graph` 用 `GRID_STEP=3000` 直接比坐标差，必须米制坐标。`_identify_boundary_nodes` 的 `step=3000` 也一样。
- **为什么 row-major**：让节点 id = `row*n_cols+col` 可预测，测试里能算出每个节点的预期邻居。
- **方形半边长 = 100m**：远小于 `step=3000`，单元几何不重叠，且 `EDGE_DIST_TOLERANCE=150` 下 4 邻接判定只依赖 centroid 坐标差（方形尺寸不影响邻接判定）。
- **fixture 参数化**：`@pytest.fixture(params=[(3,3),(4,2),(2,2)])` 一次覆盖多种网格形状。

### 3.2 S4 守护（`tests/test_simulate_percolation.py`）

用合成 `nx.Graph`（不依赖 build_grid_graph），断言渗流曲线的数学性质：

| 测试 | 输入 | 断言 |
|---|---|---|
| `test_boundary_f0_full_graph` | 全连通 9 节点图 | `sizes[0] ≈ 1.0`（f=0 时最大分量=全部） |
| `test_boundary_f1_isolated` | 同上 | `sizes[-1] ≈ 1/9`（f=1 时全孤立） |
| `test_monotonic_non_increasing` | 3×3 网格图 | `all(sizes[i] ≥ sizes[i+1] - 1e-9)` ← **S4 守护核心** |
| `test_array_lengths` | 任意 | `len(fractions)==len(sizes)==n_steps+1` |
| `test_threshold_in_unit_interval` | 任意 | `0.0 ≤ threshold ≤ 1.0` |
| `test_line_graph_exact_curve` | 线图 0-1-2-3，`n_steps=3` | `x_interp==x_exact` 时 `sizes==[1.0, 0.75, 0.5, 0.25]`（手算值） |
| `test_empty_graph` | 0 边图 | 返回常数 1/N 曲线，threshold=0.0，不抛异常 |

**关键点**：`test_monotonic_non_increasing` 是 S4 的回归守护。如果未来重构把 `sizes_reverse`/`sizes_forward` 对齐改错，单调性必破。

#### S4 手算验证（设计依据）

线图 0-1-2-3（3 条边），`n_steps=3`：
- 边按权重升序 `[e01,e12,e23]`，`reversed` → `[e23,e12,e01]`
- 从空图归入：union(2,3)→分量2，union(1,2)→分量3，union(0,1)→分量4
- `sizes_reverse=[1/4, 2/4, 3/4, 4/4]`（长度 4 = total_edges+1 ✓）
- `sizes_forward=[4/4, 3/4, 2/4, 1/4]`
- `x_exact=[0, 0.333, 0.667, 1.0]`：f=0→1.0(全连通)，f=1→0.25(全孤立) ✓

**结论：当前对齐逻辑实际是正确的**——S4 不是活跃 bug，是「脆弱且无测试保护」的代码。测试网的价值正在于此：把这条性质钉死，未来任何重构（包括修 S3）一旦碰坏对齐，测试立刻红。

### 3.3 S3 守护（`tests/test_build_grid_graph.py`）

用 `make_grid_gdf` 生成 3×3 网格，验证 `build_grid_graph` 的节点-索引不变量：

| 测试 | 断言 |
|---|---|
| `test_node_count` | `G.number_of_nodes() == 9` |
| `test_edge_count_4adj` | `G.number_of_edges() == 12`（2·n·(n-1)，无对角边） |
| **`test_node_id_matches_row_index`** | 对每个 i：`G.nodes[i]['pos'] ≈ centroid(gdf.iloc[i])` ← **S3 不变量核心** |
| `test_no_diagonal_edges` | 节点 0 的邻居只有 {1,3}（不含对角 4） |
| `test_degree_sequence` | 角(0,2,6,8)度=2，边(1,3,5,7)度=3，中心(4)度=4 |
| `test_weight_min_mode` | 设节点 0 权=0.1、节点 1 权=0.9，边(0,1)权=0.1（min） |
| `test_weight_mean_mode` | 同上但 `weight_mode="mean"`，边权=0.5 |
| `test_isolated_node_count_logged` | 单元 gdf 含一个远离格，断言其 degree=0 但仍作为节点存在 |

**S3 的脆弱点在下游**：`identify_key_nodes` 返回 `node_idx` 列，main.py 用它 `gdf.iloc[node_idx]`。所以 S3 守护分两层——build_grid_graph 这层保证 node id==行索引；identify_key_nodes 那层（见第 4 节）保证返回的 `node_idx` 是合法 iloc 位置且 `pos` 与 `gdf.iloc[node_idx].centroid` 一致。

### 3.4 设计决策

- **测试用手算期望值，不用「跑一遍存下来」**：线图那组 `[1.0, 0.75, 0.5, 0.25]` 是手推的，如果代码错了测试会红；快照测试（方案 C）做不到这点。
- **不测 `run_percolation_pipeline`**：它带文件 IO + `load_all_periods()` 读真实数据，属方案 B 范围。本轮只测它调用的纯函数。
- **fixture 的 `weights` 可注入**：测 min/mean 模式时给相邻节点不同权重，才能区分两种模式。

---

## 4. 其余 4 个测试模块

### 4.1 UnionFind（`tests/test_unionfind.py`）

钉住并查集本身的正确性——这是 S4 的底层，若它错则渗流曲线必错。

| 测试 | 输入 | 期望 |
|---|---|---|
| `test_init_all_singletons` | `_UnionFind(5)` | `max_component_size()==1`，`find(i)==i` ∀i |
| `test_union_grows_component` | union(0,1),union(1,2) | `max_component_size()==3` |
| `test_union_idempotent` | union(0,1) 两次 | 第二次不增长，仍为 2 |
| `test_path_compression` | 链式 union(0,1),(1,2),(2,3) | `find(3)==find(0)`，且 find 后父节点被压缩 |
| `test_disjoint_groups` | union(0,1) + union(2,3) | `max==2`（两独立分量，非 4） |
| `test_empty` | `_UnionFind(0)` | `max_component_size()==0`，不抛异常 |

### 4.2 find_percolation_threshold（`tests/test_find_threshold.py`）

三种方法各一组手算输入（纯数组，无图）：

| 方法 | 输入 `fractions`/`sizes` | 期望返回 |
|---|---|---|
| `half` | `[0,0.25,0.5,0.75,1.0]` / `[1.0,0.8,0.6,0.4,0.2]` | `0.75`（首个 <0.5 的位置） |
| `half`（永不跌破） | sizes 全 ≥0.5 | `1.0` |
| `steepest`（差分最小明显） | `[0,0.25,0.5,0.75,1.0]` / `[1.0,0.9,0.8,0.3,0.2]` | `0.75`（diff=`[-0.1,-0.1,-0.5,-0.1]`，argmin=2，返回 `fractions[3]=0.75`） |
| `steepest`（<3 点） | 长度 2 的 sizes | `0.5`（fallback） |
| `span` | sizes[0]=1.0,N=100 → span=0.1 | 首个 <0.1 的 fractions |
| `unknown_method` | method="bogus" | `0.5`（默认 fallback） |

#### find_threshold steepest 手算验证（设计依据）

代码（percolation.py:240-244）：
```python
diff = np.diff(sizes)          # 一阶差分
idx = np.argmin(diff)          # 最陡下降处
return fractions[min(idx+1, len(fractions)-1)]
```
输入 `sizes=[1.0,0.9,0.8,0.3,0.2]`：`diff=[-0.1,-0.1,-0.5,-0.1]`，`argmin(diff)=2`，返回 `fractions[3]=0.75`。
> 注：不能用差分全等的数组（如 `[1,0.8,0.6,0.4,0.2]`，diff 全为 -0.2），否则 `argmin` 取第一个最小值，测试不体现 steepest 语义。

### 4.3 identify_key_nodes（`tests/test_identify_key_nodes.py`）

这是 S3 守护的**第二层**——验证返回的 `node_idx` 是合法行索引、且与 gdf 行一一对应：

| 测试 | 断言 |
|---|---|
| `test_returns_dataframe_with_expected_columns` | 列含 `node_idx, centrality, pos_x, pos_y, weight` |
| `test_node_idx_valid_row_index` | 所有 `node_idx ∈ [0, len(gdf))` ← **S3 第二层守护** |
| `test_pos_matches_gdf_centroid` | 对每行，`(pos_x,pos_y) ≈ centroid(gdf.iloc[node_idx])` ← **S3 第二层守护** |
| `test_top_n_limit` | `top_n=3` 时返回 ≤3 行 |
| `test_exclude_boundary_zeroes_centrality` | `exclude_boundary=True` 时，返回行不含角节点（3×3 的 0,2,6,8） |
| `test_empty_graph_returns_empty_df` | 0 边图 → 空 DataFrame，不抛异常 |
| `test_pagerank_vs_betweenness_both_run` | 两种模式都返回合法 DataFrame（不比较数值大小，只验可用性） |

**设计取舍**：不测具体 centrality 数值（networkx 版本敏感、数值不稳定），只测**结构不变量**（列、行索引合法性、位置一致性、边界排除语义）。

### 4.4 overlay matching（`tests/test_overlay_matching.py`）

用两个**部分重叠**的合成 gdf，测匹配纯函数：

| 测试 | 构造 | 断言 |
|---|---|---|
| `test_exact_vertex_match_overlap` | gdf_a 与 gdf_b 共享 3 个 `vertex1_x/y` | `find_exact_vertex_matches` 返回 3 行，`match_type=="exact_vertex"` |
| `test_exact_vertex_no_match` | 两 gdf 顶点完全不同 | 返回空 df，不抛异常 |
| `test_centroid_dist_within_tolerance` | gdf_b 整体偏移 1000m（<1500 容差） | 匹配数 == 重叠格数 |
| `test_centroid_dist_beyond_tolerance` | 偏移 3000m（>1500） | 返回空 df |
| `test_centroid_dist_custom_tolerance` | `max_dist=5000` 覆盖默认 | 偏移 3000m 也能匹配 |
| `test_identify_target_areas_clustering` | 合成 overlap_df：4 点聚成 1 簇 + 1 离散点，`min_cluster_size=3` | 1 个簇 + 1 个噪声(-1)，`target_area` 含「靶区1」和「散点」 |
| `test_identify_target_areas_requires_columns` | 缺 `centroid_x` | 抛 `ValueError` |

**这组同时覆盖 M5**：测试里显式传 `max_dist`/`eps_meters`，暴露这些容差是函数参数（好测、可配），为将来把硬编码 `3000/1500/4500` 统一成配置铺路——但**本轮不改业务代码**，只在测试里体现可参数化。

### 4.5 测试总量

| 模块 | 用例数 |
|---|---|
| UnionFind | 6 |
| simulate_percolation | 7 |
| find_threshold | 6 |
| build_grid_graph | 8 |
| identify_key_nodes | 7 |
| overlay_matching | 7 |
| **合计** | **~41** |

全部用合成数据，无文件 IO，单次运行预计 < 10 秒。

---

## 5. 验收标准 / 风险 / 交付清单

### 5.1 验收标准（Definition of Done）

本轮工作完成 = 以下全部为真，且每条都有**实际命令输出**佐证（不靠声称）：

| # | 验收项 | 验证命令 | 预期 |
|---|---|---|---|
| AC1 | pytest 可在 Python 3.12 下运行 | `Python312\python.exe -m pytest --version` | 输出版本号，无 ImportError |
| AC2 | 全部测试通过 | `Python312\python.exe -m pytest tests/ -v` | `~41 passed`，0 failed |
| AC3 | 单次运行快 | 同上 | wall time < 10s（无真实数据 IO） |
| AC4 | 覆盖率达标 | `--cov=program/percolation --cov=program/multiperiod_overlay --cov-report=term-missing` | 以下被测函数行覆盖 = 100%：`_UnionFind`、`simulate_percolation`、`find_percolation_threshold`、`build_grid_graph`、`identify_key_nodes`、`find_exact_vertex_matches`、`find_centroid_distance_matches`、`identify_target_areas`。percolation.py 整体覆盖率因 plot/pipeline 函数未测而较低，属预期。 |
| AC5 | S4 守护可触发 | 临时改坏 `simulate_percolation` 的 `sizes_reverse` 对齐 → 跑测试 | 单调性/边界测试**变红**，证明网有效；改回后复绿 |
| AC6 | S3 守护可触发 | 临时让 `build_grid_graph` 用非顺序节点 id → 跑测试 | `test_node_id_matches_row_index` **变红**；改回后复绿 |
| AC7 | 业务代码零改动 | `git diff --stat program/` | 空（本轮不碰 program/） |

**AC5/AC6 是测试网有效性的核心验证**——不是装样子，而是证明「故意引入 S3/S4 类 bug 时测试确实会红」。这一步会在实施时实际执行并记录输出。

### 5.2 风险与缓解

| 风险 | 可能性 | 影响 | 缓解 |
|---|---|---|---|
| **合成网格的几何细节**（方形 vs 点、CRS 投影）导致 build_grid_graph 4 邻接判定与真实数据行为不一致 | 中 | 测试绿但没守住真实场景 | fixture 的方形 centroid 精确落在 `(col*step, row*step)`，与 build_grid_graph 的 `abs(dy-GRID_STEP)<EDGE_DIST_TOLERANCE` 判定完全对齐；同时在 AC6 用「非顺序节点 id」验证，贴近真实 gdf 过滤/重排场景 |
| **pytest/pytest-cov 装进 Python 3.12 后与现有依赖冲突** | 低 | 环境损坏 | 只装两个纯 Python 包，且先在装前/装后各跑一次 `import networkx,geopandas` 确认无回归 |
| **networkx 3.6 的 pagerank/betweenness 数值在测试环境波动** | 低 | centrality 测试 flaky | 已规避——identify_key_nodes 测试只验结构不变量，不验数值（见 4.3） |
| **Windows 路径/编码**（中文路径 `（-1，0）`）导致 pytest 收集失败 | 低 | 测试跑不起来 | conftest 用 `os.path.join` 不硬编码路径；`pyproject.toml` 不设可能触发编码问题的选项 |
| **误以为已修 S3/S4** | 中 | 范围蔓延 | AC7 明确要求 `program/` 零改动；修复阶段另开 brainstorming |

### 5.3 明确不做的事（Non-Goals）

- ❌ **不修任何业务代码**（S1-S4/M1-M6/L1-L5 全部留待后续）
- ❌ **不加 CI**（仓库无 CI，本轮只保证本地 `pytest` 可跑）
- ❌ **不测 GUI / pipeline 编排**（main.py、run_percolation_pipeline、run_overlay_pipeline —— 属方案 B）
- ❌ **不引入快照测试**（与「修 S3/S4」目标冲突，见方案 C 分析）
- ❌ **不改 requirements.txt**（测试依赖隔离到 requirements-dev.txt）
- ❌ **不碰 `demo.ui`（M1）/魔法数字（M5）/配置缓存（M6）** —— 这些是修复阶段的事

### 5.4 交付清单

新增 11 个文件，修改 0 个：
```
requirements-dev.txt              （新）
pyproject.toml                    （新，仅 pytest 配置）
conftest.py                       （新，根，sys.path 注入）
tests/__init__.py                 （新）
tests/conftest.py                 （新，make_grid_gdf fixture）
tests/test_unionfind.py           （新，6 用例）
tests/test_simulate_percolation.py（新，7 用例，S4 守护）
tests/test_find_threshold.py      （新，6 用例）
tests/test_build_grid_graph.py    （新，8 用例，S3 守护）
tests/test_identify_key_nodes.py  （新，7 用例，S3 第二层守护）
tests/test_overlay_matching.py    （新，7 用例）
```
**`program/` 目录：零改动。**

### 5.5 完成后的下一步

测试网就位后，针对 S3/S4（或用户选定的优先项）单独走一轮 brainstorming → spec → plan。届时每一步修复都能靠本轮建立的测试网验证回归。

---

## 附录：被钉住的问题清单（仅作索引，本轮不修）

| 编号 | 描述 | 本轮是否织网守护 |
|---|---|---|
| S3 | 节点-索引假设脆弱（percolation.py `build_grid_graph` 用 `range(len(gdf))` 作节点 id） | ✅ test_build_grid_graph + test_identify_key_nodes |
| S4 | 逆序 Union-Find 渗流对齐脆弱（percolation.py `simulate_percolation`） | ✅ test_simulate_percolation（单调性 + 边界 + 手算线图） |
| S1 | main.py God Object | ❌ 范围外 |
| S2 | 全局可变状态（main.py 模块级全局） | ❌ 范围外 |
| M1-M6, L1-L5 | 见原始审查 | ❌ 范围外 |

**重要事实**：经手算验证，S4 的当前对齐逻辑**实际是正确的**，它是「脆弱但当前正确、且无测试保护」的代码。本轮通过测试把这条性质钉死，防止后续重构破坏。
