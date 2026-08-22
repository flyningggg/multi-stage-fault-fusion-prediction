# M5 魔法数字配置化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Stage 2/3 的网格步长与空间容差硬编码收编进 `config.yaml`，纯函数参数化 + 管线入口读配置，默认行为不变。

**Architecture:** 遵循仓库既有模式（`batch_run.py` 的 `load_config()` → `cfg.get("section", {})`）：纯函数增加带默认值（=现模块常量）的可选参数，管线入口函数（`run_percolation_pipeline` / `run_overlay_pipeline`）读取 `config.yaml` 的 `grid:` 段并向下传递。同时消除 `_identify_boundary_nodes` 内与 `GRID_STEP` 脱钩的重复字面量 `step = 3000.0`。

**Tech Stack:** Python 3.12（`C:\Users\（-1，0）\AppData\Local\Programs\Python\Python312\python.exe`，下文记作 `PY312`）、pytest ≥8.0、PyYAML（补进 requirements.txt）。

**对应审查项:** M5（魔法数字），出自 `docs/superpowers/specs/2026-06-17-test-safety-net-design.md` 记录的代码审查问题清单（S1-S4/M1-M6/L1-L5）。测试安全网已就位（41 用例）。

---

## ⚠️ 全局约束

1. **解释器**：一律用完整路径 `C:\Users\（-1，0）\AppData\Local\Programs\Python\Python312\python.exe`（记作 `PY312`）。不要用默认 `python`（3.14，无项目依赖）。
2. **不碰 GUI 与无关模块**：不改 `main.py`、`demo.py`、`fusion_algorithm.py`（它有一个同名不同义的 `build_grid_graph(n_nodes=...)`，绝不触碰）。
3. **行为默认不变**：所有原 41 用例不允许修改且必须持续全绿；显式参数优先于配置；配置缺省时回退到与当前行为一致的常量。
4. **每个任务独立提交**，只 add 本任务文件。
5. **工作目录**：仓库根 `D:\PycharmProjects\multi-stage-fault-fusion-prediction`。

## M5 问题清单（现状事实）

| # | 值 | 位置 | 现状 |
|---|---|---|---|
| 1 | 3000.0 | `percolation.py:47` `GRID_STEP` | 模块常量，`build_grid_graph` 使用（:103, :113-114） |
| 2 | 150.0 | `percolation.py:48` `EDGE_DIST_TOLERANCE` | 模块常量，邻接判定使用 |
| 3 | 3000.0 | `percolation.py:275` `step = 3000.0` | `_identify_boundary_nodes` 内**重复硬编码**，与 GRID_STEP 脱钩（漂移风险） |
| 4 | 1500.0 | `multiperiod_overlay.py:37` `CENTROID_MATCH_TOLERANCE_M` | 模块常量，已是匹配函数默认参数 |
| 5 | 4500 / 3 | `multiperiod_overlay.py:562` | `run_overlay_pipeline` 内硬编码调用 `identify_target_areas(min_cluster_size=3, eps_meters=4500)` |

调用方兼容性（已核实）：`main.py:609/651/2791` 调 `run_percolation_pipeline()` 无参；`main.py:2757` 调 `run_overlay_pipeline()` 无参；`agent_model.py:166` 调 `build_grid_graph(gdf, edge_weight_col=..., weight_mode="min")`。新增带默认值的可选参数全部向后兼容。

## File Structure

| 文件 | 职责 | 创建/修改 |
|---|---|---|
| `program/config.yaml` | 新增 `grid:` 配置段（五个键） | 修改 |
| `requirements.txt` | 补 `pyyaml>=6.0`（config 基础设施实际依赖） | 修改 |
| `program/percolation.py` | 纯函数参数化 + 管线读配置 | 修改 |
| `program/multiperiod_overlay.py` | 管线读配置 | 修改 |
| `tests/test_grid_config_keys.py` | grid 段键契约测试 | 创建 |
| `tests/test_build_grid_graph.py` | 追加参数化用例 | 修改 |
| `tests/test_boundary_nodes.py` | 边界识别测试 | 创建 |

---

## Task 0: 保存本计划文档

- [ ] Step 1: 本文档已写入 `docs/superpowers/plans/2026-08-22-m5-grid-config.md`
- [ ] Step 2: 提交

```cmd
git add docs/superpowers/plans/2026-08-22-m5-grid-config.md
git commit -m "docs: M5 魔法数字配置化实施计划"
```

---

## Task 1: config.yaml grid 段 + pyyaml + 键契约测试（TDD）

**Files:**
- Create: `tests/test_grid_config_keys.py`
- Modify: `program/config.yaml`（在 `export_grid:` 段之后插入）
- Modify: `requirements.txt`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_grid_config_keys.py`：

```python
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
```

- [ ] **Step 2: 跑测试验证变红**

Run:
```cmd
PY312 -m pytest tests/test_grid_config_keys.py -v
```
Expected: FAIL（`config.yaml 缺少 grid 段`）。

- [ ] **Step 3: 在 config.yaml 的 `export_grid:` 段后插入**

```yaml

# 网格与空间容差（Stage 2 叠加 / Stage 3 渗流）
grid:
  step_m: 3000.0                   # 网格步长 (m)
  edge_dist_tolerance_m: 150.0     # 邻接判定坐标容差 (m)
  centroid_match_tolerance_m: 1500.0  # 三期叠加 centroid 匹配容差（半步长）
  target_eps_m: 4500.0             # 靶区 DBSCAN eps（1.5 倍步长）
  target_min_cluster_size: 3       # 靶区最小簇样本数
```

同时在 `requirements.txt` 的 Core dependencies 末尾追加一行：

```
pyyaml>=6.0
```

- [ ] **Step 4: 复跑验证变绿**

Run:
```cmd
PY312 -m pytest tests/test_grid_config_keys.py -v
```
Expected: `2 passed`。

- [ ] **Step 5: 提交**

```cmd
git add tests/test_grid_config_keys.py program/config.yaml requirements.txt
git commit -m "feat: config.yaml 新增 grid 容差段 + 键契约测试（M5 1/3）"
```

---

## Task 2: build_grid_graph 参数化（TDD）

**Files:**
- Modify: `tests/test_build_grid_graph.py`（追加 2 用例）
- Modify: `program/percolation.py:51-55`（签名）与 :103、:113-114（函数体）

- [ ] **Step 1: 追加失败测试到 `tests/test_build_grid_graph.py` 末尾**

```python
def test_custom_grid_step_param():
    # 6000m 步长网格 + grid_step=6000 → 4 邻接正确（12 边）
    gdf = make_grid_gdf(3, 3, step=6000.0)
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min",
                         grid_step=6000.0)
    assert G.number_of_edges() == 12
    assert set(G.neighbors(0)) == {1, 3}


def test_edge_tolerance_param_controls_adjacency():
    # 两格垂直间距 3050m：默认容差 150 可邻接；收紧到 10 后断开
    gdf = make_grid_gdf(1, 2)
    from shapely.geometry import box
    gdf.loc[1, "geometry"] = box(-100, 2950, 100, 3150)  # centroid (0,3050)
    G_default = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    assert G_default.has_edge(0, 1)
    G_tight = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min",
                               edge_dist_tolerance=10.0)
    assert not G_tight.has_edge(0, 1)
```

- [ ] **Step 2: 跑测试验证变红**

Run:
```cmd
PY312 -m pytest tests/test_build_grid_graph.py -v
```
Expected: 两个新用例 TypeError FAIL（unexpected keyword argument）；原 8 用例仍绿。

- [ ] **Step 3: 修改 `percolation.py`**

签名（:51-55）改为：

```python
def build_grid_graph(
    gdf: gpd.GeoDataFrame,
    edge_weight_col: str = "NC_A",
    weight_mode: str = "min",
    grid_step: float = GRID_STEP,
    edge_dist_tolerance: float = EDGE_DIST_TOLERANCE,
) -> "nx.Graph":
```

函数体三处替换：
- :103 `max_dist = GRID_STEP * 1.15` → `max_dist = grid_step * 1.15`
- :113 `h_adj = (dx < EDGE_DIST_TOLERANCE) and (abs(dy - GRID_STEP) < EDGE_DIST_TOLERANCE)` → `h_adj = (dx < edge_dist_tolerance) and (abs(dy - grid_step) < edge_dist_tolerance)`
- :114 `v_adj = ...` 同理替换为小写参数

- [ ] **Step 4: 复跑验证全绿**

Run:
```cmd
PY312 -m pytest tests/test_build_grid_graph.py tests/test_simulate_percolation.py -q
```
Expected: 全部 passed（含新 2 用例）。

- [ ] **Step 5: 提交**

```cmd
git add program/percolation.py tests/test_build_grid_graph.py
git commit -m "refactor: build_grid_graph 参数化 grid_step/edge_dist_tolerance（M5 2/3）"
```

---

## Task 3: _identify_boundary_nodes 参数化去重（TDD）

**Files:**
- Create: `tests/test_boundary_nodes.py`
- Modify: `program/percolation.py:260-289`（消除 `step = 3000.0` 重复字面量）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_boundary_nodes.py`：

```python
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
```

- [ ] **Step 2: 跑测试验证变红**

Run:
```cmd
PY312 -m pytest tests/test_boundary_nodes.py -v
```
Expected: FAIL（`_identify_boundary_nodes() got an unexpected keyword argument 'grid_step'`）。

- [ ] **Step 3: 修改 `percolation.py:260-289`**

签名改为 `def _identify_boundary_nodes(G: "nx.Graph", gdf, grid_step: float = GRID_STEP) -> set:`；
删除以下两行：

```python
    # 网格步长（约3000m）
    step = 3000.0
    tol = step * 0.5
```

替换为：

```python
    tol = grid_step * 0.5
```

- [ ] **Step 4: 复跑验证全绿**

Run:
```cmd
PY312 -m pytest tests/ -q
```
Expected: 全部 passed（41+2+3=46 用例）。

- [ ] **Step 5: 提交**

```cmd
git add program/percolation.py tests/test_boundary_nodes.py
git commit -m "refactor: _identify_boundary_nodes 复用 GRID_STEP 并参数化（M5 3/3 percolation 侧）"
```

---

## Task 4: run_percolation_pipeline 接线配置

**Files:**
- Modify: `program/percolation.py`（import 区 + `run_percolation_pipeline` 函数体）

说明：该函数入口即 `load_all_periods()` 读真实数据 IO，按既定范围不做集成测试；由既有单测守护被调纯函数，本任务以「导入冒烟 + 全量回归」验证。

- [ ] **Step 1: import 区添加**（在 `_THIS_DIR` sys.path 注入块之后、`from multiperiod_data import ...` 之前或之后均可）

```python
try:
    from utils.config_loader import load_config
except ImportError:
    def load_config(config_path=None):
        return {}
```

- [ ] **Step 2: `run_percolation_pipeline` 函数体内接线**

在 `logger.info("=== Stage 3: 图渗流模拟 ===")` 之后添加：

```python
    cfg = load_config()
    grid_cfg = cfg.get("grid", {}) if isinstance(cfg, dict) else {}
```

将图构建调用（:502）改为：

```python
        G = build_grid_graph(
            gdf,
            edge_weight_col=edge_weight_col,
            weight_mode="min",
            grid_step=float(grid_cfg.get("step_m", GRID_STEP)),
            edge_dist_tolerance=float(grid_cfg.get("edge_dist_tolerance_m", EDGE_DIST_TOLERANCE)),
        )
```

- [ ] **Step 3: 导入冒烟 + 全量回归**

Run:
```cmd
PY312 -c "import sys; sys.path.insert(0,'program'); import percolation; print('import OK')"
PY312 -m pytest tests/ -q
```
Expected: `import OK`；全量 passed。

- [ ] **Step 4: 提交**

```cmd
git add program/percolation.py
git commit -m "feat: 渗流管线从 config.yaml 读取网格容差"
```

---

## Task 5: run_overlay_pipeline 接线配置

**Files:**
- Modify: `program/multiperiod_overlay.py`（:30 import 区、:532-562 `run_overlay_pipeline`）

- [ ] **Step 1: 顶部 import 区添加**（紧跟现有 `from utils.logging_utils import get_logger` 之后）

```python
from utils.config_loader import load_config
```

- [ ] **Step 2: 修改签名与函数体**

签名（:532-535）改为：

```python
def run_overlay_pipeline(
    out_dir: Optional[str] = None,
    max_dist: Optional[float] = None,
) -> dict:
```

在 `os.makedirs(out_dir, exist_ok=True)` 之后添加：

```python
    cfg = load_config()
    grid_cfg = cfg.get("grid", {}) if isinstance(cfg, dict) else {}
    if max_dist is None:
        max_dist = float(grid_cfg.get("centroid_match_tolerance_m", CENTROID_MATCH_TOLERANCE_M))
```

将靶区聚类调用（:562）改为：

```python
    overlap_df = identify_target_areas(
        overlap_df,
        min_cluster_size=int(grid_cfg.get("target_min_cluster_size", 3)),
        eps_meters=float(grid_cfg.get("target_eps_m", 4500)),
    )
```

行为保持说明：main.py 无参调用时，max_dist 由常量默认值变为「config 值（=同值）回退常量」，结果一致。

- [ ] **Step 3: 导入冒烟 + 全量回归**

Run:
```cmd
PY312 -c "import sys; sys.path.insert(0,'program'); import multiperiod_overlay; print('import OK')"
PY312 -m pytest tests/ -q
```
Expected: `import OK`；全量 passed。

- [ ] **Step 4: 提交**

```cmd
git add program/multiperiod_overlay.py
git commit -m "feat: 叠加管线从 config.yaml 读取匹配/聚类容差"
```

---

## Task 6: 最终验收

- [ ] **Step 1: 全量测试**

Run:
```cmd
PY312 -m pytest tests/ -v
```
Expected: 46 passed（原 41 不改动 + 新增 5）。

- [ ] **Step 2: grep 扫描残留**

Run:
```cmd
git grep -n "step = 3000" -- program/
git grep -n "eps_meters=4500" -- program/
```
Expected: 均无输出（重复字面量与硬编码调用已消除）。注：`identify_target_areas` 签名上的 `eps_meters: float = 4500` 是命名参数默认值，属正常实践，不在清除范围。

- [ ] **Step 3: 改动范围核对**

Run:
```cmd
git diff main --stat
```
Expected: 只涉及 `program/config.yaml`、`program/percolation.py`、`program/multiperiod_overlay.py`、`requirements.txt`、`tests/*`、`docs/*`。不含 `main.py` / `demo.py` / `fusion_algorithm.py`。

## 验收标准（AC）

| AC | 内容 | 验证方式 |
|---|---|---|
| AC1 | config.yaml 有 grid 段五键且与现行为等价 | test_grid_config_keys 2 用例绿 |
| AC2 | 原 41 用例零修改且全绿；新增 ≥5 用例 | Task 6 Step 1 |
| AC3 | `_identify_boundary_nodes` 不再有字面量 3000.0，复用 GRID_STEP | Task 6 Step 2 |
| AC4 | 两条管线从容差配置读取、显式参数可覆盖、缺省回退与旧行为一致 | Task 4/5 代码 + 回归绿 |
| AC5 | 改动范围不越界（无 GUI/无关模块） | Task 6 Step 3 |

## Self-Review 记录

1. **Spec 覆盖**：M5 五个值全部有对应任务（3000×2→Task 2/3，150→Task 2，1500→Task 5，4500/3→Task 5）；pyyaml 缺口→Task 1；契约测试→Task 1；管线接线→Task 4/5；验收→Task 6。
2. **占位符扫描**：所有步骤含完整代码/命令/期望输出，无 TBD。
3. **类型一致性**：参数名 `grid_step`/`edge_dist_tolerance` 与配置键 `step_m`/`edge_dist_tolerance_m`/`centroid_match_tolerance_m`/`target_eps_m`/`target_min_cluster_size` 各任务间一致；`make_grid_gdf(step=...)` 签名已核实支持。
4. **已知风险**：
   - 默认参数在 def 时绑定模块常量——float 不可变，安全。
   - `test_edge_tolerance_param` 数学已验算（3050 间距、容差阈值两侧）。
   - `test_larger_step_widens_boundary_band` 已手算（tight=16 外圈节点、wide 含次外圈但不含中心 12）。
