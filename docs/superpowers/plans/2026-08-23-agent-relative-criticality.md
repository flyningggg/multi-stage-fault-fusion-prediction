# 代理模型期相对关键性改造 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 spec `2026-08-23-agent-relative-criticality-design.md`：新建 `agent_features.py`（图特征/按期标准化/多尺度池化），`agent_model.py` 改为编排，训练目标改为期内相对关键性 `bc_rel`，LOPO R² 从 -0.785 提升至 ≥0.2。

**Architecture:** 方案 B——新模块 `program/agent_features.py` 承载三个纯函数（可独立单测），`agent_model.py` 只做调用编排。数据流：原始特征 → 图特征并入 → 按期标准化（含 bc_rel）→ 多尺度池化。诚实评估三件套复用，评估时显式传 `target_col="bc_rel"`（保持旧测试合成 df 默认值不受影响）。

**Tech Stack:** Python 3.12（`C:\Users\（-1，0）\AppData\Local\Programs\Python\Python312\python.exe`，下文记作 `PY312`）、networkx、pandas、pytest ≥8.0。复用 `multiscale_features.build_multiscale_pyramid` 公开 API。

---

## ⚠️ 全局约束

1. **解释器**：一律用完整路径 `PY312`。不要用默认 `python`（3.14，无项目依赖）。
2. **不碰 GUI 与无关模块**：不改 `main.py`、`ml/` 包、`multiscale_features.py` 本体。
3. **原 55 用例零修改持续全绿**；新增测试 ≥8 用例，FAST 风格（<10s）。
4. **每个任务独立提交**；工作目录为仓库根。
5. 测试文件按任务逐步放开导入（避免收集期 ImportError 掩盖红绿信号——上轮教训）。

## File Structure

| 文件 | 改动 |
|---|---|
| `program/agent_features.py` | 新建：`BASE_EXCLUDE_COLS` / `GRAPH_FEATURES` / `compute_graph_features` / `period_robust_scale` / `add_multiscale_features` |
| `program/agent_model.py` | 编排接线：导入新函数、每期并图特征、df 构建后标准化+池化、目标默认 `bc_rel`、`AGENT_EXCLUDE_COLS` 增补、pipeline 显式传 target |
| `tests/test_agent_features.py` | 新建：8 用例 |
| `docs/superpowers/plans/2026-08-23-agent-relative-criticality.md` | 本文档 |

---

## Task 0: 保存计划文档

- [x] Step 1: 本文档写入仓库
- [ ] Step 2: 提交

```cmd
git add docs/superpowers/plans/2026-08-23-agent-relative-criticality.md
git commit -m "docs: 代理模型期相对关键性实施计划"
```

---

## Task 1: agent_features 骨架 + compute_graph_features（TDD）

**Files:** Create `program/agent_features.py`、Create `tests/test_agent_features.py`

- [ ] **Step 1: 写失败测试**（先只导入已规划的两个符号）

创建 `tests/test_agent_features.py`：

```python
# -*- coding: utf-8 -*-
"""agent_features 纯函数测试：图特征/按期标准化/多尺度池化。"""
import numpy as np
import pandas as pd
import pytest

from agent_features import GRAPH_FEATURES, compute_graph_features
from tests.conftest import make_grid_gdf


def test_graph_features_grid_degrees():
    gdf = make_grid_gdf(3, 3)
    feats, names = compute_graph_features(gdf)
    assert names == GRAPH_FEATURES
    assert feats.shape == (9, 3)
    deg = feats[:, 0]
    for corner in (0, 2, 6, 8):
        assert deg[corner] == pytest.approx(2.0), f"角节点 {corner} 度应为2"
    for edge in (1, 3, 5, 7):
        assert deg[edge] == pytest.approx(3.0)
    assert deg[4] == pytest.approx(4.0)


def test_graph_features_clustering_zero():
    # 网格图无三角形 → 局部聚类系数全 0；邻域均度有限值
    gdf = make_grid_gdf(3, 3)
    feats, _ = compute_graph_features(gdf)
    assert np.allclose(feats[:, 1], 0.0)
    assert np.isfinite(feats[:, 2]).all()
```

- [ ] **Step 2: 红** —— Run: `PY312 -m pytest tests/test_agent_features.py -q`，Expected: ModuleNotFoundError（符合预期）

- [ ] **Step 3: 实现** —— 创建 `program/agent_features.py`：

```python
# -*- coding: utf-8 -*-
"""
代理模型特征工程（期不变）：
  - 图结构特征（degree/clustering_coef/nbr_mean_degree）
  - 按期 robust 标准化（特征 median/IQR；标签 z-score → bc_rel）
  - 多尺度池化特征（栅格化 → 金字塔 → 回填）
设计文档: docs/superpowers/specs/2026-08-23-agent-relative-criticality-design.md
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from multiperiod_data import TOPOLOGY_ATTRIBUTES
from percolation import build_grid_graph

# 元数据列：不参与标准化、不作特征
BASE_EXCLUDE_COLS = {
    "period",
    "betweenness",
    "log1p_betweenness",
    "log_betweenness",
    "cell_x",
    "cell_y",
}

GRAPH_FEATURES = ["degree", "clustering_coef", "nbr_mean_degree"]


def compute_graph_features(gdf) -> Tuple[np.ndarray, List[str]]:
    """构建网格图，返回每格 [degree, clustering_coef, nbr_mean_degree]。
    期不变性：度有界 [0,4]；聚类系数是比率。
    ponytail: 无权聚类版本；边界效应显著时再换加权版。"""
    if not HAS_NX:
        raise ImportError("图特征计算需要 networkx")
    G = build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min")
    deg = dict(G.degree())
    clu = nx.clustering(G)
    feats = np.zeros((len(gdf), len(GRAPH_FEATURES)), dtype=float)
    for i in range(len(gdf)):
        nbrs = list(G.neighbors(i)) if i in G else []
        feats[i, 0] = float(deg.get(i, 0))
        feats[i, 1] = float(clu.get(i, 0.0))
        feats[i, 2] = float(np.mean([deg[j] for j in nbrs])) if nbrs else 0.0
    return feats, list(GRAPH_FEATURES)
```

- [ ] **Step 4: 绿 + 提交**

Run: `PY312 -m pytest tests/test_agent_features.py -q` → Expected: `2 passed`

```cmd
git add program/agent_features.py tests/test_agent_features.py
git commit -m "feat: agent_features 图结构特征 degree/clustering/邻域均度"
```

---

## Task 2: period_robust_scale + bc_rel（TDD）

**Files:** Modify 两文件（追加）

- [ ] **Step 1: 追加失败测试**

测试文件顶部导入区改为：

```python
from agent_features import (
    GRAPH_FEATURES,
    compute_graph_features,
    period_robust_scale,
)
```

文件末尾追加：

```python
def _two_scale_df(seed=0):
    """两期同分布不同量纲的合成 df（v 列差 100 倍尺度）。"""
    rng = np.random.default_rng(seed)
    parts = []
    for p, mu, sd in (("A", 100.0, 10.0), ("B", 1.0, 0.1)):
        n = 50
        parts.append(pd.DataFrame({
            "v": rng.normal(mu, sd, n),
            "const": 7.7,
            "period": p,
            "log1p_betweenness": rng.normal(5.0, 1.0, n),
        }))
    return pd.concat(parts, ignore_index=True)


def test_period_robust_scale_normalizes():
    out = period_robust_scale(_two_scale_df())
    for p in ("A", "B"):
        g = out.loc[out["period"] == p, "v"]
        assert abs(g.median()) < 1e-6            # 每期中位数居中
        assert 0.55 < g.std() < 0.95             # robust z 后 std≈0.74（IQR≈1.35σ）


def test_bc_rel_is_period_zscore():
    base = _two_scale_df()
    out = period_robust_scale(base)
    for p in ("A", "B"):
        t = out.loc[out["period"] == p, "bc_rel"]
        assert abs(t.mean()) < 1e-9
        assert abs(t.std() - 1.0) < 1e-9
    assert "bc_rel" not in base.columns          # 原 df 不新增列


def test_period_robust_scale_constant_col():
    out = period_robust_scale(_two_scale_df())
    assert np.isfinite(out["const"]).all()       # IQR=0 回退不炸


def test_period_robust_scale_does_not_mutate_input():
    base = _two_scale_df()
    before = base["v"].to_numpy().copy()
    period_robust_scale(base)
    assert np.allclose(before, base["v"].to_numpy())
```

- [ ] **Step 2: 红** —— Expected: ImportError

- [ ] **Step 3: 实现** —— 在 `agent_features.py` 的 `compute_graph_features` 之后追加：

```python
def _robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    scale = s.quantile(0.75) - s.quantile(0.25)
    if not scale or scale <= 0:
        scale = s.std()
    return (s - med) / (scale if scale and scale > 0 else 1.0)


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / (sd if sd and sd > 0 else 1.0)


def period_robust_scale(
    df,
    period_col: str = "period",
    target_col: str = "log1p_betweenness",
    exclude_cols: Optional[set] = None,
) -> pd.DataFrame:
    """数值特征列按期 median/IQR 标准化（覆盖同名列）；
    标签按期 z-score 写入新列 bc_rel。
    返回副本，不修改传入 df。元数据列与 target 本身不参与特征标准化。"""
    exclude = set(BASE_EXCLUDE_COLS) | {target_col}
    if exclude_cols:
        exclude |= set(exclude_cols)

    def _scale_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        num_cols = [c for c in g.select_dtypes(include=[np.number]).columns
                    if c not in exclude]
        for c in num_cols:
            g[c] = _robust_z(pd.to_numeric(g[c], errors="coerce"))
        g["bc_rel"] = _z(pd.to_numeric(g[target_col], errors="coerce"))
        return g

    return df.groupby(period_col, group_keys=False).apply(_scale_group)
```

- [ ] **Step 4: 绿 + 提交** —— Expected: `6 passed`

```cmd
git add program/agent_features.py tests/test_agent_features.py
git commit -m "feat: period_robust_scale 按期标准化 + bc_rel 期相对标签"
```

---

## Task 3: add_multiscale_features（TDD）

**Files:** Modify 两文件（追加）

- [ ] **Step 1: 追加失败测试**

导入区增补 `add_multiscale_features`；末尾追加：

```python
def test_multiscale_known_values():
    # 2x2 栅格单列 [[1,2],[3,4]]，scale=2 全局均值池化 → 每格 2.5
    rows = [{"cell_x": c * 3000.0, "cell_y": r * 3000.0, "period": "A",
             "NC_A": float(2 * r + c + 1)}
            for r in range(2) for c in range(2)]
    df = pd.DataFrame(rows)
    out = add_multiscale_features(df, step=3000.0, scales=(2,), topo_cols=["NC_A"])
    assert "NC_A_ms2" in out.columns
    assert np.allclose(out["NC_A_ms2"], 2.5)


def test_multiscale_holed_grid():
    # 3x3 缺中心格 → 不炸；ms 列存在且有限
    coords = [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]
    df = pd.DataFrame([{"cell_x": c * 3000.0, "cell_y": r * 3000.0,
                        "period": "A", "NC_A": float(r + c)} for r, c in coords])
    out = add_multiscale_features(df, step=3000.0, scales=(2, 4), topo_cols=["NC_A"])
    assert {"NC_A_ms2", "NC_A_ms4"} <= set(out.columns)
    assert out[["NC_A_ms2", "NC_A_ms4"]].notna().all().all()
```

- [ ] **Step 2: 红** —— Expected: ImportError

- [ ] **Step 3: 实现** —— `agent_features.py` 末尾追加：

```python
def add_multiscale_features(
    df,
    step: float = 3000.0,
    scales: Tuple[int, ...] = (2, 4),
    topo_cols: Optional[List[str]] = None,
    period_col: str = "period",
) -> pd.DataFrame:
    """每期：cell_x/cell_y 栅格化到 step 格点（缺失填0）→
    multiscale_features.build_multiscale_pyramid mean 池化 → 回填 <col>_ms{scale} 列。
    只对拓扑属性列池化；在按期标准化之后调用，保证上下文量纲无关。
    ponytail: 同格点碰撞取后者；真实步长格点无碰撞。"""
    from multiscale_features import build_multiscale_pyramid

    if topo_cols is None:
        topo_cols = [c for c in TOPOLOGY_ATTRIBUTES if c in df.columns]
    out = df.copy()

    def _pool_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        x = pd.to_numeric(g["cell_x"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(g["cell_y"], errors="coerce").to_numpy(float)
        col_i = np.round((x - x.min()) / step).astype(int)
        row_i = np.round((y - y.min()) / step).astype(int)
        n_rows, n_cols = int(row_i.max()) + 1, int(col_i.max()) + 1
        flat = row_i * n_cols + col_i
        k = len(topo_cols)
        dense = np.zeros((n_rows * n_cols, k), dtype=float)
        dense[flat] = g[topo_cols].to_numpy(dtype=float)
        ms_arr, suffixes = build_multiscale_pyramid(
            dense, n_rows, n_cols, scales=list(scales), mode="mean")
        for si, suf in enumerate(suffixes):
            block = ms_arr[:, si * k:(si + 1) * k]
            for j, cname in enumerate(topo_cols):
                g[f"{cname}_{suf}"] = block[flat, j]
        return g

    return out.groupby(period_col, group_keys=False).apply(_pool_group)
```

注意：确认文件顶部 typing 导入含 `Tuple`。

- [ ] **Step 4: 绿 + 提交** —— Expected: `8 passed`

```cmd
git add program/agent_features.py tests/test_agent_features.py
git commit -m "feat: add_multiscale_features 栅格化金字塔池化回填"
```

---

## Task 4: agent_model 编排接线

**Files:** Modify `program/agent_model.py`（六处小改）

- [ ] **Edit A — 导入**（紧跟 `from percolation import build_grid_graph, GRID_STEP` 之后）：

```python
from agent_features import (
    add_multiscale_features,
    compute_graph_features,
    period_robust_scale,
)
```

- [ ] **Edit B — AGENT_EXCLUDE_COLS** 集合末行加 `"bc_rel",`

- [ ] **Edit C — build_agent_training_data 循环内**：在 `cell_xs/cell_ys` 计算之后添加：

```python
        # 图结构特征（期不变）
        gf, gf_cols = compute_graph_features(gdf)
```

行构造处（`row["cell_y"] = ...` 与 `row["betweenness"] = ...` 之间）插入：

```python
            for j, gc in enumerate(gf_cols):
                row[gc] = float(gf[i, j])
```

- [ ] **Edit D — 函数尾部**：将 `_add_interaction_features` 后的 `return df` 替换为：

```python
    # 期相对变换：按期标准化 + 多尺度池化（步长接 config grid.step_m）
    try:
        from utils.config_loader import load_config
    except ImportError:
        def load_config(config_path=None):
            return {}
    grid_step = float((load_config().get("grid") or {}).get("step_m", GRID_STEP))

    df = period_robust_scale(df)
    df = add_multiscale_features(df, step=grid_step)
    logger.info("期相对变换完成: bc_rel 目标 + 多尺度列, %d 行 × %d 列", len(df), len(df.columns))
    return df
```

- [ ] **Edit E — train_agent_model 签名**：`target_col: str = "log1p_betweenness"` → `target_col: str = "bc_rel"`

- [ ] **Edit F — run_agent_pipeline 显式传参**：

```python
    spatial_cv = spatial_cv_evaluate(df, target_col="bc_rel")
    lopo = leave_one_period_out_evaluate(df, target_col="bc_rel")
```

（两个评估函数自身默认值不动——保住旧测试合成 df 兼容性。）

- [ ] **验证 + 提交**：

```cmd
PY312 -c "import sys; sys.path.insert(0,'program'); import agent_model; print('import OK')"
PY312 -m pytest tests/ -q
```
Expected: `import OK`；63 passed（55+8）零回归。

```cmd
git add program/agent_model.py
git commit -m "feat: 代理模型切换期相对关键性目标 bc_rel 并接入新特征"
```

---

## Task 5: 真实数据验收运行

- [ ] **Step 1**: 运行完整管线（精确 betweenness 约 40s），记录对比：

基线（改造前）：随机划分 0.779 / 空间块CV 0.424±0.187 / LOPO -0.785±0.707

验收命令（临时脚本放 `D:\temp\opencode\`，不入库）：

```cmd
py -3.12 D:\temp\opencode\run_agent_eval.py
```

- [ ] **Step 2**: 核对 `agent_eval_summary.json` 已更新且含 spatial_cv/lopo 键
- [ ] **Step 3**: 全量测试最终绿；记录 LOPO 实际值 vs 成功线 0.2 / 守护线（空间块CV ≥0.37）
- [ ] **Step 4**: 如有修复类改动单独提交；结果数字写入会话报告

---

## 验收标准（AC）

| AC | 内容 | 验证 |
|---|---|---|
| AC1 | 新增 8 用例绿；原 55 用例零修改绿 | 各任务步骤 |
| AC2 | 真实数据产出新 agent_eval_summary.json（spatial_cv/lopo 为 bc_rel 口径） | Task 5 |
| AC3 | LOPO r2_mean ≥ 0.2 达成功线；未达则如实记录实际值与分析 | Task 5 |
| AC4 | 空间块 CV r2_mean ≥ 0.37 守护线 | Task 5 |
| AC5 | GUI 无参路径兼容（import 冒烟 + result 键只增不减） | Task 4 |

## Self-Review 记录

1. **Spec 覆盖**：spec 4.1 三函数→Task 1/2/3；4.2 编排六处→Task 4 Edit A-F；step 接 config→Edit D；AC1-5 对应 spec 第 7 节。
2. **占位符扫描**：全部步骤含完整代码/命令/期望输出；无 TBD。
3. **类型一致性**：`period_robust_scale`/`add_multiscale_features` 签名与 spec 4.1 一致；评估函数默认值不改、pipeline 显式传 `bc_rel`——旧 55 用例不受影响的推理成立（make_synth_df 无 bc_rel，但两评估仅在被显式传参时才需要它）。
4. **已知风险**：groupby.apply 在新版 pandas 可能有 UserWarning（pyproject 已过滤）；真实数据 cell 坐标浮点漂移由 round 吸收；LOPO 若仍不达标，下一候选=去 x_norm/y_norm 类期漂移特征（本轮明确不做）。
