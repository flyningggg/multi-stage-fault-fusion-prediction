# 代理模型跨期泛化改进（期相对关键性）设计

- **日期**: 2026-08-23
- **状态**: 已批准（brainstorming 完成，待实施）
- **范围**: Stage 4 代理模型特征工程与任务定义改造；不碰 GUI、不动 `ml/` 包

## 1. 背景

诚实评估（2026-08-23 已落地）发现代理模型（XGBoost 预测 `log1p(betweenness)`）：

| 口径 | R² | 结论 |
|---|---|---|
| 随机划分 | 0.779 | 相邻网格空间自相关泄漏，虚高 |
| 空间块 CV | 0.424 ± 0.187 | 真实空间泛化约打对折 |
| 留一期外推（LOPO） | -0.785 ± 0.707 | 跨期完全不泛化 |

诊断定位三个根因：

1. **特征量纲跨期漂移**：长度类拓扑属性绝对值差异巨大（`NL_A` 三期均值 3.8e7 / 1.3e7 / 9.6e6），XGBoost 按绝对阈值分裂，跨期阈值语义不同。
2. **目标分布各期差异大**：喜山期 log1p(BC) std=0.69、max=10.2；海西/印支 std≈3.0、max≈40。留出喜山期 R²=-1.78 直接由此导致。
3. **缺图结构信号**：现有 32 维特征全是局部统计（属性值+邻域均值+交互项），无任何图结构量；且部分特征与目标的各期相关符号不稳（`x_norm`: -0.41~-0.11）。

## 2. 目标与成功标准

- 任务重定义为**预测期内相对关键性**：`bc_rel` = 每期内 log1p(BC) 的 z-score。应用语义 =「找出该期网络中的关键节点排名」。
- **成功线**: LOPO r2_mean ≥ 0.2（现 -0.785）；理想 ≥ 0.5。
- **守护线**: 空间块 CV r2_mean 不明显回退（现 0.424，允许 ±0.05）；随机划分口径继续报告但不作为验收依据。
- 诚实评估三件套照跑并继续写入 `agent_eval_summary.json`。

## 3. 方案决策记录

- **方案 B（选定）**：新建独立特征模块 `program/agent_features.py`，`agent_model.py` 只做训练编排。备选 A（全部内联）因 agent_model.py 已超 750 行被否。
- 用户选定全量范围：按期标准化 + 图结构特征 + 多尺度池化三者同轮实施。

## 4. 详细设计

### 4.1 新模块 `program/agent_features.py`

三个纯函数 + 常量，全部可合成数据单测：

```python
GRAPH_FEATURES = ["degree", "clustering_coef", "nbr_mean_degree"]

def compute_graph_features(gdf) -> Tuple[np.ndarray, List[str]]:
    """从 build_grid_graph(gdf, edge_weight_col="NC_A", weight_mode="min") 构图后，
    返回每格 [degree, local_clustering_coefficient, neighbor_mean_degree]。
    期不变性：度有界 [0,4]；聚类系数是比率。"""

def period_robust_scale(df, period_col="period",
                        target_col="log1p_betweenness",
                        exclude_cols=None) -> pd.DataFrame:
    """数值特征列按期 median/IQR 标准化（标准化值覆盖同名列，元数据列除外）；
    标签按期 z-score 写入新列 bc_rel。IQR=0 的列回退除以 std=1。
    返回标准化后的副本 df，不修改调用方传入的 df。"""

def add_multiscale_features(df, step=3000.0, scales=(2, 4),
                            topo_cols=None) -> pd.DataFrame:
    """对每期：
    1) 由 cell_x/cell_y 栅格化：row=round((y-ymin)/step), col=round((x-xmin)/step)
    2) 6 个标准化后的拓扑属性铺成 dense 数组（缺失格填 0）
    3) 复用 multiscale_features.build_multiscale_pyramid(X, n_rows, n_cols, scales, mode="mean")
    4) 回填 present 格，新增 <col>_ms2 / <col>_ms4 列
    topo_cols 默认 TOPOLOGY_ATTRIBUTES 中 df 实际存在的列。"""
```

要点：

- **执行顺序**：原始特征 → 图特征并入 → 按期标准化 → 多尺度池化（池化作用在标准化后的特征上，保证上下文量纲无关）。
- **多尺度只对 6 个拓扑属性池化**（+12 列），不对全部 ~37 列——克制列数膨胀。
- **bc_rel 是部署合法的变换**：新期到来时可用该期自身分布计算归一化，符合「期内排名」任务定义。已知边界：空间块 CV 内该期统计含测试块样本，属 transductive 变换，在 spec 中明示接受。

### 4.2 `agent_model.py` 编排改动

- `build_agent_training_data`：每期循环内调 `compute_graph_features(gdf)` 并入行数据；df 构建完成后依次调 `period_robust_scale(df)` 与 `add_multiscale_features(df, step=<config grid.step_m>)`——步长取自 config.yaml（延续 M5 配置化成果），缺省回退 `GRID_STEP`。
- 训练目标默认改为 `target_col="bc_rel"`；`AGENT_EXCLUDE_COLS` 增补 `"bc_rel"`。
- `train_agent_model` / `spatial_cv_evaluate` / `leave_one_period_out_evaluate` 签名不变（已有 target_col 参数）。
- `run_agent_pipeline()` 无参调用路径与返回 dict 结构兼容（main.py:695/2831 不改）。

### 4.3 数据流总览

```
load_all_periods() → 每期 gdf
  ├─ build_grid_graph(G) ──→ betweenness（不变）
  ├─ get_topology_matrix / spatial / neighbor / interaction（不变）
  └─ compute_graph_features(G)            [+3 列]
→ df (9872 × ~40)
  ├─ period_robust_scale                  特征原地标准化 + bc_rel
  └─ add_multiscale_features              [+≤12 列]
→ train_agent_model(target_col="bc_rel")
→ spatial_cv_evaluate / leave_one_period_out_evaluate（诚实评估不变）
```

## 5. 测试计划（`tests/test_agent_features.py`）

合成数据延续现有 fixture 风格，FAST 参数，全程 <10s：

| 测试 | 断言 |
|---|---|
| `test_graph_features_grid_degrees` | 3×3 规则网格 degree：角=2/边=3/心=4 |
| `test_graph_features_clustering_zero` | 网格图无三角形 → clustering_coef≈0 |
| `test_period_robust_scale_normalizes` | 两期不同尺度同分布特征 → 各期 mean≈0/std≈1 |
| `test_period_robust_scale_constant_col` | 常数列（IQR=0）回退不炸、输出有限 |
| `test_bc_rel_is_period_zscore` | bc_rel 各期 mean≈0/std≈1 |
| `test_multiscale_known_values` | 小栅格 2×2 池化手算期望值一致 |
| `test_multiscale_holed_grid` | 带洞网格（缺格）填 0 不炸、列名后缀正确 |

回归：原 55 用例零修改持续全绿。

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 栅格化假设格点在步长整数位 | round + clip 到 [0, n_side]；浮点漂移 ≤ 半步长可容忍 |
| 喜山期非零格仅 651，池化窗口多空 | 属预期信息量低；ms 列以 0 填充，由树模型自行忽略 |
| LOPO 提升不达标 | 保留诊断脚本路径；下一候选=去 x_norm/y_norm 类期漂移特征（本轮不做，避免混淆归因） |

## 7. 验收标准（AC）

1. AC1: 新增测试 ≥7 用例绿；原 55 用例零修改绿。
2. AC2: 真实数据运行产出新 `agent_eval_summary.json`；LOPO r2_mean ≥ 0.2 达成功线（否则记录实际值并分析，不隐瞒）。
3. AC3: 空间块 CV r2_mean ≥ 0.37（守护线）。
4. AC4: GUI 无参调用路径不回归（import 冒烟 + result 键只增不减）。

## 8. 非目标（本轮不做）

- 不迁移/重写既有 `_compute_neighbor_features` 等特征函数。
- 不引入 GNN/torch、不做超参搜索、不加 CI。
- 不动 `ml/` 包与 `multiscale_features.py` 本体（只复用其公开 API）。
