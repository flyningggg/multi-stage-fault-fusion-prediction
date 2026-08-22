# 代理模型诚实评估（空间块 CV + 留一期外推）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Stage 4 代理模型（XGBoost 预测 `log1p(betweenness)`）补上两种诚实评估——空间分块交叉验证（消除随机划分的空间自相关泄漏）与 Leave-One-Period-Out 跨期外推验证，量化真实泛化能力，默认行为不变。

**Architecture:** 在 `agent_model.py` 新增纯函数 `_build_block_ids`（等宽分箱空间分块，通用坐标数组版）、`spatial_cv_evaluate`（GroupKFold 按空间块分组）、`leave_one_period_out_evaluate`（按 period 外推）。`AGENT_XGB_PARAMS` 提为模块常量供三处共用；`_feature_cols` 统一排除元数据列。`build_agent_training_data` 携带原始质心坐标 `cell_x/cell_y`（不进特征）。`run_agent_pipeline` 接线两个评估并导出 JSON。

**Tech Stack:** Python 3.12（`C:\Users\（-1，0）\AppData\Local\Programs\Python\Python312\python.exe`，下文记作 `PY312`）、xgboost、scikit-learn GroupKFold、pytest ≥8.0。

---

## ⚠️ 全局约束

1. **解释器**：一律用完整路径 `PY312`。不要用默认 `python`（3.14，无项目依赖）。
2. **不碰 GUI 与无关模块**：不改 `main.py`、`ml/` 包。
3. **向后兼容**：`run_agent_pipeline()` 无参调用行为兼容（main.py:695/2831），result dict 只增键不减键。
4. **每个任务独立提交**；全量测试持续绿。
5. **测试必须快**：合成数据 + 小树参数（`n_estimators=10, max_depth=2`），单测文件总时长 < 10s。

## 背景与问题

| 问题 | 位置 | 后果 |
|---|---|---|
| 随机划分 train/test（`random_state=42`） | `agent_model.py` `train_agent_model` :268-276 | 相邻网格特征与 BC 高度空间自相关 → R² 虚高 |
| 三期混合训练，无跨期检验 | 同上 | 无法回答「能否预测新断裂网络」 |
| XGB 超参内联硬编码 | :278-290 | CV 折无法复用同参数 |

参考实现：`ml/train.py:_build_spatial_block_ids`(:101-140，需 vertex1..4 列——agent df 不携带)、`_spatial_block_cv_regression`(:143-173, GroupKFold)。因列结构不同且避免跨包私有引用，本计划在 agent_model 本地实现通用坐标版算法。

## File Structure

| 文件 | 改动 |
|---|---|
| `program/agent_model.py` | 新增 `AGENT_XGB_PARAMS` / `AGENT_EXCLUDE_COLS` / `_feature_cols` / `_build_block_ids` / `_lopo_split` / `spatial_cv_evaluate` / `leave_one_period_out_evaluate`；`build_agent_training_data` 加 `cell_x/cell_y`；`train_agent_model` 改用常量；`run_agent_pipeline` 接线 + JSON 导出 |
| `tests/test_agent_evaluation.py` | 新建：合成 df fixture + 7 个纯逻辑测试 |

## Tasks

### Task 0: 保存计划文档
- [ ] 写入本文档并提交 `git commit -m "docs: 代理模型诚实评估实施计划"`

### Task 1: 常量提取 + 元数据列 + _feature_cols（TDD）
- [ ] 失败测试：`test_feature_cols_excludes_metadata`
- [ ] 实现：`AGENT_XGB_PARAMS` 模块常量；`train_agent_model` 的 exclude_cols 与超参改用常量；`build_agent_training_data` 循环中加 `row["cell_x"]=float(xs[i])`, `row["cell_y"]=float(ys[i])`（每期先算 centroid 数组）
- [ ] 绿 + 提交

### Task 2: _build_block_ids（TDD）
- [ ] 失败测试：3×3 网格坐标 n_blocks=9 → 9 个唯一块、确定性；退化输入（全部重合）→ None
- [ ] 实现：pd.cut 等宽分箱 n_side²，<2 唯一块返回 None
- [ ] 绿 + 提交

### Task 3: spatial_cv_evaluate（TDD）
- [ ] 失败测试：合成 df + FAST 参数 → 返回含 `r2_mean/r2_std/rmse_mean/rmse_std/mae_mean/mae_std/n_blocks_used/n_splits_used` 且数值有限
- [ ] 实现：GroupKFold(groups=block_id)，折内 XGBRegressor(**params) 无 early stopping（ponytail: 升级路径=折内早停）
- [ ] 绿 + 提交

### Task 4: leave_one_period_out_evaluate（TDD）
- [ ] 失败测试：`_lopo_split` 划分互斥且覆盖全集；LOPO 结果 per_period 键 == 全部时期，聚合键存在
- [ ] 实现：逐期留出训练/测试，per_period 含 r2/rmse/mae/n_test，聚合 r2_mean/std
- [ ] 绿 + 提交

### Task 5: run_agent_pipeline 接线
- [ ] 训练后追加两个评估；日志对比三种口径 R²；导出 `agent_eval_summary.json`（random_split/spatial_cv/lopo）
- [ ] 导入冒烟 + 全量回归 + 提交

### Task 6: 真实数据验收
- [ ] CLI 运行 `run_agent_pipeline()`（精确 betweenness 计算耗时较长），记录随机划分 vs 空间块CV vs 留一期外推的对比数字
- [ ] 全量测试最终绿

## 验收标准（AC）

| AC | 内容 |
|---|---|
| AC1 | 新增测试 ≥7 用例全绿，原 48 用例零修改全绿 |
| AC2 | 真实数据产出 agent_eval_summary.json，三个口径可对比 |
| AC3 | GUI 无参调用路径不受影响（result 只增键） |
| AC4 | 测试文件运行 <10s（FAST 参数） |

## Self-Review 记录

1. **Spec 覆盖**：A1 空间块CV→Task 2/3/5；A2 留一期外推→Task 4/5；元数据坐标→Task 1；诚实对比输出→Task 5/6。
2. **占位符扫描**：所有步骤含完整代码，无 TBD。
3. **类型一致性**：`xgb_params` 可选覆盖参数在三处签名一致；`_feature_cols(df)` 单一来源；JSON 键名 random_split/spatial_cv/lopo 固定。
4. **已知风险**：真实数据运行耗时长（精确 BC O(V·E)）；period 少于 2 时 LOPO 返回 {}；cell 坐标缺失时 spatial CV 返回 {} 并告警，不抛异常中断管线。
