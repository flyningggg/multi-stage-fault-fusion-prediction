# 合成真值恢复与鲁棒性分析计划

## 父对象与证据问题

- 父对象：`target-screening-mvp-v1` 正式精确拓扑筛选流程。
- 父主张：该流程能够识别跨时期持续、具有较高网络关键性且对距离变换稳定的内部候选区。
- 本轮问题：在真值位置预先已知的合成三期规则网络中，流程能否找回持续高连通区，并拒绝只在单一期出现的诱饵区？
- 主张边界：本轮只验证受控合成网络上的算法行为，不代表塔里木工区有效性，不构成油气发现概率或储层物理验证。

## 执行边界

- 设备：本机 CPU；不使用 GPU、网络服务或外部数据。
- 规模：13×13 网格、3 个时期、8 个确定性场景；固定随机种子。
- 依赖：复用正式 `screening_pipeline`、默认评分权重和默认候选阈值。
- 可比性：所有场景保持网格、评分、距离变换、匹配与聚类合同不变，每次只改变计划中声明的输入扰动。
- 停止条件：全套场景完成或同一实现失败连续两次且无新诊断证据；不为通过门槛临时降低候选阈值。

## 预注册场景

| 场景 | 唯一变化 | 类型 |
|---|---|---|
| `baseline` | 低噪声、三期同位持续高连通区 | claim-carrying |
| `baseline_repeat` | 与 baseline 完全相同，用于确定性检查 | claim-carrying |
| `weight_noise_15` | 所有时期权重加入 15% 乘性噪声 | supporting |
| `weight_noise_30` | 所有时期权重加入 30% 乘性噪声 | supporting |
| `one_period_attenuated` | 第三期真值区增强显著衰减 | claim-carrying |
| `one_period_shifted` | 第三期真值区向东平移一个网格 | supporting |
| `single_period_decoy` | 第一期增加远离真值区的高权重诱饵 | claim-carrying |
| `combined_stress` | 30% 噪声、第三期偏移并随机削弱部分节点 | limitation-boundary |

## 指标与门槛

- 真值区：中心 3×3 网格；诱饵区：右上角 3×3 网格。
- `stable_target_hit`：稳定候选靶区质心距真值中心不超过 1.5 个网格步长。
- `localization_error_m`：最近稳定候选靶区质心到真值中心的距离。
- `candidate_cell_precision/recall`：已聚类候选单元与真值 3×3 网格的精确坐标重合率。
- `nearest_target_rank`：离真值中心最近的稳定候选在全部稳定候选中的综合分排名。
- `decoy_rejected`：单期诱饵中心 1.5 个网格步长内不存在稳定候选靶区。
- `deterministic`：baseline 与 baseline_repeat 的核心指标完全一致。

验收：baseline 必须命中；baseline 重复必须确定；`baseline`、两档噪声、单期衰减、单期偏移五个场景至少 4 个命中；单期诱饵必须被拒绝。`combined_stress` 只用于暴露边界，不计入通过分母。

## 计划产物

- `result.json`：总门槛、逐场景指标、主张更新与限制。
- `scenario_metrics.csv`：逐场景可比较指标。
- `maps/synthetic_recovery.png`：真值、基准恢复和扰动指标图。
- `report.md`：可直接复用的中文结果摘要。
- `runs/<scenario>/`：每个场景的正式流程清单、CSV、JSON、GIS 和图件。
