# 项目背景

多时期断裂网络连通性分析与预测系统（multi-stage-fault-fusion-prediction）。

三套地层，每套6个拓扑属性，通过多阶段流程融合分析断裂网络连通性、识别关键节点、预测渗流阈值。

## 数据

- 三个 CSV 文件（`program/` 目录下）：
  | 文件 | 地质时期 | 网格数 | 非零网格 |
  |------|---------|-------|---------|
  | 海西.csv | 海西期（~4亿-2.5亿年前） | 3599 | 941 |
  | 喜山.csv | 喜山期（~6500万年前至今） | 2793 | 651 |
  | 印支燕山.csv | 印支期+燕山期（~2.5亿-1亿年前） | 3480 | 1607 |

- 网格步长：3000m，三个时期起点偏移不同，格子不重叠

## 每个CSV的6个拓扑属性（最后6列）

| # | 代号 | 含义 |
|---|------|------|
| 1 | NC_NB | 每条分支平均连接节点数 → 分支连通性 |
| 2 | NC_NL | 每条裂缝平均连接节点数 → 裂缝连通性 |
| 3 | NB_NL | 每条裂缝分几段分支 → 裂缝复杂程度 |
| 4 | NC_A | 单位面积连接节点数 → 连接密度 |
| 5 | NB_A | 单位面积分支数 → 分支密度 |
| 6 | NL_A | 单位面积裂缝数 → 裂缝密度 |

## 项目结构

```
program/
├── main.py                 # GUI主程序（PyQt5）
├── batch_run.py            # Stage 1: 三期独立融合对比
├── multiperiod_data.py     # Stage 0: 数据加载与预处理
├── multiperiod_overlay.py  # Stage 2: 空间叠加与靶区识别
├── percolation.py          # Stage 3: 渗流模拟与关键节点
├── agent_model.py          # Stage 4: 代理模型（XGBoost + SHAP）
├── topology_fusion.py      # 拓扑融合算法
├── fusion_algorithm.py     # 加权/GAT融合
├── utils/                  # 工具模块
├── ml/                     # 机器学习模块
└── data/                   # 数据与结果
```

## 五个阶段（Stage 0-4）

### Stage 0: 数据预处理
- 列名标准化（NC/NB → NC_NB）
- 顶点坐标转几何对象
- 文件：`multiperiod_data.py`

### Stage 1: 三期独立融合对比
- PCA降维 + KMeans聚类（4类）
- 加权融合（NC_NB/NC_NL/NC_A高权重）
- XGBoost特征重要性分析
- 文件：`batch_run.py`

### Stage 2: 空间叠加与靶区识别
- 基于centroid距离的空间匹配（容差1500m）
- 仅保留三期均有非零数据的重叠网格（147个）
- DBSCAN靶区聚类
- 文件：`multiperiod_overlay.py`

### Stage 3: 渗流模拟与关键节点
- 4连通网格图构建
- Union-Find渗流阈值计算
- PageRank关键节点识别（排除边界效应）
- 边界节点排除（减少边界效应约6-7%）
- 文件：`percolation.py`

### Stage 4: 代理模型
- 32维特征：6拓扑 + 3空间 + 13邻域 + 10交互
- 精确betweenness centrality计算（非k=500近似）
- XGBoost回归（R²=0.78, 82%±0.5准确率）
- SHAP可解释性分析（蜂群图+柱状图）
- 文件：`agent_model.py`

## 关键技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 融合方法 | PCA + 加权 + XGBoost | 三种方法互补 |
| 聚类数 | 4 | 轮廓系数最优 |
| 空间匹配 | centroid距离（1500m） | 网格不重叠，无法精确匹配 |
| 渗流权重 | NC_A | 连接密度最能反映连通性 |
| 关键节点算法 | PageRank（排除边界） | 对边界效应更鲁棒 |
| 代理模型目标 | log(betweenness+1) | 解决偏态分布 |
| 特征空间 | 32维 | 拓扑+空间+邻域+交互 |

## 运行命令

```bash
# Stage 1: 三期融合对比
python program/batch_run.py --multiperiod

# Stage 2-4: 全部运行
python program/batch_run.py --multiperiod  # Stage 1
python -c "from multiperiod_overlay import run_overlay_pipeline; run_overlay_pipeline()"  # Stage 2
python -c "from percolation import run_percolation_pipeline; run_percolation_pipeline()"  # Stage 3
python -c "from agent_model import run_agent_pipeline; run_agent_pipeline()"  # Stage 4

# GUI
python program/main.py
```

## 依赖

见 `requirements.txt`

## 已完成的工作（2026-06-10）

- 修复中文字体显示问题
- 修复边界效应（PageRank + 排除边界节点）
- 修复节点尺寸归一化
- 修复聚类图标题笔误（"硅=" → "轮廓系数="）
- 修复渗流曲线文字重叠
- 代理模型从6特征扩展到32特征
- 从k=500近似改为精确betweenness计算
- 新增SHAP可解释性分析
- 创建独立仓库，与原项目分离
