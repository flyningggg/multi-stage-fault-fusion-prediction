# Multi-Stage Fault Fusion Prediction

多时期断裂网络连通性分析与预测系统

## 项目简介

本项目实现了一套完整的多时期断裂网络拓扑分析流程，包括：

- **Stage 0**: 数据预处理与列名标准化
- **Stage 1**: 三期独立融合对比（PCA + 加权融合 + XGBoost特征重要性）
- **Stage 2**: 三期空间叠加与靶区识别
- **Stage 3**: 图渗流模拟与关键节点识别
- **Stage 4**: 代理模型（XGBoost预测betweenness centrality）

## 目录结构

```
program/
├── main.py                 # GUI主程序
├── batch_run.py            # 批量处理与Stage 1
├── multiperiod_data.py     # 数据加载与预处理
├── multiperiod_overlay.py  # 空间叠加与靶区识别
├── percolation.py          # 渗流模拟与关键节点
├── agent_model.py          # 代理模型（XGBoost + SHAP）
├── topology_fusion.py      # 拓扑融合算法
├── fusion_algorithm.py     # 加权/GAT融合
├── utils/                  # 工具函数
└── data/                   # 数据与结果
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行批量分析
python program/batch_run.py --multiperiod

# 运行GUI
python program/main.py
```

## 核心功能

### 三期独立融合对比
- PCA降维 + KMeans聚类
- 加权融合（NC_NB/NC_NL/NC_A高权重）
- XGBoost特征重要性分析

### 空间叠加与靶区识别
- 基于centroid距离的空间匹配
- DBSCAN靶区聚类
- 三期重叠区域可视化

### 渗流模拟
- 4连通网格图构建
- Union-Find渗流阈值计算
- PageRank关键节点识别（排除边界效应）

### 代理模型
- 32维特征（6拓扑 + 3空间 + 13邻域 + 10交互）
- 精确betweenness centrality计算
- SHAP可解释性分析

## 数据说明

- 海西期：3599网格，941非零
- 喜山期：2793网格，651非零
- 印支燕山期：3480网格，1607非零
- 网格步长：3000m

## 许可证

MIT License
