# -*- coding: utf-8 -*-
"""
图级表示学习（Graph-level / Node-level Embedding）

实现要点：
- 使用 GraphSAGE / GAT / GIN 等图神经网络，对断裂网络图直接学习嵌入；
- 支持节点级嵌入 Z（可与工程特征拼接），以及简单的图级嵌入（全局平均池化）。

说明：
- 这里默认使用规则网格图的邻接（可由 fusion_algorithm.build_grid_graph 构造），
  节点特征为已有的空间–拓扑特征矩阵 X；
- 对 fractopo 原始断裂网络若需更精细的图结构，可以在前置脚本中构建 edge_index，
  再直接调用本模块的 gnn_embedding_* 接口。
"""

from typing import Tuple, Dict, Optional

import numpy as np


def _require_torch_geometric():
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "需要 PyTorch + PyTorch Geometric 才能使用 GNN 嵌入：pip install torch torch_geometric"
        ) from e


def _to_data(X: np.ndarray, edge_index: np.ndarray):
    import torch
    from torch_geometric.data import Data

    x = torch.tensor(X, dtype=torch.float32)
    edge = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edge)


def gnn_embedding_graphsage(
    X: np.ndarray,
    edge_index: np.ndarray,
    hidden_dim: int = 64,
    out_dim: int = 32,
    epochs: int = 200,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    使用 GraphSAGE 做节点级嵌入，再通过全局平均池化得到图级嵌入。

    返回：
        Z: 节点嵌入 (n_nodes, out_dim)
        g: 图级嵌入 (out_dim,)
        metrics: 训练损失等信息
    """
    _require_torch_geometric()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv, global_mean_pool

    data = _to_data(X, edge_index)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)

    class GraphSAGEEncoder(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, out_dim)
            # 嵌入维 out_dim 与输入维 in_dim 通常不同，不能直接 MSE(z, x)；用线性解码重构
            self.dec = nn.Linear(out_dim, in_dim)

        def forward(self, x, edge_index):
            h = F.relu(self.conv1(x, edge_index))
            z = self.conv2(h, edge_index)
            recon = self.dec(z)
            return z, recon

    in_dim = data.num_features
    model = GraphSAGEEncoder(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    # 无监督自编码式目标：解码后重构自身特征
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        z, recon = model(data.x, data.edge_index)
        loss = F.mse_loss(recon, data.x)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
    model.eval()
    with torch.no_grad():
        Z, _ = model(data.x, data.edge_index)
        # 单图场景下，batch 全 0
        batch = Z.new_zeros(Z.size(0), dtype=torch.long)
        g = global_mean_pool(Z, batch)
    Z_np = Z.cpu().numpy()
    g_np = g.squeeze(0).cpu().numpy()
    return Z_np, g_np, {"loss_final": loss_history[-1], "loss_history": loss_history}


def gnn_embedding_gat(
    X: np.ndarray,
    edge_index: np.ndarray,
    hidden_dim: int = 32,
    out_dim: int = 16,
    heads: int = 2,
    epochs: int = 200,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    使用 GAT（图注意力网络）做节点嵌入 + 图级平均池化。
    """
    _require_torch_geometric()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv, global_mean_pool

    data = _to_data(X, edge_index)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)

    class GATEncoder(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=0.1)
            self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=0.1)
            self.dec = nn.Linear(out_dim, in_dim)

        def forward(self, x, edge_index):
            h = F.elu(self.conv1(x, edge_index))
            z = self.conv2(h, edge_index)
            recon = self.dec(z)
            return z, recon

    in_dim = data.num_features
    model = GATEncoder(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        z, recon = model(data.x, data.edge_index)
        loss = F.mse_loss(recon, data.x)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
    model.eval()
    with torch.no_grad():
        Z, _ = model(data.x, data.edge_index)
        batch = Z.new_zeros(Z.size(0), dtype=torch.long)
        g = global_mean_pool(Z, batch)
    Z_np = Z.cpu().numpy()
    g_np = g.squeeze(0).cpu().numpy()
    return Z_np, g_np, {"loss_final": loss_history[-1], "loss_history": loss_history}


def gnn_embedding_gin(
    X: np.ndarray,
    edge_index: np.ndarray,
    hidden_dim: int = 64,
    out_dim: int = 32,
    epochs: int = 200,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    使用 GIN（Graph Isomorphism Network）做节点嵌入 + 图级平均池化。
    """
    _require_torch_geometric()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GINConv, global_mean_pool

    data = _to_data(X, edge_index)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)

    class GINEncoder(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            nn1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            nn2 = nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
            self.dec = nn.Linear(out_dim, in_dim)

        def forward(self, x, edge_index):
            h = F.relu(self.conv1(x, edge_index))
            z = self.conv2(h, edge_index)
            recon = self.dec(z)
            return z, recon

    in_dim = data.num_features
    model = GINEncoder(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        z, recon = model(data.x, data.edge_index)
        loss = F.mse_loss(recon, data.x)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
    model.eval()
    with torch.no_grad():
        Z, _ = model(data.x, data.edge_index)
        batch = Z.new_zeros(Z.size(0), dtype=torch.long)
        g = global_mean_pool(Z, batch)
    Z_np = Z.cpu().numpy()
    g_np = g.squeeze(0).cpu().numpy()
    return Z_np, g_np, {"loss_final": loss_history[-1], "loss_history": loss_history}

