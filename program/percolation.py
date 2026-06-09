# -*- coding: utf-8 -*-
"""
断裂网络图渗流模拟模块。
基于网格邻接图，逐步删除低连通性边，模拟断裂网络连通性的崩塌过程，
识别渗流阈值与关键节点。

理论依据：Stauffer & Aharony 渗流理论。
地质类比：边删除 = 裂缝闭合/充填；分量崩解 = 断裂网络失去跨区域连通性。

核心功能：
  1. 网格邻接图构建（4邻接）
  2. 递增删边渗流模拟
  3. 渗流阈值识别
  4. 关键节点（高 betweenness centrality）定位
  5. 可视化（渗流曲线 + 关键节点地图）
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from multiperiod_data import (
    load_all_periods, load_period_csv,
    get_topology_matrix, TOPOLOGY_ATTRIBUTES,
    DEFAULT_CSV_PATHS,
)
from utils.logging_utils import get_logger

logger = get_logger("percolation")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------
GRID_STEP = 3000.0        # 网格步长 (m)
EDGE_DIST_TOLERANCE = 150.0  # 邻接判定的坐标容差 (m)


def build_grid_graph(
    gdf: gpd.GeoDataFrame,
    edge_weight_col: str = "NC_A",
    weight_mode: str = "min",
) -> "nx.Graph":
    """
    将网格 GeoDataFrame 构建为邻接图。

    参数：
      gdf:             含 vertex1_x, vertex1_y 与拓扑属性的 GeoDataFrame
      edge_weight_col: 边权重的拓扑属性列名（如 NC_A, NC_NB, NL_A 等）
      weight_mode:     边权重聚合方式
                        "min"  — 取两邻接网格的最小值（更保守）
                        "mean" — 取平均值

    返回：
      networkx.Graph
        - 节点属性: pos=(x, y), weight=该网格的拓扑属性值
        - 边属性:   weight=min(两节点连通性), frac_deleted=None

    4邻接判定：两网格 centroid 在 X 方向差 3000m 且 Y 方向差约 0，
    或在 Y 方向差 3000m 且 X 方向差约 0。
    """
    if not HAS_NX:
        raise ImportError("图渗流需要 networkx: pip install networkx")

    if edge_weight_col not in gdf.columns:
        available = [c for c in TOPOLOGY_ATTRIBUTES if c in gdf.columns]
        if available:
            edge_weight_col = available[0]
            logger.warning("列 %s 不存在，改用 %s", edge_weight_col, edge_weight_col)
        else:
            raise ValueError(f"无可用拓扑属性列: {list(gdf.columns)}")

    G = nx.Graph()
    centroids = np.column_stack([
        np.array([p.x for p in gdf.geometry.centroid]),
        np.array([p.y for p in gdf.geometry.centroid]),
    ])

    weights = pd.to_numeric(gdf[edge_weight_col], errors="coerce").fillna(0.0).values

    for i in range(len(gdf)):
        G.add_node(i, pos=(centroids[i, 0], centroids[i, 1]), weight=float(weights[i]))

    # 使用 cKDTree 找每对邻接网格
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError("图构建需要 scipy: pip install scipy")

    tree = cKDTree(centroids)
    max_dist = GRID_STEP * 1.15

    for i in range(len(centroids)):
        neighbors = tree.query_ball_point(centroids[i], max_dist)
        for j in neighbors:
            if j <= i:
                continue
            dx = abs(centroids[i, 0] - centroids[j, 0])
            dy = abs(centroids[i, 1] - centroids[j, 1])
            # 只连接正交方向（4邻接），排除对角
            h_adj = (dx < EDGE_DIST_TOLERANCE) and (abs(dy - GRID_STEP) < EDGE_DIST_TOLERANCE)
            v_adj = (dy < EDGE_DIST_TOLERANCE) and (abs(dx - GRID_STEP) < EDGE_DIST_TOLERANCE)
            if h_adj or v_adj:
                w_i = float(weights[i])
                w_j = float(weights[j])
                if weight_mode == "min":
                    w = min(w_i, w_j)
                else:
                    w = (w_i + w_j) / 2.0
                G.add_edge(i, j, weight=w, frac_deleted=None)

    # 过滤孤立节点（degree=0 的节点对渗流无意义，但保留用于完整计数）
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    n_isolated = sum(1 for _, deg in G.degree() if deg == 0)
    logger.info("构建图：%d 节点, %d 边, %d 孤立节点", n_nodes, n_edges, n_isolated)
    return G


# ---------------------------------------------------------------------------
# Union-Find（并查集）— 高效增量连通分量追踪
# ---------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self._size = [1] * n
        self._max = 1 if n > 0 else 0

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._size[rx] < self._size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self._size[rx] += self._size[ry]
        if self._size[rx] > self._max:
            self._max = self._size[rx]

    def max_component_size(self) -> int:
        return self._max


# ---------------------------------------------------------------------------
# 渗流模拟（Union-Find 优化，O(E log E + E·α(N))）
# ---------------------------------------------------------------------------
def simulate_percolation(
    G: "nx.Graph",
    n_steps: int = 100,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    递增删边渗流模拟（逆序 Union-Find 实现，精确 O(E log E)）。

    策略：
      - 按边权重从弱到强排序
      - 从空图开始，逆序归入最强边 → 用并查集增量追踪最大分量
      - 得到精确的渗流曲线（每条边对应一个状态点）
      - 线形插值到 n_steps 个均匀分位点

    返回：
      fractions:  (n_steps+1,) 已删除边占比
      sizes:      (n_steps+1,) 最大连通分量节点占比
      threshold:  渗流阈值（最大分量跌至 50% 时的删边占比）
    """
    edges = sorted(G.edges(data="weight"), key=lambda e: e[2])
    total_edges = len(edges)
    total_nodes = G.number_of_nodes()

    if total_edges == 0:
        fractions = np.linspace(0, 1, n_steps + 1)
        sizes = np.full(n_steps + 1, 1.0 / max(total_nodes, 1))
        return fractions, sizes, 0.0

    # 逆序：从空图开始，逐条加入最强边
    uf = _UnionFind(total_nodes)
    sizes_reverse = [1.0 / max(total_nodes, 1)]  # f=1.0: 无边

    for u, v, _ in reversed(edges):
        uf.union(int(u), int(v))
        sizes_reverse.append(uf.max_component_size() / max(total_nodes, 1))

    # sizes_reverse[i] = 在归入 i 条最强边后的最大分量占比
    # = 在删除 total_edges - i 条最弱边后的状态
    # →  delete_frac = (total_edges - i) / total_edges
    sizes_forward = np.array(list(reversed(sizes_reverse)), dtype=np.float64)
    x_exact = np.linspace(0, 1, total_edges + 1)  # 精确分位

    # 插值到 n_steps 个均匀点
    x_interp = np.linspace(0, 1, n_steps + 1)
    sizes = np.interp(x_interp, x_exact, sizes_forward)

    # 渗流阈值
    threshold = 1.0
    for i, s in enumerate(sizes):
        if s < 0.5:
            threshold = float(x_interp[i])
            break

    return x_interp, sizes, threshold


def find_percolation_threshold(
    fractions: np.ndarray,
    sizes: np.ndarray,
    method: str = "half",
) -> float:
    """
    计算渗流阈值（最大连通分量崩塌的临界点）。

    方法：
      "half"     — sizes 首次低于 0.5 时的 fractions
      "steepest" — sizes 下降最快处（一阶差分极小值）
      "span"     — sizes 首次低于 1/sqrt(总节点数)（跨越判定）
    """
    if method == "half":
        for i in range(len(sizes)):
            if sizes[i] < 0.5:
                return float(fractions[i])
        return 1.0

    if method == "steepest":
        if len(sizes) < 3:
            return 0.5
        diff = np.diff(sizes)
        idx = np.argmin(diff)
        return float(fractions[min(idx + 1, len(fractions) - 1)])

    if method == "span":
        n_nodes = 1.0 / max(sizes[0], 1e-10)
        span_threshold = 1.0 / max(np.sqrt(n_nodes), 1e-10)
        for i in range(len(sizes)):
            if sizes[i] < span_threshold:
                return float(fractions[i])
        return 1.0

    return 0.5


# ---------------------------------------------------------------------------
# 关键节点识别
# ---------------------------------------------------------------------------
def _identify_boundary_nodes(G: "nx.Graph", gdf) -> set:
    """
    识别边界节点：位于研究区矩形边界上的节点。
    这些节点的 betweenness 会因边界效应而虚高。
    """
    import numpy as np
    centroids = gdf.geometry.centroid
    xs = np.array([p.x for p in centroids])
    ys = np.array([p.y for p in centroids])

    # 找到边界范围
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # 网格步长（约3000m）
    step = 3000.0
    tol = step * 0.5

    boundary_nodes = set()
    for i in range(len(gdf)):
        # 检查是否在矩形边界上
        on_left = abs(xs[i] - x_min) < tol
        on_right = abs(xs[i] - x_max) < tol
        on_bottom = abs(ys[i] - y_min) < tol
        on_top = abs(ys[i] - y_max) < tol

        if on_left or on_right or on_bottom or on_top:
            boundary_nodes.add(i)

    return boundary_nodes


def identify_key_nodes(
    G: "nx.Graph",
    top_n: int = 20,
    k_sample: int = 200,
    exclude_boundary: bool = False,
    gdf=None,
    use_pagerank: bool = False,
) -> pd.DataFrame:
    """
    识别图中的关键节点。

    参数：
      exclude_boundary: 排除边界节点（减少边界效应）
      use_pagerank: 使用 PageRank 替代 betweenness（对边界效应更鲁棒）
      gdf: 用于识别边界节点的 GeoDataFrame（exclude_boundary=True 时必须）

    返回 DataFrame: [node_idx, centrality, pos_x, pos_y, weight]
    """
    if G.number_of_edges() == 0:
        return pd.DataFrame(columns=["node_idx", "centrality", "pos_x", "pos_y", "weight"])

    n_nodes = G.number_of_nodes()

    # 计算中心性
    if use_pagerank:
        centrality = nx.pagerank(G, weight="weight", alpha=0.85)
        metric_name = "pagerank"
    else:
        k = min(k_sample, n_nodes)
        centrality = nx.betweenness_centrality(G, k=k, weight="weight", normalized=True, seed=42)
        metric_name = "betweenness"

    # 排除边界节点
    if exclude_boundary and gdf is not None:
        boundary_nodes = _identify_boundary_nodes(G, gdf)
        logger.info("边界节点: %d / %d (%.1f%%)", len(boundary_nodes), n_nodes,
                    len(boundary_nodes) / n_nodes * 100)
        # 将边界节点的中心性设为0，不参与排序
        for node_idx in boundary_nodes:
            if node_idx in centrality:
                centrality[node_idx] = 0.0

    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]

    rows = []
    for node_idx, c_val in sorted_nodes:
        pos = G.nodes[node_idx].get("pos", (0, 0))
        w = G.nodes[node_idx].get("weight", 0)
        rows.append({
            "node_idx": node_idx,
            "centrality": round(c_val, 8),
            "pos_x": pos[0],
            "pos_y": pos[1],
            "weight": w,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def plot_percolation_curves(
    results: Dict[str, dict],
    out_path: str,
    title: str = "三期渗流曲线对比",
) -> str:
    """绘制三期渗流曲线对比图（最大连通分量大小 vs 删边占比）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    try:
        from utils.matplotlib_chinese import setup_matplotlib_chinese
        setup_matplotlib_chinese()
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#E63946", "#457B9D", "#2A9D8F"]

    # 不同的y位置避免文字重叠
    y_positions = [0.85, 0.75, 0.65]

    for i, ((period_name, r), color) in enumerate(zip(results.items(), colors)):
        frac = r["fractions"]
        sizes = r["sizes"]
        threshold = r["threshold"]
        ax.plot(frac, sizes, color=color, linewidth=2, label=period_name)
        ax.axvline(x=threshold, color=color, linestyle="--", alpha=0.6, linewidth=1.5)
        ax.annotate(f"p_c={threshold:.3f}", (threshold + 0.01, y_positions[i]),
                    color=color, fontsize=9, fontweight="bold")

    # 50% 参考线
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.text(0.01, 0.53, "50% 参考线", color="gray", fontsize=8)

    ax.set_xlabel("已删除边占比 (f)")
    ax.set_ylabel("最大连通分量节点占比")
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_key_nodes_map(
    gdf: gpd.GeoDataFrame,
    key_nodes_df: pd.DataFrame,
    out_path: str,
    period_name: str = "",
    title_prefix: str = "",
) -> str:
    """绘制关键节点空间分布图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    try:
        from utils.matplotlib_chinese import setup_matplotlib_chinese
        setup_matplotlib_chinese()
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 10))

    # 底图：网格轮廓
    _, X, _ = get_topology_matrix(gdf)
    nonzero = X.sum(axis=1) > 0
    gdf[nonzero].boundary.plot(ax=ax, color="#457B9D", linewidth=0.5, alpha=0.4)
    gdf[~nonzero].boundary.plot(ax=ax, color="#cccccc", linewidth=0.2, alpha=0.1)

    # 关键节点（归一化 centrality 用于尺寸和颜色）
    bc = key_nodes_df["centrality"].values
    bc_min, bc_max = bc.min(), bc.max()
    if bc_max - bc_min > 1e-10:
        bc_norm = (bc - bc_min) / (bc_max - bc_min)
    else:
        bc_norm = np.ones_like(bc) * 0.5  # 所有值相等时给默认尺寸

    sc = ax.scatter(
        key_nodes_df["pos_x"], key_nodes_df["pos_y"],
        c=bc_norm,
        s=bc_norm * 300 + 30,  # 归一化后尺寸范围 30~330
        cmap="YlOrRd",
        edgecolors="#333333",
        linewidths=0.5,
        zorder=5,
        vmin=0, vmax=1,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Centrality (normalized)", fontsize=10)

    title = f"{period_name} 关键节点分布" if period_name else "关键节点分布"
    if title_prefix:
        title = f"{title_prefix} — {title}"
    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_percolation_pipeline(
    out_dir: Optional[str] = None,
    edge_weight_col: str = "NC_A",
    n_steps: int = 100,
) -> dict:
    """
    Stage 3：图渗流模拟主流程。

    对三期数据分别：
      1. 构建邻接图
      2. 模拟递增删边渗流
      3. 计算渗流阈值
      4. 识别关键节点
      5. 可视化

    返回 dict，key 为时期名。
    """
    if not HAS_NX:
        raise ImportError("图渗流需要 networkx: pip install networkx")

    if out_dir is None:
        out_dir = os.path.join(_THIS_DIR, "data", "processed", "multiperiod")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=== Stage 3: 图渗流模拟 ===")

    period_gdfs = load_all_periods()
    results = {}

    for period_name, gdf in period_gdfs.items():
        logger.info("--- %s ---", period_name)

        # 1) 构建图
        G = build_grid_graph(gdf, edge_weight_col=edge_weight_col, weight_mode="min")

        # 2) 渗流模拟
        fractions, sizes, threshold = simulate_percolation(G, n_steps=n_steps)

        # 3) 渗流阈值（多种方法）
        threshold_steepest = find_percolation_threshold(fractions, sizes, method="steepest")

        # 4) 关键节点（排除边界节点 + 使用 PageRank 减少边界效应）
        key_nodes = identify_key_nodes(
            G, top_n=30, exclude_boundary=True, use_pagerank=True, gdf=gdf
        )

        logger.info("  渗流阈值(50%%): %.3f  最陡点: %.3f  关键节点: %d",
                    threshold, threshold_steepest, len(key_nodes))

        # 5) 保存
        results[period_name] = {
            "graph": G,
            "fractions": fractions,
            "sizes": sizes,
            "threshold": threshold,
            "threshold_steepest": threshold_steepest,
            "key_nodes": key_nodes,
            "gdf": gdf,
        }

    # 汇总表
    summary_rows = []
    for period_name, r in results.items():
        summary_rows.append({
            "时期": period_name,
            "渗流阈值_pc_50%": round(r["threshold"], 4),
            "渗流阈值_pc_最陡": round(r["threshold_steepest"], 4),
            "关键节点数": len(r["key_nodes"]),
            "最大Centrality": round(r["key_nodes"]["centrality"].max(), 8)
            if len(r["key_nodes"]) > 0 else 0,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "percolation_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info("汇总表: %s", summary_path)

    # 可视化
    plot_paths = {}
    try:
        curve_path = os.path.join(out_dir, "percolation_curves.png")
        plot_percolation_curves(results, curve_path)
        plot_paths["curves"] = curve_path
        logger.info("渗流曲线: %s", curve_path)

        for period_name, r in results.items():
            kn_path = os.path.join(out_dir,
                                   f"key_nodes_{period_name}.png")
            plot_key_nodes_map(r["gdf"], r["key_nodes"], kn_path,
                              period_name=period_name)
            plot_paths[f"key_nodes_{period_name}"] = kn_path
        logger.info("关键节点图: %d 张", len(results))
    except Exception as e:
        logger.warning("可视化失败: %s", e)

    results["_summary"] = summary_df
    results["_plot_paths"] = plot_paths
    results["_out_dir"] = out_dir
    return results


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_percolation_pipeline()
    summary = results["_summary"]
    print("\n=== 渗流模拟结果 ===")
    print(summary.to_string(index=False))
