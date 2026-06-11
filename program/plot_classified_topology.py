# -*- coding: utf-8 -*-
"""
单张「分类+拓扑」图：分支/节点按类型着色（CC/CI/II 分支 + X/Y/I 节点），保存 plot13.pdf
新增：三期数据分类拓扑图函数
"""
import math
from fractopo.general import CC_branch, CI_branch, II_branch, X_node, Y_node, I_node
from fractopo import Network
import geopandas as gpd
import warnings
import matplotlib.pyplot as plt
import numpy as np

from utils.matplotlib_chinese import setup_matplotlib_chinese
setup_matplotlib_chinese()


def assign_colors(feature_type: str):
    """
    根据 fractopo 分支/节点类型分配颜色：
    CC_branch / X_node → 绿色（贯通型，最高连通）
    CI_branch / Y_node → 蓝色（半贯通型）
    II_branch / I_node → 黑色（孤立型）
    其他 → 红色
    """
    if feature_type in (CC_branch, X_node):
        return "green"
    if feature_type in (CI_branch, Y_node):
        return "blue"
    if feature_type in (II_branch, I_node):
        return "black"
    return "red"


def plot_classified_for_period(gdf, period_name: str, out_path: str = None) -> plt.Figure:
    """
    为单个时期绘制分类拓扑图。

    参数：
        gdf: GeoDataFrame，包含geometry列
        period_name: 时期名称
        out_path: 保存路径（可选）

    返回：
        matplotlib Figure对象
    """
    from multiperiod_data import get_topology_matrix

    # 获取拓扑矩阵
    _, X, cols = get_topology_matrix(gdf)
    nonzero = np.array(X.sum(axis=1)).ravel() > 0

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制网格（非零网格着色，零值网格灰色）
    gdf[~nonzero].plot(ax=ax, facecolor='#f0f0f0', edgecolor='#cccccc', linewidth=0.3, alpha=0.5)
    gdf[nonzero].plot(ax=ax, facecolor='#457B9D', edgecolor='none', alpha=0.6)

    # 添加拓扑属性标注
    for j, col in enumerate(cols):
        vals = X[nonzero, j]
        if len(vals) > 0:
            mean_val = vals.mean()
            ax.text(0.02, 0.98 - j * 0.05, f'{col}: {mean_val:.4f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(f'{period_name} 分类拓扑图\n(非零网格: {nonzero.sum()}/{len(gdf)})',
                fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')

    return fig


def plot_all_periods(period_gdfs: dict, out_dir: str = None) -> list:
    """
    为所有时期绘制分类拓扑图。

    参数：
        period_gdfs: {"海西期": gdf, "喜山期": gdf, "印支燕山期": gdf}
        out_dir: 保存目录（可选）

    返回：
        list of matplotlib Figure对象
    """
    figs = []
    for period_name, gdf in period_gdfs.items():
        out_path = None
        if out_dir:
            import os
            out_path = os.path.join(out_dir, f'classified_topology_{period_name}.png')
        fig = plot_classified_for_period(gdf, period_name, out_path)
        figs.append(fig)
    return figs


# ── 原有代码（保留向后兼容）──
if __name__ == "__main__":
    # ── 加载 THK 区数据 ──
    trace_data_url = "THK/thkceshi-landmark1.geojson"
    area_data_url = "THK/my_area.geojson"
    traces = gpd.read_file(trace_data_url)
    area = gpd.read_file(area_data_url)
    name = "MY"

    # 计算迹线空间范围，用于设置图幅比例与边距
    geometry = traces.geometry.tolist()
    left, right, down, up = math.inf, -math.inf, math.inf, -math.inf
    for one in geometry:
        left = min(left, one.boundary.bounds[0])
        right = max(right, one.boundary.bounds[2])
        down = min(down, one.boundary.bounds[1])
        up = max(up, one.boundary.bounds[3])
    rate = (up - down) / (right - left)  # 图幅宽高比
    width, height = 0.01 * (right - left), 0.01 * (up - down)  # 边距

    warnings.filterwarnings("ignore")

    # ── 构建 fractopo Network，自动识别分支/节点类型 ──
    network = Network(traces, area, name=name, determine_branches_nodes=True,
                      truncate_traces=True, circular_target_area=False, snap_threshold=0.001)

    # ── 绘制：分支按类型着色 + 节点按类型着色 ──
    fig, ax = plt.subplots(figsize=(9, 9 * rate))
    network.branch_gdf.plot(colors=[assign_colors(bt) for bt in network.branch_types], ax=ax)
    network.trace_gdf.plot(ax=ax, linewidth=0.5)
    network.node_gdf.plot(
        c=[assign_colors(bt) for bt in network.node_types], ax=ax, markersize=10
    )
    # 研究区边界
    area.boundary.plot(ax=ax, color="red")

    # 图例
    handles = [
        plt.Line2D([0], [0], color="green", lw=2, label="CC_branch / X_node"),
        plt.Line2D([0], [0], color="blue", lw=2, label="CI_branch / Y_node"),
        plt.Line2D([0], [0], color="black", lw=2, label="II_branch / I_node"),
        plt.Line2D([0], [0], color="red", lw=2, label="Other / Boundary"),
    ]
    ax.legend(handles=handles, loc='lower left')
    plt.xlim((left - width, right + width))
    plt.ylim((down - height, up + height))
    ax.set_aspect('equal')
    plt.savefig('plot13.pdf')
    plt.show()
