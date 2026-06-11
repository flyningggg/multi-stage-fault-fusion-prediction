# -*- coding: utf-8 -*-
"""
两网络拓扑参数对比：KB11 vs THK，画柱状图保存 plot13.pdf
新增：三期数据拓扑参数对比函数
"""
import warnings
import geopandas as gpd
from fractopo import Network
from fractopo.analysis.parameters import plot_parameters_plot
from matplotlib import pyplot as plt
import numpy as np

from utils.matplotlib_chinese import setup_matplotlib_chinese
setup_matplotlib_chinese()
warnings.filterwarnings("ignore")


def compare_three_periods(period_gdfs: dict, out_path: str = None) -> tuple:
    """
    对比三期数据的拓扑参数。

    参数：
        period_gdfs: {"海西期": gdf, "喜山期": gdf, "印支燕山期": gdf}
        out_path: 保存路径（可选）

    返回：
        (fig, ax) matplotlib图形对象
    """
    from multiperiod_data import get_topology_matrix

    # 收集所有时期的拓扑参数
    all_params = {}
    for period_name, gdf in period_gdfs.items():
        _, X, cols = get_topology_matrix(gdf)
        nonzero = np.array(X.sum(axis=1)).ravel() > 0
        params = {}
        for j, col in enumerate(cols):
            vals = X[nonzero, j]
            if len(vals) > 0:
                params[col] = {
                    'mean': vals.mean(),
                    'std': vals.std(),
                    'min': vals.min(),
                    'max': vals.max(),
                }
        all_params[period_name] = params

    # 绘制对比柱状图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, col in enumerate(cols):
        ax = axes[idx]
        period_names = []
        means = []
        stds = []
        for period_name in period_gdfs.keys():
            if period_name in all_params and col in all_params[period_name]:
                period_names.append(period_name)
                means.append(all_params[period_name][col]['mean'])
                stds.append(all_params[period_name][col]['std'])

        x = np.arange(len(period_names))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                     color=['#E63946', '#457B9D', '#2A9D8F'][:len(period_names)],
                     alpha=0.8)
        ax.set_xlabel('时期')
        ax.set_ylabel(col)
        ax.set_title(f'{col} 对比')
        ax.set_xticks(x)
        ax.set_xticklabels(period_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('三期拓扑参数对比', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')

    return fig, axes


# ── 原有代码（保留向后兼容）──
if __name__ == "__main__":
    # ── 加载 KB11 区网络 ──
    traces = gpd.read_file("KB11/KB11_traces.geojson")
    area = gpd.read_file("KB11/my_area1.geojson")
    name = "KB11"
    KB11 = Network(traces, area, name=name, determine_branches_nodes=True,
                   truncate_traces=True, circular_target_area=False, snap_threshold=0.001)

    # ── 加载 THK 区网络（作为 MY 对比）──
    traces = gpd.read_file("THK/thkceshi-landmark1.geojson")
    area = gpd.read_file("THK/my_area.geojson")
    name = "MY"
    MY = Network(traces, area, name=name, determine_branches_nodes=True,
                 truncate_traces=True, circular_target_area=False, snap_threshold=0.001)

    # ── 选定对比的拓扑参数 ──
    b22 = "Dimensionless Intensity B22"  # 无量纲强度
    cpb = "Connections per Branch"        # 每条分支连接节点数
    selected = {b22, cpb}

    # 提取两个网络的选定参数
    kb11_network_selected_params = {
        param: value for param, value in KB11.parameters.items() if param in selected
    }
    kb7_network_selected_params = {
        param: value for param, value in MY.parameters.items() if param in selected
    }

    # fractopo 内置的参数对比柱状图
    figs, axes = plot_parameters_plot(
        topology_parameters_list=[kb11_network_selected_params, kb7_network_selected_params],
        labels=["KB11", "MY"],
        colors=["red", "blue"],
    )
    plt.savefig('plot13.pdf')
    plt.show()
