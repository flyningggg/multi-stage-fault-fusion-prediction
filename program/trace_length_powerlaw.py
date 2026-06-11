# -*- coding: utf-8 -*-
"""
多网络迹线长度分布与幂律拟合：KB11、MY、THK 三区对比，保存 plot14.pdf
新增：三期数据长度分布对比函数
"""
import warnings
import matplotlib as mpl
import geopandas as gpd
import joblib
import shutil
from fractopo import Network
from fractopo.analysis import length_distributions
from fractopo import general
from matplotlib import pyplot as plt
import numpy as np

from utils.matplotlib_chinese import setup_matplotlib_chinese
setup_matplotlib_chinese()

# 清除 fractopo 的缓存（避免先前运行影响）
shutil.rmtree('.cache/fractopo', ignore_errors=True)

warnings.filterwarnings("ignore")


def plot_three_period_length_distribution(period_gdfs: dict, out_path: str = None) -> tuple:
    """
    绘制三期数据的长度分布对比图。

    参数：
        period_gdfs: {"海西期": gdf, "喜山期": gdf, "印支燕山期": gdf}
        out_path: 保存路径（可选）

    返回：
        (fig, axes) matplotlib图形对象
    """
    from multiperiod_data import get_topology_matrix

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#E63946', '#457B9D', '#2A9D8F']

    for ax, (period_name, gdf), color in zip(axes, period_gdfs.items(), colors):
        _, X, cols = get_topology_matrix(gdf)
        nonzero = np.array(X.sum(axis=1)).ravel() > 0

        # 提取长度相关属性
        length_cols = ['NC_NB', 'NC_NL', 'NB_NL']
        for col in length_cols:
            if col in cols:
                idx = cols.index(col)
                vals = X[nonzero, idx]
                if len(vals) > 0:
                    ax.hist(vals, bins=30, alpha=0.5, label=col, edgecolor='white')

        ax.set_xlabel('值')
        ax.set_ylabel('频次')
        ax.set_title(f'{period_name}\n(非零网格: {nonzero.sum()})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('三期长度分布对比', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')

    return fig, axes


# ── 原有代码（保留向后兼容）──
if __name__ == "__main__":
    mpl.rcParams["figure.figsize"] = (5, 5)
    mpl.rcParams["font.size"] = 8

    # ── 加载 KB11 区数据 ──
    traces = gpd.read_file("KB11/KB11_traces.geojson")
    area = gpd.read_file("KB11/my_area1.geojson")
    name = "KB11"
    KB11 = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True,
                   circular_target_area=False, snap_threshold=0.001)

    # ── 加载 MY（英买2）区数据 ──
    traces = gpd.read_file("MY/11.geojson")
    area = gpd.read_file("MY/my_area1.geojson")
    name = "MY"
    MY = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True,
                 circular_target_area=False, snap_threshold=0.001)

    # ── 加载 THK 区数据 ──
    traces = gpd.read_file("THK/thkceshi-landmark1.geojson")
    area = gpd.read_file("THK/my_area.geojson")
    name = "THK"
    THK = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True,
                  circular_target_area=False, snap_threshold=0.001)

    # ── 计算三个网络的迹线长度分布 ──
    networks = [KB11, MY, THK]
    distributions = [netw.trace_length_distribution(azimuth_set=None) for netw in networks]
    # MultiLengthDistribution：多网络长度分布联合对比 + 幂律拟合
    mld = length_distributions.MultiLengthDistribution(
        distributions=distributions,
        using_branches=False,
        fitter=length_distributions.scikit_linear_regression
    )

    # ── 自动优化截断阈值，使幂律拟合 R² 最大 ──
    shgo_kwargs = dict(sampling_method="sobol")  # SHGO 全局优化采样方法
    scorer = general.r2_scorer  # 用 R² 评估拟合质量
    opt_result, opt_mld = mld.optimize_cut_offs(scorer=scorer)

    # 绘制多网络长度分布 + 最优幂律拟合曲线
    polyfit, fig, ax = opt_mld.plot_multi_length_distributions(
        automatic_cut_offs=False, scorer=scorer, plot_truncated_data=True
    )
    print(f""" Optimized cut-offs: {opt_result.optimize_result.x}
    Resulting power-law exponent: {opt_result.polyfit.m_value}
    Resulting {scorer.__name__} score: {opt_result.polyfit.score} """)

    plt.tight_layout()
    plt.savefig('plot14.pdf')
    plt.show()
