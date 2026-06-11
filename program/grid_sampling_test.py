# -*- coding: utf-8 -*-
"""
网格采样测试：检查网格设置与参数，输出 PDF 含拓扑参数等值线图
新增：三期数据网格采样分析函数
"""
import math
import warnings
import geopandas as gpd
from fractopo import Network
from matplotlib import pyplot as plt
import time
import numpy as np

from utils.matplotlib_chinese import setup_matplotlib_chinese
setup_matplotlib_chinese()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_three_period_grid_sampling(period_gdfs: dict, width: int = 1000, out_dir: str = None) -> dict:
    """
    为三期数据运行网格采样分析。

    参数：
        period_gdfs: {"海西期": gdf, "喜山期": gdf, "印支燕山期": gdf}
        width: 网格步长（米）
        out_dir: 保存目录（可选）

    返回：
        dict: {period_name: {'stats': {...}, 'density': float, ...}}
    """
    from multiperiod_data import get_topology_matrix
    import os

    results = {}
    for period_name, gdf in period_gdfs.items():
        _, X, cols = get_topology_matrix(gdf)
        nonzero = np.array(X.sum(axis=1)).ravel() > 0

        # 计算每个属性的统计信息
        stats = {}
        for j, col in enumerate(cols):
            vals = X[nonzero, j]
            if len(vals) > 0:
                stats[col] = {
                    'mean': vals.mean(),
                    'std': vals.std(),
                    'min': vals.min(),
                    'max': vals.max(),
                    'median': np.median(vals),
                    'count': len(vals),
                }

        # 计算网格密度
        total_count = len(gdf)
        nonzero_count = nonzero.sum()
        density = nonzero_count / total_count if total_count > 0 else 0

        results[period_name] = {
            'stats': stats,
            'nonzero_count': nonzero_count,
            'total_count': total_count,
            'density': density,
            'cols': cols,
        }

        # 生成等值线图（如果需要）
        if out_dir:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            for idx, col in enumerate(cols):
                ax = axes[idx]
                vals = X[nonzero, idx]
                if len(vals) > 0:
                    ax.hist(vals, bins=30, color='#457B9D', alpha=0.7, edgecolor='white')
                    ax.set_xlabel(col)
                    ax.set_ylabel('频次')
                    ax.set_title(f'{period_name} {col}')
                    ax.grid(True, alpha=0.3)

            plt.suptitle(f'{period_name} 网格采样分析', fontsize=14, fontweight='bold')
            plt.tight_layout()

            out_path = os.path.join(out_dir, f'grid_sampling_{period_name}.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    return results


# ── 原有代码（保留向后兼容）──
if __name__ == "__main__":
    time_start = time.time()

    # data_option 切换要测试的区域
    data_option = 2
    if data_option == 1:
        traces = gpd.read_file("KB11/KB11_traces.geojson")
        area = gpd.read_file("KB11/my_area1.geojson")
        name = "KB11"
    elif data_option == 2:
        traces = gpd.read_file("THK/thkceshi-landmark1.geojson")
        area = gpd.read_file("THK/my_area.geojson")
        name = "THK"
    elif data_option == 3:
        traces = gpd.read_file("MY/11.geojson")
        area = gpd.read_file("MY/my_area1.geojson")
        name = "MY"

    # ── 构建 fractopo Network ──
    network = Network(traces, area, name=name, determine_branches_nodes=True,
                      truncate_traces=True, circular_target_area=False, snap_threshold=0.001)

    # ── 统计迹线空间范围，辅助选择合适的 cell_width ──
    width = 1000  # 网格步长（米）
    geometry = traces.geometry.tolist()
    left, right, down, up = math.inf, -math.inf, math.inf, -math.inf
    for one in geometry:
        left = min(left, one.boundary.bounds[0])
        right = max(right, one.boundary.bounds[2])
        down = min(down, one.boundary.bounds[1])
        up = max(up, one.boundary.bounds[3])
    print('左边界:', left, '右边界:', right, '上边界:', up, '下边界:', down)
    print('X轴范围:', right - left, 'Y轴范围:', up - down)
    print('网格数:', int((right - left) / width), '*', int((up - down) / width), '=',
          int((right - left) / width * (up - down) / width))

    # ── 网格采样 + 等值线图 ──
    sampled_grid = network.contour_grid(cell_width=width)
    parameter = "Number of Traces (Real)"  # 待可视化的拓扑参数
    network.plot_contour(parameter=parameter, sampled_grid=sampled_grid)
    plt.savefig(name + '-' + str(width) + '-' + parameter + '.pdf')
    plt.show()
    print('用时:', time.time() - time_start, '秒')
