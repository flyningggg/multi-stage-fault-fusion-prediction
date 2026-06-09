"""两网络拓扑参数对比：KB11 vs THK，画柱状图保存 plot13.pdf"""
import warnings
import geopandas as gpd
from fractopo import Network
from fractopo.analysis.parameters import plot_parameters_plot
from matplotlib import pyplot as plt

from utils.matplotlib_chinese import setup_matplotlib_chinese
setup_matplotlib_chinese()
warnings.filterwarnings("ignore")

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
