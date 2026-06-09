"""网格采样测试：检查网格设置与参数，输出 PDF 含拓扑参数等值线图"""
import math
import warnings
import geopandas as gpd
from fractopo import Network
from matplotlib import pyplot as plt
import time

from utils.matplotlib_chinese import setup_matplotlib_chinese
setup_matplotlib_chinese()

time_start = time.time()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
