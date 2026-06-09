# 导入
import sys
import os
_PROGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)

import numpy as np
from fractopo.general import CC_branch, CI_branch, II_branch, X_node, Y_node, I_node
import fractopo
from fractopo import Network
import geopandas as gpd
import warnings
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

from utils.matplotlib_chinese import setup_matplotlib_chinese
setup_matplotlib_chinese()
from shapely.geometry import Polygon, MultiPolygon
import geojson

# 无视警告
warnings.filterwarnings("ignore")
# 读取本地
trace_data_url = "thkceshi-landmark1.geojson"
traces = gpd.read_file(trace_data_url)
area_data_url = "my_area.geojson"
area = gpd.read_file(area_data_url)
name = "KB11"


def assign_colors(feature_type: str):
    if feature_type in (CC_branch, X_node):
        return "green"
    if feature_type in (CI_branch, Y_node):
        return "blue"
    if feature_type in (II_branch, I_node):
        return "black"
    return "red"


trace_data_url = "thkceshi-landmark1.geojson"
traces = gpd.read_file(trace_data_url)
area_data_url = "my_area.geojson"
area = gpd.read_file(area_data_url)
name = "The top surface fault of the Ordovician Yijianfang Formation in a three-dimensional area of northern Tarim Basin"

# Drop duplicates from the trace GeoDataFrame
traces.drop_duplicates(subset="geometry", inplace=True)

# Reset the index of the GeoDataFrame
traces.reset_index(drop=True, inplace=True)

# Initialize the figure and ax in which data is plotted
fig, ax = plt.subplots(figsize=(9, 9))

# Plot the loaded trace dataset consisting of fracture traces.
traces.plot(ax=ax, color="blue")

# Plot the loaded area dataset that consists of a single polygon that delineates the traces.
area.boundary.plot(ax=ax, color="red")

# Give the figure a title
ax.set_title(f"{name}, Coordinate Reference System = {traces.crs}")

# fractopo处理
kb11_network = Network(
    traces,
    area,
    name=name,
    determine_branches_nodes=True,
    truncate_traces=True,
    circular_target_area=False,
    snap_threshold=0.001,
)

# 按颜色绘制处理后的线条
fig, ax = plt.subplots(figsize=(9, 9))
# Traces
kb11_network.trace_gdf.plot(
    colors=[assign_colors(bt) for bt in kb11_network.branch_types], ax=ax, linewidth=0.5
)
# Nodes
kb11_network.node_gdf.plot(
    c=[assign_colors(bt) for bt in kb11_network.node_types], ax=ax, markersize=10
)
area.boundary.plot(ax=ax, color="red")
plt.show()

# #fractopo处理
# kb11_network = Network(
#     traces,
#     area ,
#     name=name,
#     determine_branches_nodes=True,
#     truncate_traces=True,
#     circular_target_area=False,
#     snap_threshold=0.001,
# )
#
# print(kb11_network.parameters)
#
#
#
# #定义线条颜色
# def assign_colors(feature_type: str):
#     if feature_type in (CC_branch, X_node):
#         return "green"
#     if feature_type in (CI_branch, Y_node):
#         return "blue"
#     if feature_type in (II_branch, I_node):
#         return "black"
#     return "red"
#
#
#
# #按颜色绘制处理后的线条
# fix, ax = plt.subplots(figsize=(9, 9))
# # Traces
# kb11_network.trace_gdf.plot(ax=ax, linewidth=0.5)
# # Nodes
# kb11_network.node_gdf.plot(
#     c=[assign_colors(bt) for bt in kb11_network.node_types], ax=ax, markersize=10
# )
# area.boundary.plot(ax=ax, color="red")
# plt.show()
#
