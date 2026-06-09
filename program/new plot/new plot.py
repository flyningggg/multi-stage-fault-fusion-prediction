import sys
import os
# 确保能导入 program 下的 utils
_PROGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)

import geopandas as gpd
from fractopo import Network
from scipy.spatial import distance_matrix
from skimage import filters
from preprocessing import *
from metrics import *
from edits import *
from plots import *
from utils.matplotlib_chinese import setup_matplotlib_chinese
setup_matplotlib_chinese()

from analyis import FracAnalysisPoly, FancyPlot
from analyis_test import geojson2shp
from matplotlib.colors import ListedColormap

trace_data_url = "thkceshi-landmark1.geojson"
area_data_url = "my_area.geojson"
traces = gpd.read_file(trace_data_url)
area = gpd.read_file(area_data_url)
traces.drop_duplicates(subset="geometry", inplace=True)
traces.reset_index(drop=True, inplace=True)
name = "The top surface fault of the Ordovician Yijianfang Formation in a three-dimensional area of northern Tarim Basin"

kb11_network = Network(traces, area, name=name, determine_branches_nodes=True, truncate_traces=True, circular_target_area=True, snap_threshold=0.001)

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_aspect('equal')
traces.plot(ax=ax, color='blue', aspect=1)
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.savefig('myplot01.png')

img = Image.open('myplot01.png')
region = img.crop((140, 140, 730, 730))

region.save('myplot02.png')
img = Image.open('myplot02.png').convert('L')

data = np.array(img)
data = 1 - (data - np.min(data)) / (np.max(data) - np.min(data))
smoothed = filters.gaussian(data, sigma=1)
threshold = simple_threshold_binary(smoothed, 0.5)
skeleton = skeleton_guo_hall(threshold)
ret, markers = cv2.connectedComponents(skeleton)
G = nx.Graph()
node = 0
for comp in tqdm(range(1, ret)):
    points = np.transpose(np.vstack((np.where(markers == comp))))
    for point in points:
        G.add_node(node)
        G.nodes[node]['pos'] = (point[1], point[0])
        G.nodes[node]['component'] = comp
        node += 1
for comp in tqdm(range(1, ret)):
    points = [G.nodes[node]['pos'] for node in G if G.nodes[node]['component'] == comp]
    nodes = [node for node in G if G.nodes[node]['component'] == comp]
    dm = distance_matrix(points, points)
    for n in range(len(points)):
        for m in range(len(points)):
            if dm[n, m] < 1.5 and n != m:
                G.add_edge(nodes[n], nodes[m])
G = label_components(G)
G = split_triple_junctions(G, 25)
G = label_components(G)
G = compute_edge_length(G)
G = calculate_strike(G, 3)
top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('Blues', 128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
orange_blue = ListedColormap(newcolors, name='OrangeBlue')

fig, ax = plt.subplots(figsize=(15, 15))
im = ax.imshow(data, cmap='Greys', vmin=0, vmax=1)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(im, cax=cax)
cbar.ax.set_ylabel('Intensity')

plot_attribute(G, 'strike', cmap=orange_blue, vmin=0, vmax=180, ax=ax)

plt.savefig('myplot12.png')
plt.show()

out_shp_url = 'json2shp3.shp'
geojson2shp(area_data_url, out_shp_url)
cell_size, angle_bins = 40, 5
b = FracAnalysisPoly(out_shp_url, cell_size, angle_bins)
FancyPlot(b, Patches="Number", Circles=True)
