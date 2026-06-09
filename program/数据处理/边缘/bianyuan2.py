import numpy as np
from shapely.geometry import MultiPoint, Polygon, LineString
import json
from shapely.geometry import LineString, MultiLineString, Point

# 边缘识别2
m = 0
with open('断裂数据-测试用2.xyz', 'r') as f:
    # 读取原子数量
    natoms = len(f.readlines())
    f.seek(0)
    # 读取原子坐标
    coords = np.zeros((natoms, 2))
    for i in range(natoms):
        line = f.readline().split(',')
        if line[0] != '\n':
            new_line = [string.replace('\n', '') for string in line]
            coords[m] = np.array([float(new_line[0]), float(new_line[1])])
            m = m + 1

# 将浮点型坐标转换为相同维度的列表并删除0
coords1 = [pt.tolist() for pt in coords]
# 创建多点对象
mp = MultiPoint(coords1)

# 创建一个包含所有点的边缘多边形
buffer_distance = 1  # 边缘距离，单位为度，根据数据的投影而定
polygon = mp.buffer(buffer_distance).convex_hull

# 导出边缘为GeoJSON线段文件
geometry = []
if isinstance(polygon.boundary, LineString):
    geometry.append(list(polygon.boundary.coords))
else:
    for line in polygon.boundary:
        print(1)
        geometry.append(list(line.coords))

# 将线段对象写入GeoJSON文件
feature_collection = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": geometry}, "properties": {}}]}
with open("output2.geojson", "w") as f:
    json.dump(feature_collection, f)
