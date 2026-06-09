import geopandas as gpd
import pandas as pd

# 读取数据文件
df = pd.read_csv('chemogulongqibei.dat', sep='\s+', header=None, names=['x', 'y', 'type'])

# 按type分类，将同一type的点坐标连接成线，形成LineString
crack_lines = []
for t in df['type'].unique():
    line_coords = df[df['type'] == t][['x', 'y']].values.tolist()
    crack_lines.append({'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': line_coords}, 'properties': {'type': t}})

# 创建GeoDataFrame对象，输出为GeoJSON格式
gdf = gpd.GeoDataFrame.from_features(crack_lines)
gdf.to_file('thkceshi-landmark1.geojson', driver='GeoJSON')
