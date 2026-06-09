import json
import geojson

# 读取 GeoJSON 文件
with open("thkceshi-landmark1.geojson", "r") as f:
    data = json.load(f)

# 确保文件符合要求
if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
    raise ValueError("Invalid GeoJSON file format")

# 提取断层线坐标
coordinates = []
for feature in data.get("features"):
    if feature.get("geometry", {}).get("type") != "LineString":
        continue
    coordinates += feature.get("geometry", {}).get("coordinates", [])

# 构建最小边界框
min_x = min(x for x, y in coordinates)
max_x = max(x for x, y in coordinates)
min_y = min(y for x, y in coordinates)
max_y = max(y for x, y in coordinates)

# 生成区域边界
polygon = geojson.Polygon([[[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y], [min_x, min_y], ]])

# 生成 area 参数
area = {
    "id": 1,
    "type": "puzzle",
    "geometry": polygon,
}

# 输出结果
with open("my_area.geojson", "w") as f:
    json.dump(geojson.Feature(geometry=area["geometry"], id=area["id"], properties={"type": area["type"]}), f)
