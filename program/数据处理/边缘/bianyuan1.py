import geojson
import cv2
import numpy as np

# 读取XYZ文件
m = 0
with open('断裂数据-测试用1.xyz', 'r') as f:
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
print(coords1)
# 转换为numpy数组并缩放点坐标以适应图像大小
scale_factor = 10000
points_scaled = np.array(coords1, dtype=np.float32) * scale_factor

# 将NaN值替换为0
points_scaled = np.delete(points_scaled, np.where(~points_scaled.any(axis=1))[0], axis=0)
# 创建图像并将点绘制为白色
img_size = (800, 800)
img = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
# print(points_scaled)
cv2.drawContours(img, [np.round(points_scaled).astype(np.int32)], 0, (255, 255, 255), -1)

# 使用Canny边缘检测检测边缘
approx = cv2.Canny(img, 100, 200)

# 将线段坐标缩放回原始比例
approx = approx / scale_factor

# 将线段坐标转换为geojson格式
features = []
for i in range(len(approx) - 1):
    start = approx[i]
    end = approx[i + 1]
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [start[0], start[1]],
                [end[0], end[1]]
            ]
        },
        "properties": {}
    }
    features.append(feature)

# 将features列表作为参数创建FeatureCollection对象
feature_collection = geojson.FeatureCollection(features)

# 将geojson写入文件
with open("output1.geojson", "w") as f:
    geojson.dump(feature_collection, f)
