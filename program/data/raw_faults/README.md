# 原始同位数据投放合同

此目录用于接收三期原始断裂线，不存放由当前网格 CSV 反推的伪原始数据。

建议文件名：

- `海西期.geojson`
- `喜山期.geojson`
- `印支燕山期.geojson`
- `physical_properties.yaml`

三期断裂文件须满足：

1. 使用 `LineString` 或 `MultiLineString` 几何；
2. 三期具有相同的投影米制 CRS，不能直接用经纬度计算距离和面积；
3. 每条断裂最好包含稳定的 `trace_id`，并保留可追溯的时期、来源和质量字段；
4. 支持 GeoJSON、JSON、GeoPackage 和 Shapefile，优先使用 GeoPackage 或 GeoJSON。

物性文件至少应给出基质渗透率、裂缝/基质渗透率关系、残余开度、压差或边界条件，并记录数值单位、数据来源和适用层段。可复制 `physical_properties.example.yaml` 后填写。文件存在只代表数据入口具备；在完成坐标、单位、范围和来源复核并实际运行同位模型前，不得表述为“已完成物理验证”。

从仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts\check_project_readiness.py
```
