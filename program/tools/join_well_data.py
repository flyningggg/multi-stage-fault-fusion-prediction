# -*- coding: utf-8 -*-
"""
将井位点与网格结果做空间连接（Spatial Join），输出带网格预测/簇属性的井点文件。

用途：把 XGBoost 预测结果（含 cluster_id、cluster_name、prediction、risk_level、
     uncertainty_score 等）通过空间位置关联到井位点上，指导勘探目标优选。

示例：
python program/tools/join_well_data.py \
  --grid "/path/to/predictions_xgb.gpkg" --layer predictions_xgb \
  --wells "/path/to/wells.geojson" \
  --out "/path/to/wells_joined.gpkg"
"""
from __future__ import annotations

import argparse
import os
import geopandas as gpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="网格 GPKG 路径")
    ap.add_argument("--layer", default=None, help="网格图层名（默认读第一层）")
    ap.add_argument("--wells", required=True, help="井位点文件（geojson/shp/gpkg）")
    ap.add_argument("--out", required=True, help="输出 gpkg 路径")
    ap.add_argument("--predicate", default="within", help="空间关系：within/intersects/contains")
    args = ap.parse_args()

    # 读取网格结果（按图层）
    g_grid = gpd.read_file(args.grid, layer=args.layer) if args.layer else gpd.read_file(args.grid)
    g_well = gpd.read_file(args.wells)
    # CRS 对齐：井位点重投影到与网格相同的坐标系
    if g_grid.crs is not None and g_well.crs is not None and g_grid.crs != g_well.crs:
        g_well = g_well.to_crs(g_grid.crs)
    # 保留与勘探决策相关的列
    keep = [c for c in [
        "cluster_id", "cluster_name", "prediction_xgboost",
        "prediction_rank", "risk_level", "uncertainty_score"
    ] if c in g_grid.columns]
    # 空间连接：每个井位点匹配其所在的网格单元
    joined = gpd.sjoin(g_well, g_grid[keep + ["geometry"]], how="left", predicate=args.predicate)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    joined.to_file(args.out, layer="wells_joined", driver="GPKG")
    print(f"OK: {args.out} ({len(joined)} rows)")


if __name__ == "__main__":
    main()
