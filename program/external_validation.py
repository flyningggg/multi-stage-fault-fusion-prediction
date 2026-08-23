# -*- coding: utf-8 -*-
"""候选靶区的独立外部点位验证；外部证据绝不回写内部评分。"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def validate_external_points(
    targets: pd.DataFrame,
    points: Optional[pd.DataFrame] = None,
    buffer_m: float = 0.0,
) -> Dict:
    """计算外部点是否落入候选靶区；无数据时返回显式未验证状态。"""
    if points is None or len(points) == 0:
        return {
            "status": "not_validated",
            "n_points": 0,
            "n_hits": 0,
            "hit_rate": None,
            "message": "尚未提供井位、油气显示或专家盲评外部证据。",
        }
    if targets is None or len(targets) == 0:
        return {
            "status": "no_targets",
            "n_points": int(len(points)),
            "n_hits": 0,
            "hit_rate": 0.0,
            "message": "当前运行未生成候选靶区。",
        }
    if "geometry" not in targets.columns:
        raise ValueError("targets 缺少 geometry 列")

    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError("外部空间验证需要 geopandas 和 shapely") from exc

    if isinstance(points, gpd.GeoDataFrame) and "geometry" in points.columns:
        point_geometries = points.geometry
    else:
        missing = {"x", "y"} - set(points.columns)
        if missing:
            raise ValueError(f"外部点表缺少列: {sorted(missing)}")
        point_geometries = [
            Point(float(x), float(y)) for x, y in zip(points["x"], points["y"])
        ]

    geometries = [geometry for geometry in targets["geometry"] if geometry is not None]
    if not geometries:
        return {
            "status": "no_target_geometry",
            "n_points": int(len(points)),
            "n_hits": 0,
            "hit_rate": 0.0,
            "message": "候选靶区没有可用空间几何。",
        }
    target_union = unary_union(geometries)
    if float(buffer_m) > 0:
        target_union = target_union.buffer(float(buffer_m))
    hits = [bool(target_union.covers(geometry)) for geometry in point_geometries]
    n_hits = int(sum(hits))

    by_outcome = {}
    if "outcome" in points.columns:
        for outcome, group in points.assign(_hit=hits).groupby("outcome", dropna=False):
            by_outcome[str(outcome)] = {
                "n": int(len(group)),
                "n_hits": int(group["_hit"].sum()),
                "hit_rate": float(group["_hit"].mean()),
            }
    return {
        "status": "evaluated",
        "n_points": int(len(points)),
        "n_hits": n_hits,
        "hit_rate": float(n_hits / len(points)),
        "buffer_m": float(buffer_m),
        "by_outcome": by_outcome,
        "message": "外部点位仅用于独立验证，未参与内部靶区评分。",
    }
