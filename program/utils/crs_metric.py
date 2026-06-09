# -*- coding: utf-8 -*-
"""迹线/研究区 CRS：对齐同一坐标系，并在地理 CRS 下投到 UTM，避免 fractopo 内 .length 在度单位下告警与偏差。"""

import geopandas as gpd


def _looks_like_mislabeled_projected(gdf: gpd.GeoDataFrame) -> bool:
    """
    某些数据文件的坐标值明显是米制投影坐标，但 CRS 却被错误标为 EPSG:4326。
    这种情况下 estimate_utm_crs() 会失败，且 fractopo 会把 length 当“度”处理。
    """
    try:
        crs = gdf.crs
        if crs is None or not getattr(crs, "is_geographic", False):
            return False
        minx, miny, maxx, maxy = gdf.total_bounds
        return max(abs(minx), abs(maxx)) > 180 or max(abs(miny), abs(maxy)) > 90
    except Exception:
        return False


def _force_metric_label_if_needed(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    对“坐标值已是米制、但 CRS 被误标为地理坐标”的数据，只覆盖 CRS 标签，不改坐标值。
    这里用 EPSG:3857 仅作为 meter-based 占位标签，目的是让下游长度/网格运算按米处理。
    """
    if _looks_like_mislabeled_projected(gdf):
        try:
            return gdf.set_crs(3857, allow_override=True)
        except Exception:
            return gdf
    return gdf


def unify_traces_area_crs(traces: gpd.GeoDataFrame, area: gpd.GeoDataFrame):
    """
    将迹线与研究区对齐到同一 CRS。
    fractopo 在 crop 前不会把两套不同 CRS 自动变换到同一空间，易导致裁剪为空。
    """
    from pyproj import CRS

    traces = _force_metric_label_if_needed(traces)
    area = _force_metric_label_if_needed(area)
    tc, ac = traces.crs, area.crs
    if tc is not None and ac is not None:
        if not CRS.from_user_input(tc).equals(CRS.from_user_input(ac)):
            traces = traces.to_crs(ac)
    elif ac is not None and tc is None:
        traces = traces.set_crs(ac)
    elif tc is not None and ac is None:
        area = area.set_crs(tc)
    return traces, area


def reproject_to_metric_crs(traces: gpd.GeoDataFrame, area: gpd.GeoDataFrame):
    """
    地理坐标系（如 EPSG:4326）下 geometry.length 单位为度，geopandas/fractopo 会大量报警告且阈值含义不对。
    自动估计 UTM 并重投影（与网格 cell_width 等单位一致为米）。
    """
    try:
        traces = _force_metric_label_if_needed(traces)
        area = _force_metric_label_if_needed(area)
        if traces.crs is None:
            traces = traces.set_crs(4326)
        if area.crs is None:
            area = area.set_crs(traces.crs)
        elif not traces.crs.equals(area.crs):
            area = area.to_crs(traces.crs)
        crs = traces.crs
        if crs is not None and getattr(crs, "is_geographic", False):
            utm = traces.estimate_utm_crs()
            traces = traces.to_crs(utm)
            area = area.to_crs(utm)
    except Exception:
        pass
    return traces, area
