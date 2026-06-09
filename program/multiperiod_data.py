# -*- coding: utf-8 -*-
"""
多时期断裂拓扑数据加载与预处理模块。
处理海西期、印支燕山期、喜山期三套地层的网格 CSV 数据。

阶段0核心功能：
  1. 列名归一化：将三期CSV中命名不一致的6个拓扑属性列重命名为统一名称
  2. vertex → geometry：利用顶点坐标构建 shapely Polygon 几何列
  3. 拓扑属性矩阵提取：返回 (GeoDataFrame, 特征矩阵, 列名列表)
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in __import__("sys").path:
    __import__("sys").path.insert(0, _THIS_DIR)

from utils.export_utils import build_geometry_from_vertices, VERTEX_COLUMNS
from utils.logging_utils import get_logger

# ---------------------------------------------------------------------------
# 归一化后的6个拓扑属性列名（短名，符合 Python 命名惯例）
# ---------------------------------------------------------------------------
TOPOLOGY_ATTRIBUTES = ("NC_NB", "NC_NL", "NB_NL", "NC_A", "NB_A", "NL_A")

TOPOLOGY_ATTR_DISPLAY = {
    "NC_NB": "每条分支平均连接节点数 (NC/NB)",
    "NC_NL": "每条裂缝平均连接节点数 (NC/NL)",
    "NB_NL": "每条裂缝分几段分支 (NB/NL)",
    "NC_A":  "单位面积连接节点数 (NC/A)",
    "NB_A":  "单位面积分支数 (NB/A)",
    "NL_A":  "单位面积裂缝数 (NL/A)",
}

# 时期显示名
PERIOD_NAMES = {
    "海西":   "海西期",
    "喜山":   "喜山期",
    "印支燕山": "印支燕山期",
}

# 三期 CSV 默认路径（与程序同目录）
DEFAULT_CSV_PATHS = {
    "海西":   os.path.join(_THIS_DIR, "海西.csv"),
    "喜山":   os.path.join(_THIS_DIR, "喜山.csv"),
    "印支燕山": os.path.join(_THIS_DIR, "印支燕山.csv"),
}

# ---------------------------------------------------------------------------
# 列名映射
# ---------------------------------------------------------------------------
def _map_topology_col(raw_name: str) -> Optional[str]:
    """
    将原始拓扑属性列名映射到归一化名称。
    三期 CSV 中这6列的名称不一致：
      - 海西:   5(NB/A)..., 6(NL/A)...
      - 喜山:   5(NB/A)..., 6(N_LA)...
      - 印支燕山: 5(N_BA)..., 6(N_LA)...
    按特征字符串匹配（顺序敏感：先匹配更具体的模式）。
    非拓扑列返回 None。
    """
    name = raw_name.strip()
    if "NC/NB" in name:
        return "NC_NB"
    if "NC/NL" in name:
        return "NC_NL"
    if "NB/NL" in name:
        return "NB_NL"
    if "NC/A" in name:
        return "NC_A"
    if "NB/A" in name or "N_BA" in name:
        return "NB_A"
    if "NL/A" in name or "N_LA" in name:
        return "NL_A"
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理 DataFrame 列名：
      1. 去除所有列名的首尾空格
      2. 将6个拓扑属性列重命名为归一化短名（NC_NB 等）
    返回修改后的 DataFrame 副本。
    """
    df = df.copy()

    # 去首尾空格
    rename_map = {}
    for col in df.columns:
        stripped = col.strip()
        if stripped != col:
            rename_map[col] = stripped
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # 重命名拓扑属性列
    topo_rename = {}
    for col in df.columns:
        mapped = _map_topology_col(col)
        if mapped is not None:
            topo_rename[col] = mapped
    df.rename(columns=topo_rename, inplace=True)

    return df


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def load_period_csv(csv_path: str) -> Tuple[gpd.GeoDataFrame, str]:
    """
    加载并清洗单个时期的网格 CSV。
    返回 (GeoDataFrame with geometry, 时期显示名)。

    处理流程：
      1. pandas 读 CSV
      2. normalize_columns 清洗列名
      3. build_geometry_from_vertices 构建 geometry 列
      4. 添加 period 标记列
    """
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    period_name = PERIOD_NAMES.get(basename, basename)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"时期 CSV 未找到: {csv_path}")

    df = pd.read_csv(csv_path)
    df = normalize_columns(df)
    gdf = build_geometry_from_vertices(df)
    gdf["period"] = period_name
    return gdf, period_name


def load_all_periods(
    csv_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, gpd.GeoDataFrame]:
    """
    加载三期全部数据。
    csv_paths 为 {"海西": "path", ...}，不传则用默认路径。

    返回 {"海西期": GeoDataFrame, "印支燕山期": GeoDataFrame, "喜山期": GeoDataFrame}
    """
    paths = csv_paths or DEFAULT_CSV_PATHS
    result: Dict[str, gpd.GeoDataFrame] = {}
    for period_key, path in paths.items():
        gdf, name = load_period_csv(path)
        result[name] = gdf
    return result


# ---------------------------------------------------------------------------
# 拓扑属性矩阵提取
# ---------------------------------------------------------------------------
def get_topology_matrix(
    gdf: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, np.ndarray, List[str]]:
    """
    从 GeoDataFrame 中提取6个拓扑属性矩阵。
    返回 (GeoDataFrame, 特征矩阵 float64 (n_samples, n_features), 实际可用的列名列表)。

    与 topology_fusion.load_and_prepare() 返回约定兼容。
    """
    available = [c for c in TOPOLOGY_ATTRIBUTES if c in gdf.columns]
    if not available:
        raise ValueError(
            f"未找到任何拓扑属性列（期望 {TOPOLOGY_ATTRIBUTES}），"
            f"当前列: {list(gdf.columns)}"
        )
    X = gdf[available].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return gdf, X.values.astype(np.float64), available


def get_topology_summary(
    gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    输出6个拓扑属性列的统计摘要（count / mean / std / min / 25% / 50% / 75% / max）。
    """
    available = [c for c in TOPOLOGY_ATTRIBUTES if c in gdf.columns]
    if not available:
        return pd.DataFrame()
    X = gdf[available].apply(pd.to_numeric, errors="coerce")
    summary = X.describe().T
    summary.index.name = "属性"
    return summary


# ---------------------------------------------------------------------------
# 自检入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger = get_logger("multiperiod_data", level="INFO")

    for label, path in DEFAULT_CSV_PATHS.items():
        if not os.path.isfile(path):
            logger.warning("跳过 %s：文件不存在 %s", label, path)
            continue
        gdf, name = load_period_csv(path)
        _, X, cols = get_topology_matrix(gdf)
        logger.info(
            "%s: 网格数=%d  拓扑属性列=%s  非零网格=%d/%d",
            name,
            len(gdf),
            cols,
            int((X.sum(axis=1) > 0).sum()),
            X.shape[0],
        )

    logger.info(
        "可用的6个归一化拓扑属性列: %s",
        list(TOPOLOGY_ATTRIBUTES),
    )
