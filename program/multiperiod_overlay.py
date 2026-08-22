# -*- coding: utf-8 -*-
"""
多时期网格空间叠加分析模块。
处理三期（海西期、印支燕山期、喜山期）非对齐网格的空间配准，
识别跨时期重叠区域（靶区）。

核心功能：
  1. 基于顶点坐标的精确网格匹配
  2. 基于 centroid 距离的空间连接（容差半径匹配）
  3. 三期全重叠区域识别
  4. 靶区空间聚类与可视化
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from multiperiod_data import (
    load_all_periods, load_period_csv,
    get_topology_matrix, TOPOLOGY_ATTRIBUTES,
    DEFAULT_CSV_PATHS, PERIOD_NAMES,
)
from utils.logging_utils import get_logger

logger = get_logger("multiperiod_overlay")

# ---------------------------------------------------------------------------
# 容差设置
# ---------------------------------------------------------------------------
CENTROID_MATCH_TOLERANCE_M = 1500.0   # centroid 距离容差（半网格步长）
MIN_OVERLAP_RATIO = 0.0               # 最小重叠面积比（0 = 只要有交集就算）


# ---------------------------------------------------------------------------
# 精确顶点匹配
# ---------------------------------------------------------------------------
def find_exact_vertex_matches(
    gdf_a: gpd.GeoDataFrame,
    gdf_b: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    通过 vertex1_x + vertex1_y 精确匹配两个时期的网格。
    返回 DataFrame，含 gdf_a 和 gdf_b 的行索引及对应顶点坐标。
    """
    key_cols = ["vertex1_x", "vertex1_y"]
    missing = [c for c in key_cols if c not in gdf_a.columns or c not in gdf_b.columns]
    if missing:
        raise ValueError(f"缺少顶点列: {missing}")

    merged = pd.merge(
        gdf_a[["vertex1_x", "vertex1_y"]].reset_index().rename(columns={"index": "idx_a"}),
        gdf_b[["vertex1_x", "vertex1_y"]].reset_index().rename(columns={"index": "idx_b"}),
        on=["vertex1_x", "vertex1_y"],
        how="inner",
    )
    if not merged.empty:
        merged["match_type"] = "exact_vertex"
    return merged


# ---------------------------------------------------------------------------
# centroid 距离匹配
# ---------------------------------------------------------------------------
def find_centroid_distance_matches(
    gdf_a: gpd.GeoDataFrame,
    gdf_b: gpd.GeoDataFrame,
    max_dist: float = CENTROID_MATCH_TOLERANCE_M,
) -> pd.DataFrame:
    """
    通过 centroid 距离匹配两个时期的网格。
    对 gdf_a 中每个网格，找 gdf_b 中 centroid 最近的网格，
    距离 < max_dist 则视为空间重叠。

    返回 DataFrame: [idx_a, idx_b, dist, vertex1_x, vertex1_y]
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError("centroid 匹配需要 scipy: pip install scipy")

    centroids_a = np.column_stack([
        [p.x for p in gdf_a.geometry.centroid],
        [p.y for p in gdf_a.geometry.centroid],
    ])
    centroids_b = np.column_stack([
        [p.x for p in gdf_b.geometry.centroid],
        [p.y for p in gdf_b.geometry.centroid],
    ])

    tree = cKDTree(centroids_b)
    distances, indices = tree.query(centroids_a, k=1)

    mask = distances < max_dist
    if not mask.any():
        return pd.DataFrame(columns=["idx_a", "idx_b", "dist", "vertex1_x", "vertex1_y"])

    result = pd.DataFrame({
        "idx_a": np.where(mask)[0],
        "idx_b": indices[mask],
        "dist": distances[mask],
    })
    # 附加 gdf_a 的顶点坐标（用于空间定位）
    result["vertex1_x"] = result["idx_a"].apply(
        lambda i: gdf_a.iloc[i]["vertex1_x"])
    result["vertex1_y"] = result["idx_a"].apply(
        lambda i: gdf_a.iloc[i]["vertex1_y"])
    result["match_type"] = "centroid_dist"
    return result


# ---------------------------------------------------------------------------
# 三期全重叠查找
# ---------------------------------------------------------------------------
def find_three_period_overlaps(
    period_gdfs: Dict[str, gpd.GeoDataFrame],
    max_dist: float = CENTROID_MATCH_TOLERANCE_M,
    prefer: str = "exact",
    nonzero_only: bool = True,
) -> pd.DataFrame:
    """
    找到三期网格中空间重叠的单元。

    策略：
      1. 优先尝试精确顶点匹配（同一个 vertex1_x, vertex1_y）
      2. 未精确匹配的区域使用 centroid 距离匹配
      3. 找到三期均存在的重叠组

    参数：
      period_gdfs: {"海西期": gdf, ...}
      max_dist: centroid 匹配容差 (m)
      prefer: "exact" 优先精确匹配 / "centroid" 只用 centroid

    返回：
      DataFrame 每行是一个三期重叠组：
        [hx_idx, xs_idx, yzys_idx, vertex1_x, vertex1_y, hx_centroid, xs_centroid, yzys_centroid, match_type]
    """
    periods = list(period_gdfs.keys())
    if len(periods) < 3:
        raise ValueError(f"需要3期数据，当前只有 {len(periods)} 期")

    p0, p1, p2 = periods[0], periods[1], periods[2]
    gdf0, gdf1, gdf2 = period_gdfs[p0], period_gdfs[p1], period_gdfs[p2]

    all_overlaps = []

    # 尝试精确匹配
    if prefer != "centroid":
        match_01 = find_exact_vertex_matches(gdf0, gdf1)
        match_02 = find_exact_vertex_matches(gdf0, gdf2)
        if not match_01.empty and not match_02.empty:
            three = pd.merge(match_01, match_02,
                             on=["vertex1_x", "vertex1_y", "match_type"],
                             suffixes=("_01", "_02"))
            if not three.empty:
                three = three.rename(columns={
                    "idx_a_01": "idx_a", "idx_b_01": "idx_b",
                    "idx_a_02": "drop_a", "idx_b_02": "idx_c",
                })
                three.drop(columns=["drop_a"], inplace=True)
                all_overlaps.append(three)
                logger.info("精确顶点匹配: %d 组三期重叠", len(three))

    # centroid 距离匹配 (补全非精确匹配的区域)
    if prefer != "exact" or not all_overlaps:
        match_01 = find_centroid_distance_matches(gdf0, gdf1, max_dist=max_dist)
        match_02 = find_centroid_distance_matches(gdf0, gdf2, max_dist=max_dist)
        if not match_01.empty and not match_02.empty:
            three = pd.merge(match_01, match_02,
                             on=["idx_a", "vertex1_x", "vertex1_y", "match_type"],
                             suffixes=("_01", "_02"))
            if not three.empty:
                three = three.rename(columns={
                    "idx_b_01": "idx_b", "idx_b_02": "idx_c", "dist_01": "dist_b",
                    "dist_02": "dist_c",
                })
                all_overlaps.append(three)
                logger.info("centroid 距离匹配: %d 组三期重叠", len(three))

    if not all_overlaps:
        return pd.DataFrame(columns=["idx_a", "idx_b", "idx_c", "vertex1_x",
                                      "vertex1_y", "match_type"])

    combined = pd.concat(all_overlaps, ignore_index=True)
    combined.drop_duplicates(subset=["idx_a", "idx_b", "idx_c"], inplace=True)

    # 附加三期的 centroid 坐标
    for i, gdf in enumerate([gdf0, gdf1, gdf2]):
        col = f"{['a', 'b', 'c'][i]}_centroid_x"
        combined[col] = 0.0
    # 填充 centroid（从 gdf0.geometry.centroid 提取）
    combined["centroid_x"] = combined["idx_a"].apply(lambda idx: gdf0.geometry.centroid.iloc[idx].x)
    combined["centroid_y"] = combined["idx_a"].apply(lambda idx: gdf0.geometry.centroid.iloc[idx].y)

    combined.rename(columns={
        "idx_a": f"{p0}_idx",
        "idx_b": f"{p1}_idx",
        "idx_c": f"{p2}_idx",
    }, inplace=True)

    # ---- 可选：仅保留三期均有非零拓扑数据的重叠格 ----
    if nonzero_only:
        before = len(combined)
        for pname, gdf in period_gdfs.items():
            _, X, _ = get_topology_matrix(gdf)
            nonzero_idx = set(np.where(X.sum(axis=1) > 0)[0])
            col = f"{pname}_idx"
            if col in combined.columns:
                combined = combined[combined[col].isin(nonzero_idx)]
        logger.info("nonzero_only 过滤: %d → %d (去除仅坐标重叠的零值格)", before, len(combined))

    return combined


# ---------------------------------------------------------------------------
# 靶区识别（空间聚类）
# ---------------------------------------------------------------------------
def identify_target_areas(
    overlap_df: pd.DataFrame,
    min_cluster_size: int = 3,
    eps_meters: float = 4500,
) -> pd.DataFrame:
    """
    将三期重叠网格按空间邻近聚类，划分靶区。
    使用 DBSCAN（eps = 1.5 倍网格步长 → 相邻网格连成一片）。

    返回 overlap_df 附加 cluster_id 和 target_area 列。
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise ImportError("靶区聚类需要 scikit-learn")

    if "centroid_x" not in overlap_df.columns or "centroid_y" not in overlap_df.columns:
        raise ValueError("overlap_df 缺少 centroid_x/centroid_y 列")

    coords = overlap_df[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    db = DBSCAN(eps=eps_meters, min_samples=min_cluster_size)
    labels = db.fit_predict(coords)

    result = overlap_df.copy()
    result["target_cluster"] = labels

    # 按簇中心 Y 坐标排序命名（从上到下：靶区1, 靶区2, ...）
    cluster_names = {}
    unique_clusters = sorted(set(labels) - {-1})
    cluster_centers = {}
    for cid in unique_clusters:
        mask = labels == cid
        cluster_centers[cid] = coords[mask].mean(axis=0)

    for rank, cid in enumerate(
        sorted(unique_clusters, key=lambda c: -cluster_centers[c][1]), start=1
    ):
        cluster_names[cid] = f"靶区{rank}"

    result["target_area"] = result["target_cluster"].map(
        lambda c: cluster_names.get(c, "散点")
    )

    return result


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def plot_overlap_map(
    period_gdfs: Dict[str, gpd.GeoDataFrame],
    overlap_df: pd.DataFrame,
    out_path: str,
    title: str = "三期空间重叠与靶区分布",
    use_interpolation: bool = False,
) -> str:
    """绘制三期重叠 + 靶区分布图。

    参数：
      use_interpolation: 使用插值热力图（True）或传统网格图（False）

    图例含义：
      - 深红面：三期均有非零数据的重叠区（核心靶区）
      - 各期基色面：仅该期有数据的非重叠区
      - 浅灰面：零值区（无断裂数据）
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        logger.warning("matplotlib 不可用，跳过绘图")
        return ""

    try:
        from utils.matplotlib_chinese import setup_matplotlib_chinese
        setup_matplotlib_chinese()
    except Exception:
        pass

    if use_interpolation:
        return _plot_overlap_interpolated(period_gdfs, overlap_df, out_path, title)
    else:
        return _plot_overlap_grid(period_gdfs, overlap_df, out_path, title)


def _plot_overlap_interpolated(
    period_gdfs: Dict[str, gpd.GeoDataFrame],
    overlap_df: pd.DataFrame,
    out_path: str,
    title: str,
) -> str:
    """使用插值热力图绘制三期重叠分布。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.interpolate import griddata

    periods = list(period_gdfs.keys())

    # 收集所有坐标范围
    all_x, all_y = [], []
    for gdf in period_gdfs.values():
        centroids = gdf.geometry.centroid
        all_x.extend([p.x for p in centroids])
        all_y.extend([p.y for p in centroids])
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # 创建细网格（200m分辨率）
    resolution = 200
    xi = np.arange(x_min, x_max + resolution, resolution)
    yi = np.arange(y_min, y_max + resolution, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # 自定义颜色映射：灰(0) -> 基色(1) -> 深红(2)
    period_colors = ["#E63946", "#457B9D", "#2A9D8F"]
    overlap_color = "#B71C1C"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (period_name, gdf), base_color in zip(axes, period_gdfs.items(), period_colors):
        _, X, _ = get_topology_matrix(gdf)
        nonzero_mask = np.array(X.sum(axis=1)).ravel() > 0

        period_col = f"{period_name}_idx"
        overlap_indices = set()
        if period_col in overlap_df.columns:
            overlap_indices = set(
                overlap_df[period_col].dropna().astype(int).tolist()
            )

        # 提取质心坐标和值
        centroids = gdf.geometry.centroid
        points = np.array([[p.x, p.y] for p in centroids])

        # 分配值：0=零值, 1=非零非重叠, 2=重叠
        values = np.zeros(len(gdf))
        nz_only_mask = nonzero_mask & ~gdf.index.isin(overlap_indices)
        values[nz_only_mask] = 1.0
        if overlap_indices:
            ov_indices = [i for i in range(len(gdf)) if i in overlap_indices]
            values[ov_indices] = 2.0

        # 插值到细网格
        zi = griddata(points, values, (xi_grid, yi_grid), method='nearest', fill_value=0)

        # 创建自定义颜色映射
        cmap_colors = ["#F0F0F0", base_color, overlap_color]
        cmap = LinearSegmentedColormap.from_list("custom", cmap_colors, N=256)

        # 绘制热力图
        im = ax.imshow(zi, extent=[x_min, x_max, y_min, y_max],
                       origin='lower', cmap=cmap, vmin=0, vmax=2,
                       alpha=0.85, interpolation='bilinear')

        # 靶区标注
        if "target_area" in overlap_df.columns:
            for target_name in overlap_df["target_area"].unique():
                if target_name == "散点":
                    continue
                sub = overlap_df[overlap_df["target_area"] == target_name]
                if period_col in sub.columns:
                    idxs = sub[period_col].dropna().astype(int).tolist()
                    if not idxs:
                        continue
                    centroids_ov = gdf.iloc[idxs].geometry.centroid
                    xs = [p.x for p in centroids_ov]
                    ys = [p.y for p in centroids_ov]
                    cx, cy = np.mean(xs), np.mean(ys)
                    ax.annotate(target_name, (cx, cy),
                                fontsize=9, fontweight="bold", color="#1a1a1a",
                                bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor="white", edgecolor="#333",
                                          alpha=0.85),
                                ha="center", va="center", zorder=5)

        n_overlap = len(overlap_indices)
        n_nz = int(nonzero_mask.sum())
        n_total = len(gdf)
        ax.set_title(f"{period_name}\n(非零={n_nz}/{n_total} | 重叠={n_overlap})",
                     fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

        # 图例
        legend_elements = [
            Patch(facecolor=overlap_color, edgecolor="none",
                  label=f"三期重叠({n_overlap})"),
            Patch(facecolor=base_color, edgecolor="none",
                  label=f"仅本期非零({n_nz - n_overlap})"),
            Patch(facecolor="#F0F0E0", edgecolor="none",
                  label=f"零值({n_total - n_nz})"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=7,
                  framealpha=0.9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_overlap_grid(
    period_gdfs: Dict[str, gpd.GeoDataFrame],
    overlap_df: pd.DataFrame,
    out_path: str,
    title: str,
) -> str:
    """使用传统网格图绘制三期重叠分布（备用方案）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    periods = list(period_gdfs.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    period_colors = ["#E63946", "#457B9D", "#2A9D8F"]
    overlap_face  = "#B71C1C"
    overlap_edge  = "#FF5722"
    zero_face     = "#E0E0E0"

    for ax, (period_name, gdf), base_color in zip(axes, period_gdfs.items(), period_colors):
        _, X, _ = get_topology_matrix(gdf)
        nonzero_mask = np.array(X.sum(axis=1)).ravel() > 0

        period_col = f"{period_name}_idx"
        overlap_indices = set()
        if period_col in overlap_df.columns:
            overlap_indices = set(
                overlap_df[period_col].dropna().astype(int).tolist()
            )

        # 1) 先画零值网格 — 最底层
        zero_mask = ~nonzero_mask
        gdf_z = gdf[zero_mask]
        if len(gdf_z) > 0:
            gdf_z.plot(ax=ax, facecolor=zero_face, edgecolor="none",
                       alpha=0.25, zorder=0)

        # 2) 非零、非重叠 — 各期基色，中层
        nz_only_mask = nonzero_mask & ~gdf.index.isin(overlap_indices)
        gdf_nz = gdf[nz_only_mask]
        if len(gdf_nz) > 0:
            gdf_nz.plot(ax=ax, facecolor=base_color, edgecolor="none",
                        alpha=0.50, zorder=1)

        # 3) 重叠网格 — 深红面 + 亮橙边框，最顶层
        if overlap_indices:
            gdf_ov = gdf.iloc[sorted(overlap_indices)]
            gdf_ov.plot(ax=ax, facecolor=overlap_face, edgecolor=overlap_edge,
                        linewidth=0.8, alpha=0.70, zorder=2)

        # 4) 靶区标注
        if "target_area" in overlap_df.columns:
            for target_name in overlap_df["target_area"].unique():
                if target_name == "散点":
                    continue
                sub = overlap_df[overlap_df["target_area"] == target_name]
                if period_col in sub.columns:
                    idxs = sub[period_col].dropna().astype(int).tolist()
                    if not idxs:
                        continue
                    centroids = gdf.iloc[idxs].geometry.centroid
                    xs = [p.x for p in centroids]
                    ys = [p.y for p in centroids]
                    cx, cy = np.mean(xs), np.mean(ys)
                    ax.annotate(target_name, (cx, cy),
                                fontsize=9, fontweight="bold", color="#1a1a1a",
                                bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor="white", edgecolor="#333",
                                          alpha=0.85),
                                ha="center", va="center", zorder=5)

        n_overlap = len(overlap_indices)
        n_nz = int(nonzero_mask.sum())
        n_total = len(gdf)
        ax.set_title(f"{period_name}\n(非零={n_nz}/{n_total} | 重叠={n_overlap})",
                     fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

        legend_elements = [
            Patch(facecolor=overlap_face, edgecolor=overlap_edge, linewidth=0.8,
                  label=f"三期重叠({n_overlap})"),
            Patch(facecolor=base_color, edgecolor="none", alpha=0.50,
                  label=f"仅本期非零({n_nz - n_overlap})"),
            Patch(facecolor=zero_face, edgecolor="none", alpha=0.25,
                  label=f"零值({n_total - n_nz})"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=7,
                  framealpha=0.9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_overlay_pipeline(
    out_dir: Optional[str] = None,
    max_dist: Optional[float] = None,
) -> dict:
    """
    Stage 2：三期空间叠加主流程。
    1. 加载三期数据
    2. 查找三期全重叠网格
    3. 靶区空间聚类
    4. 导出重叠表 + 可视化

    匹配/聚类容差默认从 config.yaml 的 grid 段读取；显式传参优先于配置。
    """
    if out_dir is None:
        out_dir = os.path.join(_THIS_DIR, "data", "processed", "multiperiod")
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config()
    grid_cfg = cfg.get("grid", {}) if isinstance(cfg, dict) else {}
    if max_dist is None:
        max_dist = float(grid_cfg.get("centroid_match_tolerance_m", CENTROID_MATCH_TOLERANCE_M))

    logger.info("=== Stage 2: 三期空间叠加 ===")

    period_gdfs = load_all_periods()
    for name, gdf in period_gdfs.items():
        logger.info("  %s: %d 网格", name, len(gdf))

    # 查找三期重叠
    overlap_df = find_three_period_overlaps(period_gdfs, max_dist=max_dist)
    logger.info("三期全重叠网格数: %d", len(overlap_df))

    if len(overlap_df) == 0:
        logger.warning("未找到三期全重叠区域")
        return {"period_gdfs": period_gdfs, "overlap_df": overlap_df}

    # 靶区聚类（eps/最小簇样本数可经 config.yaml grid 段覆盖）
    overlap_df = identify_target_areas(
        overlap_df,
        min_cluster_size=int(grid_cfg.get("target_min_cluster_size", 3)),
        eps_meters=float(grid_cfg.get("target_eps_m", 4500)),
    )
    target_counts = overlap_df["target_area"].value_counts()
    logger.info("靶区分布:\n%s", target_counts.to_string())

    # 导出
    overlap_csv = os.path.join(out_dir, "three_period_overlap.csv")
    overlap_df.to_csv(overlap_csv, index=False, encoding="utf-8-sig")
    logger.info("重叠表导出: %s", overlap_csv)

    # 可视化
    plot_path = os.path.join(out_dir, "three_period_overlap_map.png")
    try:
        plot_overlap_map(period_gdfs, overlap_df, plot_path)
        logger.info("重叠地图: %s", plot_path)
    except Exception as e:
        logger.warning("可视化失败: %s", e)
        plot_path = ""

    return {
        "period_gdfs": period_gdfs,
        "overlap_df": overlap_df,
        "overlap_csv": overlap_csv,
        "plot_path": plot_path,
    }


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = run_overlay_pipeline()
    print(f"\n三期重叠网格数: {len(result['overlap_df'])}")
    if "target_area" in result["overlap_df"].columns:
        print(result["overlap_df"]["target_area"].value_counts().to_string())
