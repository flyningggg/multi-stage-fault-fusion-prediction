from __future__ import annotations

import os
import json
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

# GeoPackage 图层名（与 QGIS/ArcGIS 衔接）
LAYER_PREDICTIONS_XGB = "predictions_xgb"

_METHOD_TO_LAYER_SLUG = {
    "PCA": "pca",
    "自编码器": "ae",
    "UMAP": "umap",
    "VAE": "vae",
}


def clusters_gpkg_layer_name(method_display_name: str) -> str:
    """根据融合方法得到标准图层名，如 PCA → clusters_pca。"""
    key = (method_display_name or "").strip()
    slug = _METHOD_TO_LAYER_SLUG.get(key) or key.lower().replace(" ", "_")[:12] or "fusion"
    return f"clusters_{slug}"


def enrich_predictions_for_gis(
    df: pd.DataFrame,
    *,
    pred_col_preferred: str = "prediction_xgboost",
) -> pd.DataFrame:
    """为导出的预测表增加 prediction_rank、risk_level（按预测值分位三等分：低/中/高）。"""
    out = df.copy()
    pred_col = None
    for c in (pred_col_preferred, "prediction_xgboost", "xgb_pred"):
        if c in out.columns:
            pred_col = c
            break
    if pred_col is None:
        return out
    s = pd.to_numeric(out[pred_col], errors="coerce")
    valid = s.notna()
    out["prediction_rank"] = np.nan
    out.loc[valid, "prediction_rank"] = s[valid].rank(ascending=False, method="first")
    out["prediction_rank"] = pd.to_numeric(out["prediction_rank"], errors="coerce").astype("Int64")
    risk = pd.Series("", index=out.index, dtype=object)
    sv = s[valid]
    if len(sv) >= 3:
        try:
            bins = pd.qcut(sv, q=3, labels=["低", "中", "高"], duplicates="drop")
            risk.loc[bins.index] = bins.astype(str)
        except ValueError:
            risk.loc[valid] = "中"
    elif len(sv) > 0:
        risk.loc[valid] = "中"
    out["risk_level"] = risk
    lower_col = next((c for c in ("prediction_lower", "xgb_pred_lower") if c in out.columns), None)
    upper_col = next((c for c in ("prediction_upper", "xgb_pred_upper") if c in out.columns), None)
    if lower_col and upper_col:
        lo = pd.to_numeric(out[lower_col], errors="coerce")
        up = pd.to_numeric(out[upper_col], errors="coerce")
        width = up - lo
        out["uncertainty_width"] = width
        wv = width[width.notna()]
        if len(wv) >= 3:
            try:
                ubin = pd.qcut(wv, q=3, labels=["低", "中", "高"], duplicates="drop")
                out.loc[ubin.index, "uncertainty_level"] = ubin.astype(str)
            except ValueError:
                out.loc[width.notna(), "uncertainty_level"] = "中"
        elif len(wv) > 0:
            out.loc[width.notna(), "uncertainty_level"] = "中"
        else:
            out["uncertainty_level"] = ""
        # 0~1，越高表示区间越宽（不确定性越高）
        w_min = float(wv.min()) if len(wv) else 0.0
        w_max = float(wv.max()) if len(wv) else 0.0
        if w_max > w_min:
            out["uncertainty_score"] = (width - w_min) / (w_max - w_min)
        else:
            out["uncertainty_score"] = 0.0
    return out


def config_file_hash(config_path: str) -> str:
    if not config_path or not os.path.isfile(config_path):
        return ""
    h = hashlib.sha256()
    with open(config_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def build_run_metadata(
    *,
    run_id: Optional[str] = None,
    config_path: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    rid = run_id or str(uuid.uuid4())
    meta = {
        "processing_run_id": rid,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash_sha256": config_file_hash(config_path or ""),
    }
    if extra:
        meta.update(extra)
    return meta


def write_run_manifest(
    out_dir: str,
    *,
    run_id: Optional[str] = None,
    config_path: Optional[str] = None,
    kind: str = "run",
    artifacts: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def _to_rel(v):
        if not isinstance(v, str):
            return v
        if "://" in v:
            return v
        av = os.path.abspath(v)
        if av.startswith(project_root):
            return os.path.relpath(av, project_root)
        return v

    art = {}
    for k, v in (artifacts or {}).items():
        art[k] = _to_rel(v)
    payload = {
        "kind": kind,
        "metadata": build_run_metadata(run_id=run_id, config_path=config_path, extra=extra or {}),
        "artifacts": art,
    }
    out_path = os.path.join(out_dir, f"{kind}_run_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


VERTEX_COLUMNS = (
    "vertex1_x",
    "vertex1_y",
    "vertex2_x",
    "vertex2_y",
    "vertex3_x",
    "vertex3_y",
    "vertex4_x",
    "vertex4_y",
)


def has_vertex_columns(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in VERTEX_COLUMNS)


def build_geometry_from_vertices(df: pd.DataFrame) -> gpd.GeoDataFrame:
    if not has_vertex_columns(df):
        raise ValueError("DataFrame 缺少顶点列，无法构建 GeoDataFrame。")
    polygons = []
    for row in df.loc[:, VERTEX_COLUMNS].itertuples(index=False, name=None):
        pts = [
            (float(row[0]), float(row[1])),
            (float(row[2]), float(row[3])),
            (float(row[4]), float(row[5])),
            (float(row[6]), float(row[7])),
        ]
        polygons.append(Polygon(pts))
    return gpd.GeoDataFrame(df.copy(), geometry=polygons, crs=None)


def export_spatial_dataframe(
    df: pd.DataFrame,
    out_dir: str,
    stem: str,
    *,
    export_csv: bool = True,
    export_gpkg: bool = True,
    layer_name: str = "results",
) -> Dict[str, Optional[str]]:
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, Optional[str]] = {"csv": None, "gpkg": None}
    if export_csv:
        csv_path = os.path.join(out_dir, f"{stem}.csv")
        df.to_csv(csv_path, index=False)
        paths["csv"] = csv_path
    if export_gpkg and has_vertex_columns(df):
        gpkg_path = os.path.join(out_dir, f"{stem}.gpkg")
        gdf = build_geometry_from_vertices(df)
        gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG")
        paths["gpkg"] = gpkg_path
    return paths


def export_table(
    df: pd.DataFrame,
    out_dir: str,
    stem: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}.csv")
    df.to_csv(out_path, index=False)
    return out_path
