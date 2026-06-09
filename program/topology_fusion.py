# -*- coding: utf-8 -*-
"""
拓扑属性融合模块：将多个高级拓扑属性通过 PCA / 自编码器 / UMAP / VAE 融合成新属性。
用于机器学习扩展：输入为网格 CSV 中的拓扑指标，输出为融合主成分、簇标签等。
"""
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils.config_loader import load_config
from utils.export_utils import export_spatial_dataframe, clusters_gpkg_layer_name, export_table
from utils.logging_utils import get_logger

try:
    from feature_engineering import CONNECTIVITY_FEATURE_COLUMNS
except ImportError:
    CONNECTIVITY_FEATURE_COLUMNS = (
        "Connections per Branch",
        "Connections per Trace",
        "Connection Frequency",
    )

DEFAULT_FEATURE_COLUMNS = [
    "Fracture Intensity B21",
    "Fracture Intensity P21",
    "Dimensionless Intensity B22",
    "Dimensionless Intensity P22",
    "Areal Frequency B20",
    "Areal Frequency P20",
    "Connections per Branch",
    "Connections per Trace",
    "Connection Frequency",
    "Number of Traces (Real)",
    "Number of Branches (Real)",
    "Branch Mean Length",
    "Trace Mean Length",
    "Trace Min Length",
    "Trace Max Length",
]


def _runtime_cfg():
    cfg = load_config()
    clustering_cfg = (cfg.get("clustering") or {}) if isinstance(cfg, dict) else {}
    fusion_cfg = (cfg.get("fusion") or {}) if isinstance(cfg, dict) else {}
    log_cfg = (cfg.get("logging") or {}) if isinstance(cfg, dict) else {}
    logger = get_logger(
        "topology_fusion",
        level=log_cfg.get("level", "INFO"),
        log_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), log_cfg.get("file", "logs/pipeline.log")),
    )
    return clustering_cfg, fusion_cfg, logger


def _clean_numeric_string(s):
    """清理字符串形式的数值，如"[7.743774E-4]"，返回纯数值。"""
    if isinstance(s, str):
        # 移除括号和空格
        s = s.strip().strip('[]')
    return s

def load_and_prepare(
    csv_path: str,
    feature_columns: Optional[List[str]] = None,
    drop_all_nan: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    从网格 CSV 加载数据并预处理：
    1. 清洗括号包裹的字符串数值（如 "[7.74e-4]"）
    2. 筛选出实际存在于 CSV 中的特征列
    3. 可选丢弃全 NaN 行（保留含部分有效值的行）
    4. 统一转换为 float64 矩阵

    返回 (完整DataFrame, 特征矩阵, 实际使用的列名列表)
    """
    df = pd.read_csv(csv_path)
    # 清洗所有 object 类型列中包裹在方括号里的字符串数值
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(_clean_numeric_string)
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS
    # 只保留 CSV 中实际存在的列
    available = [c for c in feature_columns if c in df.columns]
    if not available:
        raise ValueError(f"CSV 中未找到任何特征列，请检查列名: {feature_columns}")
    X_raw = df[available].copy()
    # 丢弃所有特征列全为 NaN 的行（与不含拓扑数据的空网格对应）
    if drop_all_nan:
        all_nan_mask = X_raw.isna().all(axis=1)
        valid = ~all_nan_mask
        X_raw = X_raw.loc[valid]
        df = df.loc[valid].copy()
    X = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return df, X.values.astype(np.float64), available


def fuse_with_pca(
    X: np.ndarray,
    n_components: int = 2,
    standardize: bool = True,
) -> Tuple[np.ndarray, Optional[StandardScaler], Optional[PCA]]:
    """
    PCA 降维融合：将高维拓扑属性压缩到低维主成分空间。
    第一步 StandardScaler 消除量纲差异，第二步 PCA 保留最大方差方向。
    返回 (降维后的潜向量, scaler, pca模型)
    """
    scaler = StandardScaler() if standardize else None
    if scaler is not None:
        X = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, scaler, pca


def cluster_labels(
    X_latent: np.ndarray,
    n_clusters: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, KMeans]:
    """对融合后的潜向量执行 KMeans 聚类，得到每个网格的分类标签。"""
    clustering_cfg, _, _ = _runtime_cfg()
    if n_clusters is None:
        n_clusters = int(clustering_cfg.get("n_clusters", 4))
    if random_state is None:
        random_state = int(clustering_cfg.get("random_state", 42))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_latent)
    return labels, kmeans


def interpret_clusters(
    df: pd.DataFrame,
    feature_columns: List[str],
    cluster_id: np.ndarray,
    n_clusters: int,
) -> pd.DataFrame:
    """计算各簇在各特征上的均值，用于解释每个簇的地质含义（高/低连通、高/低强度等）。"""
    df = df.copy()
    df["_cluster"] = cluster_id
    means = df.groupby("_cluster")[feature_columns].mean()
    df.drop(columns=["_cluster"], inplace=True)
    return means


def compute_cluster_quality_metrics(X_latent: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    潜空间上的聚类质量评估：
    - 轮廓系数（silhouette）：越大越好（簇内紧密、簇间分离），范围 [-1,1]
    - Davies–Bouldin 指数：越小越好（簇间相似度越低），无上界
    要求至少 3 个样本且 2 个以上簇才有意义。
    """
    X_latent = np.asarray(X_latent, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    out: Dict[str, float] = {}
    uniq = np.unique(labels)
    if len(X_latent) < 3 or len(uniq) < 2 or len(uniq) >= len(X_latent):
        return out
    try:
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        out["silhouette_score"] = float(silhouette_score(X_latent, labels))
        out["davies_bouldin_index"] = float(davies_bouldin_score(X_latent, labels))
    except Exception:
        pass
    return out


def compute_cluster_stability_ari(
    X_latent: np.ndarray,
    n_clusters: int,
    seeds: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    通过不同随机种子重复 KMeans，计算标签间 ARI 平均值，评估簇稳定性。
    """
    X_latent = np.asarray(X_latent, dtype=np.float64)
    if len(X_latent) < max(4, n_clusters):
        return {}
    if seeds is None:
        seeds = [7, 13, 23, 37, 41]
    try:
        from sklearn.metrics import adjusted_rand_score
    except Exception:
        return {}
    labels_list: List[np.ndarray] = []
    for sd in seeds:
        km = KMeans(n_clusters=int(n_clusters), random_state=int(sd), n_init=10)
        labels_list.append(km.fit_predict(X_latent))
    aris: List[float] = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            aris.append(float(adjusted_rand_score(labels_list[i], labels_list[j])))
    if not aris:
        return {}
    arr = np.asarray(aris, dtype=np.float64)
    return {
        "cluster_stability_ari_mean": float(arr.mean()),
        "cluster_stability_ari_std": float(arr.std(ddof=0)),
        "cluster_stability_pairs": int(len(aris)),
    }


def build_cluster_name_map(
    cluster_means: pd.DataFrame,
    connectivity_cols: Optional[List[str]] = None,
    intensity_cols: Tuple[str, ...] = ("Fracture Intensity B21", "Fracture Intensity P21"),
) -> Dict[int, str]:
    """
    根据各簇在「连通性特征组」与强度指标上相对全体簇中位数的偏高/偏低，
    自动生成简短的业务可读簇名（如"高连通-高强度簇"）。
    规则：连通性列中超过半数高于全体中位数→"高连通"，否则→"低连通"；
          强度同理分"高强度"/"低强度"。
    """
    connectivity_cols = [
        c
        for c in (connectivity_cols or list(CONNECTIVITY_FEATURE_COLUMNS))
        if c in cluster_means.columns
    ]
    int_col = next((c for c in intensity_cols if c in cluster_means.columns), None)
    name_map: Dict[int, str] = {}
    for cid in cluster_means.index:
        cid_i = int(cid)
        row = cluster_means.loc[cid]
        if connectivity_cols:
            highs = sum(
                1
                for c in connectivity_cols
                if float(row[c]) >= float(cluster_means[c].median())
            )
            conn_lbl = "高连通" if highs * 2 >= len(connectivity_cols) else "低连通"
        else:
            conn_lbl = "连通性未定"
        if int_col is not None:
            hi = float(row[int_col]) >= float(cluster_means[int_col].median())
            int_lbl = "高强度" if hi else "低强度"
            name_map[cid_i] = f"{conn_lbl}-{int_lbl}簇"
        else:
            name_map[cid_i] = f"{conn_lbl}簇"
    return name_map


def attach_cluster_names(df: pd.DataFrame, name_map: Dict[int, str]) -> pd.DataFrame:
    """将自动生成的簇名（如"高连通-高强度簇"）加到 DataFrame 新列 cluster_name。"""
    out = df.copy()
    if "cluster_id" not in out.columns:
        return out
    out["cluster_name"] = out["cluster_id"].map(lambda i: name_map.get(int(i), "")).fillna("")
    return out


def build_cluster_summary_rows(
    df_out: pd.DataFrame,
    cluster_means: pd.DataFrame,
    name_map: Dict[int, str],
    report_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """生成每簇汇总表：簇内网格数、空间面积占比、连通性/强度等选定属性均值。"""
    total = len(df_out)
    # 默认报告列：连通性特征组 + 强度指标
    report_cols = report_cols or [
        c
        for c in list(CONNECTIVITY_FEATURE_COLUMNS)
        if c in cluster_means.columns
    ][:6]
    extra = [c for c in ("Fracture Intensity B21", "Fracture Intensity P21") if c in cluster_means.columns]
    for c in extra:
        if c not in report_cols:
            report_cols = list(report_cols) + [c]
    rows: List[Dict[str, Any]] = []
    for cid in sorted(df_out["cluster_id"].unique()):
        cid = int(cid)
        sub = df_out[df_out["cluster_id"] == cid]
        row: Dict[str, Any] = {
            "cluster_id": cid,
            "cluster_name": name_map.get(cid, ""),
            "n_cells": len(sub),
            "spatial_fraction": (len(sub) / total) if total else 0.0,
        }
        if cid in cluster_means.index:
            for c in report_cols:
                if c in cluster_means.columns:
                    row[f"mean_{c}"] = float(cluster_means.loc[cid, c])
        rows.append(row)
    return pd.DataFrame(rows)


def _check_min_data(df: pd.DataFrame, X: np.ndarray, used_cols: list, min_samples: int = 2, min_features: int = 2):
    """前置校验：样本数和特征数是否满足融合/聚类的最低要求，不足时给出明确提示。"""
    n_samples, n_features = X.shape
    if n_samples < min_samples or n_features < min_features:
        raise ValueError(
            f"有效样本数或特征数不足（当前 {n_samples} 样本、{n_features} 特征）。"
            f"融合/聚类至少需要 {min_samples} 样本和 {min_features} 特征。请检查英买 2 网格 CSV 是否齐全或重新运行 export_grid_csv.py MY。"
        )


def run_fusion_pipeline(
    csv_path: str,
    feature_columns: Optional[List[str]] = None,
    n_components: Optional[int] = None,
    n_clusters: Optional[int] = None,
) -> Tuple[pd.DataFrame, StandardScaler, PCA, KMeans, pd.DataFrame]:
    """
    PCA 降维 + KMeans 聚类 一站式流水线：
    1. 加载数据 → 2. PCA 降维到潜空间 → 3. KMeans 聚类 → 4. 簇解释
    返回 (含PC列和cluster_id的DataFrame, scaler, pca, kmeans, 簇均值表)
    """
    clustering_cfg, _, logger = _runtime_cfg()
    if n_components is None:
        n_components = int(clustering_cfg.get("pca_components", 2))
    df, X, used_cols = load_and_prepare(csv_path, feature_columns=feature_columns)
    n_samples, n_features = X.shape
    if n_samples < 2 or n_features < 2:
        raise ValueError(
            f"有效样本数或特征数不足（当前 {n_samples} 样本、{n_features} 特征）。"
            f"PCA/聚类至少需要 2 样本和 2 特征。请先运行 export_grid_csv.py MY 生成英买 2 网格 CSV 并确保特征列非空。"
        )
    # PCA 降维：n_components 不能超过样本数和特征数
    n_components = min(n_components, n_samples, n_features)
    X_pca, scaler, pca = fuse_with_pca(X, n_components=n_components, standardize=True)
    labels, kmeans = cluster_labels(X_pca, n_clusters=n_clusters)
    # 将 PCA 各主分量写入 DataFrame
    for i in range(n_components):
        df[f"PC{i+1}"] = X_pca[:, i]
    df["cluster_id"] = labels
    cluster_means = interpret_clusters(df, used_cols, labels, n_clusters)
    logger.info("PCA 融合完成：samples=%s features=%s clusters=%s", n_samples, n_features, int(df["cluster_id"].nunique()))
    return df, scaler, pca, kmeans, cluster_means


def fuse_with_autoencoder(
    X: np.ndarray,
    n_latent: int = 2,
    standardize: bool = True,
    epochs: int = 100,
    lr: float = 1e-2,
) -> Tuple[np.ndarray, Optional[StandardScaler], Optional[object]]:
    """
    自编码器（AutoEncoder）降维融合：
    通过"编码→低维瓶颈→解码→重构"的无监督训练，迫使瓶颈层学习有效低维表示。
    编码器：输入→隐藏层（维度=特征数/2）→瓶颈（n_latent 维）；解码器为对称结构。
    返回 (潜向量, scaler, model)
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("自编码器需要 PyTorch，请执行: pip install torch")
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None
    n_features = X.shape[1]
    X_t = torch.tensor(X, dtype=torch.float32)

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            # 编码器：逐层压缩到瓶颈层
            self.encoder = nn.Sequential(
                nn.Linear(n_features, max(n_features // 2, n_latent)),
                nn.ReLU(),
                nn.Linear(max(n_features // 2, n_latent), n_latent),
            )
            # 解码器：从瓶颈层对称还原到原始维度
            self.decoder = nn.Sequential(
                nn.Linear(n_latent, max(n_features // 2, n_latent)),
                nn.ReLU(),
                nn.Linear(max(n_features // 2, n_latent), n_features),
            )
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    model = AE()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # MSE 重构损失：鼓励瓶颈层保留尽可能多的原始信息
    for _ in range(epochs):
        opt.zero_grad()
        recon, _ = model(X_t)
        loss = ((X_t - recon) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, Z = model(X_t)
    return Z.numpy(), scaler, model


def run_fusion_pipeline_ae(
    csv_path: str,
    feature_columns: Optional[List[str]] = None,
    n_latent: Optional[int] = None,
    n_clusters: Optional[int] = None,
    ae_epochs: Optional[int] = None,
) -> Tuple[pd.DataFrame, StandardScaler, KMeans, pd.DataFrame]:
    """自编码器降维 + KMeans 聚类 一站式流水线。返回 (含潜向量和cluster_id的DataFrame, scaler, kmeans, 簇均值表)"""
    clustering_cfg, _, _ = _runtime_cfg()
    if n_latent is None:
        n_latent = int(clustering_cfg.get("ae_latent", 2))
    if ae_epochs is None:
        ae_epochs = int(clustering_cfg.get("ae_epochs", 100))
    df, X, used_cols = load_and_prepare(csv_path, feature_columns=feature_columns)
    _check_min_data(df, X, used_cols)
    Z, scaler, _ = fuse_with_autoencoder(X, n_latent=n_latent, standardize=True, epochs=ae_epochs)
    labels, kmeans = cluster_labels(Z, n_clusters=n_clusters)
    for i in range(n_latent):
        df[f"Z{i+1}"] = Z[:, i]
    df["cluster_id"] = labels
    cluster_means = interpret_clusters(df, used_cols, labels, n_clusters)
    return df, scaler, kmeans, cluster_means


def fuse_with_umap(
    X: np.ndarray,
    n_components: int = 2,
    standardize: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, Optional[StandardScaler], Optional[object]]:
    """
    UMAP 降维融合：保留局部邻域结构的前提下学习低维嵌入。
    相比 PCA（线性），UMAP 能捕获非线性流形结构；相比 t-SNE，速度更快且能更好地保留全局距离关系。
    n_neighbors 控制局部/全局平衡（越小越关注局部细节），min_dist 控制嵌入点的最小距离。
    """
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP 需要 umap-learn，请执行: pip install umap-learn")
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, max(2, len(X) - 1)),
        min_dist=min_dist,
        random_state=random_state,
    )
    X_umap = reducer.fit_transform(X)
    return X_umap, scaler, reducer


def run_fusion_pipeline_umap(
    csv_path: str,
    feature_columns: Optional[List[str]] = None,
    n_components: Optional[int] = None,
    n_clusters: Optional[int] = None,
    n_neighbors: Optional[int] = None,
) -> Tuple[pd.DataFrame, StandardScaler, object, KMeans, pd.DataFrame]:
    """UMAP 降维 + KMeans 聚类 一站式流水线。返回 (含U1/U2和cluster_id的DataFrame, scaler, reducer, kmeans, 簇均值表)"""
    clustering_cfg, _, _ = _runtime_cfg()
    if n_components is None:
        n_components = int(clustering_cfg.get("umap_components", 2))
    if n_neighbors is None:
        n_neighbors = int(clustering_cfg.get("umap_neighbors", 15))
    df, X, used_cols = load_and_prepare(csv_path, feature_columns=feature_columns)
    _check_min_data(df, X, used_cols)
    X_umap, scaler, reducer = fuse_with_umap(
        X, n_components=n_components, standardize=True, n_neighbors=n_neighbors
    )
    labels, kmeans = cluster_labels(X_umap, n_clusters=n_clusters)
    for i in range(n_components):
        df[f"U{i+1}"] = X_umap[:, i]
    df["cluster_id"] = labels
    cluster_means = interpret_clusters(df, used_cols, labels, n_clusters)
    return df, scaler, reducer, kmeans, cluster_means


def fuse_with_vae(
    X: np.ndarray,
    n_latent: int = 2,
    standardize: bool = True,
    epochs: int = 150,
    lr: float = 1e-3,
) -> Tuple[np.ndarray, Optional[StandardScaler], Optional[object]]:
    """
    变分自编码器（VAE）降维融合：
    在 AE 基础上引入 KL 散度正则项，迫使潜空间服从标准正态分布，
    使得潜空间更平滑（相近的潜向量对应相近的重建），适合后续聚类。
    损失 = MSE(重构) + 0.5 × KL(正态先验)。
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("VAE 需要 PyTorch，请执行: pip install torch")
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None
    n_features = X.shape[1]
    X_t = torch.tensor(X, dtype=torch.float32)
    hidden = max(n_features // 2, n_latent)

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            # 编码器输出 2×n_latent：前半为均值 μ，后半为对数方差 log(σ²)
            self.enc_fc1 = nn.Linear(n_features, hidden)
            self.enc_fc2 = nn.Linear(hidden, n_latent * 2)
            self.dec_fc1 = nn.Linear(n_latent, hidden)
            self.dec_fc2 = nn.Linear(hidden, n_features)

        def encode(self, x):
            h = torch.relu(self.enc_fc1(x))
            mu_logvar = self.enc_fc2(h)
            mu, logvar = mu_logvar[:, :n_latent], mu_logvar[:, n_latent:]
            return mu, logvar

        def reparameterize(self, mu, logvar):
            # 重参数化技巧：z = μ + ε·σ，使得采样操作可导
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h = torch.relu(self.dec_fc1(z))
            return self.dec_fc2(h)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar, z

    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        recon, mu, logvar, _ = model(X_t)
        mse = ((X_t - recon) ** 2).mean()
        # KL 散度：D_KL(q(z|x) || p(z))，其中 p(z) = N(0,1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        loss = mse + 0.5 * kl
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, _, _, Z = model(X_t)
    return Z.numpy(), scaler, model


def run_fusion_pipeline_vae(
    csv_path: str,
    feature_columns: Optional[List[str]] = None,
    n_latent: Optional[int] = None,
    n_clusters: Optional[int] = None,
    vae_epochs: Optional[int] = None,
) -> Tuple[pd.DataFrame, StandardScaler, KMeans, pd.DataFrame]:
    """VAE 降维 + KMeans 聚类 一站式流水线。返回 (含潜向量和cluster_id的DataFrame, scaler, kmeans, 簇均值表)"""
    clustering_cfg, _, _ = _runtime_cfg()
    if n_latent is None:
        n_latent = int(clustering_cfg.get("vae_latent", 2))
    if vae_epochs is None:
        vae_epochs = int(clustering_cfg.get("vae_epochs", 150))
    df, X, used_cols = load_and_prepare(csv_path, feature_columns=feature_columns)
    _check_min_data(df, X, used_cols)
    Z, scaler, _ = fuse_with_vae(X, n_latent=n_latent, standardize=True, epochs=vae_epochs)
    labels, kmeans = cluster_labels(Z, n_clusters=n_clusters)
    for i in range(n_latent):
        df[f"Z{i+1}"] = Z[:, i]
    df["cluster_id"] = labels
    cluster_means = interpret_clusters(df, used_cols, labels, n_clusters)
    return df, scaler, kmeans, cluster_means


def export_cluster_results(
    df_out: pd.DataFrame,
    csv_path: str,
    method_name: str,
    *,
    out_dir: Optional[str] = None,
    cluster_summary: Optional[pd.DataFrame] = None,
    quality_metrics: Optional[Dict[str, float]] = None,
) -> dict:
    """
    将聚类结果导出为 CSV + GPKG（含网格几何面要素），并附带：
    - 每簇统计汇总表 (per_cluster_stats.csv)
    - 聚类质量 JSON (silhouette, Davies–Bouldin, 稳定性 ARI 等)
    """
    _, fusion_cfg, logger = _runtime_cfg()
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "data", "processed")
    stem = f"{os.path.splitext(os.path.basename(csv_path))[0]}_{method_name.lower()}_clusters"
    layer = clusters_gpkg_layer_name(method_name)
    paths = export_spatial_dataframe(
        df_out,
        out_dir,
        stem,
        export_csv=bool(fusion_cfg.get("export_cluster_csv", True)),
        export_gpkg=bool(fusion_cfg.get("export_cluster_gpkg", True)),
        layer_name=layer,
    )
    aux = {}
    if cluster_summary is not None and not cluster_summary.empty:
        aux["cluster_summary_csv"] = export_table(cluster_summary, out_dir, f"{stem}_per_cluster_stats")
    if quality_metrics:
        import json

        qpath = os.path.join(out_dir, f"{stem}_fusion_quality.json")
        with open(qpath, "w", encoding="utf-8") as f:
            json.dump(quality_metrics, f, ensure_ascii=False, indent=2)
        aux["quality_json"] = qpath
    logger.info("聚类结果导出：method=%s csv=%s gpkg=%s layer=%s", method_name, paths.get("csv"), paths.get("gpkg"), layer)
    return {**paths, **aux}


if __name__ == "__main__":
    import sys
    csv_path = "Yingmai 2 area in Tarim Basin.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    df_out, scaler, pca, kmeans, means = run_fusion_pipeline(csv_path, n_components=2, n_clusters=4)
    print("PCA 融合后新增列:", [c for c in df_out.columns if c.startswith("PC") or c == "cluster_id"])
    out_path = csv_path.replace(".csv", "_fused.csv")
    df_out.to_csv(out_path, index=False)
    print("已写出:", out_path)
