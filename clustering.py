from __future__ import annotations

"""
clustering.py
=============
Unsupervised behavioural profiling of ETH whales.

Steps:
  1. Load address_features.csv
  2. Normalise numeric features (RobustScaler — handles outliers well)
  3. KMeans (k=4..6, chosen by silhouette score)
  4. DBSCAN (noise / fringe whale detection)
  5. Dimensionality reduction: PCA 2D scatter + t-SNE 2D scatter
  6. Cluster profiling: radar (spider) chart per cluster
  7. Save results to data/clustered_addresses.csv
     and figures to data/cluster_*.png

Run:
  python clustering.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
FEATURES_CSV = DATA_DIR / "address_features.csv"
LABELED_CSV = DATA_DIR / "labeled_addresses.csv"
OUTPUT_CSV = DATA_DIR / "clustered_addresses.csv"

FEATURE_COLS = [
    "tx_count_out",
    "total_eth_out",
    "avg_tx_eth",
    "median_tx_eth",
    "max_tx_eth",
    "std_tx_eth",
    "unique_receivers",
    "exchange_ratio",
    "top1_receiver_ratio",
    "round_number_ratio",
    "avg_gas_gwei",
    "gas_variability",
    "hour_entropy",
    "active_days",
    "net_flow_eth",
]

CLUSTER_PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

CLUSTER_NAMES = {
    0: "Large One-Shot Movers",
    1: "Mega-Whale Traders",
    2: "Small Whales",
    3: "Mid-Range Regulars",
    4: "Cluster 4",
    5: "Cluster 5",
}


def load_and_prepare(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, RobustScaler]:
    """Load features, impute NaNs, log-transform heavy-tailed columns, then scale."""
    source = LABELED_CSV if LABELED_CSV.exists() else csv_path
    df = pd.read_csv(source)
    print(f"Loaded {len(df)} addresses from {source.name}")

    df_feat = df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        if df_feat[col].isna().any():
            df_feat[col] = df_feat[col].fillna(df_feat[col].median())

    LOG_COLS = [
        "tx_count_out", "total_eth_out", "avg_tx_eth", "median_tx_eth",
        "max_tx_eth", "std_tx_eth", "unique_receivers", "avg_gas_gwei",
        "gas_variability", "active_days",
    ]
    for col in LOG_COLS:
        if col in df_feat.columns:
            df_feat[col] = np.log1p(df_feat[col].clip(lower=0))

    if "net_flow_eth" in df_feat.columns:
        nf = df_feat["net_flow_eth"]
        df_feat["net_flow_eth"] = np.sign(nf) * np.log1p(nf.abs())

    scaler = RobustScaler()
    X = scaler.fit_transform(df_feat.values)
    return df, X, scaler


def find_best_k(X: np.ndarray, k_range=range(2, 9), preferred_k: int = 4) -> int:
    """Return k with the highest silhouette score.

    If *preferred_k* has a silhouette within 20% of the best, choose it
    instead — more clusters often yield more interpretable behavioural
    profiles even at a small silhouette cost.
    """
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        scores[k] = silhouette_score(X, labels)
        print(f"  k={k}  silhouette={scores[k]:.3f}")

    best_k = max(scores, key=scores.get)
    best_sil = scores[best_k]

    # Prefer a richer segmentation when the quality cost is small
    if preferred_k in scores and preferred_k != best_k:
        pref_sil = scores[preferred_k]
        gap = (best_sil - pref_sil) / (best_sil + 1e-9)
        if gap <= 0.20:
            print(f"Best silhouette at k={best_k} ({best_sil:.3f}), but k={preferred_k} "
                  f"({pref_sil:.3f}) is within {gap*100:.1f}% — choosing k={preferred_k} "
                  f"for richer segmentation")
            return preferred_k

    print(f"Best k = {best_k}  (silhouette={best_sil:.3f})")
    return best_k


def run_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    return km.fit_predict(X)


def run_dbscan(X: np.ndarray, eps: float = 1.5, min_samples: int = 3) -> np.ndarray:
    """DBSCAN: label -1 = noise (outlier whale)."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    return db.fit_predict(X)


def find_optimal_eps(X: np.ndarray, min_samples: int = 5, out_path: Path | None = None) -> float:
    """Find optimal DBSCAN eps using the kneedle method on k-distance curve.

    Draws a line from the first to the last point of the sorted k-distance
    curve, then picks the point with the maximum perpendicular distance
    to that line — this is the "knee" where density drops off.
    """
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])[::-1]  # descending

    # Kneedle: max perpendicular distance from the line connecting endpoints
    n_pts = len(k_distances)
    x_norm = np.linspace(0, 1, n_pts)
    y_norm = (k_distances - k_distances[-1]) / (k_distances[0] - k_distances[-1] + 1e-9)

    # Line from (0, 1) to (1, 0) in normalised space
    # Perpendicular distance = |y_norm[i] - (1 - x_norm[i])| / sqrt(2)
    perp_dist = np.abs(y_norm - (1.0 - x_norm))

    # Skip the first 1% of points to avoid edge effects from extreme outliers
    skip = max(1, n_pts // 100)
    elbow_idx = skip + int(np.argmax(perp_dist[skip:]))
    optimal_eps = float(k_distances[elbow_idx])

    # Sanity clamp
    optimal_eps = max(0.3, min(optimal_eps, 5.0))
    print(f"  k-distance knee at index {elbow_idx} / {n_pts}, optimal eps = {optimal_eps:.3f}")

    if out_path:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")
        ax.plot(k_distances, color="#3498db", linewidth=1.5, label="k-distance curve")
        ax.axhline(optimal_eps, color="#e74c3c", linestyle="--", lw=1.5,
                   label=f"eps = {optimal_eps:.3f}")
        ax.axvline(elbow_idx, color="#f39c12", linestyle=":", lw=1, alpha=0.7,
                   label=f"knee index = {elbow_idx}")
        ax.set_xlabel("Points (sorted by distance)", color="white")
        ax.set_ylabel(f"{min_samples}-NN Distance", color="white")
        ax.set_title("k-Distance Plot (DBSCAN eps selection — kneedle method)",
                     color="white", fontsize=13)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.legend(framealpha=0.3, labelcolor="white", facecolor="#1a1a1a")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved {out_path.name}")

    return optimal_eps


def plot_pca(X: np.ndarray, labels: np.ndarray, title: str, out_path: Path, label_names: dict | None = None):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    unique_labels = sorted(set(labels))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] if lbl != -1 else "#666666"
        name = (label_names or {}).get(lbl, f"Cluster {lbl}" if lbl != -1 else "Noise")
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=55, alpha=0.8,
                   edgecolors="white", linewidths=0.3, label=name, zorder=3)

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", color="white")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", color="white")
    ax.set_title(title, color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    legend = ax.legend(framealpha=0.3, labelcolor="white", facecolor="#1a1a1a")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out_path.name}")


def plot_tsne(X: np.ndarray, labels: np.ndarray, title: str, out_path: Path, label_names: dict | None = None):
    n = len(X)
    perp = min(30, max(5, n // 4))
    # sklearn >= 1.2 uses max_iter; n_iter was removed
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    unique_labels = sorted(set(labels))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] if lbl != -1 else "#666666"
        name = (label_names or {}).get(lbl, f"Cluster {lbl}" if lbl != -1 else "Noise")
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=55, alpha=0.8,
                   edgecolors="white", linewidths=0.3, label=name, zorder=3)

    ax.set_xlabel("t-SNE dim 1", color="white")
    ax.set_ylabel("t-SNE dim 2", color="white")
    ax.set_title(title, color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.legend(framealpha=0.3, labelcolor="white", facecolor="#1a1a1a")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out_path.name}")


def plot_radar(df: pd.DataFrame, cluster_col: str, out_path: Path):
    """Radar / spider chart showing average feature values per cluster."""
    radar_features = [
        "tx_count_out", "total_eth_out", "avg_tx_eth", "unique_receivers",
        "exchange_ratio", "round_number_ratio", "hour_entropy",
        "active_days", "gas_variability",
    ]
    avail = [f for f in radar_features if f in df.columns]
    if not avail:
        return

    # Normalise each feature 0..1 across the whole dataset
    norm = df[avail].copy()
    for col in avail:
        rng = norm[col].max() - norm[col].min()
        norm[col] = (norm[col] - norm[col].min()) / (rng if rng > 0 else 1)
    norm[cluster_col] = df[cluster_col]

    cluster_means = norm.groupby(cluster_col)[avail].mean()
    N = len(avail)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    for i, (cluster_id, row) in enumerate(cluster_means.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), avail, color="white", fontsize=9)
    ax.set_title("Cluster Behaviour Profiles", color="white", fontsize=14, pad=20)
    ax.tick_params(colors="white")
    ax.grid(color="#333", linewidth=0.5)
    ax.spines["polar"].set_color("#333")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              framealpha=0.3, labelcolor="white", facecolor="#1a1a1a")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out_path.name}")


def print_cluster_profiles(df: pd.DataFrame, cluster_col: str):
    profile_cols = [
        "tx_count_out", "total_eth_out", "avg_tx_eth", "unique_receivers",
        "exchange_ratio", "round_number_ratio", "active_days", "net_flow_eth",
    ]
    avail = [c for c in profile_cols if c in df.columns]
    print("\n=== Cluster Profiles (medians) ===")
    print(df.groupby(cluster_col)[avail].median().round(3).to_string())
    print("\n=== Cluster Sizes ===")
    print(df[cluster_col].value_counts().sort_index().to_string())


def main():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"{FEATURES_CSV} not found. Run feature_engineering.py first.")

    df, X, scaler = load_and_prepare(FEATURES_CSV)

    if len(df) < 4:
        print("Too few addresses for clustering (need ≥ 4). Collect more data first.")
        return

    # ── KMeans ──────────────────────────────────────────────────────────────
    print("\n[KMeans] Finding best k ...")
    best_k = find_best_k(X, k_range=range(2, min(9, len(df))))
    km_labels = run_kmeans(X, best_k)
    df["kmeans_cluster"] = km_labels

    label_names = {k: CLUSTER_NAMES.get(k, f"Cluster {k}") for k in range(best_k)}

    print("\n[PCA] Plotting KMeans clusters ...")
    plot_pca(X, km_labels, f"KMeans Clusters (k={best_k}) — PCA projection",
             DATA_DIR / "cluster_pca_kmeans.png", label_names)

    if len(df) >= 10:
        print("[t-SNE] Plotting KMeans clusters (may take a moment) ...")
        plot_tsne(X, km_labels, f"KMeans Clusters (k={best_k}) — t-SNE projection",
                  DATA_DIR / "cluster_tsne_kmeans.png", label_names)

    print("[Radar] Plotting cluster behaviour profiles ...")
    plot_radar(df, "kmeans_cluster", DATA_DIR / "cluster_radar.png")

    print_cluster_profiles(df, "kmeans_cluster")

    # ── DBSCAN ──────────────────────────────────────────────────────────────
    print("\n[DBSCAN] Auto-tuning eps via k-distance kneedle ...")
    optimal_eps = find_optimal_eps(X, min_samples=5, out_path=DATA_DIR / "cluster_kdistance.png")
    print(f"\n[DBSCAN] Detecting noise / outlier whales (eps={optimal_eps:.3f}, min_samples=5) ...")
    db_labels = run_dbscan(X, eps=optimal_eps, min_samples=5)
    df["dbscan_cluster"] = db_labels
    n_noise = (db_labels == -1).sum()
    n_core = (db_labels >= 0).sum()
    print(f"  Core points: {n_core}   Noise (outliers): {n_noise}")

    db_label_names = {k: f"DBSCAN-{k}" for k in set(db_labels) if k != -1}
    db_label_names[-1] = "Outlier Whale"
    plot_pca(X, db_labels, "DBSCAN Clusters — PCA projection",
             DATA_DIR / "cluster_pca_dbscan.png", db_label_names)

    # ── Save ────────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved clustered data to {OUTPUT_CSV}")

    sil = silhouette_score(X, km_labels)
    print(f"\nFinal silhouette score (KMeans k={best_k}): {sil:.4f}")
    if sil >= 0.3:
        print("  Clustering quality: GOOD (>= 0.3)")
    elif sil >= 0.15:
        print("  Clustering quality: FAIR — consider collecting more data")
    else:
        print("  Clustering quality: WEAK — collect significantly more data")


if __name__ == "__main__":
    main()
