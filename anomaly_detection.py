from __future__ import annotations

"""
anomaly_detection.py
====================
Detects anomalous whale behaviour using a two-layer approach:

Layer 1 — Global anomalies (Isolation Forest, auto threshold):
    Addresses whose overall feature profile is statistically unusual
    compared to the entire whale population.  Uses contamination="auto"
    so the threshold is derived from the score distribution rather than
    an arbitrary percentage.

Layer 2 — Intra-cluster anomalies (Mahalanobis distance):
    Addresses that belong to a KMeans cluster but deviate significantly
    from that cluster's centroid.  A whale can look "normal" globally
    but behave oddly *for its type* (e.g. a Small Whale with 100%
    exchange_ratio).

Both layers are combined into a single anomaly flag with an explanation
of which features contributed most to the anomaly score.

Outputs:
  data/anomaly_scores.csv      — per-address scores, labels, explanations
  data/anomaly_scores_plot.png — score distribution with auto threshold
  data/anomaly_timeline.png    — whale anomaly count vs ETH price over time

Run:
  python anomaly_detection.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.spatial.distance import mahalanobis

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
FEATURES_CSV = DATA_DIR / "address_features.csv"
LABELED_CSV = DATA_DIR / "labeled_addresses.csv"
CLUSTERED_CSV = DATA_DIR / "clustered_addresses.csv"
RAW_CSV = DATA_DIR / "raw_whale_transactions.csv"
ANOMALY_CSV = DATA_DIR / "anomaly_scores.csv"

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

# Financial columns that need log1p (must match clustering.py / classification.py)
LOG_COLS = [
    "tx_count_out", "total_eth_out", "avg_tx_eth", "median_tx_eth",
    "max_tx_eth", "std_tx_eth", "unique_receivers", "avg_gas_gwei",
    "gas_variability", "active_days",
]

CLUSTER_NAMES = {
    0: "Large One-Shot Movers",
    1: "Mega-Whale Traders",
    2: "Small Whales",
    3: "Mid-Range Regulars",
}


# ── Preprocessing (synced with clustering.py) ──────────────────────────────

def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to heavy-tailed columns."""
    df = df.copy()
    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    if "net_flow_eth" in df.columns:
        nf = df["net_flow_eth"]
        df["net_flow_eth"] = np.sign(nf) * np.log1p(nf.abs())
    return df


def load_address_features() -> pd.DataFrame:
    """Load the best available feature source."""
    if CLUSTERED_CSV.exists():
        source = CLUSTERED_CSV
    elif LABELED_CSV.exists():
        source = LABELED_CSV
    else:
        source = FEATURES_CSV
    df = pd.read_csv(source)
    for col in FEATURE_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df, source.name


# ── Layer 1: Global anomalies (Isolation Forest) ───────────────────────────

def detect_global_anomalies(df: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
    """
    Isolation Forest with contamination='auto'.

    Instead of forcing a fixed % of anomalies, the algorithm uses the
    theoretical threshold from the original paper (Liu et al., 2008):
    score < 0 → anomaly.  This means the number of anomalies is
    data-driven, not arbitrarily chosen.
    """
    iso = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    df = df.copy()
    df["iso_label"] = iso.fit_predict(X_scaled)       # -1 = anomaly
    df["iso_score"] = iso.score_samples(X_scaled)      # lower → more anomalous

    n_anom = (df["iso_label"] == -1).sum()
    pct = n_anom / len(df) * 100
    print(f"  Global anomalies (Isolation Forest, auto): {n_anom} / {len(df)} ({pct:.1f}%)")
    return df, iso


# ── Layer 2: Intra-cluster anomalies (Mahalanobis) ─────────────────────────

def detect_cluster_anomalies(df: pd.DataFrame, X_scaled: np.ndarray,
                             cluster_col: str = "kmeans_cluster",
                             threshold_percentile: float = 95.0) -> pd.DataFrame:
    """
    For each cluster, compute Mahalanobis distance of every member from
    the cluster centroid.  Points beyond the per-cluster percentile
    threshold are flagged as intra-cluster anomalies.

    This catches whales that look normal globally but are unusual
    *for their behavioural type*.
    """
    df = df.copy()
    df["mahal_distance"] = np.nan
    df["cluster_anomaly"] = False

    if cluster_col not in df.columns:
        print("  [Skip] No cluster column found — skipping intra-cluster anomalies")
        return df

    for cid in sorted(df[cluster_col].unique()):
        mask = df[cluster_col] == cid
        X_cluster = X_scaled[mask]

        if len(X_cluster) < X_cluster.shape[1] + 2:
            continue

        centroid = X_cluster.mean(axis=0)
        cov = np.cov(X_cluster, rowvar=False)

        # Regularise covariance to avoid singularity
        cov += np.eye(cov.shape[0]) * 1e-6

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        distances = np.array([
            mahalanobis(x, centroid, cov_inv) for x in X_cluster
        ])

        threshold = np.percentile(distances, threshold_percentile)
        cluster_name = CLUSTER_NAMES.get(cid, f"Cluster {cid}")

        n_outliers = (distances > threshold).sum()
        print(f"  Cluster {cid} ({cluster_name}): {n_outliers} intra-cluster outliers "
              f"(Mahalanobis > {threshold:.2f}, p{threshold_percentile:.0f})")

        idx = df.index[mask]
        df.loc[idx, "mahal_distance"] = distances
        df.loc[idx, "cluster_anomaly"] = distances > threshold

    n_cluster_anom = df["cluster_anomaly"].sum()
    print(f"  Total intra-cluster anomalies: {n_cluster_anom}")
    return df


# ── Feature contribution (why is this address anomalous?) ──────────────────

def compute_feature_contributions(df: pd.DataFrame, X_scaled: np.ndarray,
                                  feature_names: list[str]) -> pd.DataFrame:
    """
    For each anomalous address, identify the top 3 features that deviate
    most from the population median (in scaled space).  This gives a
    human-readable explanation of *why* the address was flagged.
    """
    df = df.copy()
    median_vec = np.median(X_scaled, axis=0)

    explanations = []
    top_features = []

    for i in range(len(df)):
        is_anom = (df.iloc[i].get("iso_label", 1) == -1) or df.iloc[i].get("cluster_anomaly", False)
        if not is_anom:
            explanations.append("")
            top_features.append("")
            continue

        deviations = np.abs(X_scaled[i] - median_vec)
        top3_idx = np.argsort(deviations)[-3:][::-1]
        parts = []
        feat_list = []
        for idx in top3_idx:
            fname = feature_names[idx]
            direction = "high" if X_scaled[i, idx] > median_vec[idx] else "low"
            parts.append(f"{fname} ({direction}, {deviations[idx]:.1f}σ)")
            feat_list.append(fname)
        explanations.append("; ".join(parts))
        top_features.append(", ".join(feat_list))

    df["anomaly_explanation"] = explanations
    df["top_anomaly_features"] = top_features
    return df


# ── Combine layers ─────────────────────────────────────────────────────────

def combine_anomaly_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final anomaly label:
      -1 = anomalous (flagged by Isolation Forest OR intra-cluster Mahalanobis)
       1 = normal

    anomaly_type column explains which layer(s) triggered:
      'global'        — Isolation Forest only
      'intra-cluster' — Mahalanobis only
      'both'          — flagged by both methods
      'normal'        — not anomalous
    """
    df = df.copy()
    is_global = df.get("iso_label", pd.Series(1, index=df.index)) == -1
    is_cluster = df.get("cluster_anomaly", pd.Series(False, index=df.index)).astype(bool)

    df["anomaly_label"] = np.where(is_global | is_cluster, -1, 1)
    df["anomaly_score"] = df.get("iso_score", 0.0)

    conditions = [
        is_global & is_cluster,
        is_global & ~is_cluster,
        ~is_global & is_cluster,
    ]
    choices = ["both", "global", "intra-cluster"]
    df["anomaly_type"] = np.select(conditions, choices, default="normal")

    n_total = (df["anomaly_label"] == -1).sum()
    n_global = is_global.sum()
    n_cluster = is_cluster.sum()
    n_both = (is_global & is_cluster).sum()
    print(f"\n  Combined anomalies: {n_total} / {len(df)} ({n_total/len(df)*100:.1f}%)")
    print(f"    Global only:        {n_global - n_both}")
    print(f"    Intra-cluster only: {n_cluster - n_both}")
    print(f"    Both:               {n_both}")
    return df


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_anomaly_score_distribution(df: pd.DataFrame, out_path: Path):
    normal = df.loc[df["anomaly_label"] == 1, "iso_score"]
    anomaly = df.loc[df["anomaly_label"] == -1, "iso_score"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    all_scores = df["iso_score"].dropna()
    bins = np.linspace(all_scores.min(), all_scores.max(), 50)
    ax.hist(normal, bins=bins, color="#3498db", alpha=0.7, label="Normal whale")
    ax.hist(anomaly, bins=bins, color="#e74c3c", alpha=0.8, label="Anomalous whale")

    # Show the auto threshold (score = 0 boundary for IF "auto")
    if len(anomaly) > 0:
        threshold = anomaly.max()
        ax.axvline(threshold, color="#f39c12", linestyle="--", lw=2,
                   label=f"IF threshold ({threshold:.3f})")

    ax.set_xlabel("Isolation Forest Score (lower = more anomalous)", color="white")
    ax.set_ylabel("Number of addresses", color="white")
    ax.set_title("Whale Anomaly Score Distribution (auto threshold)", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.legend(framealpha=0.3, labelcolor="white", facecolor="#1a1a1a")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out_path.name}")


def plot_anomaly_by_cluster(df: pd.DataFrame, out_path: Path):
    """Bar chart: anomaly count per cluster, split by anomaly type."""
    if "kmeans_cluster" not in df.columns:
        return

    anom = df[df["anomaly_label"] == -1].copy()
    if len(anom) == 0:
        return

    cluster_ids = sorted(df["kmeans_cluster"].unique())
    types = ["global", "intra-cluster", "both"]
    colors = {"global": "#e74c3c", "intra-cluster": "#f39c12", "both": "#9b59b6"}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    x = np.arange(len(cluster_ids))
    width = 0.25

    for i, atype in enumerate(types):
        counts = [len(anom[(anom["kmeans_cluster"] == c) & (anom["anomaly_type"] == atype)])
                  for c in cluster_ids]
        labels = [CLUSTER_NAMES.get(c, f"Cluster {c}") for c in cluster_ids]
        ax.bar(x + i * width, counts, width, label=atype.title(),
               color=colors[atype], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([CLUSTER_NAMES.get(c, f"Cluster {c}") for c in cluster_ids],
                       rotation=20, ha="right", color="white")
    ax.set_ylabel("Anomaly count", color="white")
    ax.set_title("Anomalies by Cluster and Detection Method", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.legend(framealpha=0.3, labelcolor="white", facecolor="#1a1a1a")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out_path.name}")


def build_timeline(raw_df: pd.DataFrame, addr_anomalies: set) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    raw_df["is_anomalous"] = raw_df["from_address"].isin(addr_anomalies)
    raw_df["hour"] = raw_df["timestamp"].dt.floor("h")

    timeline = raw_df.groupby("hour").agg(
        total_tx=("tx_hash", "count"),
        anomalous_tx=("is_anomalous", "sum"),
        total_eth_moved=("value_eth", "sum"),
        avg_eth_price=("eth_price_usd", "mean"),
    ).reset_index()

    timeline["anomaly_ratio"] = timeline["anomalous_tx"] / timeline["total_tx"].clip(lower=1)
    return timeline


def plot_anomaly_timeline(timeline: pd.DataFrame, out_path: Path):
    has_price = "avg_eth_price" in timeline.columns and timeline["avg_eth_price"].notna().any()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    ax1.bar(timeline["hour"], timeline["anomalous_tx"],
            color="#e74c3c", alpha=0.7, label="Anomalous whale txs", width=0.03)
    ax1.set_xlabel("Time (UTC)", color="white")
    ax1.set_ylabel("Anomalous tx count", color="#e74c3c")
    ax1.tick_params(axis="y", colors="#e74c3c")
    ax1.tick_params(axis="x", colors="white")
    ax1.spines[:].set_color("#333")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=30, ha="right", color="white")

    ax1.bar(timeline["hour"], timeline["total_tx"],
            color="#3498db", alpha=0.2, label="All whale txs", width=0.03)

    if has_price:
        ax2 = ax1.twinx()
        price_data = timeline.dropna(subset=["avg_eth_price"])
        ax2.plot(price_data["hour"], price_data["avg_eth_price"],
                 color="#f1c40f", linewidth=2, label="ETH/USD price", zorder=5)
        ax2.set_ylabel("ETH Price (USD)", color="#f1c40f")
        ax2.tick_params(axis="y", colors="#f1c40f")
        ax2.spines[:].set_color("#333")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   framealpha=0.3, labelcolor="white", facecolor="#1a1a1a", loc="upper left")
    else:
        ax1.legend(framealpha=0.3, labelcolor="white", facecolor="#1a1a1a")

    ax1.set_title("Whale Anomaly Activity vs ETH Price", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out_path.name}")


def compute_price_correlation(timeline: pd.DataFrame) -> None:
    if "avg_eth_price" not in timeline.columns:
        return
    t = timeline.dropna(subset=["avg_eth_price"]).copy()
    if len(t) < 4:
        return
    t["price_change_pct"] = t["avg_eth_price"].pct_change() * 100
    t = t.dropna(subset=["price_change_pct"])
    if len(t) < 3:
        return

    corr = t["anomalous_tx"].corr(t["price_change_pct"])
    corr_total = t["total_tx"].corr(t["price_change_pct"])
    print(f"\n  Pearson correlation — anomalous whale tx vs ETH price change: {corr:.3f}")
    print(f"  Pearson correlation — all whale tx vs ETH price change:       {corr_total:.3f}")
    if abs(corr) > 0.3:
        direction = "positive" if corr > 0 else "negative"
        print(f"  Notable {direction} correlation detected.")


def print_top_anomalies(df: pd.DataFrame, n: int = 10):
    anomalies = df[df["anomaly_label"] == -1].sort_values("anomaly_score")
    cols = ["address", "anomaly_score", "anomaly_type", "anomaly_explanation",
            "tx_count_out", "total_eth_out", "exchange_ratio", "unique_receivers"]
    avail_cols = [c for c in cols if c in anomalies.columns]
    print(f"\n=== Top {n} Most Anomalous Whale Addresses ===")
    pd.set_option("display.max_colwidth", 80)
    print(anomalies[avail_cols].head(n).to_string(index=False))
    pd.reset_option("display.max_colwidth")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    if not FEATURES_CSV.exists() and not CLUSTERED_CSV.exists():
        raise FileNotFoundError(
            "No feature data found. Run feature_engineering.py or clustering.py first."
        )

    print("Loading address features ...")
    df, source = load_address_features()
    print(f"  {len(df)} addresses from {source}")
    has_clusters = "kmeans_cluster" in df.columns

    # Preprocessing: log1p + RobustScaler (same as clustering.py)
    avail = [c for c in FEATURE_COLS if c in df.columns]
    df_feat = log_transform(df[avail])
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_feat.values)

    # Layer 1: Global anomalies
    print("\n[Layer 1] Isolation Forest (contamination=auto) ...")
    df, iso = detect_global_anomalies(df, X_scaled)

    # Layer 2: Intra-cluster anomalies
    if has_clusters:
        print("\n[Layer 2] Intra-cluster Mahalanobis distance ...")
        df = detect_cluster_anomalies(df, X_scaled, threshold_percentile=95)
    else:
        print("\n[Layer 2] Skipped — no cluster labels found. Run clustering.py first.")

    # Combine both layers
    print("\n[Combining anomaly layers]")
    df = combine_anomaly_labels(df)

    # Feature contributions for anomalous addresses
    print("\n[Feature Contributions] Explaining anomalies ...")
    df = compute_feature_contributions(df, X_scaled, avail)

    # Plots
    plot_anomaly_score_distribution(df, DATA_DIR / "anomaly_scores_plot.png")
    if has_clusters:
        plot_anomaly_by_cluster(df, DATA_DIR / "anomaly_by_cluster.png")

    print_top_anomalies(df)

    # Save
    df.to_csv(ANOMALY_CSV, index=False)
    print(f"\nSaved anomaly scores to {ANOMALY_CSV}")

    # Timeline analysis
    if RAW_CSV.exists():
        print("\n[Timeline] Building anomaly-vs-price timeline ...")
        raw_df = pd.read_csv(RAW_CSV)
        anomalous_addresses = set(df[df["anomaly_label"] == -1]["address"])
        timeline = build_timeline(raw_df, anomalous_addresses)
        plot_anomaly_timeline(timeline, DATA_DIR / "anomaly_timeline.png")
        compute_price_correlation(timeline)
    else:
        print("raw_whale_transactions.csv not found — skipping timeline analysis.")


if __name__ == "__main__":
    main()
