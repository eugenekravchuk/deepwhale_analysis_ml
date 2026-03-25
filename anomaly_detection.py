"""
anomaly_detection.py
====================
Detects anomalous whale behaviour using Isolation Forest.

Two angles of analysis:
  A) Address-level anomalies — which addresses have atypical feature profiles
     compared to the rest of the whale population.
  B) Market-event correlation — do anomaly spikes in whale activity precede
     or coincide with sharp ETH price movements?

Outputs:
  data/anomaly_scores.csv      — per-address anomaly score (-1=anomaly, 1=normal)
  data/anomaly_timeline.png    — whale anomaly count vs ETH price over time
  data/anomaly_scores_plot.png — score distribution

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

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
FEATURES_CSV = DATA_DIR / "address_features.csv"
LABELED_CSV = DATA_DIR / "labeled_addresses.csv"
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

# Isolation Forest contamination — expected fraction of anomalies in data
CONTAMINATION = 0.10


def load_address_features(path: Path) -> pd.DataFrame:
    source = LABELED_CSV if LABELED_CSV.exists() else path
    df = pd.read_csv(source)
    for col in FEATURE_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def detect_address_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit Isolation Forest on address-level features.
    Adds columns: anomaly_label (-1 or 1), anomaly_score (continuous, lower = more anomalous).
    """
    avail = [c for c in FEATURE_COLS if c in df.columns]
    X = df[avail].values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    df = df.copy()
    df["anomaly_label"] = iso.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
    df["anomaly_score"] = iso.score_samples(X_scaled)  # lower → more anomalous

    n_anomalies = (df["anomaly_label"] == -1).sum()
    print(f"  Anomalous addresses: {n_anomalies} / {len(df)} "
          f"({n_anomalies/len(df)*100:.1f}%)")
    return df


def plot_anomaly_score_distribution(df: pd.DataFrame, out_path: Path):
    normal = df.loc[df["anomaly_label"] == 1, "anomaly_score"]
    anomaly = df.loc[df["anomaly_label"] == -1, "anomaly_score"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    bins = np.linspace(df["anomaly_score"].min(), df["anomaly_score"].max(), 40)
    ax.hist(normal, bins=bins, color="#3498db", alpha=0.7, label="Normal whale")
    ax.hist(anomaly, bins=bins, color="#e74c3c", alpha=0.8, label="Anomalous whale")
    ax.axvline(df.loc[df["anomaly_label"] == -1, "anomaly_score"].max(),
               color="orange", linestyle="--", lw=1.5, label="Anomaly threshold")

    ax.set_xlabel("Isolation Forest Score (lower = more anomalous)", color="white")
    ax.set_ylabel("Number of addresses", color="white")
    ax.set_title("Whale Anomaly Score Distribution", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.legend(framealpha=0.3, labelcolor="white", facecolor="#1a1a1a")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out_path.name}")


def build_timeline(raw_df: pd.DataFrame, addr_anomalies: pd.Series) -> pd.DataFrame:
    """
    Join raw transactions with anomaly labels to build an hourly timeline of:
    - anomalous_tx_count: number of transactions from anomalous addresses
    - total_tx_count: total whale transactions
    - avg_eth_price: average ETH/USD price (if available)
    """
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
    """Dual-axis chart: whale anomaly count + ETH price over time."""
    has_price = "avg_eth_price" in timeline.columns and timeline["avg_eth_price"].notna().any()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # Anomalous transaction count
    ax1.bar(timeline["hour"], timeline["anomalous_tx"],
            color="#e74c3c", alpha=0.7, label="Anomalous whale txs", width=0.03)
    ax1.set_xlabel("Time (UTC)", color="white")
    ax1.set_ylabel("Anomalous tx count", color="#e74c3c")
    ax1.tick_params(axis="y", colors="#e74c3c")
    ax1.tick_params(axis="x", colors="white")
    ax1.spines[:].set_color("#333")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=30, ha="right", color="white")

    # Total whale volume (faded)
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

        # Merge legends
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
    """Print Pearson correlation between anomaly count and ETH price change."""
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
    cols = ["address", "anomaly_score", "tx_count_out", "total_eth_out",
            "exchange_ratio", "unique_receivers"]
    avail_cols = [c for c in cols if c in anomalies.columns]
    print(f"\n=== Top {n} Most Anomalous Whale Addresses ===")
    print(anomalies[avail_cols].head(n).to_string(index=False))


def main():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"{FEATURES_CSV} not found. Run feature_engineering.py first.")

    print("Loading address features ...")
    df = load_address_features(FEATURES_CSV)
    print(f"  {len(df)} addresses")

    print("\n[Isolation Forest] Detecting anomalous whale addresses ...")
    df = detect_address_anomalies(df)

    plot_anomaly_score_distribution(df, DATA_DIR / "anomaly_scores_plot.png")
    print_top_anomalies(df)

    # Save anomaly-annotated addresses
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
