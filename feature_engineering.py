"""
feature_engineering.py
======================
Aggregates raw per-transaction data into per-address behavioural features
over a rolling time window (default 30 days).

Input:  data/raw_whale_transactions.csv
Output: data/address_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy as scipy_entropy

DATA_DIR = Path(__file__).parent / "data_new"
RAW_CSV = DATA_DIR / "raw_whale_transactions.csv"
FEATURES_CSV = DATA_DIR / "address_features.csv"
KNOWN_CSV = DATA_DIR / "known_addresses.csv"

WINDOW_DAYS = 30
ROUND_ETH_VALUES = {10, 20, 25, 30, 50, 75, 100, 150, 200, 250, 500, 1000}


def load_known_exchanges(path: Path) -> set[str]:
    """Return a set of lowercased known exchange addresses."""
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(df["address"].str.lower())


def is_round_number(value_eth: float, tolerance: float = 0.01) -> bool:
    """True if value_eth is within `tolerance` of a round whale number."""
    for r in ROUND_ETH_VALUES:
        if abs(value_eth - r) / r < tolerance:
            return True
    return False


def hour_entropy(timestamps: pd.Series) -> float:
    """Shannon entropy of transaction hour distribution (0-23).
    High entropy → transactions spread across all hours (bot / exchange).
    Low entropy  → concentrated at specific hours (human behaviour)."""
    hours = pd.to_datetime(timestamps).dt.hour
    counts = hours.value_counts().reindex(range(24), fill_value=0).values.astype(float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return float(scipy_entropy(probs + 1e-9, base=2))


def avg_interval_hours(timestamps: pd.Series) -> float:
    """Average time gap between consecutive transactions in hours."""
    sorted_ts = pd.to_datetime(timestamps).sort_values()
    if len(sorted_ts) < 2:
        return np.nan
    diffs = sorted_ts.diff().dropna().dt.total_seconds() / 3600
    return float(diffs.mean())


def build_address_features(df_raw: pd.DataFrame, exchange_set: set[str]) -> pd.DataFrame:
    """
    For each unique from_address compute 22 behavioural features.

    Returns a DataFrame indexed by address.
    """
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["to_lower"] = df["to_address"].str.lower()
    df["is_exchange_dest"] = df["to_lower"].isin(exchange_set)
    df["is_round"] = df["value_eth"].apply(is_round_number)

    # Restrict to the last WINDOW_DAYS of data
    cutoff = df["timestamp"].max() - pd.Timedelta(days=WINDOW_DAYS)
    df = df[df["timestamp"] >= cutoff]

    records = []

    for address, grp in df.groupby("from_address"):
        grp = grp.sort_values("timestamp")

        tx_count_out = len(grp)
        total_eth_out = grp["value_eth"].sum()
        total_usd_out = grp["value_usd"].sum() if "value_usd" in grp else np.nan
        avg_tx_eth = grp["value_eth"].mean()
        median_tx_eth = grp["value_eth"].median()
        max_tx_eth = grp["value_eth"].max()
        std_tx_eth = grp["value_eth"].std(ddof=0) if tx_count_out > 1 else 0.0

        unique_receivers = grp["to_address"].nunique()
        exchange_volume = grp.loc[grp["is_exchange_dest"], "value_eth"].sum()
        exchange_ratio = exchange_volume / total_eth_out if total_eth_out > 0 else 0.0

        # Concentration: fraction of volume going to single top receiver
        top_receiver_vol = grp.groupby("to_address")["value_eth"].sum().max()
        top1_receiver_ratio = top_receiver_vol / total_eth_out if total_eth_out > 0 else 0.0

        round_number_ratio = grp["is_round"].mean()

        avg_gas_gwei = grp["gas_price_gwei"].mean()
        gas_variability = grp["gas_price_gwei"].std(ddof=0) if tx_count_out > 1 else 0.0

        h_entropy = hour_entropy(grp["timestamp"])
        active_days = grp["timestamp"].dt.date.nunique()
        avg_interval = avg_interval_hours(grp["timestamp"])

        # Incoming ETH for net-flow (from_address receives in to_address column of other rows)
        incoming = df[df["to_address"].str.lower() == address.lower()]["value_eth"].sum()
        net_flow_eth = incoming - total_eth_out

        # Blocks span
        block_span = int(grp["block_number"].max() - grp["block_number"].min()) if "block_number" in grp else 0

        records.append(
            {
                "address": address,
                "tx_count_out": tx_count_out,
                "total_eth_out": round(total_eth_out, 4),
                "total_usd_out": round(total_usd_out, 2) if pd.notna(total_usd_out) else np.nan,
                "avg_tx_eth": round(avg_tx_eth, 4),
                "median_tx_eth": round(median_tx_eth, 4),
                "max_tx_eth": round(max_tx_eth, 4),
                "std_tx_eth": round(std_tx_eth, 4),
                "unique_receivers": unique_receivers,
                "exchange_ratio": round(exchange_ratio, 4),
                "top1_receiver_ratio": round(top1_receiver_ratio, 4),
                "round_number_ratio": round(round_number_ratio, 4),
                "avg_gas_gwei": round(avg_gas_gwei, 4),
                "gas_variability": round(gas_variability, 4),
                "hour_entropy": round(h_entropy, 4),
                "active_days": active_days,
                "avg_interval_hours": round(avg_interval, 4) if pd.notna(avg_interval) else np.nan,
                "net_flow_eth": round(net_flow_eth, 4),
                "block_span": block_span,
                # Derived convenience flags
                "is_known_exchange": int(address.lower() in exchange_set),
                "first_seen": grp["timestamp"].min().isoformat(),
                "last_seen": grp["timestamp"].max().isoformat(),
            }
        )

    return pd.DataFrame(records)


def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Run data_collection.py first."
        )

    print(f"Loading {RAW_CSV} ...")
    df_raw = pd.read_csv(RAW_CSV)
    print(f"  {len(df_raw)} raw transactions, {df_raw['from_address'].nunique()} unique senders")

    exchange_set = load_known_exchanges(KNOWN_CSV)
    print(f"  Loaded {len(exchange_set)} known exchange addresses")

    print("Building address features ...")
    df_features = build_address_features(df_raw, exchange_set)
    print(f"  {len(df_features)} addresses with features")

    df_features.to_csv(FEATURES_CSV, index=False)
    print(f"Saved to {FEATURES_CSV}")
    print(df_features.describe().to_string())


if __name__ == "__main__":
    main()
