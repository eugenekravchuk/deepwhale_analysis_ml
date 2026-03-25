"""
labeling.py
===========
Assigns heuristic behavioural labels to each whale address based on its
aggregated features.  Labels are *weak* — derived from transparent rules,
not ground-truth annotations — and are used as training targets for the
supervised classifier.

Input:  data/address_features.csv  (produced by feature_engineering.py)
        data/known_addresses.csv   (known exchange hot/cold wallets)
Output: data/labeled_addresses.csv

Label taxonomy (4 classes):
  exchange_depositor  – regularly sends large ETH volumes to exchange wallets
  accumulator         – net ETH inflow, infrequent outbound activity (HODLer)
  active_trader       – high-frequency, many unique counterparties, varied gas
  unknown_whale       – does not fit the above patterns clearly
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
FEATURES_CSV = DATA_DIR / "address_features.csv"
KNOWN_CSV = DATA_DIR / "known_addresses.csv"
LABELED_CSV = DATA_DIR / "labeled_addresses.csv"

# ── Thresholds (tunable) ────────────────────────────────────────────────────
EXCHANGE_RATIO_THRESHOLD = 0.50      # >50 % volume flows to known exchanges
ACCUMULATOR_NET_FLOW_MIN = 0.0       # net ETH inflow ≥ 0
ACCUMULATOR_MAX_ACTIVE_DAYS = 7      # trades on fewer than 7 days
ACCUMULATOR_LOW_TX_MAX = 5           # fewer than 5 outbound transactions
TRADER_MIN_TX = 8                    # at least 8 outbound transactions
TRADER_MIN_RECEIVERS = 5             # at least 5 unique receivers
TRADER_MAX_TOP1_RATIO = 0.70         # no single receiver dominates >70 %


def assign_label(row: pd.Series) -> str:
    """Deterministic rule-based label assignment."""

    # Override: address is itself a known exchange (shouldn't appear much as sender,
    # but handle gracefully)
    if row.get("is_known_exchange", 0) == 1:
        return "exchange_depositor"

    # --- Exchange Depositor ---
    # High fraction of outbound volume goes to known exchange addresses
    if row["exchange_ratio"] >= EXCHANGE_RATIO_THRESHOLD:
        return "exchange_depositor"

    # --- Accumulator / Long-term HODLer ---
    # Positive net flow (receives more than sends), low activity
    if (
        row["net_flow_eth"] >= ACCUMULATOR_NET_FLOW_MIN
        and row["active_days"] <= ACCUMULATOR_MAX_ACTIVE_DAYS
        and row["tx_count_out"] <= ACCUMULATOR_LOW_TX_MAX
    ):
        return "accumulator"

    # --- Active Trader ---
    # High tx count, many different counterparties, no single dominant receiver
    if (
        row["tx_count_out"] >= TRADER_MIN_TX
        and row["unique_receivers"] >= TRADER_MIN_RECEIVERS
        and row["top1_receiver_ratio"] <= TRADER_MAX_TOP1_RATIO
    ):
        return "active_trader"

    return "unknown_whale"


def compute_label_confidence(row: pd.Series, label: str) -> float:
    """
    Rough confidence score [0, 1] based on how strongly the address satisfies
    the rules for its assigned label.  Useful for filtering low-confidence
    samples before training.
    """
    if label == "exchange_depositor":
        return min(1.0, float(row["exchange_ratio"]) / 1.0)

    if label == "accumulator":
        # Normalise: fewer active days → more confident
        day_score = max(0, 1 - row["active_days"] / ACCUMULATOR_MAX_ACTIVE_DAYS)
        tx_score = max(0, 1 - row["tx_count_out"] / ACCUMULATOR_LOW_TX_MAX)
        return float((day_score + tx_score) / 2)

    if label == "active_trader":
        tx_score = min(1.0, row["tx_count_out"] / (TRADER_MIN_TX * 3))
        recv_score = min(1.0, row["unique_receivers"] / (TRADER_MIN_RECEIVERS * 3))
        return float((tx_score + recv_score) / 2)

    return 0.3  # unknown_whale: low confidence by default


def label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df.apply(assign_label, axis=1)
    df["label_confidence"] = df.apply(
        lambda r: compute_label_confidence(r, r["label"]), axis=1
    )
    return df


def print_label_summary(df: pd.DataFrame):
    counts = df["label"].value_counts()
    print("\nLabel distribution:")
    for lbl, cnt in counts.items():
        pct = cnt / len(df) * 100
        avg_conf = df.loc[df["label"] == lbl, "label_confidence"].mean()
        print(f"  {lbl:<22} {cnt:>4} ({pct:5.1f}%)  avg_confidence={avg_conf:.2f}")
    print()


def main():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"{FEATURES_CSV} not found. Run feature_engineering.py first."
        )

    print(f"Loading features from {FEATURES_CSV} ...")
    df = pd.read_csv(FEATURES_CSV)
    print(f"  {len(df)} addresses")

    df = label_dataframe(df)

    print_label_summary(df)

    df.to_csv(LABELED_CSV, index=False)
    print(f"Saved labeled data to {LABELED_CSV}")

    # Print a few examples per class
    for lbl in df["label"].unique():
        sample = df[df["label"] == lbl].head(2)[
            ["address", "tx_count_out", "exchange_ratio", "active_days",
             "net_flow_eth", "unique_receivers", "label", "label_confidence"]
        ]
        print(f"\n--- {lbl} (sample) ---")
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
