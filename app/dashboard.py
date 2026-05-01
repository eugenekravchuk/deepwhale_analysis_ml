"""
dashboard.py  —  DeepWhale Interactive Dashboard
=================================================
Run:  streamlit run dashboard.py

Four tabs:
  1. Live Whales      — latest on-chain whale transactions (filterable)
  2. Address Profiler — classify any ETH address + SHAP explanation
  3. Anomaly Explorer — browse anomalous whales with explanations
  4. Market Context   — ETH price vs whale activity + cluster map
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

RAW_CSV = RAW_DIR / "raw_whale_transactions.csv"
LABELED_CSV = PROCESSED_DIR / "labeled_addresses.csv"
ANOMALY_CSV = PROCESSED_DIR / "anomaly_scores.csv"
CLUSTERED_CSV = PROCESSED_DIR / "clustered_addresses.csv"
KNOWN_CSV = DATA_DIR / "external" / "known_addresses.csv"
MODEL_PKL = MODELS_DIR / "whale_classifier.pkl"

FEATURE_COLS = [
    "tx_count_out", "total_eth_out", "avg_tx_eth", "median_tx_eth",
    "max_tx_eth", "std_tx_eth", "unique_receivers", "exchange_ratio",
    "top1_receiver_ratio", "round_number_ratio", "avg_gas_gwei",
    "gas_variability", "hour_entropy", "active_days", "net_flow_eth",
]

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

CLUSTER_COLORS = {
    "0": "#e74c3c",
    "1": "#3498db",
    "2": "#2ecc71",
    "3": "#f39c12",
}

CLUSTER_ICONS = {
    0: "🎯",
    1: "🐋",
    2: "🐟",
    3: "🔄",
}


# ── Data loaders (cached) ────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_raw_transactions() -> pd.DataFrame:
    if not RAW_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
    return df.sort_values("timestamp", ascending=False)


@st.cache_data(ttl=300)
def load_anomaly() -> pd.DataFrame:
    if not ANOMALY_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(ANOMALY_CSV)


@st.cache_data(ttl=300)
def load_clustered() -> pd.DataFrame:
    if not CLUSTERED_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(CLUSTERED_CSV)


@st.cache_resource
def load_model():
    if not MODEL_PKL.exists():
        return None
    with open(MODEL_PKL, "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=30)
def fetch_eth_price_binance(n_candles: int = 168) -> pd.DataFrame:
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "ETHUSDT", "interval": "1h", "limit": n_candles}
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "n_trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df[["timestamp", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_known_addresses() -> set:
    if not KNOWN_CSV.exists():
        return set()
    return set(pd.read_csv(KNOWN_CSV)["address"].str.lower())


def log_transform_row(values: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Apply log1p to a single feature row (for real-time inference)."""
    result = values.copy().astype(float)
    for i, fname in enumerate(feature_names):
        if fname in LOG_COLS:
            result[i] = np.log1p(max(0, result[i]))
        elif fname == "net_flow_eth":
            result[i] = np.sign(result[i]) * np.log1p(abs(result[i]))
    return result


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepWhale Dashboard",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main { background-color: #0d1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] { background-color: #1f6feb; color: white; }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    .anomaly-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 600;
    }
    .badge-global { background: #e74c3c; color: white; }
    .badge-cluster { background: #f39c12; color: white; }
    .badge-both { background: #9b59b6; color: white; }
    .badge-normal { background: #2ecc71; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def cluster_badge(cluster_id: int) -> str:
    icon = CLUSTER_ICONS.get(cluster_id, "❓")
    name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
    color = list(CLUSTER_COLORS.values())[cluster_id % len(CLUSTER_COLORS)]
    return (
        f'<span style="background:{color};color:white;padding:4px 12px;'
        f'border-radius:12px;font-size:0.85em;font-weight:600">'
        f'{icon} {name}</span>'
    )


def anomaly_badge(anomaly_type: str) -> str:
    mapping = {
        "global": ("badge-global", "🌐 Global Anomaly"),
        "intra-cluster": ("badge-cluster", "🎯 Intra-Cluster Anomaly"),
        "both": ("badge-both", "⚠️ Double Anomaly"),
        "normal": ("badge-normal", "✅ Normal"),
    }
    css_class, text = mapping.get(anomaly_type, ("badge-normal", "✅ Normal"))
    return f'<span class="anomaly-badge {css_class}">{text}</span>'


def build_feature_vector(address: str, raw_df: pd.DataFrame, known_set: set) -> list[float] | None:
    """Build a 15-feature vector for a single address from raw transactions."""
    sub = raw_df[raw_df["from_address"].str.lower() == address.lower()]
    if sub.empty:
        return None

    tx_count_out = len(sub)
    total_eth_out = sub["value_eth"].sum()
    avg_tx_eth = sub["value_eth"].mean()
    median_tx_eth = sub["value_eth"].median()
    max_tx_eth = sub["value_eth"].max()
    std_tx_eth = sub["value_eth"].std(ddof=0) if tx_count_out > 1 else 0.0
    unique_receivers = sub["to_address"].nunique()

    exchange_dest = sub["to_address"].str.lower().isin(known_set)
    exchange_ratio = (
        sub.loc[exchange_dest, "value_eth"].sum() / total_eth_out
        if total_eth_out > 0 else 0.0
    )

    top1 = sub.groupby("to_address")["value_eth"].sum().max()
    top1_receiver_ratio = top1 / total_eth_out if total_eth_out > 0 else 0.0

    ROUND_SET = {10, 20, 25, 30, 50, 75, 100, 150, 200, 250, 500, 1000}
    def is_round(v):
        return any(abs(v - r) / r < 0.01 for r in ROUND_SET if r > 0)
    round_number_ratio = sub["value_eth"].apply(is_round).mean()

    avg_gas_gwei = sub["gas_price_gwei"].mean() if "gas_price_gwei" in sub.columns else 0.0
    gas_variability = (
        sub["gas_price_gwei"].std(ddof=0) if tx_count_out > 1 and "gas_price_gwei" in sub.columns
        else 0.0
    )

    sub_ts = pd.to_datetime(sub["timestamp"])
    hours = sub_ts.dt.hour
    counts = hours.value_counts().reindex(range(24), fill_value=0).values.astype(float)
    probs = counts / (counts.sum() + 1e-9)
    hour_entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))

    active_days = sub_ts.dt.date.nunique()

    recv_eth = raw_df[raw_df["to_address"].str.lower() == address.lower()]["value_eth"].sum()
    net_flow_eth = recv_eth - total_eth_out

    return [
        tx_count_out, total_eth_out, avg_tx_eth, median_tx_eth, max_tx_eth,
        std_tx_eth, unique_receivers, exchange_ratio, top1_receiver_ratio,
        round_number_ratio, avg_gas_gwei, gas_variability, hour_entropy,
        active_days, net_flow_eth,
    ]


def predict_cluster(feature_vector: list[float], model_bundle: dict) -> dict:
    """Classify address into a cluster and return probabilities."""
    feat_cols = model_bundle["feature_cols"]
    scaler = model_bundle["scaler"]
    xgb = model_bundle["xgb"]
    cluster_names = model_bundle["cluster_names"]

    X = np.array(feature_vector[:len(feat_cols)]).reshape(1, -1)
    X_log = log_transform_row(X[0], feat_cols).reshape(1, -1)
    X_scaled = scaler.transform(X_log)

    proba = xgb.predict_proba(X_scaled)[0]
    pred_idx = int(np.argmax(proba))
    pred_name = cluster_names.get(pred_idx, f"Cluster {pred_idx}")

    return {
        "cluster_id": pred_idx,
        "cluster_name": pred_name,
        "confidence": float(proba[pred_idx]),
        "all_probs": {cluster_names.get(i, f"Cluster {i}"): float(p) for i, p in enumerate(proba)},
        "X_scaled": X_scaled,
    }


# ── Load all data ────────────────────────────────────────────────────────────
raw_df = load_raw_transactions()
anomaly_df = load_anomaly()
clustered_df = load_clustered()
model_bundle = load_model()
known_set = load_known_addresses()


# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 🐋 DeepWhale — Crypto Whale Behaviour Dashboard")
st.caption("Real-time classification, anomaly detection, and market analysis of ETH whales")

# Top KPIs
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Whale Transactions", f"{len(raw_df):,}" if not raw_df.empty else "—")
with c2:
    n_addr = raw_df["from_address"].nunique() if not raw_df.empty else 0
    st.metric("Unique Addresses", f"{n_addr:,}")
with c3:
    n_anomaly = int((anomaly_df["anomaly_label"] == -1).sum()) if not anomaly_df.empty else 0
    st.metric("Anomalies Detected", f"{n_anomaly}")
with c4:
    total_eth = raw_df["value_eth"].sum() if not raw_df.empty else 0
    st.metric("Total ETH Moved", f"{total_eth:,.0f}")
with c5:
    price_df = fetch_eth_price_binance(1)
    if not price_df.empty:
        st.metric("ETH Price", f"${price_df['close'].iloc[-1]:,.2f}")
    else:
        st.metric("ETH Price", "—")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 Live Whales", "🔍 Address Profiler", "🚨 Anomaly Explorer", "📈 Market Context"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE WHALES
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Recent Whale Transactions")

    if raw_df.empty:
        st.warning("No transaction data found. Run `python data_collection.py` first.")
    else:
        # Filters
        col_f1, col_f2, col_f3, col_f4 = st.columns([1.5, 1.5, 1.5, 1.5])
        with col_f1:
            min_eth = st.slider("Min ETH", 10, 1000, 10, step=10, key="t1_min")
        with col_f2:
            page_size = st.selectbox("Rows per page", [25, 50, 100], index=1, key="t1_page")
        with col_f3:
            show_anom_only = st.checkbox("Anomalous only", value=False, key="t1_anom")
        with col_f4:
            cluster_filter = st.multiselect(
                "Filter by cluster",
                options=list(CLUSTER_NAMES.values()),
                default=[],
                key="t1_cluster",
            )

        filtered = raw_df[raw_df["value_eth"] >= min_eth].copy()

        # Enrich with anomaly and cluster info
        if not anomaly_df.empty:
            anomaly_set = set(anomaly_df[anomaly_df["anomaly_label"] == -1]["address"])
        else:
            anomaly_set = set()

        filtered["anomaly"] = filtered["from_address"].apply(
            lambda a: "🚨" if a in anomaly_set else ""
        )
        filtered["to_exchange"] = filtered["to_address"].apply(
            lambda a: "🏦" if str(a).lower() in known_set else ""
        )

        # Cluster info
        if not clustered_df.empty and "kmeans_cluster" in clustered_df.columns:
            cluster_map = dict(zip(clustered_df["address"], clustered_df["kmeans_cluster"]))
            filtered["cluster"] = filtered["from_address"].map(cluster_map)
            filtered["cluster_name"] = filtered["cluster"].map(CLUSTER_NAMES).fillna("Unknown")

            if cluster_filter:
                filtered = filtered[filtered["cluster_name"].isin(cluster_filter)]
        else:
            filtered["cluster_name"] = "Unknown"

        if show_anom_only:
            filtered = filtered[filtered["anomaly"] == "🚨"]

        # Pagination
        total_rows = len(filtered)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)

        if "whale_page" not in st.session_state:
            st.session_state.whale_page = 1
        if st.session_state.whale_page > total_pages:
            st.session_state.whale_page = total_pages

        pc1, pc2, pc3 = st.columns([1, 3, 1])
        with pc1:
            if st.button("← Prev", disabled=st.session_state.whale_page <= 1, key="t1_prev"):
                st.session_state.whale_page -= 1
                st.rerun()
        with pc2:
            st.markdown(
                f"<div style='text-align:center;color:#8b949e'>"
                f"Page {st.session_state.whale_page}/{total_pages} "
                f"({total_rows:,} txs)</div>",
                unsafe_allow_html=True,
            )
        with pc3:
            if st.button("Next →", disabled=st.session_state.whale_page >= total_pages, key="t1_next"):
                st.session_state.whale_page += 1
                st.rerun()

        start = (st.session_state.whale_page - 1) * page_size
        display = filtered.iloc[start:start + page_size]

        show_cols = ["timestamp", "from_address", "to_address", "value_eth",
                     "value_usd", "cluster_name", "to_exchange", "anomaly"]
        avail = [c for c in show_cols if c in display.columns]

        st.dataframe(
            display[avail].rename(columns={
                "value_eth": "ETH",
                "value_usd": "USD",
                "cluster_name": "Whale Type",
                "to_exchange": "Exch",
                "anomaly": "⚠️",
            }),
            use_container_width=True,
            height=420,
        )

        # Hourly activity chart
        if "timestamp" in raw_df.columns:
            hourly = raw_df.copy()
            hourly["hour"] = hourly["timestamp"].dt.floor("h")
            hourly_agg = hourly.groupby("hour").agg(
                tx_count=("tx_hash", "count"),
                total_eth=("value_eth", "sum"),
            ).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hourly_agg["hour"], y=hourly_agg["tx_count"],
                name="Transactions", marker_color="#3498db", opacity=0.8,
            ))
            fig.add_trace(go.Scatter(
                x=hourly_agg["hour"], y=hourly_agg["total_eth"],
                name="ETH Volume", mode="lines",
                line=dict(color="#f39c12", width=2), yaxis="y2",
            ))
            fig.update_layout(
                title="Hourly Whale Activity",
                template="plotly_dark",
                yaxis=dict(title="Transactions"),
                yaxis2=dict(title="ETH Volume", overlaying="y", side="right"),
                legend=dict(orientation="h", y=1.08),
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ADDRESS PROFILER
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Whale Address Profiler")
    st.caption("Classify any ETH address into a behavioural cluster with SHAP explanation")

    # Address input
    col_input, col_select = st.columns([2, 2])
    with col_input:
        addr_input = st.text_input(
            "ETH Address (0x...)",
            placeholder="0x28c6c06298d514db089934071355e5743bf21d60",
            key="t2_addr",
        )
    with col_select:
        if not raw_df.empty and not addr_input:
            top_addrs = raw_df["from_address"].value_counts().head(20).index.tolist()
            selected = st.selectbox(
                "Or pick from top whales in dataset",
                [""] + top_addrs, key="t2_select",
            )
            if selected:
                addr_input = selected

    if addr_input:
        addr_lower = addr_input.strip().lower()

        # Build feature vector
        feat_vec = build_feature_vector(addr_input.strip(), raw_df, known_set) if not raw_df.empty else None

        col_class, col_status = st.columns([1, 1])

        with col_class:
            st.markdown("#### Classification")
            if model_bundle and feat_vec:
                result = predict_cluster(feat_vec, model_bundle)
                st.markdown(
                    f"**Predicted type:** {cluster_badge(result['cluster_id'])}",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Confidence:** {result['confidence']*100:.1f}%")

                # Probability bars
                prob_df = pd.DataFrame(
                    list(result["all_probs"].items()), columns=["Cluster", "Probability"]
                ).sort_values("Probability", ascending=True)
                fig = px.bar(
                    prob_df, x="Probability", y="Cluster", orientation="h",
                    template="plotly_dark", color="Probability",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(height=200, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

                # SHAP explanation
                try:
                    import shap
                    explainer = shap.TreeExplainer(model_bundle["xgb"])
                    shap_values = explainer.shap_values(result["X_scaled"])

                    if isinstance(shap_values, list):
                        sv = np.array(shap_values[result["cluster_id"]])[0]
                    elif shap_values.ndim == 3:
                        sv = shap_values[0, :, result["cluster_id"]]
                    else:
                        sv = shap_values[0]

                    feat_names = model_bundle["feature_cols"]
                    shap_df = pd.DataFrame({
                        "Feature": feat_names[:len(sv)],
                        "SHAP Impact": sv[:len(feat_names)],
                    }).sort_values("SHAP Impact", key=abs, ascending=True)

                    fig_shap = px.bar(
                        shap_df.tail(8), x="SHAP Impact", y="Feature",
                        orientation="h", template="plotly_dark",
                        title="Top features driving this prediction",
                        color="SHAP Impact",
                        color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=0,
                    )
                    fig_shap.update_layout(height=260, coloraxis_showscale=False)
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception:
                    st.info("SHAP explanation unavailable (install `shap` package)")

            elif not model_bundle:
                st.info("No model found. Run `python classification.py` first.")
            else:
                st.warning("Address not found in collected transactions.")

        with col_status:
            st.markdown("#### Anomaly Status")
            # Check pre-computed anomaly
            if not anomaly_df.empty and "address" in anomaly_df.columns:
                row = anomaly_df[anomaly_df["address"].str.lower() == addr_lower]
                if not row.empty:
                    atype = row.iloc[0].get("anomaly_type", "normal")
                    st.markdown(anomaly_badge(atype), unsafe_allow_html=True)

                    explanation = row.iloc[0].get("anomaly_explanation", "")
                    if explanation:
                        st.markdown("**Why anomalous:**")
                        for part in str(explanation).split(";"):
                            part = part.strip()
                            if part:
                                st.markdown(f"- {part}")

                    score = row.iloc[0].get("iso_score", None)
                    if score is not None:
                        st.metric("LOF Score", f"{score:.4f}")

                    mahal = row.iloc[0].get("mahal_distance", None)
                    if mahal is not None and not np.isnan(mahal):
                        st.metric("Mahalanobis Distance", f"{mahal:.2f}")
                else:
                    st.markdown(anomaly_badge("normal"), unsafe_allow_html=True)
            else:
                st.info("Run `python anomaly_detection.py` for anomaly data.")

            # Cluster assignment from pre-computed data
            if not clustered_df.empty and "address" in clustered_df.columns:
                row = clustered_df[clustered_df["address"].str.lower() == addr_lower]
                if not row.empty:
                    cid = int(row.iloc[0].get("kmeans_cluster", -1))
                    st.markdown("**Cluster (pre-computed):**")
                    st.markdown(cluster_badge(cid), unsafe_allow_html=True)

        # Transaction history
        if not raw_df.empty:
            addr_txs = raw_df[raw_df["from_address"].str.lower() == addr_lower]
            if not addr_txs.empty:
                st.markdown("---")
                st.markdown(f"**Transaction History** — {len(addr_txs)} txs in dataset")

                col_h1, col_h2 = st.columns([2, 1])
                with col_h1:
                    ts_sorted = addr_txs.sort_values("timestamp").copy()
                    ts_sorted["cumulative_eth"] = ts_sorted["value_eth"].cumsum()
                    fig = px.area(
                        ts_sorted, x="timestamp", y="cumulative_eth",
                        template="plotly_dark",
                        title="Cumulative ETH Sent",
                        color_discrete_sequence=["#3498db"],
                    )
                    fig.update_layout(height=220)
                    st.plotly_chart(fig, use_container_width=True)

                with col_h2:
                    st.metric("Total ETH Sent", f"{addr_txs['value_eth'].sum():,.1f}")
                    st.metric("Transactions", f"{len(addr_txs)}")
                    st.metric("Unique Receivers", f"{addr_txs['to_address'].nunique()}")
                    st.metric("Active Days", f"{pd.to_datetime(addr_txs['timestamp']).dt.date.nunique()}")

                st.dataframe(
                    addr_txs[["timestamp", "to_address", "value_eth", "value_usd"]].head(15),
                    use_container_width=True, height=200,
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ANOMALY EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Anomaly Explorer")
    st.caption(
        "Two-layer detection: **Layer 1** (Local Outlier Factor) finds locally unusual addresses. "
        "**Layer 2** (Mahalanobis) finds addresses that deviate from their own cluster."
    )

    if anomaly_df.empty:
        st.warning("No anomaly data. Run `python anomaly_detection.py` first.")
    else:
        # Summary metrics
        n_total = len(anomaly_df)
        n_anom = int((anomaly_df["anomaly_label"] == -1).sum())
        n_global = int((anomaly_df.get("anomaly_type", pd.Series()) == "global").sum())
        n_cluster = int((anomaly_df.get("anomaly_type", pd.Series()) == "intra-cluster").sum())
        n_both = int((anomaly_df.get("anomaly_type", pd.Series()) == "both").sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Anomalies", f"{n_anom} / {n_total}", f"{n_anom/n_total*100:.1f}%")
        m2.metric("Global (LOF)", f"{n_global + n_both}")
        m3.metric("Intra-Cluster", f"{n_cluster + n_both}")
        m4.metric("Both Layers", f"{n_both}")

        st.markdown("---")

        # Filters
        col_af1, col_af2, col_af3 = st.columns([1.5, 1.5, 2])
        with col_af1:
            type_filter = st.multiselect(
                "Anomaly type",
                ["global", "intra-cluster", "both"],
                default=["global", "intra-cluster", "both"],
                key="t3_type",
            )
        with col_af2:
            sort_by = st.selectbox(
                "Sort by",
                ["anomaly_score", "mahal_distance", "total_eth_out"],
                index=0, key="t3_sort",
            )
        with col_af3:
            n_show = st.slider("Show top N", 10, 100, 30, step=10, key="t3_n")

        # Filter anomalies
        anom_view = anomaly_df[anomaly_df["anomaly_label"] == -1].copy()
        if "anomaly_type" in anom_view.columns:
            anom_view = anom_view[anom_view["anomaly_type"].isin(type_filter)]

        if sort_by in anom_view.columns:
            ascending = True if sort_by == "anomaly_score" else False
            anom_view = anom_view.sort_values(sort_by, ascending=ascending)

        anom_view = anom_view.head(n_show)

        # Display table
        display_cols = ["address", "anomaly_type", "anomaly_score", "anomaly_explanation",
                        "kmeans_cluster", "total_eth_out", "tx_count_out", "exchange_ratio"]
        avail_cols = [c for c in display_cols if c in anom_view.columns]

        st.dataframe(
            anom_view[avail_cols].rename(columns={
                "anomaly_type": "Type",
                "anomaly_score": "LOF Score",
                "anomaly_explanation": "Explanation",
                "kmeans_cluster": "Cluster",
                "total_eth_out": "ETH Out",
                "tx_count_out": "Txs",
                "exchange_ratio": "Exch %",
            }),
            use_container_width=True,
            height=400,
        )

        # Visualizations
        st.markdown("---")
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            # Score distribution
            fig_dist = go.Figure()
            normal_scores = anomaly_df.loc[anomaly_df["anomaly_label"] == 1, "iso_score"].dropna()
            anom_scores = anomaly_df.loc[anomaly_df["anomaly_label"] == -1, "iso_score"].dropna()

            fig_dist.add_trace(go.Histogram(
                x=normal_scores, name="Normal", marker_color="#3498db",
                opacity=0.7, nbinsx=40,
            ))
            fig_dist.add_trace(go.Histogram(
                x=anom_scores, name="Anomalous", marker_color="#e74c3c",
                opacity=0.8, nbinsx=40,
            ))
            if len(anom_scores) > 0:
                threshold = anom_scores.max()
                fig_dist.add_vline(x=threshold, line_dash="dash", line_color="#f39c12",
                                   annotation_text=f"Threshold: {threshold:.3f}")

            fig_dist.update_layout(
                title="LOF Score Distribution",
                template="plotly_dark",
                xaxis_title="Score (more negative = more anomalous)",
                yaxis_title="Count",
                barmode="overlay",
                height=320,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_v2:
            # Anomalies by cluster
            if "kmeans_cluster" in anomaly_df.columns and "anomaly_type" in anomaly_df.columns:
                anom_only = anomaly_df[anomaly_df["anomaly_label"] == -1].copy()
                anom_only["cluster_name"] = anom_only["kmeans_cluster"].map(CLUSTER_NAMES)

                fig_bar = px.histogram(
                    anom_only, x="cluster_name", color="anomaly_type",
                    template="plotly_dark",
                    title="Anomalies by Cluster & Detection Method",
                    color_discrete_map={
                        "global": "#e74c3c",
                        "intra-cluster": "#f39c12",
                        "both": "#9b59b6",
                    },
                    barmode="group",
                )
                fig_bar.update_layout(height=320, xaxis_title="", yaxis_title="Count")
                st.plotly_chart(fig_bar, use_container_width=True)

        # Top anomaly features heatmap
        if "top_anomaly_features" in anomaly_df.columns:
            st.markdown("#### Most Common Anomaly-Driving Features")
            all_feats = anomaly_df.loc[
                anomaly_df["anomaly_label"] == -1, "top_anomaly_features"
            ].dropna().str.split(", ").explode()
            feat_counts = all_feats.value_counts().head(10).reset_index()
            feat_counts.columns = ["Feature", "Times Flagged"]

            fig_feat = px.bar(
                feat_counts, x="Times Flagged", y="Feature", orientation="h",
                template="plotly_dark",
                color="Times Flagged", color_continuous_scale="Reds",
            )
            fig_feat.update_layout(height=280, coloraxis_showscale=False)
            st.plotly_chart(fig_feat, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MARKET CONTEXT
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Market Context — ETH Price vs Whale Activity")

    col_m1, col_m2 = st.columns([1, 3])
    with col_m1:
        hours_back = st.selectbox(
            "Price window", [24, 48, 168, 336], index=2,
            format_func=lambda h: f"{h}h ({h//24}d)", key="t4_hours",
        )

    price_df = fetch_eth_price_binance(hours_back)

    if price_df.empty:
        st.warning("Cannot reach Binance API. Check internet connection.")
    else:
        current_price = price_df["close"].iloc[-1]
        open_price = price_df["close"].iloc[0]
        change_pct = (current_price - open_price) / open_price * 100

        pm1, pm2, pm3 = st.columns(3)
        pm1.metric("Current ETH", f"${current_price:,.2f}", f"{change_pct:+.2f}%")
        pm2.metric("Period High", f"${price_df['close'].max():,.2f}")
        pm3.metric("Period Low", f"${price_df['close'].min():,.2f}")

        # Price + whale volume overlay
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_df["timestamp"], y=price_df["close"],
            name="ETH Price", mode="lines",
            line=dict(color="#f1c40f", width=2),
            fill="tozeroy", fillcolor="rgba(241,196,15,0.08)",
        ))

        if not raw_df.empty:
            wh = raw_df.copy()
            wh["hour"] = wh["timestamp"].dt.floor("h")
            wh_agg = wh.groupby("hour")["value_eth"].sum().reset_index()
            wh_agg.columns = ["timestamp", "whale_eth"]

            merged = pd.merge(
                price_df[["timestamp"]], wh_agg, on="timestamp", how="left"
            ).fillna(0)

            if merged["whale_eth"].sum() > 0:
                fig.add_trace(go.Bar(
                    x=merged["timestamp"], y=merged["whale_eth"],
                    name="Whale ETH Volume", yaxis="y2",
                    marker_color="rgba(52,152,219,0.5)",
                ))

            # Anomaly volume overlay
            if not anomaly_df.empty:
                anom_set = set(anomaly_df[anomaly_df["anomaly_label"] == -1]["address"])
                wh_anom = raw_df[raw_df["from_address"].isin(anom_set)].copy()
                if not wh_anom.empty:
                    wh_anom["hour"] = wh_anom["timestamp"].dt.floor("h")
                    anom_agg = wh_anom.groupby("hour")["value_eth"].sum().reset_index()
                    anom_agg.columns = ["timestamp", "anom_eth"]
                    fig.add_trace(go.Bar(
                        x=anom_agg["timestamp"], y=anom_agg["anom_eth"],
                        name="Anomalous Whale ETH", yaxis="y2",
                        marker_color="rgba(231,76,60,0.7)",
                    ))

        fig.update_layout(
            template="plotly_dark",
            title="ETH Price + Whale Activity Overlay",
            yaxis=dict(title="ETH Price (USD)"),
            yaxis2=dict(title="ETH Volume", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.08),
            height=420,
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Cluster Distribution ─────────────────────────────────────────────────
    if not clustered_df.empty and "kmeans_cluster" in clustered_df.columns:
        st.markdown("---")
        st.subheader("Whale Population by Cluster")

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            cluster_counts = clustered_df["kmeans_cluster"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster ID", "Count"]
            cluster_counts["Name"] = cluster_counts["Cluster ID"].map(CLUSTER_NAMES)

            fig_pie = px.pie(
                cluster_counts, names="Name", values="Count",
                template="plotly_dark",
                title="Cluster Distribution",
                color="Name",
                color_discrete_map={v: list(CLUSTER_COLORS.values())[k % 4]
                                    for k, v in CLUSTER_NAMES.items()},
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_c2:
            # ETH volume by cluster
            if not raw_df.empty:
                cluster_map = dict(zip(clustered_df["address"], clustered_df["kmeans_cluster"]))
                vol_df = raw_df.copy()
                vol_df["cluster"] = vol_df["from_address"].map(cluster_map)
                vol_df["cluster_name"] = vol_df["cluster"].map(CLUSTER_NAMES).fillna("Unknown")
                vol_by_cluster = vol_df.groupby("cluster_name")["value_eth"].sum().reset_index()
                vol_by_cluster.columns = ["Cluster", "ETH Volume"]

                fig_vol = px.bar(
                    vol_by_cluster.sort_values("ETH Volume", ascending=True),
                    x="ETH Volume", y="Cluster", orientation="h",
                    template="plotly_dark",
                    title="ETH Volume by Whale Cluster",
                    color="Cluster",
                    color_discrete_map={v: list(CLUSTER_COLORS.values())[k % 4]
                                        for k, v in CLUSTER_NAMES.items()},
                )
                fig_vol.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_vol, use_container_width=True)

        # Interactive PCA scatter
        st.markdown("---")
        st.subheader("Cluster Map (PCA Projection)")

        from sklearn.decomposition import PCA

        avail_feat = [c for c in FEATURE_COLS if c in clustered_df.columns]
        if len(avail_feat) >= 2 and len(clustered_df) >= 10:
            # Apply same preprocessing as clustering
            pca_df = clustered_df[avail_feat].copy()
            for col in LOG_COLS:
                if col in pca_df.columns:
                    pca_df[col] = np.log1p(pca_df[col].clip(lower=0))
            if "net_flow_eth" in pca_df.columns:
                nf = pca_df["net_flow_eth"]
                pca_df["net_flow_eth"] = np.sign(nf) * np.log1p(nf.abs())

            from sklearn.preprocessing import RobustScaler as RS
            X_pca = RS().fit_transform(pca_df.fillna(0).values)
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_pca)

            scatter_df = pd.DataFrame({
                "PC1": coords[:, 0],
                "PC2": coords[:, 1],
                "Cluster": clustered_df["kmeans_cluster"].map(CLUSTER_NAMES),
                "Address": clustered_df["address"],
                "ETH Out": clustered_df.get("total_eth_out", 0),
            })

            # Mark anomalies
            if not anomaly_df.empty:
                anom_addrs = set(anomaly_df[anomaly_df["anomaly_label"] == -1]["address"])
                scatter_df["Anomaly"] = scatter_df["Address"].isin(anom_addrs).map(
                    {True: "Anomalous", False: "Normal"}
                )
            else:
                scatter_df["Anomaly"] = "Normal"

            fig_sc = px.scatter(
                scatter_df, x="PC1", y="PC2",
                color="Cluster", symbol="Anomaly",
                hover_data=["Address", "ETH Out"],
                template="plotly_dark",
                title=f"PCA Projection (var explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)",
                height=450,
                symbol_map={"Normal": "circle", "Anomalous": "diamond-open"},
            )
            fig_sc.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0.3, color="white")))
            st.plotly_chart(fig_sc, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "DeepWhale — Behavioral Analysis of Crypto Whales | "
    "Data: Ethereum blockchain (Alchemy) + Binance API | "
    "ML: KMeans clustering → XGBoost classification → SHAP explanations → LOF anomaly detection | "
    "For educational purposes only."
)
