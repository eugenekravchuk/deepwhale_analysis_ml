"""
dashboard.py  —  DeepWhale Interactive Dashboard
=================================================
Run:  streamlit run dashboard.py

Three tabs:
  1. Live Whales   — latest on-chain whale transactions (auto-refresh)
  2. Address Profiler — classify and profile any ETH address
  3. Market Context  — ETH price vs whale activity overlay
"""

import pickle
import time
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
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

RAW_CSV = DATA_DIR / "raw_whale_transactions.csv"
LABELED_CSV = DATA_DIR / "labeled_addresses.csv"
ANOMALY_CSV = DATA_DIR / "anomaly_scores.csv"
CLUSTERED_CSV = DATA_DIR / "clustered_addresses.csv"
KNOWN_CSV = DATA_DIR / "known_addresses.csv"
MODEL_PKL = MODELS_DIR / "whale_classifier.pkl"

FEATURE_COLS = [
    "tx_count_out", "total_eth_out", "avg_tx_eth", "median_tx_eth",
    "max_tx_eth", "std_tx_eth", "unique_receivers", "exchange_ratio",
    "top1_receiver_ratio", "round_number_ratio", "avg_gas_gwei",
    "gas_variability", "hour_entropy", "active_days", "net_flow_eth",
]

LABEL_COLORS = {
    "exchange_depositor": "#3498db",
    "accumulator": "#2ecc71",
    "active_trader": "#e74c3c",
    "unknown_whale": "#95a5a6",
}

LABEL_ICONS = {
    "exchange_depositor": "🏦",
    "accumulator": "🐋",
    "active_trader": "⚡",
    "unknown_whale": "❓",
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepWhale Dashboard",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data loaders (cached) ─────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_raw_transactions() -> pd.DataFrame:
    if not RAW_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
    return df.sort_values("timestamp", ascending=False)


@st.cache_data(ttl=300)
def load_labeled() -> pd.DataFrame:
    src = LABELED_CSV if LABELED_CSV.exists() else (DATA_DIR / "address_features.csv")
    if not src.exists():
        return pd.DataFrame()
    return pd.read_csv(src)


@st.cache_data(ttl=300)
def load_anomaly() -> pd.DataFrame:
    if not ANOMALY_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(ANOMALY_CSV)


@st.cache_data(ttl=300)
def load_clustered() -> pd.DataFrame:
    src = CLUSTERED_CSV if CLUSTERED_CSV.exists() else LABELED_CSV
    if not src.exists():
        return pd.DataFrame()
    return pd.read_csv(src)


@st.cache_resource
def load_model():
    if not MODEL_PKL.exists():
        return None
    with open(MODEL_PKL, "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=30)
def fetch_eth_price_binance(n_candles: int = 168) -> pd.DataFrame:
    """Fetch last n_candles hourly ETH/USDT candles from Binance public API."""
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def tag_address(addr: str, known: set) -> str:
    return "Known Exchange" if addr.lower() in known else "Unknown"


def label_badge(label: str) -> str:
    icon = LABEL_ICONS.get(label, "❓")
    color = LABEL_COLORS.get(label, "#95a5a6")
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.85em">{icon} {label.replace("_", " ").title()}</span>'
    )


def predict_address(address: str, model_bundle: dict) -> dict | None:
    """
    Classify a single address using the trained model.
    Returns dict with label, probability, and top features or None if not enough data.
    """
    df = load_raw_transactions()
    if df.empty:
        return None

    # Build minimal feature vector from raw transactions for this address
    sub = df[df["from_address"].str.lower() == address.lower()]
    if sub.empty:
        return None

    known = load_known_addresses()

    tx_count_out = len(sub)
    total_eth_out = sub["value_eth"].sum()
    avg_tx_eth = sub["value_eth"].mean()
    median_tx_eth = sub["value_eth"].median()
    max_tx_eth = sub["value_eth"].max()
    std_tx_eth = sub["value_eth"].std(ddof=0) if tx_count_out > 1 else 0.0
    unique_receivers = sub["to_address"].nunique()
    exchange_dest = sub["to_address"].str.lower().isin(known)
    exchange_ratio = sub.loc[exchange_dest, "value_eth"].sum() / total_eth_out if total_eth_out > 0 else 0.0
    top1 = sub.groupby("to_address")["value_eth"].sum().max()
    top1_receiver_ratio = top1 / total_eth_out if total_eth_out > 0 else 0.0

    ROUND_SET = {10, 20, 25, 30, 50, 75, 100, 150, 200, 250, 500, 1000}
    def is_round(v):
        return any(abs(v - r) / r < 0.01 for r in ROUND_SET)
    round_number_ratio = sub["value_eth"].apply(is_round).mean()

    avg_gas_gwei = sub["gas_price_gwei"].mean()
    gas_variability = sub["gas_price_gwei"].std(ddof=0) if tx_count_out > 1 else 0.0

    sub_ts = pd.to_datetime(sub["timestamp"])
    hours = sub_ts.dt.hour
    counts = hours.value_counts().reindex(range(24), fill_value=0).values.astype(float)
    probs = counts / (counts.sum() + 1e-9)
    from scipy.stats import entropy as scipy_entropy
    h_entropy = float(scipy_entropy(probs + 1e-9, base=2))

    active_days = sub_ts.dt.date.nunique()

    recv_col = df[df["to_address"].str.lower() == address.lower()]["value_eth"].sum()
    net_flow_eth = recv_col - total_eth_out

    feat_values = [
        tx_count_out, total_eth_out, avg_tx_eth, median_tx_eth, max_tx_eth,
        std_tx_eth, unique_receivers, exchange_ratio, top1_receiver_ratio,
        round_number_ratio, avg_gas_gwei, gas_variability, h_entropy,
        active_days, net_flow_eth,
    ]

    scaler = model_bundle["scaler"]
    le = model_bundle["label_encoder"]
    xgb = model_bundle["xgb"]
    feat_cols = model_bundle.get("feature_cols", FEATURE_COLS)

    X = np.array(feat_values[: len(feat_cols)]).reshape(1, -1)
    X_scaled = scaler.transform(X)
    proba = xgb.predict_proba(X_scaled)[0]
    pred_idx = int(np.argmax(proba))
    label = le.inverse_transform([pred_idx])[0]

    return {
        "label": label,
        "confidence": float(proba[pred_idx]),
        "all_probs": dict(zip(le.classes_, proba.tolist())),
        "tx_count": tx_count_out,
        "total_eth": total_eth_out,
        "exchange_ratio": exchange_ratio,
        "active_days": active_days,
        "net_flow_eth": net_flow_eth,
    }


# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 🐋 DeepWhale — Crypto Whale Behaviour Dashboard")
st.caption("Behavioral analysis and classification of ETH whales using on-chain data")

# Top KPIs
raw_df = load_raw_transactions()
labeled_df = load_labeled()
anomaly_df = load_anomaly()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Whale Txs", f"{len(raw_df):,}" if not raw_df.empty else "—")
with c2:
    n_addr = raw_df["from_address"].nunique() if not raw_df.empty else 0
    st.metric("Unique Whale Addresses", f"{n_addr:,}")
with c3:
    n_anomaly = int((anomaly_df["anomaly_label"] == -1).sum()) if not anomaly_df.empty else 0
    st.metric("Anomalous Addresses", f"{n_anomaly}")
with c4:
    total_eth = raw_df["value_eth"].sum() if not raw_df.empty else 0
    st.metric("Total ETH Moved", f"{total_eth:,.0f} ETH")
with c5:
    price_df = fetch_eth_price_binance(1)
    if not price_df.empty:
        st.metric("ETH Price (live)", f"${price_df['close'].iloc[-1]:,.2f}")
    else:
        st.metric("ETH Price", "—")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔴 Live Whales", "🔍 Address Profiler", "📈 Market Context"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE WHALES
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Recent Whale Transactions (> 10 ETH)")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    with col_ctrl1:
        min_eth = st.slider("Min ETH value", 10, 500, 10, step=10)
    with col_ctrl2:
        page_size = st.selectbox("Rows per page", [25, 50, 100, 200], index=1)
    with col_ctrl3:
        show_anomalies_only = st.checkbox("Show anomalous addresses only", value=False)

    if raw_df.empty:
        st.warning("No transaction data found. Run `python data_collection.py` first.")
    else:
        filtered = raw_df[raw_df["value_eth"] >= min_eth].copy()

        known_set = load_known_addresses()
        if not anomaly_df.empty:
            anomaly_set = set(anomaly_df[anomaly_df["anomaly_label"] == -1]["address"])
        else:
            anomaly_set = set()

        filtered["exchange_flag"] = filtered["to_address"].apply(
            lambda a: "🏦" if str(a).lower() in known_set else ""
        )
        filtered["anomaly_flag"] = filtered["from_address"].apply(
            lambda a: "🚨" if a in anomaly_set else ""
        )

        if show_anomalies_only:
            filtered = filtered[filtered["anomaly_flag"] != ""]

        total_rows = len(filtered)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)

        if "whale_page" not in st.session_state:
            st.session_state.whale_page = 1
        if st.session_state.whale_page > total_pages:
            st.session_state.whale_page = total_pages

        page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
        with page_col1:
            if st.button("← Previous", disabled=st.session_state.whale_page <= 1):
                st.session_state.whale_page -= 1
                st.rerun()
        with page_col2:
            st.markdown(
                f"<div style='text-align:center'>Page {st.session_state.whale_page} / {total_pages}"
                f" &nbsp;({total_rows:,} transactions)</div>",
                unsafe_allow_html=True,
            )
        with page_col3:
            if st.button("Next →", disabled=st.session_state.whale_page >= total_pages):
                st.session_state.whale_page += 1
                st.rerun()

        start_idx = (st.session_state.whale_page - 1) * page_size
        display = filtered.iloc[start_idx : start_idx + page_size]

        show_cols = ["timestamp", "from_address", "to_address", "value_eth",
                     "value_usd", "gas_price_gwei", "exchange_flag", "anomaly_flag"]
        avail = [c for c in show_cols if c in display.columns]

        st.dataframe(
            display[avail].rename(columns={
                "exchange_flag": "Exchange Dest",
                "anomaly_flag": "Anomaly",
                "value_eth": "ETH",
                "value_usd": "USD",
                "gas_price_gwei": "Gas (Gwei)",
            }),
            use_container_width=True,
            height=420,
        )

        # Recent activity bar chart
        if "timestamp" in raw_df.columns:
            hourly = raw_df.copy()
            hourly["hour"] = pd.to_datetime(hourly["timestamp"]).dt.floor("h")
            hourly_agg = hourly.groupby("hour").agg(
                tx_count=("tx_hash", "count"),
                total_eth=("value_eth", "sum"),
            ).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(x=hourly_agg["hour"], y=hourly_agg["tx_count"],
                                 name="# Txs", marker_color="#3498db", opacity=0.8))
            fig.add_trace(go.Scatter(x=hourly_agg["hour"], y=hourly_agg["total_eth"],
                                     name="Total ETH", mode="lines+markers",
                                     line=dict(color="#f39c12", width=2),
                                     yaxis="y2"))
            fig.update_layout(
                title="Hourly Whale Activity",
                template="plotly_dark",
                yaxis=dict(title="# Transactions"),
                yaxis2=dict(title="ETH Volume", overlaying="y", side="right"),
                legend=dict(orientation="h", y=1.1),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ADDRESS PROFILER
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Whale Address Profiler")
    st.caption("Enter an ETH address to classify its behaviour and see its on-chain profile.")

    addr_input = st.text_input(
        "ETH Address (0x...)",
        placeholder="0x28c6c06298d514db089934071355e5743bf21d60",
    )

    # Also let user pick from known whales in the dataset
    if not raw_df.empty and not addr_input:
        all_addresses = raw_df["from_address"].value_counts().head(30).index.tolist()
        selected = st.selectbox("...or choose a known whale from collected data",
                                [""] + all_addresses)
        if selected:
            addr_input = selected

    if addr_input:
        model_bundle = load_model()
        known_set = load_known_addresses()
        addr_lower = addr_input.strip().lower()

        # Check known exchange
        is_exchange = addr_lower in known_set
        if is_exchange:
            st.info("This address is a **known exchange hot/cold wallet**.")

        # Try to get pre-computed data from labeled/anomaly CSVs
        precomp_label = None
        precomp_anomaly = None
        precomp_cluster = None

        if not labeled_df.empty and "address" in labeled_df.columns:
            row = labeled_df[labeled_df["address"].str.lower() == addr_lower]
            if not row.empty:
                precomp_label = row.iloc[0].get("label", None)

        if not anomaly_df.empty and "address" in anomaly_df.columns:
            row = anomaly_df[anomaly_df["address"].str.lower() == addr_lower]
            if not row.empty:
                precomp_anomaly = int(row.iloc[0].get("anomaly_label", 1))

        clustered_df = load_clustered()
        if not clustered_df.empty and "address" in clustered_df.columns:
            row = clustered_df[clustered_df["address"].str.lower() == addr_lower]
            if not row.empty:
                precomp_cluster = int(row.iloc[0].get("kmeans_cluster", -1))

        # --- Classification ---
        col_a, col_b = st.columns([1, 1])

        with col_a:
            if model_bundle:
                with st.spinner("Classifying..."):
                    result = predict_address(addr_input.strip(), model_bundle)

                if result:
                    label = result["label"]
                    conf = result["confidence"]
                    st.markdown(f"**Predicted Whale Type:** {label_badge(label)}",
                                unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** {conf*100:.1f}%")

                    # Probability breakdown
                    prob_df = pd.DataFrame(
                        list(result["all_probs"].items()), columns=["Type", "Probability"]
                    ).sort_values("Probability", ascending=True)
                    fig = px.bar(prob_df, x="Probability", y="Type", orientation="h",
                                 template="plotly_dark", title="Classification Probabilities",
                                 color="Probability",
                                 color_continuous_scale="Blues")
                    fig.update_layout(height=220, showlegend=False,
                                      coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Address not found in collected data. Run data_collection.py first.")
            elif precomp_label:
                st.markdown(f"**Whale Type (pre-computed):** {label_badge(precomp_label)}",
                            unsafe_allow_html=True)
            else:
                st.info("No trained model found. Run `python classification.py` first.")

        with col_b:
            st.markdown("**Anomaly Status**")
            if precomp_anomaly is not None:
                if precomp_anomaly == -1:
                    st.error("🚨 ANOMALOUS — behaviour is atypical vs whale population")
                else:
                    st.success("✅ NORMAL — behaviour matches typical whale pattern")
            else:
                st.info("Run anomaly_detection.py to compute anomaly score.")

            st.markdown("**Cluster Assignment**")
            if precomp_cluster is not None and precomp_cluster >= 0:
                from clustering import CLUSTER_NAMES
                cluster_name = CLUSTER_NAMES.get(precomp_cluster, f"Cluster {precomp_cluster}")
                st.info(f"Cluster {precomp_cluster}: **{cluster_name}**")
            else:
                st.info("Run clustering.py to compute cluster assignment.")

        # Transaction history for this address
        if not raw_df.empty:
            addr_txs = raw_df[raw_df["from_address"].str.lower() == addr_lower]
            if not addr_txs.empty:
                st.markdown(f"**Transaction history** — {len(addr_txs)} whale txs in dataset")
                st.dataframe(
                    addr_txs[["timestamp", "to_address", "value_eth", "value_usd", "gas_price_gwei"]]
                    .head(20),
                    use_container_width=True,
                    height=250,
                )

                # Cumulative ETH moved over time
                ts_sorted = addr_txs.sort_values("timestamp").copy()
                ts_sorted["cumulative_eth"] = ts_sorted["value_eth"].cumsum()
                fig = px.area(ts_sorted, x="timestamp", y="cumulative_eth",
                              template="plotly_dark",
                              title=f"Cumulative ETH Sent — {addr_input[:12]}...",
                              color_discrete_sequence=["#3498db"])
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MARKET CONTEXT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Market Context — ETH Price vs Whale Activity")

    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        hours_back = st.selectbox("Price history window", [24, 48, 168, 336], index=2,
                                  format_func=lambda h: f"{h}h ({h//24}d)")

    price_df = fetch_eth_price_binance(hours_back)

    if price_df.empty:
        st.warning("Cannot reach Binance API. Check your internet connection.")
    else:
        # Compute price change stats
        current_price = price_df["close"].iloc[-1]
        open_price = price_df["close"].iloc[0]
        change_pct = (current_price - open_price) / open_price * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Current ETH", f"${current_price:,.2f}", f"{change_pct:+.2f}%")
        m2.metric("Period High", f"${price_df['close'].max():,.2f}")
        m3.metric("Period Low", f"${price_df['close'].min():,.2f}")

        fig = go.Figure()

        # ETH price candlestick-style area
        fig.add_trace(go.Scatter(
            x=price_df["timestamp"], y=price_df["close"],
            name="ETH Price (USD)", mode="lines",
            line=dict(color="#f1c40f", width=2),
            fill="tozeroy", fillcolor="rgba(241,196,15,0.1)",
        ))

        # Overlay whale transaction volume (hourly bins)
        if not raw_df.empty:
            wh = raw_df.copy()
            wh["hour"] = pd.to_datetime(wh["timestamp"]).dt.floor("h")
            wh_agg = wh.groupby("hour")["value_eth"].sum().reset_index()
            wh_agg.columns = ["timestamp", "whale_eth"]

            # Only keep hours within price window
            merged = pd.merge(
                price_df[["timestamp"]], wh_agg, on="timestamp", how="left"
            ).fillna(0)

            if merged["whale_eth"].sum() > 0:
                fig.add_trace(go.Bar(
                    x=merged["timestamp"], y=merged["whale_eth"],
                    name="Whale ETH Moved", yaxis="y2",
                    marker_color="rgba(52,152,219,0.6)",
                ))

            # Anomaly markers
            if not anomaly_df.empty:
                anomalous_set = set(anomaly_df[anomaly_df["anomaly_label"] == -1]["address"])
                wh_anom = raw_df[raw_df["from_address"].isin(anomalous_set)].copy()
                wh_anom["hour"] = pd.to_datetime(wh_anom["timestamp"]).dt.floor("h")
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
            yaxis=dict(title="ETH Price (USD)", side="left"),
            yaxis2=dict(title="ETH Volume", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.08),
            height=450,
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Whale Type Distribution ─────────────────────────────────────────
        if not labeled_df.empty and "label" in labeled_df.columns:
            st.markdown("---")
            st.subheader("Whale Population Breakdown")
            lc1, lc2 = st.columns(2)

            with lc1:
                label_counts = labeled_df["label"].value_counts().reset_index()
                label_counts.columns = ["Type", "Count"]
                label_counts["Color"] = label_counts["Type"].map(LABEL_COLORS)
                fig_pie = px.pie(
                    label_counts, names="Type", values="Count",
                    template="plotly_dark", title="Whale Types (% of population)",
                    color="Type",
                    color_discrete_map=LABEL_COLORS,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                fig_pie.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)

            with lc2:
                # ETH volume by whale type
                if not raw_df.empty:
                    merged_vol = raw_df.merge(
                        labeled_df[["address", "label"]],
                        left_on="from_address", right_on="address", how="left"
                    )
                    merged_vol["label"] = merged_vol["label"].fillna("unlabeled")
                    vol_by_type = merged_vol.groupby("label")["value_eth"].sum().reset_index()
                    vol_by_type.columns = ["Type", "ETH Volume"]
                    fig_vol = px.bar(
                        vol_by_type.sort_values("ETH Volume", ascending=True),
                        x="ETH Volume", y="Type", orientation="h",
                        template="plotly_dark",
                        title="ETH Volume Moved by Whale Type",
                        color="Type",
                        color_discrete_map=LABEL_COLORS,
                    )
                    fig_vol.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig_vol, use_container_width=True)

        # ── Cluster scatter ─────────────────────────────────────────────────
        clustered_df = load_clustered()
        if not clustered_df.empty and "kmeans_cluster" in clustered_df.columns:
            st.markdown("---")
            st.subheader("Whale Cluster Map (PCA Projection)")

            # Compute PCA on the fly for visualisation
            from sklearn.preprocessing import RobustScaler
            from sklearn.decomposition import PCA

            avail = [c for c in FEATURE_COLS if c in clustered_df.columns]
            X = clustered_df[avail].fillna(clustered_df[avail].median()).values
            if len(X) >= 3:
                sc = RobustScaler()
                X_sc = sc.fit_transform(X)
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(X_sc)
                pca_df = pd.DataFrame({
                    "PC1": coords[:, 0], "PC2": coords[:, 1],
                    "cluster": clustered_df["kmeans_cluster"].astype(str),
                    "address": clustered_df["address"],
                    "label": clustered_df.get("label", "unknown"),
                })
                fig_sc = px.scatter(
                    pca_df, x="PC1", y="PC2",
                    color="cluster", hover_data=["address", "label"],
                    template="plotly_dark",
                    title="Whale Clusters — PCA 2D",
                    height=420,
                )
                fig_sc.update_traces(marker=dict(size=9, opacity=0.8,
                                                  line=dict(width=0.5, color="white")))
                st.plotly_chart(fig_sc, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "DeepWhale — Behavioral Analysis of Crypto Whales | "
    "Data: Ethereum blockchain (Alchemy) + Binance public API | "
    "For educational purposes only. Not financial advice."
)
