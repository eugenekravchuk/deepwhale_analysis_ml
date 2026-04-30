# DeepWhale — Project Context

## Project Goal
Behavioral analysis and classification of Ethereum crypto whales using on-chain transaction data.
The system identifies whale archetypes, detects anomalous behavior, and correlates whale activity with ETH market dynamics.

---

## Tech Stack
- **Runtime**: Python 3.11+, `.venv`
- **Blockchain**: `web3.py` (Alchemy Ethereum RPC), `requests` (Binance public API)
- **Data**: `pandas`, `numpy`, `scipy`
- **ML**: `scikit-learn`, `xgboost`, `shap`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Dashboard**: `streamlit`
- **Graph**: `networkx`
- **Orchestration**: Jupyter (`pipeline_runner.ipynb`)

---

## Pipeline — Step-by-Step

### 1. `data_collection.py`
- Connects to Ethereum via `ALCHEMY_URL` (`.env`)
- Scans the latest N blocks (default `20,000`) backwards
- Filters transactions with value > **10 ETH** (whale threshold)
- Enriches each tx with ETH/USD price from Binance (1-min candles, cached per minute bucket)
- Deduplicates by `tx_hash` and appends to existing data
- **Output**: `data/raw_whale_transactions.csv`
- **Columns**: `block_number`, `timestamp`, `unix_timestamp`, `tx_hash`, `from_address`, `to_address`, `value_eth`, `gas_price_gwei`, `eth_price_usd`, `value_usd`

### 2. `feature_engineering.py`
- Input: `data/raw_whale_transactions.csv` + `data/known_addresses.csv`
- Rolling 30-day window; aggregates per `from_address`
- **22 features engineered** (15 used in models):
  - `tx_count_out`, `total_eth_out`, `total_usd_out`
  - `avg_tx_eth`, `median_tx_eth`, `max_tx_eth`, `std_tx_eth`
  - `unique_receivers`, `exchange_ratio`, `top1_receiver_ratio`
  - `round_number_ratio` (round ETH amounts: 10, 20, 25, 50, 100, 250, 500, 1000)
  - `avg_gas_gwei`, `gas_variability`
  - `hour_entropy` (Shannon entropy of tx hour distribution — high = bot/exchange, low = human)
  - `active_days`, `avg_interval_hours`
  - `net_flow_eth` (incoming − outgoing)
  - `block_span`, `is_known_exchange`, `first_seen`, `last_seen`
- **Output**: `data/address_features.csv`

### 3. `labeling.py`
- Input: `data/address_features.csv`
- Rule-based **weak labeling** into 4 classes (used as supervised training targets):
  - `exchange_depositor` — exchange_ratio ≥ 0.50
  - `accumulator` — net_flow_eth ≥ 0, active_days ≤ 7, tx_count_out ≤ 5
  - `active_trader` — tx_count_out ≥ 8, unique_receivers ≥ 5, top1_receiver_ratio ≤ 0.70
  - `unknown_whale` — does not fit above
- Also computes `label_confidence` score [0, 1] per address
- **Output**: `data/labeled_addresses.csv`

### 4. `clustering.py`
- Input: `data/address_features.csv` (or `labeled_addresses.csv` if available)
- Preprocessing: `RobustScaler`, median imputation for NaNs
- **KMeans**: best k selected by silhouette score (range 2–6), 15 features
- **DBSCAN**: noise/outlier whale detection (eps=1.5, min_samples=3)
- Visualizations saved to `data/`:
  - `cluster_pca_kmeans.png` — PCA 2D scatter (KMeans)
  - `cluster_tsne_kmeans.png` — t-SNE 2D scatter (KMeans)
  - `cluster_pca_dbscan.png` — PCA 2D scatter (DBSCAN, noise = "Outlier Whale")
  - `cluster_radar.png` — radar/spider chart of cluster behavioral profiles
- Named clusters: `Exchange Movers`, `Silent Accumulators`, `Active Traders`, `Irregular Whales`
- **Output**: `data/clustered_addresses.csv` (adds columns `kmeans_cluster`, `dbscan_cluster`)

### 5. `classification.py`
- Input: `data/labeled_addresses.csv`
- Drops samples with `label_confidence` < 0.25
- **Time-based split**: 80% earliest (by `first_seen`) → train, 20% latest → test (no data leakage)
- Models:
  - **XGBoost** (primary): 200 trees, max_depth=4, lr=0.1, multi:softprob
  - **Random Forest** (comparison): 200 trees, max_depth=6
- Evaluation: F1 macro, confusion matrix, classification report
- Explainability: `feature_importance.png` (XGBoost gain), `shap_summary.png` (mean |SHAP|)
- Saves `data/cm_xgboost.png`, `data/cm_randomforest.png`
- **Output**: `models/whale_classifier.pkl` — bundle with keys: `xgb`, `rf`, `scaler`, `label_encoder`, `feature_cols`

### 6. `anomaly_detection.py`
- Input: `data/address_features.csv` (or `labeled_addresses.csv`)
- **Isolation Forest**: 200 estimators, contamination=0.10, `RobustScaler`
- Adds `anomaly_label` (−1 = anomaly, 1 = normal) and `anomaly_score` (lower = more anomalous)
- Timeline analysis: joins raw txs with anomaly labels → hourly `anomalous_tx` count vs ETH price
- Pearson correlation: anomalous whale tx count vs ETH price change %
- Outputs: `data/anomaly_scores.csv`, `data/anomaly_scores_plot.png`, `data/anomaly_timeline.png`

---

## Dashboard — `dashboard.py`
Run: `streamlit run dashboard.py`

**Three tabs:**

| Tab | Content |
|-----|---------|
| 🔴 Live Whales | Filterable tx feed (min ETH slider, anomaly/exchange flags 🚨🏦), hourly activity bar+line chart |
| 🔍 Address Profiler | Enter any ETH address → XGBoost classification, confidence bar chart, anomaly status, cluster assignment, tx history, cumulative ETH area chart |
| 📈 Market Context | Live ETH price (Binance API, 24h–336h window), whale ETH volume overlay, anomaly markers, whale type pie + volume bar, interactive PCA cluster scatter |

**Top KPIs (always visible):** Total Whale Txs, Unique Whale Addresses, Anomalous Addresses, Total ETH Moved, ETH Price (live)

---

## Key Data Files

| File | Description |
|------|-------------|
| `data/raw_whale_transactions.csv` | Raw on-chain whale txs (>10 ETH) |
| `data/known_addresses.csv` | 31 known exchange wallets (Binance, Coinbase, Kraken, OKX, Bitfinex, Huobi, Gate.io) |
| `data/address_features.csv` | Per-address behavioral features (22 cols) |
| `data/labeled_addresses.csv` | Features + weak labels + confidence |
| `data/clustered_addresses.csv` | Features + KMeans/DBSCAN cluster assignments |
| `data/anomaly_scores.csv` | Features + anomaly label/score |
| `models/whale_classifier.pkl` | Trained model bundle (XGBoost + RF + scaler + LabelEncoder) |

---

## Orchestration

- **`pipeline_runner.ipynb`** — runs all 6 scripts in sequence end-to-end via `subprocess`
- **`data_analysis.ipynb`** — EDA notebook with visualizations

---

## Environment Setup

```bash
# 1. Copy and fill in .env
cp .env.example .env
# Set: ALCHEMY_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python data_collection.py --blocks 20000
python feature_engineering.py
python labeling.py
python clustering.py
python classification.py
python anomaly_detection.py

# 4. Launch dashboard
streamlit run dashboard.py
```

---

## Important Design Decisions

- **Whale threshold**: 10 ETH (configurable via `WHALE_THRESHOLD_ETH` in `data_collection.py`)
- **Feature window**: 30 days rolling (configurable via `WINDOW_DAYS` in `feature_engineering.py`)
- **Labeling is weak/heuristic** — not ground truth; thresholds are tunable in `labeling.py`
- **Scaler**: `RobustScaler` throughout (handles ETH value outliers well)
- **Train/test split is temporal** — prevents leakage from future data
- **Model bundle** (`whale_classifier.pkl`) includes scaler and LabelEncoder — always use together
- **Binance API prices** cached per 1-minute bucket during collection to reduce API calls
- `hour_entropy` is a key bot-detection feature: high entropy → 24/7 trading pattern (exchange/bot)
