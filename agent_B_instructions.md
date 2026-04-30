# Agent B — Instructions (ML & Dashboard Domain)

> Always read `project_context.md` first for full project architecture before making any changes.

---

## Your Ownership

You are responsible for the **ML modeling and presentation layer** of the DeepWhale pipeline:

| File | Role |
|------|------|
| `clustering.py` | KMeans + DBSCAN unsupervised profiling |
| `classification.py` | XGBoost + Random Forest supervised classification |
| `anomaly_detection.py` | Isolation Forest anomaly detection |
| `dashboard.py` | Streamlit interactive dashboard |
| `pipeline_runner.ipynb` | End-to-end orchestration notebook |
| `models/whale_classifier.pkl` | Trained model bundle (you own this file) |

You do **NOT** touch:
- `data_collection.py`, `feature_engineering.py`, `labeling.py`
- `data/known_addresses.csv`
- `data/raw_whale_transactions.csv`, `data/address_features.csv`, `data/labeled_addresses.csv`
  (these are Agent A's outputs — treat them as read-only inputs)

---

## Your Input Contracts (what Agent A gives you)

Your scripts consume these CSVs. Never hardcode column names that aren't in this contract.

### `data/labeled_addresses.csv` (primary input for classification + anomaly)
Key columns your code uses:
- `address` — unique identifier
- `label` — one of: `exchange_depositor`, `accumulator`, `active_trader`, `unknown_whale`
- `label_confidence` — float 0–1 (drop rows < 0.25 before training)
- `first_seen` — ISO datetime string (used for temporal train/test split)
- Feature columns (all float, may have NaN → impute with median before use):
  `tx_count_out`, `total_eth_out`, `avg_tx_eth`, `median_tx_eth`, `max_tx_eth`,
  `std_tx_eth`, `unique_receivers`, `exchange_ratio`, `top1_receiver_ratio`,
  `round_number_ratio`, `avg_gas_gwei`, `gas_variability`, `hour_entropy`,
  `active_days`, `net_flow_eth`

### `data/address_features.csv` (input for clustering + anomaly)
Same feature columns as above, without `label` / `label_confidence`.

### `data/raw_whale_transactions.csv` (input for dashboard + anomaly timeline)
Columns: `block_number`, `timestamp`, `unix_timestamp`, `tx_hash`, `from_address`,
`to_address`, `value_eth`, `gas_price_gwei`, `eth_price_usd`, `value_usd`

---

## Your Output Files

| File | Written by | Used by |
|------|-----------|---------|
| `data/clustered_addresses.csv` | `clustering.py` | `dashboard.py` |
| `data/anomaly_scores.csv` | `anomaly_detection.py` | `dashboard.py` |
| `models/whale_classifier.pkl` | `classification.py` | `dashboard.py` |
| `data/cluster_*.png` | `clustering.py` | EDA / reporting |
| `data/cluster_kdistance.png` | `clustering.py` | DBSCAN eps diagnostic |
| `data/cm_*.png`, `data/feature_importance.png`, `data/shap_summary.png` | `classification.py` | EDA / reporting |
| `data/anomaly_*.png` | `anomaly_detection.py` | EDA / reporting |

### `models/whale_classifier.pkl` — bundle schema (strict contract for dashboard)
```python
{
    "xgb": XGBClassifier,           # primary model
    "rf": RandomForestClassifier,   # comparison model
    "scaler": RobustScaler,         # fitted on train set
    "label_encoder": LabelEncoder,  # fitted on all labels
    "feature_cols": list[str],      # ordered list of feature column names
}
```
**Never change this bundle schema** — `dashboard.py`'s `predict_address()` function loads these exact keys.

### `data/clustered_addresses.csv` — added columns
- `kmeans_cluster` — int (0-based cluster ID)
- `dbscan_cluster` — int (-1 = noise/outlier)

### `data/anomaly_scores.csv` — added columns
- `anomaly_label` — int: `-1` (anomaly) or `1` (normal)
- `anomaly_score` — float (lower = more anomalous)

---

## Key Parameters (tunable, document any change in `progress.txt`)

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `CONTAMINATION` | `anomaly_detection.py:59` | `0.10` | Fraction of anomalies expected |
| `CONFIDENCE_THRESHOLD` | `classification.py:72` | `0.25` | Min label confidence to train on |
| `FEATURE_COLS` | all 3 ML scripts | 15 features | Must stay in sync across scripts |
| KMeans `k_range` | `clustering.py:245` | `range(2, 7)` | Silhouette-selected best k |
| DBSCAN `eps` | auto (k-distance 95th pct) | dynamic | Set by `auto_eps()` in `clustering.py`; override via `run_dbscan(X, min_samples=N)` |
| XGBoost `n_estimators` | `classification.py:253` | `200` | Tree count |
| XGBoost `max_depth` | `classification.py:254` | `4` | Tree depth |

---

## Rules

1. **`FEATURE_COLS`** must be identical across `clustering.py`, `classification.py`, and `anomaly_detection.py`. If you add/remove a feature, update all three simultaneously.
2. The model bundle keys (`xgb`, `rf`, `scaler`, `label_encoder`, `feature_cols`) must never be renamed — the dashboard loads them by exact key name.
3. Always use a **temporal train/test split** (sort by `first_seen`, 80/20) — never random split, it leaks future data.
4. Always impute NaN feature values with **column median** before fitting or predicting.
5. Use `RobustScaler` (not StandardScaler) — ETH values are heavily outlier-prone.
6. If Agent A changes a label class name, you must refit the `LabelEncoder` and retrain all models, then regenerate `whale_classifier.pkl`.
7. Do not add imports inside functions in `dashboard.py` except where already done (the `clustering` import in tab 2 is an existing exception).
8. Log every model retrain to `progress.txt` with F1 scores.

---

## How to Run Your Steps

```bash
# Run clustering (requires address_features.csv or labeled_addresses.csv)
python clustering.py

# Train classifier (requires labeled_addresses.csv)
python classification.py

# Run anomaly detection (requires address_features.csv)
python anomaly_detection.py

# Launch dashboard
streamlit run dashboard.py

# Verify model bundle integrity
python -c "
import pickle
with open('models/whale_classifier.pkl', 'rb') as f:
    b = pickle.load(f)
print('Keys:', list(b.keys()))
print('Classes:', b['label_encoder'].classes_)
print('Features:', b['feature_cols'])
"
```

---

## Dashboard Architecture (`dashboard.py`)

```
Header KPIs (5 metrics, always visible)
├── load_raw_transactions()    → raw_df
├── load_labeled()             → labeled_df
├── load_anomaly()             → anomaly_df
└── fetch_eth_price_binance()  → price_df (live, ttl=30s)

Tab 1 — Live Whales
├── Filterable tx table (min ETH, N rows, anomalies-only checkbox)
└── Hourly whale activity bar+line chart (tx count + ETH volume)

Tab 2 — Address Profiler
├── Text input or dropdown from top-30 addresses
├── predict_address() → XGBoost classification + probability bar chart
├── Anomaly status (from anomaly_scores.csv)
├── Cluster assignment (from clustered_addresses.csv, CLUSTER_NAMES dict)
├── Tx history table (last 20 txs)
└── Cumulative ETH sent area chart

Tab 3 — Market Context
├── Live ETH price (Binance API, selectable window 24h–336h)
├── Price + whale volume overlay (dual-axis)
├── Anomalous whale ETH overlay (red bars)
├── Whale type pie chart (label distribution)
├── ETH volume by whale type (horizontal bar)
└── Interactive PCA cluster scatter (Plotly, recomputed on the fly)
```

**`CLUSTER_NAMES` dict** (update if you change k or cluster semantics):
```python
{0: "Exchange Movers", 1: "Silent Accumulators", 2: "Active Traders", 3: "Irregular Whales"}
```

---

## Current Improvement Areas (Agent B scope)

- No cross-validation in `classification.py` — consider adding `StratifiedKFold` for more robust F1 estimate
- SHAP plot currently uses mean absolute SHAP across all classes — per-class SHAP breakdown would be more informative
- Dashboard tab 1 has no pagination — large datasets may be slow; consider `st.dataframe` with server-side filtering
- `pipeline_runner.ipynb` runs scripts via `subprocess` — could be refactored to import functions directly for better error propagation
- DBSCAN `eps=1.5` is hardcoded — could be tuned automatically using k-distance elbow plot
