# Agent A — Instructions (Data & Features Domain)

> Always read `project_context.md` first for full project architecture before making any changes.

---

## Your Ownership

You are responsible for the **data ingestion and feature layer** of the DeepWhale pipeline:

| File | Role |
|------|------|
| `data_collection.py` | Ethereum block scraping, whale tx filtering, price enrichment |
| `feature_engineering.py` | Per-address behavioral feature computation |
| `labeling.py` | Rule-based weak label assignment |
| `data_analysis.ipynb` | EDA, data quality checks, visualizations |
| `data/known_addresses.csv` | Exchange address registry (editable) |

You do **NOT** touch:
- `classification.py`, `clustering.py`, `anomaly_detection.py`, `dashboard.py`
- `models/` directory
- Any CSV that is output of Agent B's scripts

---

## Your Output Contracts (DO NOT break these)

Agent B depends on your outputs. Column names and dtypes are a strict contract.

### `data/raw_whale_transactions.csv`
| Column | Type | Notes |
|--------|------|-------|
| `block_number` | int | Ethereum block |
| `timestamp` | str | `YYYY-MM-DD HH:MM:SS` UTC |
| `unix_timestamp` | int | Unix epoch seconds |
| `tx_hash` | str | Hex, unique key |
| `from_address` | str | Checksummed ETH address |
| `to_address` | str | ETH address or `"Contract_Creation"` |
| `value_eth` | float | ETH transferred (> 10.0) |
| `gas_price_gwei` | float | Gas price |
| `eth_price_usd` | float \| NaN | May be null if Binance API failed |
| `value_usd` | float \| NaN | `value_eth * eth_price_usd` |

### `data/address_features.csv`
| Column | Type | Notes |
|--------|------|-------|
| `address` | str | Unique per row |
| `tx_count_out` | int | |
| `total_eth_out` | float | |
| `total_usd_out` | float \| NaN | |
| `avg_tx_eth` | float | |
| `median_tx_eth` | float | |
| `max_tx_eth` | float | |
| `std_tx_eth` | float | |
| `unique_receivers` | int | |
| `exchange_ratio` | float | 0–1 |
| `top1_receiver_ratio` | float | 0–1 |
| `round_number_ratio` | float | 0–1 |
| `avg_gas_gwei` | float | |
| `gas_variability` | float | |
| `hour_entropy` | float | Shannon entropy (base 2) |
| `active_days` | int | |
| `avg_interval_hours` | float \| NaN | |
| `net_flow_eth` | float | Can be negative |
| `block_span` | int | |
| `is_known_exchange` | int | 0 or 1 |
| `first_seen` | str | ISO datetime |
| `last_seen` | str | ISO datetime |

### `data/labeled_addresses.csv`
All columns from `address_features.csv`, plus:
| Column | Type | Notes |
|--------|------|-------|
| `label` | str | One of: `exchange_depositor`, `accumulator`, `active_trader`, `unknown_whale` |
| `label_confidence` | float | 0–1 |

---

## Key Parameters (tunable, document any change in `progress.txt`)

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `WHALE_THRESHOLD_ETH` | `data_collection.py:24` | `5` | Min ETH for tx inclusion |
| `NUM_BLOCKS_DEFAULT` | `data_collection.py:21` | `216000` | Block range spanned (~30 days) |
| `BLOCK_STEP` | `data_collection.py:22` | `50` | Sample every N-th block (~4320 RPC calls) |
| `MAX_WORKERS` | `data_collection.py:26` | `10` | Parallel RPC threads |
| `WINDOW_DAYS` | `feature_engineering.py:21` | `30` | Rolling feature window |
| `ROUND_ETH_VALUES` | `feature_engineering.py:22` | set of round amounts | Round-number detection |
| `EXCHANGE_RATIO_THRESHOLD` | `labeling.py:30` | `0.50` | Exchange depositor threshold |
| `ACCUMULATOR_MAX_ACTIVE_DAYS` | `labeling.py:32` | `7` | Accumulator max active days |
| `TRADER_MIN_TX` | `labeling.py:34` | `8` | Active trader min tx count |

---

## Rules

1. **Never rename or remove columns** from output CSVs — Agent B's code will break silently.
2. If you add a new feature column, add it **after** existing columns and document it in `project_context.md` under "Key Data Files".
3. If you change any label class name, tell Agent B immediately — it breaks the LabelEncoder in `models/whale_classifier.pkl`.
4. Run `feature_engineering.py` → `labeling.py` in sequence after any change to `data_collection.py`.
5. The `known_addresses.csv` columns are: `address`, `label`, `exchange`, `type` — keep this schema.
6. Log every significant change to `progress.txt` with date and your name.

---

## How to Run Your Steps

```bash
# Collect new data (adjust --blocks as needed)
python data_collection.py --blocks 5000

# Rebuild features
python feature_engineering.py

# Rebuild labels
python labeling.py

# Verify outputs exist and have expected shape
python -c "
import pandas as pd
print(pd.read_csv('data/address_features.csv').shape)
print(pd.read_csv('data/labeled_addresses.csv')['label'].value_counts())
"
```

---

## Current Improvement Areas (Agent A scope)

- `networkx` is in `requirements.txt` but not yet used — graph features (degree centrality, clustering coefficient per address) could be added to `feature_engineering.py`
- `avg_interval_hours` has NaN for single-tx addresses — consider a sentinel value or separate flag
- Price data (`eth_price_usd`) is sometimes null — consider a fallback or interpolation in `feature_engineering.py`
- EDA notebook (`data_analysis.ipynb`) may need updating after new data collection runs
