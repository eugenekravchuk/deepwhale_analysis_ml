import argparse
import os
import time
import pandas as pd
import requests
from web3 import Web3
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

load_dotenv()
ALCHEMY_URL = os.getenv("ALCHEMY_URL")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = DATA_DIR / "raw_whale_transactions.csv"

# Скільки останніх блоків зчитати (≈2.8 доби при ~12 с/блок; ~20k для довшого вікна)
NUM_BLOCKS_DEFAULT = 20_000

WHALE_THRESHOLD_ETH = 10

w3 = Web3(Web3.HTTPProvider(ALCHEMY_URL))

if not w3.is_connected():
    raise ConnectionError("Cannot connect to Ethereum node. Check ALCHEMY_URL in .env")
else:
    print("Connected to Ethereum")


def get_eth_price(timestamp: int, retries: int = 3, delay: float = 1.0) -> float | None:
    """Fetch ETH/USDT close price from Binance for a given unix timestamp.
    Retries up to `retries` times on failure."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "ETHUSDT",
        "interval": "1m",
        "startTime": timestamp * 1000,
        "limit": 1,
    }
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data:
                return float(data[0][4])
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"Binance API failed after {retries} attempts: {e}")
    return None


def collect_whale_data(start_block: int, num_blocks: int = 500) -> pd.DataFrame:
    """Scan `num_blocks` blocks backwards from `start_block` and collect
    all ETH transfers larger than WHALE_THRESHOLD_ETH."""
    transactions_data = []
    price_cache: dict[int, float | None] = {}

    end_block = max(0, start_block - num_blocks)

    for block_num in range(start_block, end_block, -1):
        if (start_block - block_num) % 50 == 0:
            pct = (start_block - block_num) / num_blocks * 100
            print(f"  Block {block_num}  ({pct:.0f}% done)")

        try:
            block = w3.eth.get_block(block_num, full_transactions=True)
        except Exception as e:
            print(f"  Skipping block {block_num}: {e}")
            continue

        block_time = block.timestamp

        # Cache prices per minute bucket to reduce Binance calls
        minute_bucket = block_time // 60
        if minute_bucket not in price_cache:
            price_cache[minute_bucket] = get_eth_price(block_time)
        eth_price = price_cache[minute_bucket]

        for tx in block.transactions:
            try:
                value_eth = float(w3.from_wei(tx.value, "ether"))
                if value_eth <= WHALE_THRESHOLD_ETH:
                    continue

                gas_price_gwei = float(w3.from_wei(tx.gasPrice, "gwei"))
                value_usd = value_eth * eth_price if eth_price else None

                transactions_data.append(
                    {
                        "block_number": block.number,
                        "timestamp": datetime.utcfromtimestamp(block_time).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "unix_timestamp": block_time,
                        "tx_hash": tx.hash.hex(),
                        "from_address": tx["from"],
                        "to_address": tx.to if tx.to else "Contract_Creation",
                        "value_eth": value_eth,
                        "gas_price_gwei": gas_price_gwei,
                        "eth_price_usd": eth_price,
                        "value_usd": value_usd,
                    }
                )
            except Exception as e:
                print(f"  TX parse error in block {block_num}: {e}")
                continue

        # Throttle to respect Alchemy free-tier rate limits
        time.sleep(0.2)

    return pd.DataFrame(transactions_data)


def append_or_create(new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Merge new data with existing CSV, deduplicating by tx_hash."""
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset="tx_hash", inplace=True)
        return combined
    return new_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect large ETH transfers from recent blocks.")
    parser.add_argument(
        "--blocks",
        type=int,
        default=NUM_BLOCKS_DEFAULT,
        metavar="N",
        help=f"How many recent blocks to scan (default: {NUM_BLOCKS_DEFAULT}).",
    )
    args = parser.parse_args()
    n = max(1, args.blocks)

    latest_block = w3.eth.block_number
    print(f"Latest block: {latest_block}")
    print(f"Collecting {n} blocks ({latest_block - n} → {latest_block}) ...")

    df_whales = collect_whale_data(latest_block, num_blocks=n)

    if df_whales.empty:
        print("No whale transactions found in this range.")
    else:
        df_final = append_or_create(df_whales, OUTPUT_CSV)
        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"\nDone. {len(df_final)} whale transactions saved to {OUTPUT_CSV}")
        print(df_final[["timestamp", "from_address", "value_eth", "value_usd"]].tail(5).to_string())
