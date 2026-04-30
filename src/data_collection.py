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

NUM_BLOCKS_DEFAULT = 20_000
WHALE_THRESHOLD_ETH = 10


def connect_web3(alchemy_url: str | None = None):
    """Create and verify a Web3 connection. Raises ConnectionError on failure."""
    url = alchemy_url or os.getenv("ALCHEMY_URL")
    _w3 = Web3(Web3.HTTPProvider(url))
    if not _w3.is_connected():
        raise ConnectionError("Cannot connect to Ethereum node. Check ALCHEMY_URL in .env")
    print("Connected to Ethereum")
    return _w3


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


def collect_whale_data(start_block: int, num_blocks: int = 500, w3=None) -> pd.DataFrame:
    """Scan `num_blocks` blocks backwards from `start_block` and collect
    all ETH transfers larger than WHALE_THRESHOLD_ETH."""
    if w3 is None:
        w3 = connect_web3()
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


def run(output_csv, num_blocks: int = NUM_BLOCKS_DEFAULT, alchemy_url: str | None = None) -> pd.DataFrame:
    """Module entry point: collect whale data and append to output_csv."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    w3 = connect_web3(alchemy_url)
    latest_block = w3.eth.block_number
    print(f"Latest block: {latest_block}")
    print(f"Collecting {num_blocks} blocks ({latest_block - num_blocks} → {latest_block}) ...")
    df = collect_whale_data(latest_block, num_blocks=num_blocks, w3=w3)
    if df.empty:
        print("No whale transactions found in this range.")
        return df
    df_final = append_or_create(df, output_csv)
    df_final.to_csv(output_csv, index=False)
    print(f"\nDone. {len(df_final)} whale transactions saved to {output_csv}")
    print(df_final[["timestamp", "from_address", "value_eth", "value_usd"]].tail(5).to_string())
    return df_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect large ETH transfers from recent blocks.")
    parser.add_argument(
        "--blocks", type=int, default=NUM_BLOCKS_DEFAULT, metavar="N",
        help=f"How many recent blocks to scan (default: {NUM_BLOCKS_DEFAULT}).",
    )
    parser.add_argument("--output", type=str, default="data/raw_whale_transactions.csv")
    args = parser.parse_args()
    run(output_csv=Path(args.output), num_blocks=max(1, args.blocks))
