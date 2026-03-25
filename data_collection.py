import os
import time
import pandas as pd
import requests
from web3 import Web3
from dotenv import load_dotenv
from datetime import datetime

# Завантажуємо ключі з .env файлу
load_dotenv()
ALCHEMY_URL = os.getenv("ALCHEMY_URL")

# 1. Підключення до блокчейну
w3 = Web3(Web3.HTTPProvider(ALCHEMY_URL))

if not w3.is_connected():
    raise ConnectionError("Не вдалося підключитися до Ethereum-ноди")
else:
    print("✅ Підключено до Ethereum")

def get_eth_price(timestamp):
    """Отримує ціну ETH на Binance для заданого часу (офчейн дані)"""
    url = "https://api.binance.com/api/v3/klines"
    # Binance API очікує час у мілісекундах
    params = {
        "symbol": "ETHUSDT",
        "interval": "1m",
        "startTime": timestamp * 1000,
        "limit": 1
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data:
            return float(data[0][4]) # Close price
    except Exception as e:
        print(f"Помилка API Binance: {e}")
    return None

def collect_whale_data(start_block, num_blocks=10):
    """Збирає транзакції з блоків та формує датасет"""
    transactions_data = []

    for block_num in range(start_block, start_block - num_blocks, -1):
        print(f"🔍 Парсинг блоку: {block_num}")
        block = w3.eth.get_block(block_num, full_transactions=True)
        block_time = block.timestamp
        eth_price = get_eth_price(block_time)

        for tx in block.transactions:
            # Фільтруємо "пилюку", залишаємо тільки транзакції з переказом ETH
            value_eth = w3.from_wei(tx.value, 'ether')
            
            # Для DeepWhale нас цікавлять великі об'єми, наприклад > 10 ETH
            if value_eth > 10:
                tx_info = {
                    "block_number": block.number,
                    "timestamp": datetime.fromtimestamp(block_time).strftime('%Y-%m-%d %H:%M:%S'),
                    "tx_hash": tx.hash.hex(),
                    "from_address": tx['from'],
                    "to_address": tx.to if tx.to else "Contract Creation",
                    "value_eth": float(value_eth),
                    "gas_price_gwei": float(w3.from_wei(tx.gasPrice, 'gwei')),
                    "eth_price_usd": eth_price
                }
                transactions_data.append(tx_info)
        
        # Захист від rate limit (щоб Alchemy та Binance не заблокували)
        time.sleep(0.5)

    return pd.DataFrame(transactions_data)

# Запуск збору (беремо останній блок)
latest_block = w3.eth.block_number
print(f"Починаємо збір з блоку: {latest_block}")

# Збираємо дані за останні 50 блоків (для тесту)
df_whales = collect_whale_data(latest_block, num_blocks=50)

# Зберігаємо сирі дані
df_whales.to_csv("raw_whale_transactions.csv", index=False)
print(f"✅ Зібрано {len(df_whales)} великих транзакцій. Збережено в CSV.")