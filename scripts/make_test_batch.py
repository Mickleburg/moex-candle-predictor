"""Build a candle_batch example JSON from real SBER data (last N candles of val period)."""
import sys, json
sys.path.insert(0, 'ml')

import pandas as pd
from src.data.load import load_candles

df = load_candles('data/raw', ticker='SBER', timeframe='1H')
df['begin'] = pd.to_datetime(df['begin'], utc=True)
df = df.sort_values('begin').reset_index(drop=True)

# Take 60 candles from within val period for a realistic test
n = 60
sample = df.iloc[-(n + 200):-(200)].reset_index(drop=True)

candles = []
for _, row in sample.iterrows():
    candles.append({
        "begin": row['begin'].isoformat(),
        "open": float(row['open']),
        "high": float(row['high']),
        "low": float(row['low']),
        "close": float(row['close']),
        "volume": float(row['volume']),
    })

batch = {"ticker": "SBER", "timeframe": "1H", "candles": candles}
path = 'contracts/examples/candle_batch_real.json'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(batch, f, indent=2, ensure_ascii=False)
print(f"Written {len(candles)} candles to {path}")
print(f"Range: {candles[0]['begin']} -> {candles[-1]['begin']}")
