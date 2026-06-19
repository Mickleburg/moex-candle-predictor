"""Quick data quality check on fresh SBER parquet."""
import sys
sys.path.insert(0, 'ml')

import pandas as pd
import numpy as np

from src.data.load import load_candles
from src.data.split import time_split
from src.nlp.targets import ActionTargetSpec, make_research_action_targets

df = load_candles('data/raw', ticker='SBER', timeframe='1H')
df['begin'] = pd.to_datetime(df['begin'], utc=True)
df = df.sort_values('begin').reset_index(drop=True)
print(f"Loaded: {len(df)} candles")
print(f"Range:  {df['begin'].min()} -> {df['begin'].max()}")
print(f"Columns: {list(df.columns)}")
print(f"NaNs: {df[['open','high','low','close','volume']].isna().sum().to_dict()}")

train, val, test = time_split(df, 0.70, 0.15)
print(f"\nSplit  train={len(train)}  val={len(val)}  test={len(test)}")
print(f"Train: {train['begin'].min().date()} -> {train['begin'].max().date()}")
print(f"Val:   {val['begin'].min().date()} -> {val['begin'].max().date()}")
print(f"Test:  {test['begin'].min().date()} -> {test['begin'].max().date()}")

spec = ActionTargetSpec(
    mode='triple_barrier',
    barrier_horizon=3,
    barrier_vol_window=12,
    barrier_up_k=1.25,
    barrier_down_k=1.25,
)
result = make_research_action_targets(df, spec)
targets = result.labels
valid_mask = targets != -1
print(f"\nTarget distribution (full, excluding -1):")
counts = pd.Series(targets[valid_mask]).value_counts().sort_index()
total = counts.sum()
for cls, cnt in counts.items():
    label = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}.get(cls, str(cls))
    print(f"  {label} ({cls}): {cnt}  ({100*cnt/total:.1f}%)")

# Feature check
from src.nlp.action_features import make_continuous_past_features
feat_matrix, feat_names = make_continuous_past_features(df)
print(f"\nFeatures: {len(feat_names)} columns, {feat_matrix.shape[0]} rows")
nan_cols = [(name, np.isnan(feat_matrix[:, i]).sum()) for i, name in enumerate(feat_names) if np.isnan(feat_matrix[:, i]).any()]
if nan_cols:
    print(f"NaN in features: {nan_cols[:5]}")
else:
    print("No NaN in feature matrix (after burn-in)")
