"""Analyze model quality: feature importance, probability calibration, class bias."""
import sys
sys.path.insert(0, 'ml')

import pickle, json
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.load import load_candles
from src.data.split import time_split
from src.nlp.action_features import make_continuous_past_features
from src.nlp.targets import ActionTargetSpec, make_research_action_targets

ARTIFACT_DIR = Path('ml/artifacts/research_triple_barrier_sber_h1')

df = load_candles('data/raw', ticker='SBER', timeframe='1H')
df['begin'] = pd.to_datetime(df['begin'], utc=True)
df = df.sort_values('begin').reset_index(drop=True)

train, val, test = time_split(df, 0.70, 0.15)
print(f"Splits: train={len(train)} val={len(val)} test={len(test)}")

spec = ActionTargetSpec(mode='triple_barrier', barrier_horizon=3, barrier_vol_window=12, barrier_up_k=1.25, barrier_down_k=1.25)
result_full = make_research_action_targets(df, spec)
targets_full = result_full.labels

feat_matrix, feat_names = make_continuous_past_features(df)

with open(ARTIFACT_DIR / 'feature_config.json') as f:
    fcfg = json.load(f)
feat_cols = fcfg['feature_columns']
mean = np.array(fcfg['standardization_mean'])
std = np.array(fcfg['standardization_std'])
std = np.where(std < 1e-12, 1.0, std)

name_to_idx = {n: i for i, n in enumerate(feat_names)}
col_indices = [name_to_idx[c] for c in feat_cols]

X = feat_matrix[:, col_indices]
X_norm = (X - mean) / std
X_norm = np.nan_to_num(X_norm)
y = targets_full

# Val mask
val_start_idx = len(train)
val_end_idx = len(train) + len(val)
val_mask = (np.arange(len(df)) >= val_start_idx) & (np.arange(len(df)) < val_end_idx) & (y != -1)
X_val = X_norm[val_mask]
y_val = y[val_mask]

with open(ARTIFACT_DIR / 'model.pkl', 'rb') as f:
    model = pickle.load(f)

proba = model.predict_proba(X_val)
preds = model.predict(X_val)

print(f"\nVal set: {len(y_val)} samples after filtering -1")

# Per-class metrics
from sklearn.metrics import classification_report, f1_score
label_names = ['SELL', 'HOLD', 'BUY']
print("\nClassification report:")
print(classification_report(y_val, preds, target_names=label_names))

print(f"Macro-F1: {f1_score(y_val, preds, average='macro'):.4f}")

# Predicted probability distribution
print("\nPredicted prob stats (val):")
for i, name in enumerate(label_names):
    p = proba[:, i]
    print(f"  {name}: mean={p.mean():.3f}  std={p.std():.3f}  max={p.max():.3f}")

# Confidence distribution
conf = proba.max(axis=1)
print(f"\nConfidence distribution: mean={conf.mean():.3f}  median={np.median(conf):.3f}  >0.5={( conf>0.5).mean():.2%}")

# Feature importances
importances = model.feature_importances_
top_k = 15
top_idx = np.argsort(importances)[::-1][:top_k]
print(f"\nTop-{top_k} feature importances:")
for rank, idx in enumerate(top_idx, 1):
    print(f"  {rank:2d}. {feat_cols[idx]:<30s}  {importances[idx]:.4f}")
