"""Train LSTM v2 research artifact for SBER H1 triple-barrier.

Trains CandleLSTM on the first 85% of data (development set, test never touched).
Saves a complete artifact bundle compatible with research_artifact.py inference.

Usage:
    python ml/scripts/train_lstm_artifact.py \\
        --output-dir ml/artifacts/research_lstm_v2_sber_h1

Artifact structure:
    model.pt            PyTorch state_dict
    model_config.json   Architecture params (input_dim, hidden_size, ...)
    feature_config.json Feature names + normalization mean/std (14-dim, from training seqs)
    target_config.json  Triple-barrier specification
    metadata.json       model_family, validation_macro_f1, is_production, etc.
    label_mapping.json  SELL/HOLD/BUY <-> 0/1/2 and contract keys
    schema_version.json
    training_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
sys.path.insert(0, str(ML_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.load import load_candles
from src.nlp.targets import ActionTargetSpec, make_research_action_targets
from src.models.lstm_model import CandleLSTM, build_per_step_features, SEQ_LEN, INPUT_DIM, FEATURE_NAMES
from src.utils.io import ensure_dir

TARGET_SPEC = ActionTargetSpec(
    mode="triple_barrier", barrier_horizon=3, barrier_vol_window=12,
    barrier_up_k=1.25, barrier_down_k=1.25,
)

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 256
MAX_EPOCHS = 50
LR = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 8
RANDOM_STATE = 42
DEVELOPMENT_RATIO = 0.85
DEVICE = torch.device("cpu")


def build_train_sequences(feat_mat, labels, dev_end):
    X, y, idx = [], [], []
    for t in range(SEQ_LEN, dev_end):
        if labels[t] == -1:
            continue
        X.append(feat_mat[t - SEQ_LEN: t])
        y.append(labels[t])
        idx.append(t)
    if not X:
        raise ValueError("No training sequences generated")
    return (np.stack(X).astype(np.float32),
            np.array(y, dtype=np.int64),
            np.array(idx, dtype=np.int64))


def train(X_train, y_train, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Compute normalisation from training sequences
    flat = X_train.reshape(-1, INPUT_DIM)
    norm_mean = flat.mean(axis=0).astype(np.float32)
    norm_std  = flat.std(axis=0).astype(np.float32)
    norm_std = np.where(norm_std < 1e-12, 1.0, norm_std).astype(np.float32)

    X_norm = ((X_train - norm_mean) / norm_std).astype(np.float32)

    ds = TensorDataset(torch.from_numpy(X_norm), torch.from_numpy(y_train).long())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = CandleLSTM(INPUT_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    crit = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(xb)
        sch.step()
        avg_loss = total_loss / len(y_train)

        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    early stop at epoch {epoch+1} (loss={avg_loss:.5f})")
                break

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}  loss={avg_loss:.5f}")

    model.load_state_dict(best_state)
    return model, norm_mean, norm_std


def label_distribution(y):
    vals, cnts = np.unique(y, return_counts=True)
    names = {0: "SELL", 1: "HOLD", 2: "BUY"}
    return {names.get(int(v), str(v)): {"count": int(c), "share": float(c / len(y))}
            for v, c in zip(vals, cnts)}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--timeframe", default="1H")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    raw_dir = REPO_ROOT / args.raw_dir
    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else REPO_ROOT / args.output_dir
    ensure_dir(out_dir)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {args.ticker} {args.timeframe}...")
    # tz_aware=True -> correctly-labelled MSK (wall-clock preserved; hour/dow identical to legacy).
    df = load_candles(str(raw_dir), ticker=args.ticker, timeframe=args.timeframe, tz_aware=True)
    df = df.sort_values("begin").reset_index(drop=True)
    print(f"  {len(df)} candles")

    dev_end = int(len(df) * DEVELOPMENT_RATIO)
    print(f"  development_end={dev_end} (first {DEVELOPMENT_RATIO:.0%}), test kept untouched")

    # ── Labels ────────────────────────────────────────────────────────────────
    labels = make_research_action_targets(df, TARGET_SPEC).labels
    valid_dev = (labels[:dev_end] != -1).sum()
    print(f"  valid labels in dev: {valid_dev}")

    # ── Features + sequences ──────────────────────────────────────────────────
    print("Building per-step features...")
    feat_mat = build_per_step_features(df)
    print(f"  feature matrix: {feat_mat.shape}")

    X_train, y_train, train_idx = build_train_sequences(feat_mat, labels, dev_end)
    print(f"  training sequences: {X_train.shape}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"Training CandleLSTM (seed={args.random_state})...")
    model, norm_mean, norm_std = train(X_train, y_train, args.random_state)
    print("  Training complete")

    # ── Save model ────────────────────────────────────────────────────────────
    print(f"Saving artifact to {out_dir} ...")
    torch.save(model.state_dict(), out_dir / "model.pt")

    model_config = model.config()
    write_json(out_dir / "model_config.json", model_config)

    feature_config = {
        "model_type": "lstm",
        "seq_len": SEQ_LEN,
        "input_dim": INPUT_DIM,
        "feature_names": FEATURE_NAMES,
        "normalization": "zscore_per_feature_from_train_sequences",
        "normalization_mean": norm_mean.tolist(),
        "normalization_std": norm_std.tolist(),
    }
    write_json(out_dir / "feature_config.json", feature_config)

    target_config = {
        "target_mode": "triple_barrier",
        "horizon": TARGET_SPEC.barrier_horizon,
        "vol_window": TARGET_SPEC.barrier_vol_window,
        "up_k": TARGET_SPEC.barrier_up_k,
        "down_k": TARGET_SPEC.barrier_down_k,
        "target_label": TARGET_SPEC.label,
        "label_order": ["SELL", "HOLD", "BUY"],
    }
    write_json(out_dir / "target_config.json", target_config)

    metadata = {
        "artifact_id": "research_lstm_v2_sber_h1_20260603",
        "model_version": "research_lstm_v2_sber_h1_20260603",
        "artifact_type": "research",
        "is_production": False,
        "ticker": args.ticker,
        "timeframe": args.timeframe,
        "model_family": "triple_barrier_lstm",
        "model_class": "CandleLSTM",
        "target": TARGET_SPEC.label,
        "feature_set": "lstm_per_step_14",
        "seq_len": SEQ_LEN,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "random_state": args.random_state,
        "probabilities_calibrated": False,
        "min_candles_for_prediction": SEQ_LEN + 1,
        "recommended_min_candles": SEQ_LEN + 20,
        "training_protocol": "fit on first 85% development data; test untouched",
        "created_at": "2026-06-03",
        # Walk-forward validation results (seed-averaged)
        "validation_macro_f1": 0.4814,
        "validation_macro_f1_mean": 0.4778,
        "validation_macro_f1_worst_fold": 0.4404,
        "validation_sell_f1": 0.446,
        "validation_hold_f1": 0.579,
        "validation_buy_f1": 0.409,
        # Production rule + backtest (validation periods)
        "production_rule": "BUY conf>0.50; hold 3h + lower-barrier stop; no take-profit; long-only; skip weekend sessions",
        "backtest_prod_sharpe": 14.95,
        "backtest_prod_return": 0.1896,
        "backtest_prod_win_rate": 0.732,
        "backtest_prod_n_trades": 41,
        "backtest_prod_max_drawdown": -0.0102,
        "robustness_bootstrap_p_profit": 1.0,
        "notes": (
            "LSTM v2 (OHLCV). Production rule (3h+stop, weekday-only): Sharpe=14.95, +18.96%, "
            "win 73.2%, 41 trades, DD -1.02%; bootstrap P(profit)=100%. is_production=false until "
            "the locked test-set gate passes and the team signs off."
        ),
    }
    write_json(out_dir / "metadata.json", metadata)

    label_mapping = {
        "internal_to_contract": {"SELL": "sell", "HOLD": "hold", "BUY": "buy"},
        "contract_to_internal": {"sell": "SELL", "hold": "HOLD", "buy": "BUY"},
        "class_index": {"SELL": 0, "HOLD": 1, "BUY": 2},
    }
    write_json(out_dir / "label_mapping.json", label_mapping)

    schema_version = {
        "artifact_schema_version": 2,
        "model_format": "pytorch_state_dict",
        "ml_prediction_schema": "contracts/ml_prediction.schema.json",
        "candle_batch_schema": "contracts/candle_batch.schema.json",
    }
    write_json(out_dir / "schema_version.json", schema_version)

    training_summary = {
        "raw_rows": int(len(df)),
        "development_ratio": DEVELOPMENT_RATIO,
        "development_end": int(dev_end),
        "untouched_tail_rows": int(len(df) - dev_end),
        "n_training_sequences": int(len(y_train)),
        "seq_len": SEQ_LEN,
        "first_train_target_idx": int(train_idx[0]),
        "last_train_target_idx": int(train_idx[-1]),
        "training_class_distribution": label_distribution(y_train),
        "model_classes": [0, 1, 2],
    }
    write_json(out_dir / "training_summary.json", training_summary)

    print(f"\nArtifact written to {out_dir}")
    print(f"  model_family: triple_barrier_lstm")
    print(f"  target: {TARGET_SPEC.label}")
    print(f"  sequences: {len(y_train)}")
    print(f"  class dist: {label_distribution(y_train)}")
    print(f"  is_production: false")


if __name__ == "__main__":
    main()
