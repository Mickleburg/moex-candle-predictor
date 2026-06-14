"""
Orthogonal-feature LSTM for a MOEX ticker — production-grade direct LSTM ablation.

Tests whether cross-instrument drivers (broad market, sector, rates, implicit FX) break
the OHLCV 0.48 ceiling and/or improve the production backtest. Same recipe as LSTM v2,
only the per-step feature vector grows: 14 OHLCV/time  +  orthogonal features
(src/features/orthogonal.build_combined_features). Judged by the BACKTEST (the real gate),
not F1 — per the project's hard-won lesson.

Usage:
    python ml/scripts/sber_orthogonal_research.py --ticker SBER --groups market,sector,rates
    python ml/scripts/sber_orthogonal_research.py --ticker SBER --groups market

Baseline (OHLCV-only, weekday-only, 3h+stop) for SBER: +18.96%, Sharpe 14.95, win 73.2%, 41 trades.
Result: ml/docs/research/<ticker>_orthogonal_<groups>_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
sys.path.insert(0, str(ML_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from scripts.sber_multiticker_lstm_research import TickerData, TARGET_SPEC, PRIMARY_TICKER
from scripts.sber_backtest_research import (
    build_sequences_with_indices, normalize_seqs, FEE,
    WF_INITIAL_TRAIN, WF_VAL_SIZE, WF_N_SPLITS, SEEDS,
    BATCH_SIZE, MAX_EPOCHS, LR, WEIGHT_DECAY, PATIENCE, DEVICE,
)
from scripts.sber_multihorizon_research import collect_trades, summarize
from src.data.split import walk_forward_ranges
from src.models.lstm_model import CandleLSTM
from src.features.orthogonal import load_ortho_series, build_combined_features
from src.nlp.targets import triple_barrier_details

RESULTS_DIR = ML_DIR / "docs" / "research"
DATA_DIR = REPO_ROOT / "data" / "raw"
BASELINE = {"SBER": {"return": 0.1896, "sharpe": 14.95, "win_rate": 0.732, "trades": 41}}


def train_fold(X_tr, y_tr, X_va, y_va, seed, input_dim):
    """Same as the LSTM v2 backtest trainer but input_dim-aware (no class weights)."""
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr_n, X_va_n = normalize_seqs(X_tr, X_va)
    tr_ds = TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(y_tr).long())
    va_ds = TensorDataset(torch.from_numpy(X_va_n), torch.from_numpy(y_va).long())
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE * 4)
    model = CandleLSTM(input_dim=input_dim, hidden_size=128, num_layers=2, dropout=0.3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    crit = nn.CrossEntropyLoss()
    best_f1, best_state, no_imp = 0.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in tr_ld:
            opt.zero_grad()
            crit(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                trues.extend(yb.numpy())
        macro = f1_score(trues, preds, average="macro", zero_division=0)
        if macro > best_f1:
            best_f1 = macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    out = []
    with torch.no_grad():
        for xb, yb in DataLoader(va_ds, batch_size=BATCH_SIZE * 4):
            out.append(torch.softmax(model(xb.to(DEVICE)), dim=1).cpu().numpy())
    return np.vstack(out), best_f1


def collect(feat, labels, n_trainval, input_dim):
    folds = walk_forward_ranges(n_trainval, n_splits=WF_N_SPLITS,
                                initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE)
    all_proba, all_idx = [], []
    for fold in folds:
        X_tr, y_tr, _ = build_sequences_with_indices(feat, labels, fold.train_start, fold.train_end)
        X_va, y_va, va_idx = build_sequences_with_indices(feat, labels, fold.val_start, fold.val_end)
        if len(y_tr) < 200 or len(y_va) < 50:
            continue
        print(f"  fold {fold.fold_id}: {len(y_tr)} train, {len(y_va)} val")
        seed_probas = []
        for seed in SEEDS:
            proba, f1 = train_fold(X_tr, y_tr, X_va, y_va, seed, input_dim)
            seed_probas.append(proba)
            print(f"    seed={seed}: F1={f1:.4f}")
        all_proba.append(np.mean(seed_probas, axis=0))
        all_idx.append(va_idx)
    return np.vstack(all_proba), np.concatenate(all_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SBER")
    ap.add_argument("--groups", default="market,sector,rates",
                    help="comma list of: market,sector,rates (empty = OHLCV baseline)")
    args = ap.parse_args()
    ticker = args.ticker.upper()
    groups = tuple(g.strip() for g in args.groups.split(",") if g.strip())

    run_start = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "_".join(groups) if groups else "baseline"
    out_path = RESULTS_DIR / f"{ticker.lower()}_orthogonal_{tag}_{ts}.json"

    print("=" * 74)
    print(f"Orthogonal-feature LSTM — {ticker} | groups={groups or 'OHLCV-only baseline'}")
    print("=" * 74)

    td = TickerData(ticker)
    ortho = load_ortho_series(str(DATA_DIR))
    combined, names = build_combined_features(td.df, ortho, ticker, groups=groups)
    input_dim = combined.shape[1]
    print(f"feature vector: {input_dim} dims ({len(names)-14} orthogonal)")

    proba, idx = collect(combined, td.labels, td.n_trainval, input_dim)
    macro_f1 = float(f1_score(td.labels[idx], proba.argmax(1), average="macro", zero_division=0))

    close = td.df["close"].astype(float).to_numpy()
    high = td.df["high"].astype(float).to_numpy()
    low = td.df["low"].astype(float).to_numpy()
    begin = pd.to_datetime(td.df["begin"])
    det = triple_barrier_details(td.df, TARGET_SPEC)
    trades = collect_trades(proba, idx, close, high, low, det, begin, h=3, skip_weekend=True)
    bt = summarize([t["ret"] for t in trades], [t["hold_h"] for t in trades])

    base = BASELINE.get(ticker)
    print(f"\n{'='*74}\nRESULT — {ticker} groups={tag}")
    print(f"  WF macro-F1: {macro_f1:.4f}")
    print(f"  Backtest (3h+stop, weekday-only): ret={bt['total_return']:+.2%} "
          f"Sharpe={bt['sharpe']:.2f} win={bt['win_rate']:.1%} trades={bt['n']}")
    if base:
        print(f"  OHLCV baseline:                   ret={base['return']:+.2%} "
              f"Sharpe={base['sharpe']:.2f} win={base['win_rate']:.1%} trades={base['trades']}")
        print(f"  Delta return: {bt['total_return']-base['return']:+.2%}  "
              f"Delta Sharpe: {bt['sharpe']-base['sharpe']:+.2f}")

    result = {
        "experiment": "orthogonal_lstm", "ticker": ticker, "groups": list(groups),
        "timestamp": ts, "git_branch": "ml-expirement",
        "input_dim": input_dim, "orthogonal_feature_names": names[14:],
        "wf_macro_f1": macro_f1, "backtest": bt,
        "ohlcv_baseline": base,
        "total_seconds": round(time.time() - run_start, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Total time: {(time.time()-run_start)/60:.1f} min")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
