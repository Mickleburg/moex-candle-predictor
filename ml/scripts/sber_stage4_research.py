"""
SBER Stage 4 — #5 target re-engineering (by backtest, not F1) + #6 sequence length.

Low prior (the edge is a clean simple phenomenon; complexity has diluted it 6x), run for
completeness. Config-driven: vary triple-barrier up_k/down_k/vol_window (#5) and the LSTM input
window seq_len (#6). Same LSTM v2 recipe otherwise. Judged by the production backtest
(BUY conf>0.50, hold 3h + lower-barrier stop, skip weekends) vs the baseline +18.96%/Sharpe 14.95.

Usage:
    python ml/scripts/sber_stage4_research.py --up-k 1.5 --down-k 1.0 --tag asym_long
    python ml/scripts/sber_stage4_research.py --seq-len 48 --tag seqlen48
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

from scripts.sber_multiticker_lstm_research import TickerData, PRIMARY_TICKER
from scripts.sber_backtest_research import (
    normalize_seqs, FEE, WF_INITIAL_TRAIN, WF_VAL_SIZE, WF_N_SPLITS, SEEDS,
    BATCH_SIZE, MAX_EPOCHS, LR, WEIGHT_DECAY, PATIENCE, DEVICE,
)
from scripts.sber_edge_analysis import long_stop_return, summarize
from src.data.split import walk_forward_ranges
from src.models.lstm_model import CandleLSTM, build_per_step_features
from src.nlp.targets import ActionTargetSpec, make_research_action_targets, triple_barrier_details

RESULTS_DIR = ML_DIR / "docs" / "research"
BASELINE = {"return": 0.1896, "sharpe": 14.95, "win_rate": 0.732, "trades": 41}
HORIZON = 3
BUY = 2


def build_seq(feat, labels, start, end, seq_len):
    X, y, idx = [], [], []
    for t in range(start + seq_len, end):
        if labels[t] == -1:
            continue
        X.append(feat[t - seq_len:t]); y.append(labels[t]); idx.append(t)
    if not X:
        return np.empty((0, seq_len, feat.shape[1]), np.float32), np.empty(0, np.int64), np.empty(0, np.int64)
    return np.stack(X).astype(np.float32), np.array(y, np.int64), np.array(idx, np.int64)


def train_fold(X_tr, y_tr, X_va, seed, input_dim):
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr_n, X_va_n = normalize_seqs(X_tr, X_va)
    tr = DataLoader(TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(y_tr).long()),
                    batch_size=BATCH_SIZE, shuffle=True)
    model = CandleLSTM(input_dim=input_dim, hidden_size=128, num_layers=2, dropout=0.3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    crit = nn.CrossEntropyLoss()
    best, best_state, no_imp = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train(); tot = 0.0
        for xb, yb in tr:
            opt.zero_grad(); loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE)); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item() * len(xb)
        sch.step(); avg = tot / len(y_tr)
        if avg < best - 1e-5:
            best, best_state, no_imp = avg, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in DataLoader(TensorDataset(torch.from_numpy(X_va_n), torch.zeros(len(X_va_n))),
                                batch_size=BATCH_SIZE * 4):
            out.append(torch.softmax(model(xb.to(DEVICE)), 1).cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--up-k", type=float, default=1.25)
    ap.add_argument("--down-k", type=float, default=1.25)
    ap.add_argument("--vol-window", type=int, default=12)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    run_start = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    spec = ActionTargetSpec(mode="triple_barrier", barrier_horizon=HORIZON,
                            barrier_vol_window=args.vol_window,
                            barrier_up_k=args.up_k, barrier_down_k=args.down_k)
    print("=" * 70)
    print(f"SBER Stage 4 — {args.tag} | up_k={args.up_k} down_k={args.down_k} "
          f"vol_w={args.vol_window} seq_len={args.seq_len}")
    print("=" * 70)

    td = TickerData(PRIMARY_TICKER)
    feat = build_per_step_features(td.df)
    labels = make_research_action_targets(td.df, spec).labels
    close = td.df["close"].astype(float).to_numpy()
    high = td.df["high"].astype(float).to_numpy()
    low = td.df["low"].astype(float).to_numpy()
    begin = pd.to_datetime(td.df["begin"])
    det = triple_barrier_details(td.df, spec)
    up, dn, fut = det["upper_return"], det["lower_return"], det["future_return"]

    folds = walk_forward_ranges(td.n_trainval, n_splits=WF_N_SPLITS,
                                initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE)
    all_proba, all_idx = [], []
    for fold in folds:
        X_tr, y_tr, _ = build_seq(feat, labels, fold.train_start, fold.train_end, args.seq_len)
        X_va, y_va, va_idx = build_seq(feat, labels, fold.val_start, fold.val_end, args.seq_len)
        if len(y_tr) < 200 or len(y_va) < 50:
            continue
        print(f"  fold {fold.fold_id}: {len(y_tr)} train, {len(y_va)} val")
        sp = [train_fold(X_tr, y_tr, X_va, seed, feat.shape[1]) for seed in SEEDS]
        all_proba.append(np.mean(sp, axis=0)); all_idx.append(va_idx)
    proba = np.vstack(all_proba); idx = np.concatenate(all_idx)
    macro_f1 = float(f1_score(labels[idx], proba.argmax(1), average="macro", zero_division=0))

    # Backtest: BUY conf>0.50, 3h + lower-barrier stop, skip weekends
    argmax, conf = proba.argmax(1), proba.max(1)
    rets, holds, free_at = [], [], -1
    for i, t in enumerate(idx):
        if t < free_at or argmax[i] != BUY or conf[i] <= 0.50:
            continue
        if begin.iloc[t].dayofweek >= 5 or t + HORIZON >= len(close):
            continue
        r, h, _ = long_stop_return(int(t), close, high, low, float(up[t]), float(dn[t]), float(fut[t]))
        rets.append(r); holds.append(h); free_at = t + int(np.ceil(h))
    bt = summarize(rets, holds)

    print(f"\nRESULT {args.tag}: WF_F1={macro_f1:.4f} | ret={bt['total_return']:+.2%} "
          f"Sharpe={bt['sharpe']:.2f} win={bt['win_rate']:.1%} n={bt['n']}")
    print(f"  baseline: ret={BASELINE['return']:+.2%} Sharpe={BASELINE['sharpe']:.2f} "
          f"win={BASELINE['win_rate']:.1%} n={BASELINE['trades']}")
    print(f"  delta: ret={bt['total_return']-BASELINE['return']:+.2%} Sharpe={bt['sharpe']-BASELINE['sharpe']:+.2f}")

    result = {"experiment": "sber_stage4", "tag": args.tag, "timestamp": ts,
              "config": {"up_k": args.up_k, "down_k": args.down_k, "vol_window": args.vol_window,
                         "seq_len": args.seq_len, "horizon": HORIZON},
              "wf_macro_f1": macro_f1, "backtest": bt, "baseline": BASELINE,
              "total_seconds": round(time.time() - run_start, 1)}
    out = RESULTS_DIR / f"sber_stage4_{args.tag}_{ts}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
