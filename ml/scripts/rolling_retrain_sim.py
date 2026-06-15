"""
Rolling-retrain deployment simulator — diagnoses whether ADAPTIVITY rescues the edge.

The frozen model (trained once on 2020-2025) failed the test gate on 2025-2026: it over-fired
(x4-5 trades) and lost money — distribution shift + frozen normalization. This simulates a
realistically DEPLOYED system that periodically retrains on a trailing window with ADAPTIVE
(rolling) normalization, and measures the out-of-sample P&L chunk-by-chunk through 2023→2026.

Setup:
  * Rolling window: train on the last TRAIN_SIZE candles, predict the next STEP candles (OOS),
    then roll forward by STEP and retrain. Normalization is recomputed from each train window
    (adaptive) — this is the key fix for the frozen-normalization over-confidence.
  * Production rule on each OOS chunk: BUY conf>0.50, hold 3h + lower-barrier stop, skip weekends.
  * Single seed (42) per retrain for tractability — diagnostic of the TREND, not precise numbers.

Key question: do the 2025-2026 chunks keep a sane action rate (~1%) and non-negative P&L when the
model is retrained on recent data? If yes → the failure was staleness (fixable by adaptivity).
If no → the directional 1H edge is genuinely regime-fragile → change the game (cross-sectional).

Integrity: this evaluates over 2025-2026 (the now-burned test period), so treat it as a DIAGNOSTIC,
not a clean gate. A fresh forward period is needed for any future honest gate.

Result: ml/docs/research/rolling_retrain_sim_<ts>.json
"""

from __future__ import annotations

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
from scripts.sber_edge_analysis import long_stop_return, summarize, sharpe
from scripts.sber_backtest_research import BATCH_SIZE, MAX_EPOCHS, LR, WEIGHT_DECAY, PATIENCE, DEVICE
from src.data.split import rolling_walk_forward_ranges
from src.models.lstm_model import CandleLSTM, build_per_step_features
from src.nlp.targets import triple_barrier_details

RESULTS_DIR = ML_DIR / "docs" / "research"
SEQ_LEN = 32
HORIZON = 3
BUY = 2
SEED = 42

TRAIN_SIZE = 12000   # trailing window (~3 years)
STEP = 2000          # retrain cadence / OOS chunk (~5-6 months)


def build_seq(feat, labels, start, end):
    X, y, idx = [], [], []
    for t in range(start + SEQ_LEN, end):
        if labels[t] == -1:
            continue
        X.append(feat[t - SEQ_LEN:t]); y.append(labels[t]); idx.append(t)
    if not X:
        return (np.empty((0, SEQ_LEN, feat.shape[1]), np.float32), np.empty(0, np.int64), np.empty(0, np.int64))
    return np.stack(X).astype(np.float32), np.array(y, np.int64), np.array(idx, np.int64)


def train_predict(X_tr, y_tr, X_va, input_dim):
    """Adaptive normalization from the TRAIN window; train single-seed; return val proba."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    flat = X_tr.reshape(-1, input_dim)
    mean = flat.mean(0).astype(np.float32)
    std = np.where(flat.std(0) < 1e-12, 1.0, flat.std(0)).astype(np.float32)
    Xtr = ((X_tr - mean) / std).astype(np.float32)
    Xva = ((X_va - mean) / std).astype(np.float32)        # val normalized with TRAIN-window stats (adaptive)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(y_tr).long()),
                        batch_size=BATCH_SIZE, shuffle=True)
    model = CandleLSTM(input_dim=input_dim, hidden_size=128, num_layers=2, dropout=0.3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    crit = nn.CrossEntropyLoss()
    best, best_state, no_imp = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train(); tot = 0.0
        for xb, yb in loader:
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
    with torch.no_grad():
        proba = torch.softmax(model(torch.from_numpy(Xva)), 1).numpy()
    return proba


def main():
    run_start = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 74)
    print("Rolling-retrain deployment simulator — does adaptivity rescue the edge?")
    print(f"trailing train={TRAIN_SIZE}, retrain/OOS step={STEP}, adaptive normalization, seed={SEED}")
    print("=" * 74)

    td = TickerData(PRIMARY_TICKER)
    feat = build_per_step_features(td.df)
    labels = td.labels
    close = td.df["close"].astype(float).to_numpy()
    high = td.df["high"].astype(float).to_numpy()
    low = td.df["low"].astype(float).to_numpy()
    begin = pd.to_datetime(td.df["begin"])
    det = triple_barrier_details(td.df, TARGET_SPEC)
    up, dn, fut = det["upper_return"], det["lower_return"], det["future_return"]

    folds = rolling_walk_forward_ranges(len(td.df), train_size=TRAIN_SIZE, val_size=STEP,
                                        step_size=STEP, max_folds=12)
    print(f"{len(folds)} rolling retrains\n")
    print(f"  {'OOS period':>34} | {'F1':>6} {'sig':>4} {'act%':>5} {'win':>5} {'ret':>8} {'Sharpe':>7}")
    print("-" * 90)

    chunk_records, all_rets, all_holds = [], [], []
    for fold in folds:
        X_tr, y_tr, _ = build_seq(feat, labels, fold.train_start, fold.train_end)
        X_va, y_va, va_idx = build_seq(feat, labels, fold.val_start, fold.val_end)
        if len(y_tr) < 500 or len(y_va) < 50:
            continue
        proba = train_predict(X_tr, y_tr, X_va, feat.shape[1])
        argmax, conf = proba.argmax(1), proba.max(1)
        f1 = f1_score(y_va, argmax, average="macro", zero_division=0)

        rets, holds, free_at = [], [], -1
        n_sig = int(((argmax == BUY) & (conf > 0.50)).sum())
        for i, t in enumerate(va_idx):
            if t < free_at or argmax[i] != BUY or conf[i] <= 0.50:
                continue
            if begin.iloc[t].dayofweek >= 5 or t + HORIZON >= len(close):
                continue
            r, h, _ = long_stop_return(int(t), close, high, low, float(up[t]), float(dn[t]), float(fut[t]))
            rets.append(r); holds.append(h); free_at = t + int(np.ceil(h))
        s = summarize(rets, holds)
        act = n_sig / max(1, len(va_idx))
        period = f"{begin.iloc[fold.val_start]:%Y-%m-%d}..{begin.iloc[fold.val_end-1]:%Y-%m-%d}"
        print(f"  {period:>34} | {f1:>6.3f} {n_sig:>4} {act:>5.1%} {s['win_rate']:>4.0%} "
              f"{s['total_return']:>+7.2%} {s['sharpe']:>7.2f}")
        chunk_records.append({"period": period, "val_start": int(fold.val_start),
                              "f1": float(f1), "n_signals": n_sig, "action_rate": float(act),
                              "n_trades": s["n"], "win_rate": s["win_rate"],
                              "return": s["total_return"], "sharpe": s["sharpe"]})
        all_rets += rets; all_holds += holds

    total = summarize(all_rets, all_holds)
    # compounded equity across all chunks
    eq = float(np.prod([1 + r for r in all_rets])) - 1 if all_rets else 0.0
    print("-" * 90)
    print(f"  TOTAL across chunks: trades={total['n']} compounded_return={eq:+.2%} "
          f"Sharpe={total['sharpe']:.2f} win={total['win_rate']:.1%}")

    # 2025-2026 subset (where frozen failed)
    recent = [c for c in chunk_records if c["period"] >= "2025"]
    if recent:
        rr = sum(c["return"] for c in recent)
        print(f"  2025-2026 chunks: {[c['period'] for c in recent]}")
        print(f"    summed return={rr:+.2%}  avg action_rate={np.mean([c['action_rate'] for c in recent]):.1%}  "
              f"avg win={np.mean([c['win_rate'] for c in recent]):.0%}")

    result = {"experiment": "rolling_retrain_sim", "timestamp": ts, "git_branch": "ml-expirement",
              "config": {"train_size": TRAIN_SIZE, "step": STEP, "seed": SEED,
                         "adaptive_normalization": True, "rule": "BUY conf>0.50, 3h+stop, skip weekends"},
              "chunks": chunk_records, "total": {**total, "compounded_return": eq},
              "frozen_gate_reference": {"sber_test_return": -0.0796, "sber_test_trades": 213,
                                        "note": "frozen model on 2025-09..2026-06 lost -7.96% with 213 trades"},
              "total_seconds": round(time.time() - run_start, 1)}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"rolling_retrain_sim_{ts}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Total time: {(time.time()-run_start)/60:.1f} min")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
