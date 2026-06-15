"""
Rolling-retrain ENSEMBLE simulator — the decisive test of whether the edge is salvageable.

The single-seed rolling sim showed no edge (over-fires, win 38-43% everywhere). But the original
validation that showed +18.96%/win 73%/~1% action used a 3-SEED ENSEMBLE (averaging suppresses
confidence → only rare consensus signals clear conf>0.50). We packaged a single-seed artifact —
a mismatch. This re-runs the rolling deployment simulation with a 3-seed ensemble (average proba,
trade only on consensus conf>0.50) to test:
  * does the ensemble restore the rare ~1% action rate and high win, and
  * does it stay positive through 2025-2026 (where the frozen single-seed model failed)?

Yes → salvageable: deploy as ensemble, fix packaging. No → edge is regime-fragile → cross-sectional.

Same rolling schedule (train 12000, step 2000), adaptive per-window normalization. Seeds [7,42,100].
Result: ml/docs/research/rolling_ensemble_sim_<ts>.json
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
from scripts.sber_edge_analysis import long_stop_return, summarize
from scripts.sber_backtest_research import BATCH_SIZE, MAX_EPOCHS, LR, WEIGHT_DECAY, PATIENCE, DEVICE
from scripts.rolling_retrain_sim import build_seq, TRAIN_SIZE, STEP, SEQ_LEN, HORIZON, BUY
from src.data.split import rolling_walk_forward_ranges
from src.models.lstm_model import CandleLSTM, build_per_step_features
from src.nlp.targets import triple_barrier_details

RESULTS_DIR = ML_DIR / "docs" / "research"
SEEDS = [7, 42, 100]


def train_one(X_tr, y_tr, Xva_n, input_dim, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    flat = X_tr.reshape(-1, input_dim)
    mean = flat.mean(0).astype(np.float32)
    std = np.where(flat.std(0) < 1e-12, 1.0, flat.std(0)).astype(np.float32)
    Xtr = ((X_tr - mean) / std).astype(np.float32)
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
    # normalize val with THIS seed's train-window stats, predict
    Xva = ((Xva_n - mean) / std).astype(np.float32)
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(Xva)), 1).numpy()


def main():
    run_start = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 78)
    print(f"Rolling ENSEMBLE simulator (seeds={SEEDS}, consensus conf>0.50) — salvageable?")
    print("=" * 78)

    td = TickerData(PRIMARY_TICKER)
    feat = build_per_step_features(td.df); labels = td.labels
    close = td.df["close"].astype(float).to_numpy()
    high = td.df["high"].astype(float).to_numpy(); low = td.df["low"].astype(float).to_numpy()
    begin = pd.to_datetime(td.df["begin"])
    det = triple_barrier_details(td.df, TARGET_SPEC)
    up, dn, fut = det["upper_return"], det["lower_return"], det["future_return"]

    folds = rolling_walk_forward_ranges(len(td.df), train_size=TRAIN_SIZE, val_size=STEP,
                                        step_size=STEP, max_folds=12)
    print(f"{len(folds)} rolling retrains x {len(SEEDS)} seeds\n")
    print(f"  {'OOS period':>34} | {'sig':>4} {'act%':>5} {'win':>5} {'ret':>8} {'Sharpe':>7}")
    print("-" * 80)

    chunk_records, all_rets, all_holds = [], [], []
    for fold in folds:
        X_tr, y_tr, _ = build_seq(feat, labels, fold.train_start, fold.train_end)
        X_va, y_va, va_idx = build_seq(feat, labels, fold.val_start, fold.val_end)
        if len(y_tr) < 500 or len(y_va) < 50:
            continue
        probas = [train_one(X_tr, y_tr, X_va, feat.shape[1], s) for s in SEEDS]
        proba = np.mean(probas, axis=0)                    # ENSEMBLE consensus
        argmax, conf = proba.argmax(1), proba.max(1)
        rets, holds, free_at = [], [], -1
        n_sig = int(((argmax == BUY) & (conf > 0.50)).sum())
        for i, t in enumerate(va_idx):
            if t < free_at or argmax[i] != BUY or conf[i] <= 0.50:
                continue
            if begin.iloc[t].dayofweek >= 5 or t + HORIZON >= len(close):
                continue
            r, h, _ = long_stop_return(int(t), close, high, low, float(up[t]), float(dn[t]), float(fut[t]))
            rets.append(r); holds.append(h); free_at = t + int(np.ceil(h))
        s = summarize(rets, holds); act = n_sig / max(1, len(va_idx))
        period = f"{begin.iloc[fold.val_start]:%Y-%m-%d}..{begin.iloc[fold.val_end-1]:%Y-%m-%d}"
        print(f"  {period:>34} | {n_sig:>4} {act:>5.1%} {s['win_rate']:>4.0%} "
              f"{s['total_return']:>+7.2%} {s['sharpe']:>7.2f}")
        chunk_records.append({"period": period, "n_signals": n_sig, "action_rate": float(act),
                              "n_trades": s["n"], "win_rate": s["win_rate"],
                              "return": s["total_return"], "sharpe": s["sharpe"]})
        all_rets += rets; all_holds += holds

    total = summarize(all_rets, all_holds)
    eq = float(np.prod([1 + r for r in all_rets])) - 1 if all_rets else 0.0
    print("-" * 80)
    print(f"  TOTAL: trades={total['n']} compounded={eq:+.2%} Sharpe={total['sharpe']:.2f} win={total['win_rate']:.1%}")
    recent = [c for c in chunk_records if c["period"] >= "2025"]
    if recent:
        print(f"  2025-2026: summed_ret={sum(c['return'] for c in recent):+.2%} "
              f"avg_action={np.mean([c['action_rate'] for c in recent]):.1%} "
              f"avg_win={np.mean([c['win_rate'] for c in recent]):.0%}")

    out = {"experiment": "rolling_ensemble_sim", "timestamp": ts, "seeds": SEEDS,
           "chunks": chunk_records, "total": {**total, "compounded_return": eq},
           "single_seed_reference": {"total_compounded": -0.1463, "win": 0.384},
           "total_seconds": round(time.time() - run_start, 1)}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / f"rolling_ensemble_sim_{ts}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Total time: {(time.time()-run_start)/60:.1f} min\n  Saved: {p}")


if __name__ == "__main__":
    main()
