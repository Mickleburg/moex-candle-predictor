"""
SBER multi-horizon LSTM — Stage 2 (#1): expand the opportunity set via different horizons.

The SBER edge is real but rare (~41 weekday trades / 16 mo at h=3). Different triple-barrier
horizons (h=6, h=12) give DIFFERENT entry signals on the same series — potentially more
independent high-conviction opportunities of similar quality. This trains LSTM v2 for h=6 and
h=12 (h=3 loaded from cache), backtests each with the production rule, and tests COMBINING
horizons (pooled entries, global cooldown) to see if total tradeable opportunity grows.

Identical recipe per horizon (only barrier_horizon changes): CandleLSTM v2, 14 features,
4-fold walk-forward, seeds [7,42,100], no class weights. Backtest = BUY conf>0.50, hold h bars
with lower-barrier stop, no take-profit, long-only, SKIP WEEKEND sessions (per edge analysis).

Caveat from prior research: at h>=6 the HOLD class shrinks (a barrier is almost always hit
within 6-12h), so these targets are more BUY/SELL — different signal character, tested here.

Result: ml/docs/research/sber_multihorizon_results_YYYYMMDD_HHMMSS.json
Caches predictions per horizon: ml/artifacts/lstm_v2_wf_predictions_h{H}.npz
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
from sklearn.metrics import f1_score

from scripts.sber_multiticker_lstm_research import TickerData, PRIMARY_TICKER
from scripts.sber_backtest_research import (
    train_lstm_fold, build_sequences_with_indices, FEE,
    WF_INITIAL_TRAIN, WF_VAL_SIZE, WF_N_SPLITS, SEEDS,
)
from scripts.sber_backtest_research import HOURS_PER_YEAR
from src.data.split import walk_forward_ranges
from src.nlp.targets import ActionTargetSpec, make_research_action_targets, triple_barrier_details

RESULTS_DIR = ML_DIR / "docs" / "research"
ARTIFACTS_DIR = ML_DIR / "artifacts"
H3_CACHE = ARTIFACTS_DIR / "lstm_v2_wf_predictions.npz"

HORIZONS = [3, 6, 12]
PROD_THR = 0.50


def spec_for(h):
    return ActionTargetSpec(mode="triple_barrier", barrier_horizon=h, barrier_vol_window=12,
                            barrier_up_k=1.25, barrier_down_k=1.25)


def collect(feat, labels, n_trainval):
    folds = walk_forward_ranges(n_trainval, n_splits=WF_N_SPLITS,
                                initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE)
    all_proba, all_idx = [], []
    for fold in folds:
        X_tr, y_tr, _ = build_sequences_with_indices(feat, labels, fold.train_start, fold.train_end)
        X_va, y_va, va_idx = build_sequences_with_indices(feat, labels, fold.val_start, fold.val_end)
        if len(y_tr) < 200 or len(y_va) < 50:
            continue
        print(f"    fold {fold.fold_id}: {len(y_tr)} train, {len(y_va)} val")
        seed_probas = []
        for seed in SEEDS:
            proba, f1 = train_lstm_fold(X_tr, y_tr, X_va, y_va, seed)
            seed_probas.append(proba)
        all_proba.append(np.mean(seed_probas, axis=0))
        all_idx.append(va_idx)
    return np.vstack(all_proba), np.concatenate(all_idx)


def get_predictions(h, feat, labels, n_trainval):
    cache = H3_CACHE if h == 3 else ARTIFACTS_DIR / f"lstm_v2_wf_predictions_h{h}.npz"
    if cache.exists():
        d = np.load(cache)
        print(f"  [h={h}] loaded cache ({len(d['idx'])} preds)")
        return d["proba"], d["idx"]
    print(f"  [h={h}] training walk-forward...")
    proba, idx = collect(feat, labels, n_trainval)
    np.savez(cache, proba=proba, idx=idx)
    print(f"  [h={h}] cached {len(idx)} preds")
    return proba, idx


def sharpe(rets, mean_hold):
    tr = np.asarray(rets, float)
    if len(tr) < 2 or tr.std() < 1e-12:
        return 0.0
    return float(tr.mean() / tr.std() * np.sqrt(HOURS_PER_YEAR / max(1.0, mean_hold)))


def summarize(rets, holds):
    tr = np.asarray(rets, float)
    if len(tr) == 0:
        return {"n": 0, "total_return": 0.0, "win_rate": 0.0, "sharpe": 0.0, "mean_ret": 0.0}
    eq = np.cumprod(1 + tr)
    return {"n": int(len(tr)), "total_return": float(eq[-1] - 1), "win_rate": float((tr > 0).mean()),
            "sharpe": sharpe(tr, float(np.mean(holds))), "mean_ret": float(tr.mean())}


def collect_trades(proba, idx, close, high, low, det, begin, h, skip_weekend=True):
    """BUY conf>0.50, hold h bars + lower-barrier stop, skip weekend sessions. Returns trade list."""
    argmax = proba.argmax(1); conf = proba.max(1)
    up = det["upper_return"]; dn = det["lower_return"]; fut = det["future_return"]
    trades = []
    free_at = -1
    for i, t in enumerate(idx):
        if t < free_at:
            continue
        if argmax[i] != 2 or conf[i] <= PROD_THR:
            continue
        if skip_weekend and begin.iloc[t].dayofweek >= 5:
            continue
        if t + h >= len(close):
            continue
        lower = close[t] * (1.0 - float(dn[t]))
        r, hold = None, h
        for step in range(1, h + 1):
            if low[t + step] <= lower:
                r = -float(dn[t]) - 2 * FEE; hold = step; break
        if r is None:
            r = float(fut[t]) - 2 * FEE; hold = h
        trades.append({"t": int(t), "ret": r, "hold_h": hold, "horizon": h})
        free_at = t + int(np.ceil(hold))
    return trades


def combine(trade_lists):
    """Pool entries from all horizons; greedily accept non-overlapping (global cooldown)."""
    pooled = sorted([tr for lst in trade_lists for tr in lst], key=lambda x: x["t"])
    chosen = []
    free_at = -1
    for tr in pooled:
        if tr["t"] < free_at:
            continue
        chosen.append(tr)
        free_at = tr["t"] + int(np.ceil(tr["hold_h"]))
    return chosen


def main():
    run_start = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"sber_multihorizon_results_{ts}.json"

    print("=" * 74)
    print("SBER multi-horizon LSTM — Stage 2 (#1) opportunity expansion")
    print(f"Horizons={HORIZONS} | rule: BUY conf>0.50, hold h + stop, skip weekends")
    print("=" * 74)

    sber = TickerData(PRIMARY_TICKER)
    feat = sber.feat
    close = sber.df["close"].astype(float).to_numpy()
    high = sber.df["high"].astype(float).to_numpy()
    low = sber.df["low"].astype(float).to_numpy()
    begin = pd.to_datetime(sber.df["begin"])
    n_trainval = sber.n_trainval

    per_h = {}
    trade_lists = []
    for h in HORIZONS:
        print(f"\n--- horizon h={h} ---")
        labels = make_research_action_targets(sber.df, spec_for(h)).labels
        proba, idx = get_predictions(h, feat, labels, n_trainval)
        y_true = labels[idx]
        macro_f1 = float(f1_score(y_true, proba.argmax(1), average="macro", zero_division=0))
        det = triple_barrier_details(sber.df, spec_for(h))
        trades = collect_trades(proba, idx, close, high, low, det, begin, h)
        bt = summarize([t["ret"] for t in trades], [t["hold_h"] for t in trades])
        per_h[h] = {"wf_macro_f1": macro_f1, "backtest": bt}
        trade_lists.append(trades)
        print(f"  h={h}: WF_F1={macro_f1:.4f} | trades={bt['n']} ret={bt['total_return']:+.2%} "
              f"Sharpe={bt['sharpe']:.2f} win={bt['win_rate']:.1%}")

    # combined
    chosen = combine(trade_lists)
    combo = summarize([t["ret"] for t in chosen], [t["hold_h"] for t in chosen])
    by_h = {}
    for t in chosen:
        by_h[t["horizon"]] = by_h.get(t["horizon"], 0) + 1
    print(f"\nCOMBINED (pooled, global cooldown): trades={combo['n']} ret={combo['total_return']:+.2%} "
          f"Sharpe={combo['sharpe']:.2f} win={combo['win_rate']:.1%}  composition={by_h}")

    print(f"\n{'='*74}\nSUMMARY")
    print(f"  {'config':>10} | {'WF F1':>7} | {'trades':>6} {'ret':>9} {'Sharpe':>7} {'win':>6}")
    for h in HORIZONS:
        r = per_h[h]
        print(f"  {'h='+str(h):>10} | {r['wf_macro_f1']:>7.4f} | {r['backtest']['n']:>6} "
              f"{r['backtest']['total_return']:>+8.2%} {r['backtest']['sharpe']:>7.2f} {r['backtest']['win_rate']:>5.1%}")
    print(f"  {'combined':>10} | {'--':>7} | {combo['n']:>6} {combo['total_return']:>+8.2%} "
          f"{combo['sharpe']:>7.2f} {combo['win_rate']:>5.1%}")

    result = {
        "experiment": "sber_multihorizon", "timestamp": ts, "git_branch": "ml-expirement",
        "config": {"horizons": HORIZONS, "rule": "BUY conf>0.50, hold h + lower-barrier stop, skip weekends",
                   "seeds": SEEDS},
        "per_horizon": {str(h): per_h[h] for h in HORIZONS},
        "combined": {**combo, "composition": {str(k): v for k, v in by_h.items()}},
        "baseline_h3_weekday": {"return": 0.1896, "sharpe": 14.95, "win_rate": 0.732, "trades": 41},
        "total_seconds": round(time.time() - run_start, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Total time: {(time.time()-run_start)/60:.1f} min")
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
