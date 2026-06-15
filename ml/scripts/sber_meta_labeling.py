"""
SBER meta-labeling — Stage 3 (#2). Sharpen the trade selection with a secondary model.

Lopez de Prado meta-labeling: the primary model (LSTM) gives DIRECTION; a secondary model
decides whether to ACT on each primary BUY signal (bet / no-bet). Trained time-ordered on
past-only features at each candidate, with label = "would this 3h+stop trade have won?".

All on cached SBER predictions (no LSTM retrain). Candidates = every BUY-argmax val candle.
Meta-features (past-only at t): primary probs/confidence, volatility, session (hour/dow,
is_friday, is_evening), recent OHLCV momentum. Meta-label: long 3h+stop outcome > 0.

Evaluation: time-split candidates 50/50. Train meta on the first half, then on the HELD-OUT
second half compare:
    baseline  = conf>0.50 + skip weekends (the current production rule)
    meta-rule = meta P(win) > threshold (+ skip weekends)
Judged by backtest (return/Sharpe/win/trades). If meta beats baseline out-of-sample on the
held-out half, it sharpens selection.

Result: ml/docs/research/sber_meta_labeling_<ts>.json
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from scripts.sber_multiticker_lstm_research import TickerData, TARGET_SPEC, PRIMARY_TICKER
from scripts.sber_edge_analysis import long_stop_return, summarize
from scripts.sber_backtest_research import HOURS_PER_YEAR
from src.nlp.targets import triple_barrier_details

CACHE = ML_DIR / "artifacts" / "lstm_v2_wf_predictions.npz"
RESULTS_DIR = ML_DIR / "docs" / "research"
BUY = 2
BASELINE_FULL = {"return": 0.1896, "sharpe": 14.95, "win_rate": 0.732, "trades": 41}


def build_meta(proba, idx, td, det):
    """Per BUY-argmax candidate: past-only meta-features, outcome label, return, hold, t."""
    close = td.df["close"].astype(float).to_numpy()
    high = td.df["high"].astype(float).to_numpy()
    low = td.df["low"].astype(float).to_numpy()
    begin = pd.to_datetime(td.df["begin"])
    c1 = td.df["close"].astype(float).pct_change(1).to_numpy()
    c3 = td.df["close"].astype(float).pct_change(3).to_numpy()
    up = det["upper_return"]; dn = det["lower_return"]; fut = det["future_return"]; vol = det["past_volatility"]
    argmax = proba.argmax(1); conf = proba.max(1)

    rows, y, rets, holds, ts, dows = [], [], [], [], [], []
    for i, t in enumerate(idx):
        if argmax[i] != BUY:
            continue
        if t + 3 >= len(close):
            continue
        b = begin.iloc[t]
        feat = [
            float(conf[i]), float(proba[i, 2]), float(proba[i, 1]), float(proba[i, 0]),
            float(vol[t]),
            np.sin(2 * np.pi * b.hour / 24), np.cos(2 * np.pi * b.hour / 24),
            float(b.dayofweek), 1.0 if b.dayofweek == 4 else 0.0, 1.0 if b.hour >= 19 else 0.0,
            float(np.nan_to_num(c1[t])), float(np.nan_to_num(c3[t])),
        ]
        r, h, _ = long_stop_return(t, close, high, low, float(up[t]), float(dn[t]), float(fut[t]))
        rows.append(feat); y.append(1 if r > 0 else 0)
        rets.append(r); holds.append(h); ts.append(int(t)); dows.append(int(b.dayofweek))
    names = ["conf", "p_buy", "p_hold", "p_sell", "vol", "hour_sin", "hour_cos",
             "dow", "is_friday", "is_evening", "ret_1h", "ret_3h"]
    return (np.array(rows, float), np.array(y), np.array(rets), np.array(holds),
            np.array(ts), np.array(dows), names)


def bt(mask, rets, holds, dows, skip_weekend=True):
    sel = mask & (dows < 5) if skip_weekend else mask
    return summarize(rets[sel], holds[sel])


def main():
    run_start = time.time()
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"sber_meta_labeling_{ts_str}.json"
    print("=" * 72)
    print("SBER meta-labeling — Stage 3 (#2): sharpen selection with a secondary model")
    print("=" * 72)

    d = np.load(CACHE)
    proba, idx = d["proba"], d["idx"]
    td = TickerData(PRIMARY_TICKER)
    det = triple_barrier_details(td.df, TARGET_SPEC)

    X, y, rets, holds, t_arr, dows, names = build_meta(proba, idx, td, det)
    conf = X[:, 0]
    print(f"BUY candidates: {len(y)} | base win rate: {y.mean():.1%}")

    # Time-ordered 50/50 split (candidates already in chronological order)
    order = np.argsort(t_arr, kind="stable")
    X, y, rets, holds, conf, dows, t_arr = (a[order] for a in (X, y, rets, holds, conf, dows, t_arr))
    cut = len(y) // 2
    tr, te = slice(0, cut), slice(cut, None)

    meta = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15,
                                          learning_rate=0.05, l2_regularization=1.0,
                                          random_state=42)
    meta.fit(X[tr], y[tr])
    p_win = meta.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], p_win) if len(np.unique(y[te])) > 1 else float("nan")
    print(f"meta-model test AUC: {auc:.3f}")

    rets_te, holds_te, dows_te, conf_te = rets[te], holds[te], dows[te], conf[te]

    # Baseline on the held-out half: conf>0.50 + skip weekends
    base = bt(conf_te > 0.50, rets_te, holds_te, dows_te)
    print(f"\nHeld-out half ({len(rets_te)} candidates):")
    print(f"  baseline conf>0.50:  ret={base['total_return']:+.2%} Sharpe={base['sharpe']:.2f} "
          f"win={base['win_rate']:.1%} n={base['n']}")

    # Meta-rule: sweep P(win) thresholds
    sweep = {}
    print("  meta P(win) sweep:")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
        m = bt(p_win > thr, rets_te, holds_te, dows_te)
        sweep[f"{thr:.2f}"] = m
        print(f"    P(win)>{thr:.2f}: ret={m['total_return']:+.2%} Sharpe={m['sharpe']:.2f} "
              f"win={m['win_rate']:.1%} n={m['n']}")

    # Combine: meta AND conf>0.50
    combo = {}
    for thr in [0.50, 0.55, 0.60]:
        m = bt((p_win > thr) & (conf_te > 0.50), rets_te, holds_te, dows_te)
        combo[f"{thr:.2f}"] = m
    best_meta = max(sweep.values(), key=lambda r: (r["sharpe"], r["total_return"]) if r["n"] >= 5 else (-9, -9))

    print(f"\n  Feature importances (top): training on {cut} candidates")
    # HistGBM has no native importances; skip (kept simple).

    verdict = ("meta BEATS baseline" if best_meta["sharpe"] > base["sharpe"] and best_meta["n"] >= 5
               else "meta does NOT beat baseline")
    print(f"\n  VERDICT: {verdict} (best meta Sharpe={best_meta['sharpe']:.2f} vs baseline {base['sharpe']:.2f})")

    result = {
        "experiment": "sber_meta_labeling", "timestamp": ts_str, "git_branch": "ml-expirement",
        "n_candidates": int(len(y)), "base_win_rate": float(y.mean()),
        "meta_test_auc": float(auc), "split_cut": int(cut),
        "heldout_baseline_conf050": base,
        "meta_pwin_sweep": sweep, "meta_and_conf050": combo,
        "meta_features": names,
        "full_val_baseline": BASELINE_FULL,
        "verdict": verdict,
        "total_seconds": round(time.time() - run_start, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
