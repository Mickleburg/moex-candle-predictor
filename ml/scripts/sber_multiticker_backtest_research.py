"""
Backtest: Multi-ticker LSTM vs LSTM v2 (SBER-only) vs Buy&Hold on SBER H1 val periods.

Decision gate for packaging the multi-ticker model as the new primary artifact.
The multi-ticker F1 gain (+0.0074) is modest; what matters for production is whether
its confident signals are PROFITABLE. LSTM v2's production value was Sharpe=6.38 at
conf>0.50, not its F1. This script answers: does multi-ticker training improve the
backtest too, or only the F1?

Design — apples-to-apples:
    * Both models are trained by the IDENTICAL `train_lstm_fold` (imported from
      sber_backtest_research.py) and scored by the IDENTICAL `run_backtest` engine,
      same fee (0.05% one-way), same thresholds, same 1h hold, same Sharpe annualisation.
    * The ONLY difference between the two models is the training data:
        - LSTM v2:      SBER only, per-fold [train_start, train_end]
        - Multi-ticker: SBER+LKOH+GAZP pooled, each ticker time-filtered to
                        target_ts < val_start_time (leak-free, 3x data)
    * Validation is SBER-only for BOTH, on identical walk-forward folds. Predictions
      are averaged over seeds [7,42,100], same as the documented LSTM v2 backtest.
    * Reproducing LSTM v2 here (expected Sharpe ~6.38 at conf>0.50) is a sanity anchor
      that the harness matches the original 2026-06-03 backtest.

NOTE: train_lstm_fold uses plain CrossEntropyLoss (no class weights), matching the
original backtest harness — NOT the class-weighted F1-experiment training. This keeps
the comparison to the documented 6.38 valid and isolates the data effect on Sharpe.

Result saved to:
    ml/docs/research/sber_h1_multiticker_backtest_results_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
sys.path.insert(0, str(ML_DIR))

import numpy as np

from src.data.split import walk_forward_ranges

# Reuse the EXACT trading engine + LSTM training from the documented LSTM v2 backtest
from scripts.sber_backtest_research import (
    train_lstm_fold, build_sequences_with_indices, run_backtest, buy_and_hold,
    FEE, THRESHOLDS, HOURS_PER_YEAR, SEEDS,
    WF_INITIAL_TRAIN, WF_VAL_SIZE, WF_N_SPLITS,
)
# Reuse leak-free per-ticker bundle (features, labels, timestamps, time-filtered cutoff)
from scripts.sber_multiticker_lstm_research import TickerData, ALL_TICKERS, PRIMARY_TICKER

RESULTS_DIR = ML_DIR / "docs" / "research"
LSTM_V2_DOC_SHARPE = 6.38   # documented anchor (2026-06-03 backtest, conf>0.50)


# ── Prediction collection ──────────────────────────────────────────────────────

def collect_sber_only(sber: TickerData):
    """LSTM v2 baseline: train on SBER per-fold window, predict SBER val."""
    folds = walk_forward_ranges(sber.n_trainval, n_splits=WF_N_SPLITS,
                                initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE)
    all_proba, all_idx = [], []
    for fold in folds:
        X_tr, y_tr, _ = build_sequences_with_indices(sber.feat, sber.labels,
                                                     fold.train_start, fold.train_end)
        X_va, y_va, va_idx = build_sequences_with_indices(sber.feat, sber.labels,
                                                          fold.val_start, fold.val_end)
        if len(y_tr) < 200 or len(y_va) < 50:
            continue
        print(f"  [LSTM v2]      fold {fold.fold_id}: {len(y_tr)} train, {len(y_va)} val")
        seed_probas = []
        for seed in SEEDS:
            proba, f1 = train_lstm_fold(X_tr, y_tr, X_va, y_va, seed)
            seed_probas.append(proba)
            print(f"                   seed={seed}: F1={f1:.4f}")
        all_proba.append(np.mean(seed_probas, axis=0))
        all_idx.append(va_idx)
    return np.vstack(all_proba), np.concatenate(all_idx)


def collect_multiticker(data: dict[str, TickerData]):
    """Multi-ticker: train on SBER+LKOH+GAZP (time-filtered), predict SBER val."""
    sber = data[PRIMARY_TICKER]
    folds = walk_forward_ranges(sber.n_trainval, n_splits=WF_N_SPLITS,
                                initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE)
    all_proba, all_idx = [], []
    for fold in folds:
        val_start_time = sber.ts[fold.val_start]
        X_parts, y_parts, counts = [], [], {}
        for tk in ALL_TICKERS:
            td = data[tk]
            cut = td.index_before(val_start_time)
            X, y, _ = build_sequences_with_indices(td.feat, td.labels, 0, cut)
            if len(y):
                X_parts.append(X); y_parts.append(y)
            counts[tk] = len(y)
        X_tr = np.concatenate(X_parts, axis=0)
        y_tr = np.concatenate(y_parts, axis=0)

        X_va, y_va, va_idx = build_sequences_with_indices(sber.feat, sber.labels,
                                                          fold.val_start, fold.val_end)
        if len(y_va) < 50:
            continue
        print(f"  [Multi-ticker] fold {fold.fold_id}: {len(y_tr)} train "
              f"(SBER={counts['SBER']} LKOH={counts.get('LKOH',0)} GAZP={counts.get('GAZP',0)}), "
              f"{len(y_va)} val")
        seed_probas = []
        for seed in SEEDS:
            proba, f1 = train_lstm_fold(X_tr, y_tr, X_va, y_va, seed)
            seed_probas.append(proba)
            print(f"                   seed={seed}: F1={f1:.4f}")
        all_proba.append(np.mean(seed_probas, axis=0))
        all_idx.append(va_idx)
    return np.vstack(all_proba), np.concatenate(all_idx)


def backtest_all_thresholds(close, proba, idx):
    out = {}
    for thr in THRESHOLDS:
        r = run_backtest(close, proba, idx, thr)
        r.pop("equity_curve", None)   # don't serialise the full curve
        out[f"{thr:.2f}"] = r
    return out


def best_threshold(results: dict):
    return max(results.keys(), key=lambda t: results[t]["sharpe"])


def main():
    run_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"sber_h1_multiticker_backtest_results_{timestamp}.json"

    print("=" * 70)
    print("Multi-ticker vs LSTM v2 — SBER H1 backtest (decision gate)")
    print(f"Fee={FEE:.2%} one-way | thresholds={THRESHOLDS} | 1h hold | seeds={SEEDS}")
    print(f"Output: {output_path}")
    print("=" * 70)

    print("\nLoading tickers (test split excluded)...")
    data = {tk: TickerData(tk) for tk in ALL_TICKERS}
    sber = data[PRIMARY_TICKER]
    close = sber.df["close"].astype(float).values
    for tk, td in data.items():
        print(f"  {tk}: {len(td.df)} candles, trainval={td.n_trainval}")

    print(f"\n{'='*70}\nCollecting LSTM v2 (SBER-only) predictions...")
    t0 = time.time()
    v2_proba, v2_idx = collect_sber_only(sber)
    print(f"  LSTM v2 done in {(time.time()-t0)/60:.1f} min  |  {len(v2_idx)} val predictions")

    print(f"\n{'='*70}\nCollecting Multi-ticker predictions...")
    t0 = time.time()
    mt_proba, mt_idx = collect_multiticker(data)
    print(f"  Multi-ticker done in {(time.time()-t0)/60:.1f} min  |  {len(mt_idx)} val predictions")

    # Sanity: both should validate on the same SBER candles
    assert np.array_equal(v2_idx, mt_idx), "Val index mismatch between models!"

    print(f"\n{'='*70}\nRunning backtests...")
    bh = buy_and_hold(close, v2_idx)
    print(f"  Buy&Hold: return={bh['total_return']:+.2%}  Sharpe={bh['sharpe']:.3f}  DD={bh['max_drawdown']:.2%}")

    v2_results = backtest_all_thresholds(close, v2_proba, v2_idx)
    mt_results = backtest_all_thresholds(close, mt_proba, mt_idx)

    print(f"\n  {'thr':>5} | {'LSTM v2':^34} | {'Multi-ticker':^34}")
    print(f"  {'':>5} | {'Sharpe':>8} {'ret':>9} {'trades':>7} {'win':>6} | "
          f"{'Sharpe':>8} {'ret':>9} {'trades':>7} {'win':>6}")
    for thr in THRESHOLDS:
        k = f"{thr:.2f}"
        a, b = v2_results[k], mt_results[k]
        print(f"  {thr:>5.2f} | {a['sharpe']:>8.3f} {a['total_return']:>+8.2%} "
              f"{a['n_trades']:>7} {a['win_rate']:>5.1%} | "
              f"{b['sharpe']:>8.3f} {b['total_return']:>+8.2%} "
              f"{b['n_trades']:>7} {b['win_rate']:>5.1%}")

    v2_best = best_threshold(v2_results)
    mt_best = best_threshold(mt_results)

    print(f"\n{'='*70}\nDECISION GATE")
    print(f"{'='*70}")
    print(f"  LSTM v2 best:      thr={v2_best}  Sharpe={v2_results[v2_best]['sharpe']:.3f}  "
          f"ret={v2_results[v2_best]['total_return']:+.2%}  trades={v2_results[v2_best]['n_trades']}")
    print(f"  Multi-ticker best: thr={mt_best}  Sharpe={mt_results[mt_best]['sharpe']:.3f}  "
          f"ret={mt_results[mt_best]['total_return']:+.2%}  trades={mt_results[mt_best]['n_trades']}")
    print(f"  Anchor (documented LSTM v2 conf>0.50 Sharpe): {LSTM_V2_DOC_SHARPE}")
    print(f"  Buy&Hold:          Sharpe={bh['sharpe']:.3f}  ret={bh['total_return']:+.2%}")

    mt_wins = mt_results[mt_best]["sharpe"] >= v2_results[v2_best]["sharpe"]
    print(f"\n  VERDICT: multi-ticker {'WINS' if mt_wins else 'does NOT beat'} LSTM v2 on best-threshold Sharpe.")

    total_time = time.time() - run_start
    result = {
        "experiment": "sber_h1_multiticker_backtest",
        "timestamp": timestamp,
        "git_branch": "ml-expirement",
        "system": {
            "python": sys.version, "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "config": {
            "fee_one_way": FEE, "thresholds": THRESHOLDS, "hold_hours": 1,
            "hours_per_year": HOURS_PER_YEAR, "seeds": SEEDS,
            "wf_initial_train": WF_INITIAL_TRAIN, "wf_val_size": WF_VAL_SIZE,
            "wf_n_splits": WF_N_SPLITS, "tickers": ALL_TICKERS,
            "training_loss": "CrossEntropyLoss (no class weights, matches 2026-06-03 harness)",
        },
        "buy_and_hold": bh,
        "lstm_v2": {"by_threshold": v2_results, "best_threshold": v2_best,
                    "documented_anchor_sharpe": LSTM_V2_DOC_SHARPE},
        "multiticker": {"by_threshold": mt_results, "best_threshold": mt_best},
        "verdict": {
            "multiticker_wins": bool(mt_wins),
            "lstm_v2_best_sharpe": v2_results[v2_best]["sharpe"],
            "multiticker_best_sharpe": mt_results[mt_best]["sharpe"],
        },
        "n_val_predictions": int(len(v2_idx)),
        "total_seconds": round(total_time, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Total time: {total_time/60:.1f} min")
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
