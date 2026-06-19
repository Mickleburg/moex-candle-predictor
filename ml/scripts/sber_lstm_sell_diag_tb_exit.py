"""
LSTM v2 — SELL-side diagnosis + true triple-barrier exit. All on cached predictions.

Two cheap questions, no retraining (uses ml/artifacts/lstm_v2_wf_predictions.npz):

#5 SELL diagnosis — why does a confident SELL never fire?
    - Per predicted class (argmax): confidence ceiling and how much mass clears 0.45/0.50/0.55.
    - Market fact check: realized label balance in the VAL period (upper/lower/timeout).
      The full 2020-26 set is ~33/33/33, but the val folds (2023-07..2025) may be skewed.
    - Latent SELL edge: if we DID short every SELL-leaning candle (3h and triple-barrier exit),
      is there any edge being masked by low confidence?

#1 True triple-barrier exit — match execution to the label definition.
    For each BUY signal (conf>0.50), exit at the FIRST of:
      upper barrier touched (take profit), lower barrier touched (stop), or 3h timeout.
    Barriers/outcomes come straight from triple_barrier_details (same code that makes labels).
    Compare to fixed 1h and fixed 3h exits on the same signals. Cooldown per mode
    (TB frees capital at the actual time-to-barrier, often < 3h).

Result saved to: ml/docs/research/sber_h1_sell_diag_tb_exit_results_YYYYMMDD_HHMMSS.json
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

from scripts.sber_backtest_research import FEE, HOURS_PER_YEAR
from scripts.sber_multiticker_lstm_research import TickerData, PRIMARY_TICKER, TARGET_SPEC
from src.nlp.targets import triple_barrier_details

RESULTS_DIR = ML_DIR / "docs" / "research"
CACHE_PATH = ML_DIR / "artifacts" / "lstm_v2_wf_predictions.npz"

# class indices
SELL, HOLD, BUY = 0, 1, 2
CLASS_NAME = {SELL: "SELL", HOLD: "HOLD", BUY: "BUY"}
PROD_THR = 0.50


def sharpe(trade_returns, hold_hours_mean):
    tr = np.asarray(trade_returns, float)
    if len(tr) == 0 or tr.std() < 1e-12:
        return 0.0
    ann = np.sqrt(HOURS_PER_YEAR / max(1.0, hold_hours_mean))
    return float(tr.mean() / tr.std() * ann)


def summarize(trade_returns, holds):
    tr = np.asarray(trade_returns, float)
    if len(tr) == 0:
        return {"n_trades": 0, "total_return": 0.0, "sharpe": 0.0, "win_rate": 0.0,
                "avg_trade_ret": 0.0, "max_drawdown": 0.0, "mean_hold_h": 0.0}
    eq = np.concatenate([[1.0], np.cumprod(1 + tr)])
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.min(eq / np.where(peak < 1e-12, 1.0, peak)) - 1)
    mean_hold = float(np.mean(holds)) if len(holds) else 1.0
    return {
        "n_trades": int(len(tr)),
        "total_return": float(eq[-1] - 1),
        "sharpe": sharpe(tr, mean_hold),
        "win_rate": float((tr > 0).mean()),
        "avg_trade_ret": float(tr.mean()),
        "max_drawdown": max_dd,
        "mean_hold_h": mean_hold,
    }


# ── #5 SELL diagnosis ──────────────────────────────────────────────────────────

def diagnose_sell(proba, idx, det):
    argmax = proba.argmax(1)
    conf = proba.max(1)
    labels = det["labels"][idx]            # realized triple-barrier outcome label per val candle
    outcome = det["outcome"][idx]

    print("\n" + "=" * 72)
    print("#5  SELL-SIDE DIAGNOSIS")
    print("=" * 72)

    # (a) Per predicted class: confidence ceiling
    print("\n(a) Per predicted class (argmax) — confidence ceiling:")
    print(f"  {'class':>5} {'count':>6} {'share':>7} {'meanConf':>9} {'maxConf':>8} "
          f"{'>0.45':>7} {'>0.50':>7} {'>0.55':>7}")
    per_class = {}
    for cls in (SELL, HOLD, BUY):
        m = argmax == cls
        n = int(m.sum())
        c = conf[m]
        rec = {
            "count": n, "share": float(n / len(argmax)),
            "mean_conf": float(c.mean()) if n else 0.0,
            "max_conf": float(c.max()) if n else 0.0,
            "n_gt_045": int((c > 0.45).sum()) if n else 0,
            "n_gt_050": int((c > 0.50).sum()) if n else 0,
            "n_gt_055": int((c > 0.55).sum()) if n else 0,
        }
        per_class[CLASS_NAME[cls]] = rec
        print(f"  {CLASS_NAME[cls]:>5} {n:>6} {rec['share']:>6.1%} {rec['mean_conf']:>9.3f} "
              f"{rec['max_conf']:>8.3f} {rec['n_gt_045']:>7} {rec['n_gt_050']:>7} {rec['n_gt_055']:>7}")

    # (b) Market fact: realized label balance in the val period
    print("\n(b) Realized triple-barrier outcomes in the VAL period (market fact):")
    vals, counts = np.unique(labels[labels >= 0], return_counts=True)
    realized = {}
    for v, c in zip(vals, counts):
        realized[CLASS_NAME[int(v)]] = {"count": int(c), "share": float(c / counts.sum())}
        print(f"  {CLASS_NAME[int(v)]:>5}: {c:>5} ({c/counts.sum():.1%})")

    # (c) Latent SELL edge: short every SELL-leaning candle
    print("\n(c) Latent SELL edge — if we shorted every SELL-argmax candle:")
    sell_edge = {}
    for thr in (0.0, 0.40, 0.45):
        m = (argmax == SELL) & (conf > thr)
        sell_idx = idx[m]
        rets3, holds3 = [], []
        rets_tb, holds_tb = [], []
        for t in sell_idx:
            # short P&L with fixed 3h exit: profit when price falls
            rets3.append(-float(det["future_return"][t]) - 2 * FEE); holds3.append(3)
            # short P&L with triple-barrier exit (mirror of long)
            r, h = short_tb_return(det, t)
            rets_tb.append(r); holds_tb.append(h)
        s3 = summarize(rets3, holds3)
        stb = summarize(rets_tb, holds_tb)
        sell_edge[f"conf>{thr:.2f}"] = {"fixed_3h": s3, "triple_barrier": stb}
        print(f"  SELL conf>{thr:.2f}: n={s3['n_trades']:>4} | "
              f"3h: ret={s3['total_return']:>+7.2%} win={s3['win_rate']:.1%} | "
              f"TB: ret={stb['total_return']:>+7.2%} win={stb['win_rate']:.1%}")

    return {"per_class": per_class, "realized_val_labels": realized, "sell_short_edge": sell_edge}


def short_tb_return(det, t):
    """Short trade P&L with triple-barrier exit; returns (ret, hold_hours)."""
    oc = det["outcome"][t]
    up = float(det["upper_return"][t]); dn = float(det["lower_return"][t])
    h = float(det["time_to_barrier"][t]) if np.isfinite(det["time_to_barrier"][t]) else 3.0
    if oc == "lower_first":        # price dropped → short wins
        return dn - 2 * FEE, h
    if oc == "upper_first":        # price rose → short loses
        return -up - 2 * FEE, h
    # timeout or ambiguous → realized 3h move (short = -move); ambiguous treated as adverse
    if oc == "ambiguous":
        return -up - 2 * FEE, h
    return -float(det["future_return"][t]) - 2 * FEE, h


# ── #1 True triple-barrier exit (BUY signals) ──────────────────────────────────

def long_tb_return(det, t):
    """Long trade P&L with triple-barrier exit; returns (ret, hold_hours)."""
    oc = det["outcome"][t]
    up = float(det["upper_return"][t]); dn = float(det["lower_return"][t])
    h = float(det["time_to_barrier"][t]) if np.isfinite(det["time_to_barrier"][t]) else 3.0
    if oc == "upper_first":        # take profit at upper barrier
        return up - 2 * FEE, h, "win_barrier"
    if oc == "lower_first":        # stopped out at lower barrier
        return -dn - 2 * FEE, h, "stop"
    if oc == "ambiguous":          # conservative: assume stop hit first
        return -dn - 2 * FEE, h, "stop_ambiguous"
    return float(det["future_return"][t]) - 2 * FEE, h, "timeout"   # exit at t+3


def backtest_buy(proba, idx, close, det, exit_mode):
    """exit_mode in {'1h','3h','tb'}. BUY signals with conf>PROD_THR, cooldown per mode."""
    argmax = proba.argmax(1); conf = proba.max(1)
    rets, holds, outcomes = [], [], []
    free_at = -1
    for i, t in enumerate(idx):
        if t < free_at:
            continue
        if argmax[i] != BUY or conf[i] <= PROD_THR:
            continue
        if exit_mode == "1h":
            if t + 1 >= len(close): continue
            r = (close[t + 1] - close[t]) / close[t] - 2 * FEE; h = 1; oc = "fixed1h"
        elif exit_mode == "3h":
            if t + 3 >= len(close): continue
            r = (close[t + 3] - close[t]) / close[t] - 2 * FEE; h = 3; oc = "fixed3h"
        else:  # tb
            r, h, oc = long_tb_return(det, t)
        rets.append(r); holds.append(h); outcomes.append(oc)
        free_at = t + int(np.ceil(h))
    s = summarize(rets, holds)
    s["outcome_counts"] = {o: int(outcomes.count(o)) for o in set(outcomes)}
    return s


def main():
    run_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"sber_h1_sell_diag_tb_exit_results_{timestamp}.json"

    print("=" * 72)
    print("LSTM v2 — SELL diagnosis + true triple-barrier exit (cached predictions)")
    print("=" * 72)

    if not CACHE_PATH.exists():
        print(f"ERROR: prediction cache not found at {CACHE_PATH}")
        print("Run sber_lstm_exit_horizon_backtest.py first to build it.")
        sys.exit(1)

    d = np.load(CACHE_PATH)
    proba, idx, close = d["proba"], d["idx"], d["close"]
    print(f"Loaded {len(idx)} cached val predictions.")

    # Rebuild triple-barrier details on the same SBER df (idx align)
    sber = TickerData(PRIMARY_TICKER)
    det = triple_barrier_details(sber.df, TARGET_SPEC)
    assert len(det["labels"]) == len(sber.df)

    # #5 SELL diagnosis
    sell_diag = diagnose_sell(proba, idx, det)

    # #1 True triple-barrier exit vs fixed exits (BUY, conf>0.50)
    print("\n" + "=" * 72)
    print("#1  TRUE TRIPLE-BARRIER EXIT vs fixed exits (BUY, conf>0.50)")
    print("=" * 72)
    results = {}
    print(f"\n  {'exit':>6} | {'ret':>9} {'Sharpe':>8} {'win':>6} {'avg/trd':>8} "
          f"{'maxDD':>8} {'trades':>7} {'holdH':>6}")
    for mode, label in [("1h", "1h"), ("3h", "3h"), ("tb", "TB")]:
        s = backtest_buy(proba, idx, close, det, mode)
        results[mode] = s
        print(f"  {label:>6} | {s['total_return']:>+8.2%} {s['sharpe']:>8.3f} {s['win_rate']:>5.1%} "
              f"{s['avg_trade_ret']:>+7.3%} {s['max_drawdown']:>7.2%} {s['n_trades']:>7} {s['mean_hold_h']:>6.2f}")
    print(f"\n  TB exit outcome breakdown: {results['tb']['outcome_counts']}")

    result = {
        "experiment": "sber_h1_sell_diag_tb_exit",
        "timestamp": timestamp,
        "git_branch": "ml-expirement",
        "config": {"fee_one_way": FEE, "prod_threshold": PROD_THR,
                   "target": str(TARGET_SPEC.label)},
        "sell_diagnosis": sell_diag,
        "buy_exit_comparison": results,
        "n_val_predictions": int(len(idx)),
        "total_seconds": round(time.time() - run_start, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTotal time: {time.time()-run_start:.1f}s")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
