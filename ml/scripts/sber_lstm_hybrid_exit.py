"""
LSTM v2 — complete the exit-design 2x2 matrix (BUY, conf>0.50). Cached predictions, no retraining.

We have two corners already:
    fixed 3h  (TP off, stop off) = +16.08%, Sharpe 9.56, DD -2.16%
    full TB   (TP on,  stop on)  = + 7.10%, Sharpe 7.68, DD -1.15%
This fills the two missing corners:
    hybrid A  (TP off, stop on)  = hold to t+3, but stop out if LOW touches lower barrier first
    hybrid B  (TP on,  stop off) = take profit if HIGH touches upper barrier first, else hold to t+3

Barriers are the same volatility bands used to build the labels (k=1.25 x past_vol),
from triple_barrier_details. Path is walked on intrabar high/low, horizon=3. Cooldown per mode.

Result saved to: ml/docs/research/sber_h1_hybrid_exit_results_YYYYMMDD_HHMMSS.json
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

BUY = 2
PROD_THR = 0.50
HORIZON = 3


def summarize(rets, holds):
    tr = np.asarray(rets, float)
    if len(tr) == 0:
        return {"n_trades": 0, "total_return": 0.0, "sharpe": 0.0, "win_rate": 0.0,
                "avg_trade_ret": 0.0, "max_drawdown": 0.0, "mean_hold_h": 0.0}
    eq = np.concatenate([[1.0], np.cumprod(1 + tr)])
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.min(eq / np.where(peak < 1e-12, 1.0, peak)) - 1)
    mean_hold = float(np.mean(holds)) if len(holds) else 1.0
    ann = np.sqrt(HOURS_PER_YEAR / max(1.0, mean_hold))
    shp = float(tr.mean() / tr.std() * ann) if tr.std() > 1e-12 else 0.0
    return {"n_trades": int(len(tr)), "total_return": float(eq[-1] - 1), "sharpe": shp,
            "win_rate": float((tr > 0).mean()), "avg_trade_ret": float(tr.mean()),
            "max_drawdown": max_dd, "mean_hold_h": mean_hold}


def long_exit(mode, t, close, high, low, up_ret, dn_ret, fut_ret):
    """Return (trade_ret, hold_h) for a long entered at close[t], horizon=3."""
    upper = close[t] * (1.0 + up_ret)
    lower = close[t] * (1.0 - dn_ret)
    if mode == "fixed_3h":
        return fut_ret - 2 * FEE, HORIZON
    if mode == "stop_only":          # hold to t+3 unless lower barrier hit first
        for step in range(1, HORIZON + 1):
            if low[t + step] <= lower:
                return -dn_ret - 2 * FEE, step
        return fut_ret - 2 * FEE, HORIZON
    if mode == "tp_only":            # take profit at upper barrier, else hold to t+3
        for step in range(1, HORIZON + 1):
            if high[t + step] >= upper:
                return up_ret - 2 * FEE, step
        return fut_ret - 2 * FEE, HORIZON
    if mode == "full_tb":            # first barrier wins; ambiguous -> conservative stop
        for step in range(1, HORIZON + 1):
            hit_up = high[t + step] >= upper
            hit_dn = low[t + step] <= lower
            if hit_up and hit_dn:
                return -dn_ret - 2 * FEE, step
            if hit_up:
                return up_ret - 2 * FEE, step
            if hit_dn:
                return -dn_ret - 2 * FEE, step
        return fut_ret - 2 * FEE, HORIZON
    raise ValueError(mode)


def backtest(mode, proba, idx, close, high, low, det):
    argmax = proba.argmax(1); conf = proba.max(1)
    up = det["upper_return"]; dn = det["lower_return"]; fut = det["future_return"]
    rets, holds = [], []
    free_at = -1
    for i, t in enumerate(idx):
        if t < free_at:
            continue
        if argmax[i] != BUY or conf[i] <= PROD_THR:
            continue
        if t + HORIZON >= len(close):
            continue
        r, h = long_exit(mode, t, close, high, low, float(up[t]), float(dn[t]), float(fut[t]))
        rets.append(r); holds.append(h)
        free_at = t + int(np.ceil(h))
    return summarize(rets, holds)


def main():
    run_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"sber_h1_hybrid_exit_results_{timestamp}.json"

    print("=" * 72)
    print("LSTM v2 — exit-design 2x2 matrix (BUY, conf>0.50)")
    print("=" * 72)

    if not CACHE_PATH.exists():
        print(f"ERROR: cache not found at {CACHE_PATH}"); sys.exit(1)
    d = np.load(CACHE_PATH)
    proba, idx, close = d["proba"], d["idx"], d["close"]

    sber = TickerData(PRIMARY_TICKER)
    high = sber.df["high"].astype(float).to_numpy()
    low = sber.df["low"].astype(float).to_numpy()
    det = triple_barrier_details(sber.df, TARGET_SPEC)
    print(f"Loaded {len(idx)} cached predictions.\n")

    modes = [
        ("fixed_3h", "fixed 3h          (TP off, stop off)"),
        ("stop_only", "hybrid A: stop    (TP off, stop ON )"),
        ("tp_only",   "hybrid B: take-pft(TP ON,  stop off)"),
        ("full_tb",   "full TB           (TP ON,  stop ON )"),
    ]
    print(f"  {'mode':<36} | {'ret':>9} {'Sharpe':>8} {'win':>6} {'avg/trd':>8} "
          f"{'maxDD':>8} {'trades':>7} {'holdH':>6}")
    print("-" * 100)
    results = {}
    for key, label in modes:
        s = backtest(key, proba, idx, close, high, low, det)
        results[key] = s
        print(f"  {label:<36} | {s['total_return']:>+8.2%} {s['sharpe']:>8.3f} "
              f"{s['win_rate']:>5.1%} {s['avg_trade_ret']:>+7.3%} {s['max_drawdown']:>7.2%} "
              f"{s['n_trades']:>7} {s['mean_hold_h']:>6.2f}")

    # 2x2 summary
    print("\n  Exit-design 2x2 (total return / max DD):")
    print(f"  {'':>10} | {'stop OFF':>18} | {'stop ON':>18}")
    print(f"  {'TP OFF':>10} | {results['fixed_3h']['total_return']:>+8.2%} / "
          f"{results['fixed_3h']['max_drawdown']:>6.2%} | "
          f"{results['stop_only']['total_return']:>+8.2%} / {results['stop_only']['max_drawdown']:>6.2%}")
    print(f"  {'TP ON':>10} | {results['tp_only']['total_return']:>+8.2%} / "
          f"{results['tp_only']['max_drawdown']:>6.2%} | "
          f"{results['full_tb']['total_return']:>+8.2%} / {results['full_tb']['max_drawdown']:>6.2%}")

    best_ret = max(results.items(), key=lambda kv: kv[1]["total_return"])
    best_calmar = max(results.items(),
                      key=lambda kv: kv[1]["total_return"] / (abs(kv[1]["max_drawdown"]) + 1e-9))
    print(f"\n  Best by return:  {best_ret[0]} ({best_ret[1]['total_return']:+.2%}, "
          f"DD {best_ret[1]['max_drawdown']:.2%})")
    print(f"  Best by return/DD: {best_calmar[0]} "
          f"({best_calmar[1]['total_return']:+.2%} / DD {best_calmar[1]['max_drawdown']:.2%} = "
          f"{best_calmar[1]['total_return']/(abs(best_calmar[1]['max_drawdown'])+1e-9):.1f}x)")

    result = {
        "experiment": "sber_h1_hybrid_exit",
        "timestamp": timestamp, "git_branch": "ml-expirement",
        "config": {"fee_one_way": FEE, "prod_threshold": PROD_THR, "horizon": HORIZON,
                   "target": str(TARGET_SPEC.label)},
        "results": results,
        "n_val_predictions": int(len(idx)),
        "total_seconds": round(time.time() - run_start, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTotal time: {time.time()-run_start:.1f}s")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
