"""
LSTM v2 exit-horizon + fine-threshold backtest — cheap iteration via cached predictions.

Motivation:
    LSTM v2's production edge is ~78 ultra-high-conviction BUY calls (conf>0.50) that give
    Sharpe=6.38 with a 1h hold. But the triple-barrier TARGET is defined over 3 hours — the
    signal predicts a move over [t+1, t+3]. Exiting at t+1 may leave return on the table.
    This script tests longer holds (t+2, t+3) and finer thresholds (0.50/0.55/0.60) on the
    SAME signals — the cheapest possible upside lever, with NO retraining.

How it stays cheap:
    LSTM v2 walk-forward predictions (proba + candle idx + close) are collected ONCE
    (~20 min, 4 folds × 3 seeds) and cached to ml/artifacts/lstm_v2_wf_predictions.npz
    (gitignored, regenerable). Every backtest sweep then loads the cache and runs in seconds.

Engine:
    Generalized run_backtest_h(hold_h) with a cooldown: while a position is open
    (entered at t, exits at t+hold_h), later signals are skipped — a realisable single-account
    equity curve. At hold_h=1 this reproduces the original backtest exactly (anchor: Sharpe≈6.38).

Sharpe annualisation:
    Per-trade returns × sqrt(HOURS_PER_YEAR / hold_h). A hold_h-hour trade ties up hold_h×
    the capital-time of a 1h trade, so you can place ~1/hold_h as many independent bets per
    year. Degrades to the original sqrt(1750) at hold_h=1, keeping the 6.38 anchor valid.
    Total return and win rate are exit-horizon-agnostic and reported as primary.

Result saved to:
    ml/docs/research/sber_h1_exit_horizon_results_YYYYMMDD_HHMMSS.json
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
from scripts.sber_multiticker_lstm_research import TickerData, PRIMARY_TICKER
from scripts.sber_multiticker_backtest_research import collect_sber_only

RESULTS_DIR = ML_DIR / "docs" / "research"
CACHE_PATH = ML_DIR / "artifacts" / "lstm_v2_wf_predictions.npz"

HOLD_HORIZONS = [1, 2, 3]
THRESHOLDS = [0.45, 0.50, 0.55, 0.60]
ANCHOR_HOLD, ANCHOR_THR, ANCHOR_SHARPE = 1, 0.50, 6.38


# ── Prediction cache ───────────────────────────────────────────────────────────

def get_lstm_v2_predictions():
    """Load cached LSTM v2 walk-forward predictions, or train+collect and cache them."""
    if CACHE_PATH.exists():
        d = np.load(CACHE_PATH)
        print(f"  Loaded cached predictions from {CACHE_PATH.name} "
              f"({len(d['idx'])} val predictions)")
        return d["proba"], d["idx"], d["close"]

    print("  No cache — collecting LSTM v2 walk-forward predictions (~20 min)...")
    sber = TickerData(PRIMARY_TICKER)
    close = sber.df["close"].astype(float).values
    proba, idx = collect_sber_only(sber)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, proba=proba, idx=idx, close=close)
    print(f"  Cached {len(idx)} predictions to {CACHE_PATH}")
    return proba, idx, close


# ── Generalized backtest engine (configurable hold + cooldown) ─────────────────

def run_backtest_h(close, proba, idx, threshold, hold_h):
    """Backtest with hold_h-hour exit and no overlapping positions (cooldown).

    hold_h=1 reproduces the original engine exactly. Returns summary metrics.
    """
    equity = 1.0
    trade_returns = []
    n_buy = n_sell = 0
    free_at = -1   # capital is committed until this candle index (exclusive)

    for i, t in enumerate(idx):
        if t < free_at:                       # position open → skip (cooldown)
            continue
        if t + hold_h >= len(close):          # no room to exit
            continue
        conf = proba[i].max()
        signal = int(proba[i].argmax())       # 0=SELL, 1=HOLD, 2=BUY
        if conf < threshold or signal == 1:
            continue

        raw_ret = (close[t + hold_h] - close[t]) / close[t]
        if signal == 2:                        # BUY → long
            trade_ret = raw_ret - 2 * FEE
            n_buy += 1
        else:                                  # SELL → short
            trade_ret = -raw_ret - 2 * FEE
            n_sell += 1
        equity *= (1 + trade_ret)
        trade_returns.append(trade_ret)
        free_at = t + hold_h                   # capital free again at exit candle

    n_trades = len(trade_returns)
    if n_trades == 0:
        return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "n_trades": 0, "n_buy": 0, "n_sell": 0,
                "avg_trade_ret": 0.0}

    tr = np.array(trade_returns)
    ann = np.sqrt(HOURS_PER_YEAR / hold_h)     # holding-period-adjusted annualisation
    sharpe = float(tr.mean() / (tr.std() + 1e-9) * ann)

    # equity curve over trades for drawdown
    eq = np.concatenate([[1.0], np.cumprod(1 + tr)])
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.min(eq / np.where(peak < 1e-12, 1.0, peak)) - 1)

    return {
        "sharpe": sharpe,
        "total_return": float(eq[-1] - 1),
        "max_drawdown": max_dd,
        "win_rate": float((tr > 0).mean()),
        "n_trades": n_trades,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "avg_trade_ret": float(tr.mean()),
    }


def main():
    run_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"sber_h1_exit_horizon_results_{timestamp}.json"

    print("=" * 72)
    print("LSTM v2 exit-horizon + fine-threshold backtest")
    print(f"Holds={HOLD_HORIZONS}h | thresholds={THRESHOLDS} | fee={FEE:.2%} one-way")
    print(f"Output: {output_path}")
    print("=" * 72)

    print("\nGetting LSTM v2 predictions...")
    proba, idx, close = get_lstm_v2_predictions()

    # ── Anchor check ───────────────────────────────────────────────────────────
    anchor = run_backtest_h(close, proba, idx, ANCHOR_THR, ANCHOR_HOLD)
    print(f"\nAnchor check (hold=1h, thr=0.50): Sharpe={anchor['sharpe']:.3f} "
          f"ret={anchor['total_return']:+.2%} trades={anchor['n_trades']} "
          f"(documented {ANCHOR_SHARPE})")
    anchor_ok = abs(anchor["sharpe"] - ANCHOR_SHARPE) < 0.1
    print(f"  Anchor {'OK' if anchor_ok else 'MISMATCH — investigate!'}")

    # ── Sweep ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"{'hold':>5} {'thr':>5} | {'Sharpe':>8} {'return':>9} {'avg/trade':>10} "
          f"{'maxDD':>8} {'win':>6} {'trades':>7} {'buy/sell':>9}")
    print("-" * 72)
    grid = {}
    for hold_h in HOLD_HORIZONS:
        for thr in THRESHOLDS:
            r = run_backtest_h(close, proba, idx, thr, hold_h)
            grid[f"h{hold_h}_t{thr:.2f}"] = {"hold_h": hold_h, "threshold": thr, **r}
            print(f"{hold_h:>4}h {thr:>5.2f} | {r['sharpe']:>8.3f} {r['total_return']:>+8.2%} "
                  f"{r['avg_trade_ret']:>+9.3%} {r['max_drawdown']:>7.2%} {r['win_rate']:>5.1%} "
                  f"{r['n_trades']:>7} {r['n_buy']:>4}/{r['n_sell']:<4}")
        print("-" * 72)

    # ── Best configs ───────────────────────────────────────────────────────────
    by_return = max(grid.values(), key=lambda r: r["total_return"])
    by_sharpe = max(grid.values(), key=lambda r: r["sharpe"])
    print(f"\nBest by total return: hold={by_return['hold_h']}h thr={by_return['threshold']:.2f} "
          f"→ {by_return['total_return']:+.2%} (Sharpe={by_return['sharpe']:.3f}, "
          f"{by_return['n_trades']} trades, DD={by_return['max_drawdown']:.2%})")
    print(f"Best by Sharpe:       hold={by_sharpe['hold_h']}h thr={by_sharpe['threshold']:.2f} "
          f"→ Sharpe={by_sharpe['sharpe']:.3f} (ret={by_sharpe['total_return']:+.2%}, "
          f"{by_sharpe['n_trades']} trades)")

    # Direct 1h-vs-3h on the production threshold 0.50
    h1 = grid["h1_t0.50"]; h3 = grid["h3_t0.50"]
    print(f"\n1h vs 3h exit @ conf>0.50:")
    print(f"  1h: ret={h1['total_return']:+.2%} Sharpe={h1['sharpe']:.3f} "
          f"win={h1['win_rate']:.1%} trades={h1['n_trades']}")
    print(f"  3h: ret={h3['total_return']:+.2%} Sharpe={h3['sharpe']:.3f} "
          f"win={h3['win_rate']:.1%} trades={h3['n_trades']}")

    result = {
        "experiment": "sber_h1_exit_horizon_backtest",
        "timestamp": timestamp,
        "git_branch": "ml-expirement",
        "config": {
            "holds": HOLD_HORIZONS, "thresholds": THRESHOLDS, "fee_one_way": FEE,
            "hours_per_year": HOURS_PER_YEAR,
            "sharpe_annualisation": "sqrt(HOURS_PER_YEAR / hold_h)",
            "cooldown": "no overlapping positions",
        },
        "anchor": {"hold_h": ANCHOR_HOLD, "threshold": ANCHOR_THR,
                   "documented_sharpe": ANCHOR_SHARPE, "reproduced": anchor, "ok": bool(anchor_ok)},
        "grid": grid,
        "best_by_return": by_return,
        "best_by_sharpe": by_sharpe,
        "n_val_predictions": int(len(idx)),
        "total_seconds": round(time.time() - run_start, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTotal time: {(time.time()-run_start)/60:.1f} min")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
