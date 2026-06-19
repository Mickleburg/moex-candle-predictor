"""
Per-ticker LSTM validation — does the SBER recipe transfer to GAZP and LKOH?

Architecture context: the ML block serves ml_prediction PER TICKER; the downstream
aggregator/risk_manager builds the cross-ticker portfolio. So the deliverable is a
GOOD MODEL PER TICKER. SBER is validated (WF F1=0.4778; 3h+stop backtest +17.54%,
Sharpe 11.85). This checks whether the SAME recipe (CandleLSTM v2, 14 features,
4-fold walk-forward, seeds [7,42,100], no class weights) yields tradeable edge on
GAZP and LKOH too.

Method (identical to SBER, only the ticker changes):
    * collect_sber_only(TickerData) — ticker-agnostic walk-forward predictor (4 folds x
      3 seeds, seed-averaged proba) reused from the multi-ticker backtest harness.
    * WF macro-F1 from those predictions.
    * Production backtest = the winning rule: BUY when conf>0.50, hold 3h with a
      lower-barrier stop-loss, no take-profit (hybrid 'stop_only'), plus fixed 3h for ref.
    * Predictions cached per ticker to ml/artifacts/lstm_v2_wf_predictions_<ticker>.npz.

SBER is loaded from its existing cache (no retraining). GAZP + LKOH are trained (~20 min each).

Result saved to: ml/docs/research/per_ticker_lstm_results_YYYYMMDD_HHMMSS.json
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
from sklearn.metrics import f1_score

from scripts.sber_multiticker_lstm_research import TickerData, TARGET_SPEC
from scripts.sber_multiticker_backtest_research import collect_sber_only
from scripts.sber_lstm_hybrid_exit import backtest as hybrid_backtest
from src.nlp.targets import triple_barrier_details

RESULTS_DIR = ML_DIR / "docs" / "research"
ARTIFACTS_DIR = ML_DIR / "artifacts"
SBER_CACHE = ARTIFACTS_DIR / "lstm_v2_wf_predictions.npz"

TICKERS = ["SBER", "GAZP", "LKOH"]
SBER_BASELINE_F1 = 0.4778


def get_predictions(ticker: str, td: TickerData):
    """Load cached per-ticker predictions or train+collect and cache them."""
    cache = SBER_CACHE if ticker == "SBER" else ARTIFACTS_DIR / f"lstm_v2_wf_predictions_{ticker.lower()}.npz"
    if cache.exists():
        d = np.load(cache)
        print(f"  [{ticker}] loaded cache ({len(d['idx'])} preds)")
        return d["proba"], d["idx"]
    print(f"  [{ticker}] no cache — training walk-forward (4 folds x 3 seeds, ~20 min)...")
    proba, idx = collect_sber_only(td)
    close = td.df["close"].astype(float).to_numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, proba=proba, idx=idx, close=close)
    print(f"  [{ticker}] cached {len(idx)} preds to {cache.name}")
    return proba, idx


def evaluate_ticker(ticker: str):
    td = TickerData(ticker)
    proba, idx = get_predictions(ticker, td)

    # Walk-forward F1
    y_true = td.labels[idx]
    macro_f1 = float(f1_score(y_true, proba.argmax(1), average="macro", zero_division=0))
    per_class = f1_score(y_true, proba.argmax(1), average=None, labels=[0, 1, 2], zero_division=0)

    # Production backtest (BUY/conf>0.50): hybrid stop-loss (the winning rule) + fixed 3h ref
    close = td.df["close"].astype(float).to_numpy()
    high = td.df["high"].astype(float).to_numpy()
    low = td.df["low"].astype(float).to_numpy()
    det = triple_barrier_details(td.df, TARGET_SPEC)
    bt_stop = hybrid_backtest("stop_only", proba, idx, close, high, low, det)
    bt_3h = hybrid_backtest("fixed_3h", proba, idx, close, high, low, det)

    return {
        "ticker": ticker,
        "n_val_predictions": int(len(idx)),
        "wf_macro_f1": macro_f1,
        "wf_sell_f1": float(per_class[0]),
        "wf_hold_f1": float(per_class[1]),
        "wf_buy_f1": float(per_class[2]),
        "backtest_stop_loss": bt_stop,
        "backtest_fixed_3h": bt_3h,
    }


def main():
    run_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"per_ticker_lstm_results_{timestamp}.json"

    print("=" * 76)
    print("Per-ticker LSTM validation — does the SBER recipe transfer to GAZP/LKOH?")
    print(f"Production rule = BUY conf>0.50, hold 3h + lower-barrier stop, no take-profit")
    print(f"Output: {output_path}")
    print("=" * 76)

    results = []
    for tk in TICKERS:
        print(f"\n--- {tk} ---")
        results.append(evaluate_ticker(tk))

    print(f"\n{'='*76}")
    print("SUMMARY")
    print(f"{'='*76}")
    print(f"  {'ticker':>6} | {'WF F1':>7} {'S/H/B F1':>17} | "
          f"{'stop: ret':>9} {'Sharpe':>7} {'win':>6} {'DD':>7} {'trades':>7}")
    print("-" * 76)
    for r in results:
        b = r["backtest_stop_loss"]
        print(f"  {r['ticker']:>6} | {r['wf_macro_f1']:>7.4f} "
              f"{r['wf_sell_f1']:>5.3f}/{r['wf_hold_f1']:.3f}/{r['wf_buy_f1']:.3f} | "
              f"{b['total_return']:>+8.2%} {b['sharpe']:>7.2f} {b['win_rate']:>5.1%} "
              f"{b['max_drawdown']:>6.2%} {b['n_trades']:>7}")

    # Verdict — honest bar: real edge needs Sharpe>2, win>50%, return>2%, enough trades.
    # (A near-zero Sharpe with win<50% is noise, not edge, even if total return is slightly >0.)
    tradeable = [r["ticker"] for r in results
                 if r["backtest_stop_loss"]["sharpe"] > 2.0
                 and r["backtest_stop_loss"]["win_rate"] > 0.50
                 and r["backtest_stop_loss"]["total_return"] > 0.02
                 and r["backtest_stop_loss"]["n_trades"] >= 20]
    print(f"\n  Tickers with REAL tradeable edge (Sharpe>2, win>50%, ret>2%): {tradeable}")

    result = {
        "experiment": "per_ticker_lstm",
        "timestamp": timestamp,
        "git_branch": "ml-expirement",
        "config": {
            "tickers": TICKERS, "target": str(TARGET_SPEC.label),
            "production_rule": "BUY conf>0.50, hold 3h + lower-barrier stop, no take-profit",
            "sber_baseline_f1": SBER_BASELINE_F1,
        },
        "results": results,
        "tradeable_tickers": tradeable,
        "total_seconds": round(time.time() - run_start, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Total time: {(time.time()-run_start)/60:.1f} min")
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
