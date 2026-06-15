"""Robustness (#4) for an orthogonal-feature ticker on its cached walk-forward predictions.

Usage:
    python ml/scripts/ortho_robustness.py --ticker LKOH --groups commodity,market

Reads ml/artifacts/orth_wf_preds_<ticker>_<groups>.npz (from sber_orthogonal_research.py --cache),
reconstructs the production-rule trades (BUY conf>0.50, 3h+stop, skip weekends) and runs:
  bootstrap CI on total return & Sharpe, fee stress, random-selection baseline.
Same engine as sber_edge_analysis (#4), parametrised by ticker.
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

from scripts.sber_multiticker_lstm_research import TickerData, TARGET_SPEC
from scripts.sber_edge_analysis import long_stop_return, summarize, sharpe
from src.nlp.targets import triple_barrier_details

HORIZON = 3
BUY = 2
RNG = np.random.default_rng(42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--groups", required=True)
    args = ap.parse_args()
    ticker = args.ticker.upper()
    tag = "_".join(g.strip() for g in args.groups.split(",") if g.strip())

    cache = ML_DIR / "artifacts" / f"orth_wf_preds_{ticker.lower()}_{tag}.npz"
    if not cache.exists():
        print(f"ERROR: cache not found: {cache}"); sys.exit(1)
    d = np.load(cache); proba, idx = d["proba"], d["idx"]

    td = TickerData(ticker)
    close = td.df["close"].astype(float).to_numpy()
    high = td.df["high"].astype(float).to_numpy()
    low = td.df["low"].astype(float).to_numpy()
    begin = pd.to_datetime(td.df["begin"])
    det = triple_barrier_details(td.df, TARGET_SPEC)
    up, dn, fut = det["upper_return"], det["lower_return"], det["future_return"]
    argmax, conf = proba.argmax(1), proba.max(1)

    print("=" * 70)
    print(f"Orthogonal robustness — {ticker} groups={tag}")
    print("=" * 70)

    # Production-rule trades (BUY conf>0.50, 3h+stop, skip weekends)
    rets, holds = [], []
    free_at = -1
    for i, t in enumerate(idx):
        if t < free_at or argmax[i] != BUY or conf[i] <= 0.50:
            continue
        if begin.iloc[t].dayofweek >= 5 or t + HORIZON >= len(close):
            continue
        r, h, _ = long_stop_return(int(t), close, high, low, float(up[t]), float(dn[t]), float(fut[t]))
        rets.append(r); holds.append(h); free_at = t + int(np.ceil(h))
    rets = np.array(rets); holds = np.array(holds)
    base = summarize(rets, holds)
    print(f"\nTrades: n={base['n']} ret={base['total_return']:+.2%} Sharpe={base['sharpe']:.2f} "
          f"win={base['win_rate']:.1%} mean/trade={base['mean_ret']:+.3%}")

    # Bootstrap
    B = 20000; mh = float(holds.mean()) if len(holds) else 3.0
    bt_tot = np.array([np.prod(1 + RNG.choice(rets, len(rets), replace=True)) - 1 for _ in range(B)])
    bt_shp = np.array([sharpe(RNG.choice(rets, len(rets), replace=True), mh) for _ in range(B)])
    boot = {"return_p05": float(np.percentile(bt_tot, 5)), "return_p50": float(np.percentile(bt_tot, 50)),
            "return_p95": float(np.percentile(bt_tot, 95)), "p_profit": float((bt_tot > 0).mean()),
            "sharpe_p05": float(np.percentile(bt_shp, 5)), "sharpe_p_positive": float((bt_shp > 0).mean())}
    print(f"  bootstrap: return p05={boot['return_p05']:+.2%} p50={boot['return_p50']:+.2%} "
          f"p95={boot['return_p95']:+.2%} P(profit)={boot['p_profit']:.1%}")
    print(f"             Sharpe p05={boot['sharpe_p05']:.2f} P(>0)={boot['sharpe_p_positive']:.1%}")

    # Fee stress
    print("  fee stress (one-way):")
    fee_stress = {}
    for fee in [0.0005, 0.001, 0.0015, 0.002]:
        r2, h2, fa = [], [], -1
        for i, t in enumerate(idx):
            if t < fa or argmax[i] != BUY or conf[i] <= 0.50:
                continue
            if begin.iloc[t].dayofweek >= 5 or t + HORIZON >= len(close):
                continue
            rr, hh, _ = long_stop_return(int(t), close, high, low, float(up[t]), float(dn[t]), float(fut[t]), fee=fee)
            r2.append(rr); h2.append(hh); fa = t + int(np.ceil(hh))
        s = summarize(r2, h2); fee_stress[f"{fee:.4f}"] = s
        print(f"     fee={fee:.2%}: ret={s['total_return']:+.2%} Sharpe={s['sharpe']:.2f} n={s['n']}")

    # Random-selection baseline
    all_r = []
    for t in idx:
        if t + HORIZON >= len(close):
            continue
        rr, _, _ = long_stop_return(int(t), close, high, low, float(up[t]), float(dn[t]), float(fut[t]))
        all_r.append(rr)
    all_r = np.array(all_r); n = len(rets)
    rand = np.array([RNG.choice(all_r, n, replace=False).mean() for _ in range(B)])
    pctl = float((rand < rets.mean()).mean())
    print(f"  random baseline: all-candle mean={all_r.mean():+.3%} model mean={rets.mean():+.3%} "
          f"model beats {pctl:.1%} of random-{n}")

    result = {"experiment": "ortho_robustness", "ticker": ticker, "groups": tag,
              "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
              "base": base, "bootstrap": boot, "fee_stress": fee_stress,
              "random_model_percentile": pctl, "n_trades": int(n)}
    out = ML_DIR / "docs" / "research" / f"{ticker.lower()}_ortho_robustness_{result['timestamp']}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
