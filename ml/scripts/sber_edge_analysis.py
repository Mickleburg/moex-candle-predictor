"""
SBER edge analysis — Stage 1 of the roadmap: #3 (where the edge lives) + #4 (robustness).

All on cached LSTM v2 walk-forward predictions (ml/artifacts/lstm_v2_wf_predictions.npz),
no retraining. Production rule: BUY when conf>0.50, hold 3h with a lower-barrier stop-loss,
no take-profit (the winning 'stop_only' rule). Validation periods only.

#3 — WHERE does the edge live? Decompose the ~50 winning-rule trades by:
    hour-of-day (UTC; MSK = UTC+3), day-of-week, volatility tercile at entry,
    confidence bucket, and calendar year. Reveals whether the edge clusters (a free filter).

#4 — Is the edge real, not luck? On the same trades:
    * bootstrap (B=20000) the trade returns -> CI on total return and Sharpe, P(profit).
    * per-year decomposition -> is the edge spread out or concentrated?
    * fee stress -> at what one-way cost does the edge die?
    * random-selection baseline -> does the MODEL's pick beat a random 50 long entries?

Result saved to: ml/docs/research/sber_edge_analysis_results_YYYYMMDD_HHMMSS.json
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

from scripts.sber_multiticker_lstm_research import TickerData, PRIMARY_TICKER, TARGET_SPEC
from scripts.sber_backtest_research import HOURS_PER_YEAR
from src.nlp.targets import triple_barrier_details

RESULTS_DIR = ML_DIR / "docs" / "research"
CACHE_PATH = ML_DIR / "artifacts" / "lstm_v2_wf_predictions.npz"

BUY = 2
PROD_THR = 0.50
HORIZON = 3
FEE = 0.0005
RNG = np.random.default_rng(42)


def long_stop_return(t, close, high, low, up_ret, dn_ret, fut_ret, fee=FEE):
    """3h hold + lower-barrier stop, no take-profit. Returns (ret, hold_h, outcome)."""
    lower = close[t] * (1.0 - dn_ret)
    for step in range(1, HORIZON + 1):
        if low[t + step] <= lower:
            return -dn_ret - 2 * fee, step, "stop"
    return fut_ret - 2 * fee, HORIZON, "hold_to_3h"


def sharpe(trade_rets, mean_hold_h):
    tr = np.asarray(trade_rets, float)
    if len(tr) < 2 or tr.std() < 1e-12:
        return 0.0
    return float(tr.mean() / tr.std() * np.sqrt(HOURS_PER_YEAR / max(1.0, mean_hold_h)))


def collect_trades(proba, idx, close, high, low, det, fee=FEE):
    """Reconstruct production-rule trades with per-trade context."""
    argmax = proba.argmax(1); conf = proba.max(1)
    up = det["upper_return"]; dn = det["lower_return"]; fut = det["future_return"]
    vol = det["past_volatility"]
    trades = []
    free_at = -1
    for i, t in enumerate(idx):
        if t < free_at:
            continue
        if argmax[i] != BUY or conf[i] <= PROD_THR:
            continue
        if t + HORIZON >= len(close):
            continue
        r, h, oc = long_stop_return(t, close, high, low, float(up[t]), float(dn[t]), float(fut[t]), fee)
        trades.append({"t": int(t), "conf": float(conf[i]), "ret": float(r), "hold_h": int(h),
                       "outcome": oc, "vol": float(vol[t])})
        free_at = t + int(np.ceil(h))
    return trades


def summarize(rets, holds):
    tr = np.asarray(rets, float)
    if len(tr) == 0:
        return {"n": 0, "total_return": 0.0, "win_rate": 0.0, "mean_ret": 0.0, "sharpe": 0.0}
    eq = np.cumprod(1 + tr)
    return {"n": int(len(tr)), "total_return": float(eq[-1] - 1), "win_rate": float((tr > 0).mean()),
            "mean_ret": float(tr.mean()), "sharpe": sharpe(tr, float(np.mean(holds)) if len(holds) else 3.0)}


def grp(trades, keyfn, label):
    out = {}
    for tr in trades:
        out.setdefault(keyfn(tr), []).append(tr)
    rows = {}
    for k in sorted(out):
        g = out[k]
        rows[str(k)] = {"n": len(g), "win_rate": float(np.mean([x["ret"] > 0 for x in g])),
                        "total_return": float(np.prod([1 + x["ret"] for x in g]) - 1),
                        "mean_ret": float(np.mean([x["ret"] for x in g]))}
    return rows


def main():
    run_start = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"sber_edge_analysis_results_{ts}.json"

    print("=" * 74)
    print("SBER edge analysis — #3 where the edge lives + #4 robustness")
    print("Rule: BUY conf>0.50, hold 3h + lower-barrier stop, no take-profit")
    print("=" * 74)

    if not CACHE_PATH.exists():
        print(f"ERROR: cache not found at {CACHE_PATH}"); sys.exit(1)
    d = np.load(CACHE_PATH)
    proba, idx, close = d["proba"], d["idx"], d["close"]

    sber = TickerData(PRIMARY_TICKER)
    high = sber.df["high"].astype(float).to_numpy()
    low = sber.df["low"].astype(float).to_numpy()
    begin = pd.to_datetime(sber.df["begin"])
    det = triple_barrier_details(sber.df, TARGET_SPEC)

    trades = collect_trades(proba, idx, close, high, low, det)
    rets = np.array([t["ret"] for t in trades])
    holds = np.array([t["hold_h"] for t in trades])
    base = summarize(rets, holds)
    print(f"\nProduction trades: n={base['n']}  total_return={base['total_return']:+.2%}  "
          f"win={base['win_rate']:.1%}  Sharpe={base['sharpe']:.2f}  mean/trade={base['mean_ret']:+.3%}")

    # attach calendar context
    for tr in trades:
        b = begin.iloc[tr["t"]]
        tr["hour"] = int(b.hour); tr["dow"] = int(b.dayofweek); tr["year"] = int(b.year)
    vols = np.array([t["vol"] for t in trades])
    v_lo, v_hi = np.quantile(vols, [1/3, 2/3])
    for tr in trades:
        tr["vol_bucket"] = "low" if tr["vol"] <= v_lo else ("high" if tr["vol"] > v_hi else "mid")
        c = tr["conf"]
        tr["conf_bucket"] = "0.50-0.55" if c <= 0.55 else ("0.55-0.60" if c <= 0.60 else "0.60+")

    # ── #3 decomposition ───────────────────────────────────────────────────────
    print("\n#3  WHERE the edge lives")
    decomp = {
        "by_hour_utc": grp(trades, lambda t: t["hour"], "hour"),
        "by_dow": grp(trades, lambda t: t["dow"], "dow"),
        "by_vol_bucket": grp(trades, lambda t: t["vol_bucket"], "vol"),
        "by_conf_bucket": grp(trades, lambda t: t["conf_bucket"], "conf"),
        "by_year": grp(trades, lambda t: t["year"], "year"),
    }
    for name, rows in decomp.items():
        print(f"  {name}:")
        for k, r in rows.items():
            print(f"     {k:>10}: n={r['n']:>3}  win={r['win_rate']:>5.1%}  ret={r['total_return']:>+7.2%}  "
                  f"mean={r['mean_ret']:>+6.3%}")

    # ── #4 robustness ──────────────────────────────────────────────────────────
    print("\n#4  ROBUSTNESS")

    # (a) bootstrap
    B = 20000
    boot_total, boot_sharpe = [], []
    mh = float(holds.mean())
    for _ in range(B):
        s = RNG.choice(rets, size=len(rets), replace=True)
        boot_total.append(np.prod(1 + s) - 1)
        boot_sharpe.append(sharpe(s, mh))
    boot_total = np.array(boot_total); boot_sharpe = np.array(boot_sharpe)
    bootstrap = {
        "total_return": {"p05": float(np.percentile(boot_total, 5)), "p50": float(np.percentile(boot_total, 50)),
                         "p95": float(np.percentile(boot_total, 95)), "p_positive": float((boot_total > 0).mean())},
        "sharpe": {"p05": float(np.percentile(boot_sharpe, 5)), "p50": float(np.percentile(boot_sharpe, 50)),
                   "p95": float(np.percentile(boot_sharpe, 95)), "p_positive": float((boot_sharpe > 0).mean())},
    }
    print(f"  bootstrap total_return: p05={bootstrap['total_return']['p05']:+.2%} "
          f"p50={bootstrap['total_return']['p50']:+.2%} p95={bootstrap['total_return']['p95']:+.2%} "
          f"P(profit)={bootstrap['total_return']['p_positive']:.1%}")
    print(f"  bootstrap Sharpe:       p05={bootstrap['sharpe']['p05']:.2f} "
          f"p50={bootstrap['sharpe']['p50']:.2f} p95={bootstrap['sharpe']['p95']:.2f} "
          f"P(>0)={bootstrap['sharpe']['p_positive']:.1%}")

    # (b) fee stress
    print("  fee stress (one-way):")
    fee_stress = {}
    for fee in [0.0005, 0.001, 0.0015, 0.002, 0.003]:
        tr2 = collect_trades(proba, idx, close, high, low, det, fee=fee)
        s = summarize([x["ret"] for x in tr2], [x["hold_h"] for x in tr2])
        fee_stress[f"{fee:.4f}"] = s
        print(f"     fee={fee:.2%}: ret={s['total_return']:>+7.2%}  Sharpe={s['sharpe']:>5.2f}  "
              f"win={s['win_rate']:.1%}  n={s['n']}")

    # (c) random-selection baseline: long-stop return for EVERY val candle, then random-50 subsets
    all_rets = []
    for t in idx:
        if t + HORIZON >= len(close):
            continue
        r, _, _ = long_stop_return(int(t), close, high, low,
                                   float(det["upper_return"][t]), float(det["lower_return"][t]),
                                   float(det["future_return"][t]))
        all_rets.append(r)
    all_rets = np.array(all_rets)
    n_trades = len(rets)
    rand_means = np.array([RNG.choice(all_rets, size=n_trades, replace=False).mean() for _ in range(B)])
    model_mean = rets.mean()
    pctl = float((rand_means < model_mean).mean())
    random_baseline = {
        "all_val_candles_mean_ret": float(all_rets.mean()),
        "model_selected_mean_ret": float(model_mean),
        "random_50_mean_p50": float(np.percentile(rand_means, 50)),
        "random_50_mean_p95": float(np.percentile(rand_means, 95)),
        "model_percentile_vs_random": pctl,
    }
    print(f"  random baseline: all-candle mean/trade={all_rets.mean():+.3%}  "
          f"model mean/trade={model_mean:+.3%}  "
          f"model beats {pctl:.1%} of random-{n_trades} picks")

    # (d) per-year already in decomp; surface as robustness too
    result = {
        "experiment": "sber_edge_analysis", "timestamp": ts, "git_branch": "ml-expirement",
        "rule": "BUY conf>0.50, hold 3h + lower-barrier stop, no take-profit",
        "base": base,
        "decomposition": decomp,
        "robustness": {"bootstrap": bootstrap, "fee_stress": fee_stress,
                       "random_baseline": random_baseline,
                       "per_year": decomp["by_year"]},
        "n_trades": int(n_trades),
        "vol_terciles": {"low<=": float(v_lo), "high>": float(v_hi)},
        "total_seconds": round(time.time() - run_start, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTotal time: {time.time()-run_start:.1f}s")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
