"""H1 — Cross-sectional / market-neutral baseline (NO ML).

Hypothesis (docs/RESEARCH_HYPOTHESES.md, H1): relative strength (will ticker X
out/under-perform the universe) is predictable where absolute 1H direction is not,
because the common market factor (beta) — which killed the directional approach — is
removed by going long the strongest / short the weakest names.

This script answers the CHEAP first question before any ML: does pure cross-sectional
MOMENTUM or REVERSAL on price alone carry a market-neutral edge that survives fees?
If even this shows nothing, an ML ranker won't save it. If it shows something, ML is
worth building to sharpen the ranking.

Method (honest, no lookahead, deployment-style):
  * Pivot close prices into a (time x ticker) matrix over the universe (intersection of
    timestamps so the cross-section is comparable at every t).
  * Non-overlapping rebalance every h bars. At each rebalance t:
        signal  = past return over lookback L:  close[t]/close[t-L] - 1   (PAST ONLY)
        ranks   = cross-sectional rank of signal across the universe
        MOMENTUM portfolio = long top-k (highest signal) / short bottom-k
        REVERSAL portfolio = the opposite
        realized = forward return over [t, t+h]: close[t+h]/close[t] - 1   (label only)
        port_ret = mean(realized over long) - mean(realized over short)    (market-neutral)
  * Fees: conservative full round-trip each period. gross exposure = 2 (1 long + 1 short),
    open+close => 4 units traded per period; cost = 4 * fee_oneway per period.
  * Report GROSS (pre-fee) and NET, split IN-SAMPLE (<= 2024) vs FORWARD (2025-2026),
    over a grid of (L, h, k). No cherry-picking — the whole grid is printed.

Success gate (H1): a (L, h) where market-neutral NET return is robustly > 0 in BOTH
in-sample AND forward. Otherwise: price-only intraday cross-section has no edge ->
pivot to news (H2) or a longer/daily horizon.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from src.data.load import to_moscow_time  # noqa: E402

DATA_RAW = REPO_ROOT / "data" / "raw"
REPORT_DIR = ML_DIR / "docs" / "research"

UNIVERSE = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK",
            "TATN", "MGNT", "MTSS", "SNGS", "CHMF", "ALRS"]

FEE_ONEWAY = 0.0005          # 5 bps one-way
FORWARD_START = pd.Timestamp("2025-01-01", tz="Europe/Moscow")

# Grid
LOOKBACKS = [6, 24, 72, 120]
HOLDS = [3, 6, 24, 72]
KS = [2, 3]
BARS_PER_YEAR = 247 * 9      # ~247 trading days, ~9 1H bars/day on MOEX


def load_close_matrix(universe: list[str]) -> pd.DataFrame:
    """Return a (time x ticker) close-price matrix on the timestamp intersection."""
    series = {}
    for tkr in universe:
        files = sorted(DATA_RAW.glob(f"{tkr}_1H_*.parquet"))
        if not files:
            print(f"  WARN: no parquet for {tkr}, skipping")
            continue
        df = pd.read_parquet(files[-1])
        df.columns = [c.lower() for c in df.columns]
        begin = to_moscow_time(df["begin"])
        s = pd.Series(df["close"].to_numpy(dtype=float), index=begin)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        series[tkr] = s
    mat = pd.DataFrame(series)
    before = len(mat)
    mat = mat.dropna(how="any")          # intersection: every ticker present
    print(f"  universe={list(mat.columns)}")
    print(f"  aligned rows: {len(mat)} (from {before} union), "
          f"{mat.index.min()} .. {mat.index.max()}")
    return mat


def backtest(close: pd.DataFrame, L: int, h: int, k: int):
    """Run momentum & reversal market-neutral backtests. Returns per-period frames."""
    cols = close.columns
    n = len(close)
    rows = []
    t = L
    while t + h < n:
        past = close.iloc[t].to_numpy() / close.iloc[t - L].to_numpy() - 1.0
        fwd = close.iloc[t + h].to_numpy() / close.iloc[t].to_numpy() - 1.0
        if not (np.all(np.isfinite(past)) and np.all(np.isfinite(fwd))):
            t += h
            continue
        order = np.argsort(past)                  # ascending: losers first
        bottom = order[:k]                        # weakest past performers
        top = order[-k:]                          # strongest past performers
        mom = fwd[top].mean() - fwd[bottom].mean()       # long winners / short losers
        rev = fwd[bottom].mean() - fwd[top].mean()       # long losers / short winners
        rows.append((close.index[t], mom, rev))
        t += h
    df = pd.DataFrame(rows, columns=["t", "mom_gross", "rev_gross"]).set_index("t")
    fee = 4.0 * FEE_ONEWAY                          # full round-trip, gross=2
    df["mom_net"] = df["mom_gross"] - fee
    df["rev_net"] = df["rev_gross"] - fee
    return df


def summarize(rets: pd.Series, h: int) -> dict:
    rets = rets.dropna()
    if len(rets) == 0:
        return {"n": 0}
    periods_per_year = BARS_PER_YEAR / h
    mean = float(rets.mean())
    std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (mean / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    cum = float((1.0 + rets).prod() - 1.0)
    ann = (1.0 + mean) ** periods_per_year - 1.0
    return {
        "n": int(len(rets)),
        "mean_per_period": round(mean, 6),
        "cum_return": round(cum, 4),
        "ann_return": round(float(ann), 4),
        "sharpe": round(float(sharpe), 2),
        "win_rate": round(float((rets > 0).mean()), 3),
    }


def run_daily(close_1h: pd.DataFrame) -> None:
    """Daily cross-sectional momentum/reversal — longer horizon, fees amortized.

    Classic equities cross-sectional momentum lives at the daily/weekly scale. This
    resamples the 1H matrix to daily last-close and repeats the long/short test in DAYS.
    Prints GROSS and NET so we can tell 'no signal' from 'signal killed by fees'.
    """
    daily = close_1h.resample("1D").last().dropna(how="any")
    print(f"\n--- DAILY cross-section: {len(daily)} days "
          f"{daily.index.min().date()} .. {daily.index.max().date()} ---")
    fwd_start = FORWARD_START
    fee = 4.0 * FEE_ONEWAY
    rows = []
    for L in [5, 10, 20, 60]:
        for h in [1, 5, 10, 20]:
            for k in [2, 3]:
                recs = []
                t = L
                n = len(daily)
                while t + h < n:
                    past = daily.iloc[t].to_numpy() / daily.iloc[t - L].to_numpy() - 1
                    fwd = daily.iloc[t + h].to_numpy() / daily.iloc[t].to_numpy() - 1
                    if np.all(np.isfinite(past)) and np.all(np.isfinite(fwd)):
                        order = np.argsort(past)
                        bot, top = order[:k], order[-k:]
                        mom = fwd[top].mean() - fwd[bot].mean()
                        recs.append((daily.index[t], mom))
                    t += h
                if not recs:
                    continue
                d = pd.DataFrame(recs, columns=["t", "mom"]).set_index("t")
                isd = d[d.index < fwd_start]["mom"]
                fwd = d[d.index >= fwd_start]["mom"]
                rows.append({
                    "L": L, "h": h, "k": k,
                    "all_g": float((1 + d["mom"]).prod() - 1),
                    "all_n": float((1 + d["mom"] - fee).prod() - 1),
                    "is_n": float((1 + isd - fee).prod() - 1) if len(isd) else 0.0,
                    "fwd_n": float((1 + fwd - fee).prod() - 1) if len(fwd) else 0.0,
                    "fwd_g": float((1 + fwd).prod() - 1) if len(fwd) else 0.0,
                    "n_fwd": int(len(fwd)),
                })
    rows.sort(key=lambda r: r["fwd_n"], reverse=True)
    print(f"{'L':>3} {'h':>3} {'k':>2} | {'ALL gross':>9} {'ALL net':>8} | "
          f"{'IS net':>8} | {'FWD gross':>9} {'FWD net':>8} {'n_fwd':>5}  (momentum)")
    for r in rows:
        print(f"{r['L']:>3} {r['h']:>3} {r['k']:>2} | {r['all_g']:>9.4f} {r['all_n']:>8.4f} | "
              f"{r['is_n']:>8.4f} | {r['fwd_g']:>9.4f} {r['fwd_n']:>8.4f} {r['n_fwd']:>5}")
    robust = [r for r in rows if r["is_n"] > 0 and r["fwd_n"] > 0 and r["n_fwd"] >= 10]
    print(f"DAILY momentum configs NET>0 in BOTH IS and forward: {len(robust)}")


def main() -> int:
    print("Loading universe close matrix...")
    close = load_close_matrix(UNIVERSE)
    if close.shape[1] < 5:
        print("Universe too small (<5 tickers) — download more first.")
        return 1

    is_mask = close.index < FORWARD_START
    fwd_mask = close.index >= FORWARD_START
    print(f"  in-sample rows: {is_mask.sum()}  forward rows: {fwd_mask.sum()}")

    results = []
    for L in LOOKBACKS:
        for h in HOLDS:
            for k in KS:
                df = backtest(close, L, h, k)
                if df.empty:
                    continue
                is_df = df[df.index < FORWARD_START]
                fw_df = df[df.index >= FORWARD_START]
                for strat in ("mom", "rev"):
                    rec = {
                        "strategy": strat, "L": L, "h": h, "k": k,
                        "all_gross": summarize(df[f"{strat}_gross"], h),
                        "all_net": summarize(df[f"{strat}_net"], h),
                        "is_net": summarize(is_df[f"{strat}_net"], h),
                        "fwd_net": summarize(fw_df[f"{strat}_net"], h),
                    }
                    results.append(rec)

    # ---- print a compact, honest table (sorted by forward net Sharpe) ----
    def fwd_sharpe(r):
        return r["fwd_net"].get("sharpe", 0.0)

    results.sort(key=fwd_sharpe, reverse=True)
    print("\n=== H1 cross-sectional baseline (market-neutral, net of fees) ===")
    print(f"{'strat':5} {'L':>3} {'h':>3} {'k':>2} | "
          f"{'ALL net cum':>11} {'ALL Sh':>6} {'ALL win':>7} | "
          f"{'IS net cum':>10} {'IS Sh':>6} | {'FWD net cum':>11} {'FWD Sh':>6} {'FWD win':>7}")
    for r in results:
        a, isd, fw = r["all_net"], r["is_net"], r["fwd_net"]
        print(f"{r['strategy']:5} {r['L']:>3} {r['h']:>3} {r['k']:>2} | "
              f"{a.get('cum_return',0):>11.4f} {a.get('sharpe',0):>6.2f} {a.get('win_rate',0):>7.3f} | "
              f"{isd.get('cum_return',0):>10.4f} {isd.get('sharpe',0):>6.2f} | "
              f"{fw.get('cum_return',0):>11.4f} {fw.get('sharpe',0):>6.2f} {fw.get('win_rate',0):>7.3f}")

    # ---- verdict heuristic ----
    robust = [r for r in results
              if r["is_net"].get("cum_return", -1) > 0
              and r["fwd_net"].get("cum_return", -1) > 0
              and r["fwd_net"].get("n", 0) >= 20]
    print(f"\nConfigs with NET cum>0 in BOTH in-sample AND forward (n_fwd>=20): {len(robust)}")
    for r in robust[:10]:
        print(f"  {r['strategy']} L={r['L']} h={r['h']} k={r['k']}: "
              f"IS cum={r['is_net']['cum_return']:.4f} | FWD cum={r['fwd_net']['cum_return']:.4f} "
              f"Sh={r['fwd_net']['sharpe']:.2f}")

    # Gross check: is the raw signal absent, or present but fee-killed? (best ALL-gross sharpe)
    by_g = sorted(results, key=lambda r: r["all_gross"].get("sharpe", 0), reverse=True)
    print("\nTop-3 by ALL gross Sharpe (pre-fee signal strength):")
    for r in by_g[:3]:
        g, fg = r["all_gross"], r["fwd_net"]
        print(f"  {r['strategy']} L={r['L']} h={r['h']} k={r['k']}: "
              f"ALL gross cum={g.get('cum_return',0):.4f} Sh={g.get('sharpe',0):.2f} | "
              f"FWD net cum={fg.get('cum_return',0):.4f}")

    run_daily(close)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"xsec_h1_results_{stamp}.json"
    out.write_text(json.dumps({
        "universe": list(close.columns),
        "fee_oneway": FEE_ONEWAY, "forward_start": str(FORWARD_START),
        "rows": int(len(close)), "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
