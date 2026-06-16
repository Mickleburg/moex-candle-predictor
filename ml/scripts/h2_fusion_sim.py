"""H2/H3 fusion harness — does adding news features to the price ranker help, on the gate?

Turnkey comparison: run the SAME deployment-sim gate (rolling-retrain Ridge, beta_residual
target, fees, fresh forward) with QUANT-ONLY features vs QUANT+NEWS, and report the delta.
This is the H2 test (news adds signal over price) and, with a real fused model, the H3 test
(early fusion). When the LLM/news block drops its baseline table on the decision grid, point
--news at it; with no table this runs price-only and a synthetic self-test that proves the
harness DETECTS a news contribution (positive delta for signal, ~0 for noise).

News table schema (LLM block output): tidy CSV/parquet [ticker, as_of, sentiment, impact_score, ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from src.features.cross_sectional import load_panels, relative_target  # noqa: E402
from src.features.decision_grid import NEWS_WINDOW_DAYS  # noqa: E402
from scripts.xsec_deployment_sim import build_feature_panels, deployment_sim  # noqa: E402


def news_to_panels(news: pd.DataFrame, index: pd.DatetimeIndex, cols, columns,
                   window_days: int = NEWS_WINDOW_DAYS) -> dict[str, pd.DataFrame]:
    """No-lookahead news feature panels (time x ticker): latest value with as_of<=t in window."""
    out = {f"news_{c}": pd.DataFrame(0.0, index=index, columns=columns) for c in cols}
    if news is None or len(news) == 0:
        return out
    news = news.sort_values("as_of")
    win = pd.Timedelta(days=window_days)
    for tkr, g in news.groupby("ticker"):
        if tkr not in columns:
            continue
        g = g.set_index("as_of")
        union = g.index.union(index).sort_values()
        # carried news value AND its timestamp, then zero out entries older than the window
        age_ts = pd.Series(g.index, index=g.index).reindex(union).ffill().reindex(index)
        stale = (index - age_ts) > win        # True where last news is older than window
        for c in cols:
            vals = g[c].reindex(union).ffill().reindex(index)
            vals = vals.where(~stale, 0.0).fillna(0.0)
            out[f"news_{c}"][tkr] = vals.to_numpy()
    return out


def synthetic_news(panel, target, signal: float, coverage=0.6, seed=7) -> pd.DataFrame:
    """News table with sentiment = signal*z(target) + noise (controlled rig test, lookahead)."""
    rng = np.random.default_rng(seed)
    tz = target.sub(target.mean(axis=1), axis=0).div(target.std(axis=1).replace(0, np.nan), axis=0)
    rows = []
    for i, date in enumerate(panel.index):
        for t in panel.columns:
            if rng.random() > coverage:
                continue
            base = tz.iloc[i][t]
            base = 0.0 if not np.isfinite(base) else base
            rows.append({"ticker": t, "as_of": date,
                         "sentiment": float(np.tanh(signal * base + (1 - signal) * rng.standard_normal())),
                         "impact_score": float(rng.random())})
    return pd.DataFrame(rows)


def run(panel, target, feat_panels, H, k=3, label=""):
    m = deployment_sim(panel, target, feat_panels, horizon=H, k=k)
    print(f"  {label:18} | FWD IC={m['ic_fw']:+.4f}(IR{m['ic_fw_ir']:+.2f}) "
          f"FWD net={m['bt_fw_cum']:+.4f} win={m['bt_fw_win']:.2f} n={m['n_trades_fw']}")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", type=str, default="", help="path to news table (csv/parquet)")
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()

    panel, sector_panel, market = load_panels(timeframe="1D")
    target = relative_target(panel, args.horizon, "beta_residual", sector_panel, market)
    quant = build_feature_panels(panel)
    cols = ("sentiment", "impact_score")
    print(f"H2 fusion gate (H={args.horizon}, target=beta_residual)\n")

    print("QUANT-ONLY (price baseline):")
    run(panel, target, quant, args.horizon, label="quant_only")

    if args.news:
        news = (pd.read_parquet(args.news) if args.news.endswith(".parquet")
                else pd.read_csv(args.news, parse_dates=["as_of"]))
        news["as_of"] = pd.to_datetime(news["as_of"], utc=False)
        np_panels = news_to_panels(news, panel.index, cols, panel.columns)
        print("\nQUANT + REAL NEWS:")
        run(panel, target, {**quant, **np_panels}, args.horizon, label="quant+news")
        return 0

    # self-test: synthetic news of varying signal -> harness should detect the lift
    print("\nSELF-TEST (synthetic news; harness should lift FWD IC with injected signal):")
    for sig in (0.0, 0.10, 0.25):
        news = synthetic_news(panel, target, signal=sig)
        np_panels = news_to_panels(news, panel.index, cols, panel.columns)
        run(panel, target, {**quant, **np_panels}, args.horizon, label=f"+news sig={sig:.2f}")
    print("\nWhen the LLM block drops its baseline table on data/features grid, run:")
    print("  python ml/scripts/h2_fusion_sim.py --news data/news/baseline_features.csv")
    print("News HELPS if quant+news FWD IC and net P&L beat quant_only robustly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
