"""Isolate the baseline news layer: does ANY news feature carry cross-sectional signal?

The fusion gate showed news barely moved the model (quant dominates, news weight ~0). That
could hide a weak-but-real news signal. This isolates each news feature: per decision date,
score the universe by the windowed feature value (no-lookahead) and measure rank IC vs the
beta_residual target, IS vs forward. Also reports coverage/variance so we can tell 'no signal'
from 'feature is constant/empty'. Verdict drives the LLM-quota decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from src.features.cross_sectional import load_panels, relative_target  # noqa: E402
from src.features.decision_grid import NEWS_WINDOW_DAYS  # noqa: E402

FORWARD_START = pd.Timestamp("2025-01-01", tz="Europe/Moscow")
FEATURES = ["sentiment", "impact_score", "novelty", "news_count", "recency_minutes"]


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    ra = pd.Series(a[m]).rank().to_numpy(float); rb = pd.Series(b[m]).rank().to_numpy(float)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else np.nan


def feature_panel(news, col, index, columns, window_days=NEWS_WINDOW_DAYS):
    """Windowed latest value per (date, ticker), no-lookahead (as in feature_bundle)."""
    out = pd.DataFrame(np.nan, index=index, columns=columns)
    win = pd.Timedelta(days=window_days)
    for tkr, g in news.groupby("ticker"):
        if tkr not in columns:
            continue
        g = g.sort_values("as_of").set_index("as_of")
        union = g.index.union(index).sort_values()
        ts = pd.Series(g.index, index=g.index).reindex(union).ffill().reindex(index)
        stale = (index - ts) > win
        vals = g[col].reindex(union).ffill().reindex(index).where(~stale, np.nan)
        out[tkr] = vals.to_numpy()
    return out


def main() -> int:
    panel, sector_panel, market = load_panels(timeframe="1D")
    news = pd.read_csv(ML_DIR.parent / "data/news/baseline_features.csv")
    news["as_of"] = pd.to_datetime(news["as_of"], utc=True).dt.tz_convert("Europe/Moscow")
    print(f"news rows={len(news)}, tickers={news['ticker'].nunique()}, "
          f"as_of {news['as_of'].min().date()}..{news['as_of'].max().date()}\n")

    for H in (5, 10, 20):
        target = relative_target(panel, H, "beta_residual", sector_panel, market)
        print(f"=== H={H} (beta_residual target) ===")
        print(f"{'feature':16} {'coverage':>8} {'std':>7} | {'IS IC':>7} {'FWD IC':>7} {'FWD IR':>7}")
        for col in FEATURES:
            fp = feature_panel(news, col, panel.index, panel.columns)
            cov = float(fp.notna().mean().mean())
            std = float(np.nanstd(fp.to_numpy()))
            ics = []
            for t in panel.index:
                s = fp.loc[t].to_numpy()
                if np.isfinite(s).sum() >= 3 and np.nanstd(s) > 0:
                    ics.append((t, spearman(s, target.loc[t].to_numpy())))
            ic = pd.DataFrame(ics, columns=["t", "ic"]).set_index("t")["ic"].dropna()
            is_ic = ic[ic.index < FORWARD_START]; fw_ic = ic[ic.index >= FORWARD_START]
            ir = fw_ic.mean() / (fw_ic.std() + 1e-9) if len(fw_ic) else 0.0
            print(f"{col:16} {cov:>8.2f} {std:>7.3f} | {is_ic.mean():>+7.4f} "
                  f"{fw_ic.mean():>+7.4f} {ir:>+7.2f}")
        print()
    print("Verdict guide: a baseline feature is promising if |FWD IC| > ~0.03 with stable IR.")
    print("If all ~0, the deterministic baseline carries no cross-sectional signal — the LLM run")
    print("is justified ONLY if we believe real sentiment > the crude proxy (else news is out).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
