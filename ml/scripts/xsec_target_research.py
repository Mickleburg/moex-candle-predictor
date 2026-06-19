"""(1) Target engineering — does the DEFINITION of 'relative' reveal price predictability
that universe-demeaning hid? Rigorously closes the price-only cross-section question and
picks the cleanest market-neutral target for the eventual news model.

Three targets (ml/src/features/cross_sectional.relative_target):
    universe      = fwd - cross-sectional mean
    sector        = fwd - own-sector-index fwd, then demean   (idiosyncratic vs sector)
    beta_residual = fwd - beta*market fwd, then demean         (truest market-neutral)

Diagnostic: rank IC (Spearman) of cross-sectional MOMENTUM (past L-day return) vs each
target, IN-SAMPLE (<2025) vs FORWARD (>=2025). Daily + weekly (resampled) horizons.
A target is 'better' if momentum's FORWARD IC is positive AND stable across IS/forward.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from src.features.cross_sectional import load_panels, relative_target  # noqa: E402

FORWARD_START = pd.Timestamp("2025-01-01", tz="Europe/Moscow")
MODES = ["universe", "sector", "beta_residual"]


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    ra = pd.Series(a[m]).rank().to_numpy(float); rb = pd.Series(b[m]).rank().to_numpy(float)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else np.nan


def momentum(panel, L):
    return panel / panel.shift(L) - 1.0


def ic_series(score: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    idx = score.index
    vals = [spearman(score.loc[t].to_numpy(), target.loc[t].to_numpy()) for t in idx]
    return pd.Series(vals, index=idx).dropna()


def run(timeframe_label: str, panel, sector_panel, market, horizons, lookbacks):
    print(f"\n##### {timeframe_label}: {panel.shape[1]} tickers x {len(panel)} bars "
          f"{panel.index.min().date()}..{panel.index.max().date()} #####")
    for mode in MODES:
        print(f"\n--- target = {mode} ---")
        print(f"{'L':>3} {'H':>3} | {'IC all':>7} {'IS IC':>7} {'IS IR':>6} | "
              f"{'FWD IC':>7} {'FWD IR':>6}")
        for L in lookbacks:
            for H in horizons:
                tgt = relative_target(panel, H, mode, sector_panel, market)
                sc = momentum(panel, L)
                ic = ic_series(sc, tgt)
                is_ic = ic[ic.index < FORWARD_START]; fw_ic = ic[ic.index >= FORWARD_START]
                def ir(s): return float(s.mean() / (s.std() + 1e-9)) if len(s) else 0.0
                print(f"{L:>3} {H:>3} | {ic.mean():>7.4f} {is_ic.mean():>7.4f} {ir(is_ic):>6.2f} | "
                      f"{fw_ic.mean():>7.4f} {ir(fw_ic):>6.2f}")


def main() -> int:
    panel, sector_panel, market = load_panels(timeframe="1D")
    print(f"Loaded daily: {list(panel.columns)}")
    print(f"sector indices: {list(sector_panel.columns)}  market: "
          f"{'IMOEX' if market is not None else 'MISSING'}")

    # Daily horizons
    run("DAILY", panel, sector_panel, market, horizons=[5, 10, 20], lookbacks=[10, 20, 60])

    # Weekly (resample daily -> weekly), horizons in weeks
    wk = panel.resample("1W").last().dropna(how="any")
    wk_sec = sector_panel.resample("1W").last().reindex(wk.index).ffill()
    wk_mkt = market.resample("1W").last().reindex(wk.index).ffill() if market is not None else None
    run("WEEKLY", wk, wk_sec, wk_mkt, horizons=[2, 4, 8], lookbacks=[4, 8, 12])

    print("\nReading: pick the target with the most positive+stable FORWARD IC. If ALL targets")
    print("leave momentum's forward IC ~0, price-only is closed across target definitions —")
    print("confirming the alpha must come from news. The chosen target still defines the")
    print("market-neutral label the news model will predict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
