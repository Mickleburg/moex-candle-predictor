"""The shared DECISION GRID — the contract between the ML block and the LLM/news block.

The cross-sectional model ranks the universe at a fixed cadence; the news layer must produce
features at EXACTLY those (ticker, as_of) points, or the work is wasted. This module is the
single source of truth for that grid so both chats align.

DECISION (ML block owner, 2026-06-16; cadence set to DAILY per user — finer granularity
captures news impact precisely in the days after a disclosure, and gives more training
samples; the LLM cost (~19k cells, ~10-11h) is accepted and runs offline/cached):
  cadence        DAILY — one decision per trading day. Fees are NOT a problem: we decide daily
                 but HOLD H trading days via overlapping portfolios, so daily turnover ~1/H.
                 (1H is dead to fees; weekly is the cheap fallback ~4k cells if quota bites.)
  as_of          end of that trading day's close (MSK). News usable iff published_at <= as_of.
  news_window    trailing 7 calendar days before as_of, aggregated per ticker (sparse days carry
                 the recent window; recency grows as news ages).
  target/hold    forward RELATIVE return (beta_residual) over the next H trading days
                 (evaluate H in {5, 10, 20}; overlapping books). The grid (as_of) is independent
                 of the hold horizon — the LLM only needs the as_of points.
  universe       the 12-name liquid universe (cross_sectional.UNIVERSE).
  no-lookahead   features at as_of use only price data <= as_of and news published_at <= as_of.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cross_sectional import UNIVERSE, load_panels

NEWS_WINDOW_DAYS = 7
CADENCE = "daily"
REPO_ROOT = Path(__file__).resolve().parents[3]
GRID_FILE = REPO_ROOT / "data" / "features" / "decision_grid.csv"


def decision_grid(cadence: str = CADENCE, timeframe: str = "1D") -> pd.DatetimeIndex:
    """as_of decision points for the cross-sectional model.

    cadence='daily'  -> every trading day in the aligned universe panel.
    cadence='weekly' -> last trading day of each ISO week (cheap fallback).
    """
    panel, _, _ = load_panels(timeframe=timeframe)
    idx = panel.index
    if cadence == "daily":
        return pd.DatetimeIndex(idx)
    if cadence == "weekly":
        s = pd.Series(idx, index=idx)
        last = s.groupby([idx.isocalendar().year, idx.isocalendar().week]).max()
        return pd.DatetimeIndex(sorted(last.values))
    raise ValueError(f"unknown cadence {cadence!r}")


def materialize(path: Path = GRID_FILE, cadence: str = CADENCE) -> pd.DatetimeIndex:
    """Write the as_of grid to CSV for the LLM/news block to consume; return the grid."""
    grid = decision_grid(cadence)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"as_of": [t.isoformat() for t in grid]}).to_csv(path, index=False)
    return grid


def load_grid(path: Path = GRID_FILE) -> pd.DatetimeIndex:
    """Load the materialized grid (or regenerate if missing)."""
    if not path.exists():
        return materialize(path)
    df = pd.read_csv(path)
    return pd.DatetimeIndex(pd.to_datetime(df["as_of"]))


if __name__ == "__main__":
    grid = materialize()
    cells = len(grid) * len(UNIVERSE)
    print(f"Decision grid: cadence={CADENCE}, {len(grid)} as_of points")
    print(f"  range: {grid.min().date()} .. {grid.max().date()}")
    print(f"  universe: {len(UNIVERSE)} tickers -> {cells} (ticker, as_of) cells")
    print(f"  news_window: trailing {NEWS_WINDOW_DAYS} days; no-lookahead published_at <= as_of")
    print(f"  LLM cost @ RPM 30: ~{cells/30/60:.1f} h for the full universe (baseline is free)")
    print(f"  Saved: {GRID_FILE}")
