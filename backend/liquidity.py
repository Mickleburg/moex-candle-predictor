"""Liquidity (ADTV) screen for H9 universe-expansion candidates.

Criterion 1 of the a-priori inclusion spec (ml/docs/research/h9_universe_expansion_2026-06-21.md):
a tradeable line needs trailing-12m **median daily turnover (ADTV) >= ~300 M RUB** so a single-name
slice of the ~130-190 M RUB sleeve stays a small ADV fraction and slippage can't eat the edge.

Turnover source = the candle ``value`` column (RUB traded), summed per trading day from the 1H
panel, then the median over the trailing window. Median (not mean) so a few spike days don't lift a
structurally thin name over the bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from . import store

ADTV_THRESHOLD_RUB = 300_000_000.0   # >= ~300 M RUB trailing-12m median daily turnover
TRAILING_DAYS = 365


@dataclass
class AdtvResult:
    ticker: str
    adtv_median_rub: float
    adtv_mean_rub: float
    n_days: int
    window_start: str
    window_end: str
    passed: bool


def daily_turnover(df: pd.DataFrame) -> pd.Series:
    """Sum the candle ``value`` (RUB traded) per calendar date -> daily turnover series."""
    begin = pd.to_datetime(df["begin"])
    if getattr(begin.dt, "tz", None) is not None:
        begin = begin.dt.tz_localize(None)
    val = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    return val.groupby(begin.dt.date).sum()


def adtv(ticker: str, timeframe: str = "1H", trailing_days: int = TRAILING_DAYS,
         threshold: float = ADTV_THRESHOLD_RUB,
         data_dir: Path = store.DATA_RAW) -> Optional[AdtvResult]:
    """Trailing-window median daily turnover for one ticker, or None if no candles stored."""
    df = store.load_ticker(ticker, timeframe, data_dir)
    if df is None or df.empty or "value" not in df.columns:
        return None
    turn = daily_turnover(df)
    last = pd.Timestamp(max(turn.index))
    start = (last - pd.Timedelta(days=trailing_days)).date()
    window = turn[[d >= start for d in turn.index]]
    if window.empty:
        return None
    median = float(window.median())
    return AdtvResult(
        ticker=ticker,
        adtv_median_rub=median,
        adtv_mean_rub=float(window.mean()),
        n_days=int(window.shape[0]),
        window_start=str(start),
        window_end=str(last.date()),
        passed=median >= threshold,
    )


def screen(tickers: list[str], timeframe: str = "1H",
           threshold: float = ADTV_THRESHOLD_RUB,
           data_dir: Path = store.DATA_RAW) -> list[AdtvResult]:
    out: list[AdtvResult] = []
    for tk in tickers:
        res = adtv(tk, timeframe, threshold=threshold, data_dir=data_dir)
        if res is not None:
            out.append(res)
    return out


def _fmt_mln(x: float) -> str:
    return f"{x / 1e6:,.1f}M"


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", required=True, help="tickers to screen")
    ap.add_argument("--threshold", type=float, default=ADTV_THRESHOLD_RUB)
    ap.add_argument("--timeframe", default="1H")
    args = ap.parse_args(argv)

    results = screen(args.tickers, args.timeframe, args.threshold)
    print(f"ADTV screen (trailing {TRAILING_DAYS}d median daily turnover, "
          f"threshold {_fmt_mln(args.threshold)} RUB)\n")
    print(f"  {'ticker':<8}{'ADTV median':>14}{'mean':>14}{'days':>6}  window           verdict")
    for r in sorted(results, key=lambda x: -x.adtv_median_rub):
        verdict = "PASS" if r.passed else "FAIL (<threshold)"
        print(f"  {r.ticker:<8}{_fmt_mln(r.adtv_median_rub):>14}{_fmt_mln(r.adtv_mean_rub):>14}"
              f"{r.n_days:>6}  {r.window_start}..{r.window_end}  {verdict}")
    fails = [r.ticker for r in results if not r.passed]
    if fails:
        print(f"\nFAILED screen -> exclude: {', '.join(fails)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
