"""Dividend run-up sleeve (S3-adjacent) — serving logic for the V3 multi-strategy book.

H9 found the project's first robust edge: a market-adjusted PRE-EX run-up (buy ~12 trading days
before the ex-dividend date, exit at offset -2 before the ex-gap), strongest for high-yield names,
distinct from a random-date placebo. This module turns that into a deployable, market-hedged,
inverse-vol-sized sleeve that emits TARGET POSITIONS for a given as_of (past-only). The backtest
that validates it as a portfolio (not per-event averages) is `scripts/h9_dividend_sleeve_sim.py`.

Positions are INFORMATION for risk_manager (the portfolio combiner / limits / regime gate apply on
top). is_production=false until forward-shadow accrual + a no-lookahead check vs announcement dates.

No-lookahead: a name is only entered if its ex-date is in the future window [as_of+2, as_of+ENTRY]
trading days, and the dividend was announced before as_of (board recommendation, weeks ahead — a
realistic assumption to be verified with announcement dates). Vols/weights use data through as_of.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = REPO_ROOT / "data" / "raw"

ENTRY_OFFSET = 12     # enter this many trading days before the ex-date anchor
EXIT_OFFSET = 2       # exit this many trading days before the anchor (before the ex-gap at ~-1)
VOL_WINDOW = 20       # trailing window for inverse-vol sizing (H4-style)
MAX_WEIGHT = 0.34     # per-name cap so a single active name can't dominate the long book
HEDGE_INDEX = "MARKET"  # market hedge handled by caller (beta=1 vs IMOEX in the sim)


def load_dividend_calendar(path: Path | None = None) -> pd.DataFrame:
    """Dividend events: ticker, record_date (tz Moscow), value. From MOEX ISS dump."""
    path = path or (DATA_RAW / "dividends.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("Europe/Moscow")
    df = df.dropna(subset=["value"])
    return df[df["value"] > 0].reset_index(drop=True)


def _anchor_positions(price_index: pd.DatetimeIndex, calendar: pd.DataFrame,
                      ticker: str) -> list[int]:
    """Trading-day index positions of each ex-date anchor (last day <= record date) for a ticker."""
    out = []
    sub = calendar[calendar["ticker"] == ticker]
    for rdate in sub["date"]:
        pos = price_index.searchsorted(rdate, side="right") - 1
        if 0 <= pos < len(price_index):
            out.append(pos)
    return out


def active_window_map(price_index: pd.DatetimeIndex, calendar: pd.DataFrame,
                      tickers: list[str], entry: int = ENTRY_OFFSET,
                      exit_off: int = EXIT_OFFSET) -> dict[str, set[int]]:
    """For each ticker, the set of index positions on which we HOLD the run-up (offsets [-entry,-exit])."""
    # Hold weights on day offsets [-entry, -(exit_off+1)]. With P&L using held=weight.shift(1), this
    # earns returns at offsets [-(entry-1), -exit_off] — i.e. the LAST return is INTO offset -exit_off,
    # so the ex-gap at offset -1 is NOT captured (the whole point of exiting before the ex-date).
    amap: dict[str, set[int]] = {}
    for t in tickers:
        hold: set[int] = set()
        for a in _anchor_positions(price_index, calendar, t):
            for off in range(a - entry, a - exit_off):     # exclusive end -> last weight day = a-exit_off-1
                if 0 <= off < len(price_index):
                    hold.add(off)
        amap[t] = hold
    return amap


def inverse_vol_weights(panel: pd.DataFrame, tickers: list[str], as_of_pos: int,
                        vol_window: int = VOL_WINDOW, max_weight: float = MAX_WEIGHT) -> dict[str, float]:
    """Past-only inverse-vol long weights over the active names, capped and renormalised to sum 1."""
    if not tickers:
        return {}
    rets = panel.iloc[max(0, as_of_pos - vol_window): as_of_pos + 1].pct_change()
    inv = {}
    for t in tickers:
        v = float(rets[t].std())
        inv[t] = 1.0 / v if v and np.isfinite(v) and v > 0 else 0.0
    s = sum(inv.values())
    if s <= 0:
        w = {t: 1.0 / len(tickers) for t in tickers}              # fallback: equal weight
    else:
        w = {t: inv[t] / s for t in tickers}
    # apply cap, renormalise (one pass is enough for our small books)
    w = {t: min(x, max_weight) for t, x in w.items()}
    s = sum(w.values())
    return {t: x / s for t, x in w.items()} if s > 0 else w


def target_positions(panel: pd.DataFrame, calendar: pd.DataFrame, as_of: pd.Timestamp,
                     entry: int = ENTRY_OFFSET, exit_off: int = EXIT_OFFSET,
                     vol_window: int = VOL_WINDOW, max_weight: float = MAX_WEIGHT) -> dict:
    """Target sleeve positions at as_of (past-only). Long names currently in their pre-ex window,
    inverse-vol sized; the market hedge is -sum(long weights) in IMOEX (applied by risk_manager).
    Returns {'longs': {ticker: weight}, 'market_hedge': float, 'as_of': ...}. is_production=false."""
    idx = panel.index
    as_of_pos = idx.searchsorted(as_of, side="right") - 1
    if as_of_pos < vol_window:
        return {"longs": {}, "market_hedge": 0.0, "as_of": str(as_of), "is_production": False}
    amap = active_window_map(idx, calendar, list(panel.columns), entry, exit_off)
    active = [t for t in panel.columns if as_of_pos in amap.get(t, set())]
    longs = inverse_vol_weights(panel, active, as_of_pos, vol_window, max_weight)
    return {"longs": longs, "market_hedge": -float(sum(longs.values())),
            "as_of": str(as_of), "is_production": False}
