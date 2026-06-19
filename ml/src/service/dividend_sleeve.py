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

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = REPO_ROOT / "data" / "raw"

# Shared MOEX trading calendar (RU-holiday-aware) owned by the backend block. Future ex-date
# countdowns must skip RU holidays (record dates cluster May-Jul around 1/9 May & 12 Jun) or the
# entry/exit timing drifts. Fall back to weekend-only np.busday_count if backend isn't importable
# (e.g. ML run in isolation) — LOUDLY, since that reintroduces the holiday drift.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from backend.trading_calendar import trading_days_between
except Exception:  # pragma: no cover - isolation fallback
    def trading_days_between(start, end) -> int:
        warnings.warn("backend.trading_calendar unavailable; falling back to RU-holiday-NAIVE "
                      "np.busday_count for trading-day counts", RuntimeWarning, stacklevel=2)
        return int(np.busday_count(pd.Timestamp(start).date(), pd.Timestamp(end).date()))

ENTRY_OFFSET = 12     # enter this many trading days before the ex-date anchor
EXIT_OFFSET = 2       # exit this many trading days before the anchor (before the ex-gap at ~-1)
VOL_WINDOW = 20       # trailing window for inverse-vol sizing (H4-style)
MAX_WEIGHT = 0.34     # per-name cap so a single active name can't dominate the long book
HEDGE_INDEX = "MARKET"  # market hedge handled by caller (beta=1 vs IMOEX in the sim)


UPCOMING_FEED = REPO_ROOT / "data" / "news" / "dividend_calendar_upcoming.csv"


def load_dividend_calendar(path: Path | None = None, with_upcoming: bool = True) -> pd.DataFrame:
    """Dividend events: ticker, date (record date, tz Moscow), value. The historical snapshot
    (data/raw/dividends.csv, from MOEX ISS) MERGED with the forward feed
    (data/news/dividend_calendar_upcoming.csv, board recommendations from e-disclosure, maintained by
    the LLM chat) so FUTURE ex-dates are visible before ISS publishes them. Refused dividends are
    simply absent from the feed."""
    path = path or (DATA_RAW / "dividends.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("Europe/Moscow")
    df = df.dropna(subset=["value"])
    df = df[df["value"] > 0][["ticker", "date", "value"]]
    if with_upcoming and UPCOMING_FEED.exists():
        up = pd.read_csv(UPCOMING_FEED)
        up = up.dropna(subset=["record_date", "value"])
        up = pd.DataFrame({"ticker": up["ticker"],
                           "date": pd.to_datetime(up["record_date"]).dt.tz_localize("Europe/Moscow"),
                           "value": up["value"].astype(float)})
        df = pd.concat([df, up], ignore_index=True).drop_duplicates(subset=["ticker", "date"], keep="last")
    return df.reset_index(drop=True)


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
    vol_pos = min(as_of_pos, len(idx) - 1)            # clamp for vol sizing if as_of is live (>panel)
    if vol_pos < vol_window:
        return {"longs": {}, "market_hedge": 0.0, "as_of": str(as_of), "is_production": False}
    # active = names whose record date is `td` trading days ahead with exit_off < td <= entry.
    # Hybrid: exact panel trading-day count for in-history dates; the RU-holiday-aware trading
    # calendar (backend) for FUTURE ex-dates (beyond the panel) so the sleeve produces signals off
    # the forward feed without the holiday drift a weekday-only count would introduce.
    panel_end = idx[-1]
    cols = set(panel.columns)
    active: set[str] = set()
    for tkr, rec in zip(calendar["ticker"], calendar["date"]):
        if tkr not in cols:
            continue
        if rec <= panel_end:
            td = (idx.searchsorted(rec, side="right") - 1) - as_of_pos
        else:
            td = trading_days_between(pd.Timestamp(as_of).date(), pd.Timestamp(rec).date())
        if exit_off < td <= entry:
            active.add(tkr)
    longs = inverse_vol_weights(panel, [t for t in panel.columns if t in active],
                                vol_pos, vol_window, max_weight)
    return {"longs": longs, "market_hedge": -float(sum(longs.values())),
            "as_of": str(as_of), "is_production": False}


def build_sleeve_signal(panel: pd.DataFrame, calendar: pd.DataFrame, as_of: pd.Timestamp,
                        model_version: str = "h9_dividend_runup_v1") -> dict:
    """Sleeve output for the risk_manager combiner: target positions tagged with the sleeve id.

    This is the S3-adjacent dividend run-up sleeve's contribution to the V3 multi-strategy book —
    INFORMATION for risk_manager (which nets sleeves, applies vol-targeting + regime gate + limits),
    not a trade command. Emitted as a self-describing dict (long weights + market hedge). The shared
    `aggregated_signal` schema is cross-sectional-RANKING shaped and does not fit a calendar sleeve;
    the V3 combiner handshake (extend aggregated_signal with a `sleeve`/target-position form) is a
    risk_manager-side decision — documented, not forced here. is_production=false."""
    tp = target_positions(panel, calendar, as_of)
    return {
        "sleeve": "s3_event",
        "strategy": "dividend_runup",
        "as_of": tp["as_of"],
        "market_neutral": True,
        # LONG target positions only; hedging is applied by risk_manager at BOOK level (more efficient
        # netting across sleeves). P0 robustness: SECTOR-index hedge >> broad IMOEX (Sharpe +0.92 vs
        # +0.54, DD halved) because the run-up is a name-vs-sector effect — recommended below.
        "positions": [{"ticker": t, "weight": round(w, 4), "leg": "long"}
                      for t, w in sorted(tp["longs"].items(), key=lambda kv: -kv[1])],
        "hedge_recommendation": {"method": "sector_index", "fallback": "imoex_beta_adjusted",
                                 "notional": round(sum(tp["longs"].values()), 4)},
        "gross_long": round(sum(tp["longs"].values()), 4),
        "model_version": model_version,
        "is_production": False,
    }
