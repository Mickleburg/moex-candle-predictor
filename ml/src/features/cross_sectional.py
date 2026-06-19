"""Cross-sectional building blocks for the V2 market-neutral ranker.

Reusable pieces shared by the target-engineering research and the deployment-sim gate:
  * daily close panel (native 1D parquet, fallback to resampled 1H),
  * sector mapping (ticker -> MOEX sector index),
  * relative-return TARGETS in three market-neutral flavours,
  * rolling market beta vs IMOEX.

All functions are past-only where they feed features; the forward target uses future
prices only as the label (never as an input).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = REPO_ROOT / "data" / "raw"

UNIVERSE = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK",
            "TATN", "MGNT", "MTSS", "SNGS", "CHMF", "ALRS"]

# ticker -> sector index (MOEX). Used for the sector-relative target.
SECTOR_MAP = {
    "SBER": "MOEXFN",
    "GAZP": "MOEXOG", "LKOH": "MOEXOG", "ROSN": "MOEXOG",
    "NVTK": "MOEXOG", "TATN": "MOEXOG", "SNGS": "MOEXOG",
    "GMKN": "MOEXMM", "CHMF": "MOEXMM", "ALRS": "MOEXMM",
    "MGNT": "MOEXCN",
    "MTSS": "MOEXTL",
}
MARKET_INDEX = "IMOEX"


def _to_moscow(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.tz_localize("Europe/Moscow")


def _load_close(ticker: str, timeframe: str = "1D") -> pd.Series | None:
    """Load a single instrument's close series at `timeframe`; fallback to resampled 1H."""
    files = sorted(DATA_RAW.glob(f"{ticker}_{timeframe}_*.parquet"))
    if not files:
        files = sorted(DATA_RAW.glob(f"{ticker}_1H_*.parquet"))
        if not files:
            return None
        df = pd.read_parquet(files[-1]); df.columns = [c.lower() for c in df.columns]
        s = pd.Series(df["close"].to_numpy(float), index=_to_moscow(df["begin"]))
        s = s[~s.index.duplicated(keep="last")].sort_index().resample("1D").last()
        return s
    df = pd.read_parquet(files[-1]); df.columns = [c.lower() for c in df.columns]
    s = pd.Series(df["close"].to_numpy(float), index=_to_moscow(df["begin"]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.resample("1D").last()


def load_panels(universe: list[str] = UNIVERSE, timeframe: str = "1D"):
    """Return (close panel [time x ticker], sector-index panel, market series), aligned daily."""
    closes = {t: _load_close(t, timeframe) for t in universe}
    closes = {t: s for t, s in closes.items() if s is not None}
    panel = pd.DataFrame(closes).dropna(how="any")

    sectors = sorted(set(SECTOR_MAP.values()))
    sec = {idx: _load_close(idx, timeframe) for idx in sectors}
    sec = {idx: s for idx, s in sec.items() if s is not None}
    sector_panel = pd.DataFrame(sec).reindex(panel.index).ffill()

    mkt = _load_close(MARKET_INDEX, timeframe)
    market = mkt.reindex(panel.index).ffill() if mkt is not None else None
    return panel, sector_panel, market


def forward_return(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Simple forward return over `horizon` bars (label only)."""
    return panel.shift(-horizon) / panel - 1.0


def rolling_beta(panel: pd.DataFrame, market: pd.Series, window: int = 60) -> pd.DataFrame:
    """Past-only rolling beta of each ticker vs the market index (daily returns)."""
    r = panel.pct_change()
    rm = market.pct_change()
    cov = r.rolling(window).cov(rm)
    var = rm.rolling(window).var()
    return cov.div(var, axis=0)


def relative_target(panel: pd.DataFrame, horizon: int, mode: str,
                    sector_panel: pd.DataFrame | None = None,
                    market: pd.Series | None = None,
                    beta_window: int = 60) -> pd.DataFrame:
    """Cross-sectional market-neutral forward target.

    mode:
      'universe'      -> fwd_i - mean_j(fwd_j)                (demean across the universe)
      'sector'        -> fwd_i - sector_index_fwd_i           (remove sector beta)
      'beta_residual' -> fwd_i - beta_i * market_fwd, then demean (remove market beta)
    """
    fwd = forward_return(panel, horizon)
    if mode == "universe":
        return fwd.sub(fwd.mean(axis=1), axis=0)
    if mode == "sector":
        if sector_panel is None:
            raise ValueError("sector mode needs sector_panel")
        sec_fwd = forward_return(sector_panel, horizon)
        aligned = pd.DataFrame({t: sec_fwd[SECTOR_MAP[t]] for t in panel.columns},
                               index=panel.index)
        rel = fwd - aligned
        return rel.sub(rel.mean(axis=1), axis=0)
    if mode == "beta_residual":
        if market is None:
            raise ValueError("beta_residual mode needs market series")
        beta = rolling_beta(panel, market, beta_window)
        mkt_fwd = forward_return(market.to_frame("m"), horizon)["m"]
        resid = fwd.sub(beta.mul(mkt_fwd, axis=0))
        return resid.sub(resid.mean(axis=1), axis=0)
    raise ValueError(f"unknown mode {mode!r}")
