"""Orthogonal (cross-instrument) features for MOEX tickers — strictly past-only.

The ticker's own OHLCV hits a ~0.48 WF F1 ceiling; orthogonal drivers (broad market,
sector, rates, implicit FX) add information the candles cannot contain. This module
aligns orthogonal 1H series to a target ticker's candle timeline with merge_asof
(direction="backward": only the last value at-or-before t is used -> no lookahead) and
builds per-step features the LSTM can consume alongside its 14 OHLCV/time features.

Instruments (downloaded via scripts/download_candles.py):
    IMOEX  - MOEX Russia index (RUB)        : broad-market beta
    RTSI   - RTS index (USD)                 : RTSI-IMOEX spread ~= USD/RUB driver
    MOEXFN - financials sector index         : SBER's sector
    MOEXOG - oil & gas sector index          : LKOH/GAZP sector + oil macro
    RGBI   - govt bond index                 : rates -> banks (SBER)

Feature groups (selectable for ablation):
    market : imoex/rtsi returns + vol, rtsi_imoex spread return (FX proxy)
    sector : sector index returns + vol + relative strength (ticker_ret - sector_ret)
    rates  : rgbi returns + vol
All features are dimensionless past-only returns / vols / spreads. NaNs (history start,
gaps) -> 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.load import load_candles

ORTHO_TICKERS = ("IMOEX", "RTSI", "MOEXFN", "MOEXOG", "RGBI", "BR_CONT", "NG_CONT")
SECTOR_FOR = {"SBER": "MOEXFN", "GAZP": "MOEXOG", "LKOH": "MOEXOG"}
# Genuinely-orthogonal commodity drivers (continuous front-month FORTS futures).
COMMODITY = ("BR_CONT", "NG_CONT")  # Brent oil, natural gas
_DAY_BARS = 8  # ~ index main-session length, for a "day" return


def load_ortho_series(data_dir: str, tickers=ORTHO_TICKERS) -> dict[str, pd.DataFrame]:
    """Load each available orthogonal instrument as a sorted [begin, close] frame (MSK tz-aware).

    Tolerant: instruments without a parquet (e.g. futures not yet built) are skipped.
    """
    out: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        try:
            df = load_candles(str(data_dir), ticker=tk, timeframe="1H", tz_aware=True)
        except Exception:
            continue
        df = df[["begin", "close"]].sort_values("begin").reset_index(drop=True)
        out[tk] = df.rename(columns={"close": tk})
    return out


def _per_bar_features(series: pd.DataFrame, name: str) -> pd.DataFrame:
    """Past-only return/vol features on an instrument's own timeline."""
    c = series[name].astype(float)
    ret_1 = c.pct_change(1)
    ret_3 = c.pct_change(3)
    ret_d = c.pct_change(_DAY_BARS)
    vol = ret_1.rolling(12, min_periods=4).std()
    feats = pd.DataFrame({
        "begin": series["begin"],
        f"{name}_ret_1h": ret_1,
        f"{name}_ret_3h": ret_3,
        f"{name}_ret_d": ret_d,
        f"{name}_vol": vol,
    })
    return feats


def _to_utc(s: pd.Series) -> pd.Series:
    """Normalize a datetime series to UTC tz for type-safe merging (same instants).

    Inputs may carry different tz objects (named Europe/Moscow vs fixed +03:00 from a
    contract isoformat); merge_asof rejects mismatched tz types, so we convert both to UTC.
    """
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def _align_backward(target_begin: pd.Series, feats: pd.DataFrame) -> pd.DataFrame:
    """merge_asof backward: each target candle gets the last orthogonal bar <= its begin."""
    left = pd.DataFrame({"begin": _to_utc(pd.Series(target_begin))}).sort_values("begin")
    right = feats.copy()
    right["begin"] = _to_utc(right["begin"])
    right = right.sort_values("begin")
    merged = pd.merge_asof(left, right, on="begin", direction="backward")
    return merged.set_index(left.index).sort_index()


def build_orthogonal_features(
    ticker_df: pd.DataFrame,
    ortho: dict[str, pd.DataFrame],
    ticker: str,
    groups=("market", "sector", "rates"),
) -> tuple[np.ndarray, list[str]]:
    """Return (N, K) past-only orthogonal feature matrix aligned to ticker_df rows.

    ticker_df must have tz-aware MSK `begin` and `close`. `ortho` from load_ortho_series.
    `groups` selects feature groups for ablation.
    """
    n = len(ticker_df)
    begin = pd.to_datetime(ticker_df["begin"])
    tkr_ret_1h = ticker_df["close"].astype(float).pct_change(1).to_numpy()

    cols: dict[str, np.ndarray] = {}

    def add_instrument(name: str):
        aligned = _align_backward(begin, _per_bar_features(ortho[name], name))
        for col in (f"{name}_ret_1h", f"{name}_ret_3h", f"{name}_ret_d", f"{name}_vol"):
            cols[col] = aligned[col].to_numpy()

    if "market" in groups:
        add_instrument("IMOEX")
        add_instrument("RTSI")
        # RTSI-IMOEX spread return = implicit USD/RUB driver (full-period, no FX-spot gap)
        cols["rtsi_imoex_spread_ret"] = cols["RTSI_ret_1h"] - cols["IMOEX_ret_1h"]

    if "sector" in groups:
        sector = SECTOR_FOR.get(ticker.upper(), "MOEXFN")
        add_instrument(sector)
        # relative strength: ticker vs its sector (past-only, both at t)
        cols[f"rs_vs_{sector}"] = tkr_ret_1h - cols[f"{sector}_ret_1h"]

    if "rates" in groups:
        add_instrument("RGBI")

    if "commodity" in groups:
        # Genuinely-orthogonal oil/gas drivers (not the collinear sector index)
        for name in COMMODITY:
            if name in ortho:
                add_instrument(name)

    names = list(cols.keys())
    mat = np.column_stack([cols[k] for k in names]) if names else np.empty((n, 0))
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return mat, names


def build_combined_features(
    ticker_df: pd.DataFrame,
    ortho: dict[str, pd.DataFrame],
    ticker: str,
    groups=("market", "sector", "rates"),
) -> tuple[np.ndarray, list[str]]:
    """14 OHLCV/time per-step features + orthogonal features -> (N, 14+K)."""
    from ..models.lstm_model import build_per_step_features, FEATURE_NAMES

    base = build_per_step_features(ticker_df)            # (N, 14)
    ortho_mat, ortho_names = build_orthogonal_features(ticker_df, ortho, ticker, groups)
    combined = np.concatenate([base, ortho_mat], axis=1).astype(np.float32)
    return combined, list(FEATURE_NAMES) + ortho_names
