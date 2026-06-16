"""Download OHLCV candles from MOEX ISS API and save as parquet.

Supports shares, indices, currency and futures via an instrument registry, so the same
script fetches the SBER/GAZP/LKOH candles AND orthogonal drivers (IMOEX, RTSI, sector
indices, RGBI bonds, CNY/RUB, Brent futures, ...).

Usage:
    python scripts/download_candles.py --ticker SBER  --timeframe 1H --from 2020-01-01
    python scripts/download_candles.py --ticker IMOEX --timeframe 1H --from 2020-01-01
    python scripts/download_candles.py --ticker RGBI  --timeframe 1H --from 2020-01-01
    # explicit override for an unregistered security:
    python scripts/download_candles.py --ticker BRZ5 --engine futures --market forts
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"

MOEX_ISS_BASE = "https://iss.moex.com"
PAGE_SIZE = 500

INTERVAL_MAP = {
    "1M": 1,
    "10M": 10,
    "1H": 60,
    "1D": 24,
    "1W": 7,
}

# Instrument registry: ticker -> (engine, market, board or None).
# board=None omits the /boards/<board> path segment (indices, currency, futures).
INSTRUMENT_REGISTRY: dict[str, tuple[str, str, str | None]] = {
    # Shares (TQBR board) — cross-sectional universe (liquid blue chips, sector-diverse)
    "SBER": ("stock", "shares", "TQBR"),   # banks
    "GAZP": ("stock", "shares", "TQBR"),   # oil & gas
    "LKOH": ("stock", "shares", "TQBR"),   # oil & gas
    "GMKN": ("stock", "shares", "TQBR"),   # metals (Nornickel)
    "ROSN": ("stock", "shares", "TQBR"),   # oil (Rosneft)
    "NVTK": ("stock", "shares", "TQBR"),   # gas (Novatek)
    "TATN": ("stock", "shares", "TQBR"),   # oil (Tatneft)
    "MGNT": ("stock", "shares", "TQBR"),   # retail (Magnit)
    "MTSS": ("stock", "shares", "TQBR"),   # telecom (MTS)
    "SNGS": ("stock", "shares", "TQBR"),   # oil (Surgutneftegas)
    "CHMF": ("stock", "shares", "TQBR"),   # steel (Severstal)
    "ALRS": ("stock", "shares", "TQBR"),   # diamonds (Alrosa)
    # Broad-market + sector indices (no board)
    "IMOEX": ("stock", "index", None),   # MOEX Russia Index (RUB)
    "RTSI": ("stock", "index", None),    # RTS Index (USD) -> RTSI-IMOEX ~= USD/RUB driver
    "MOEXFN": ("stock", "index", None),  # Financials sector (SBER)
    "MOEXOG": ("stock", "index", None),  # Oil & Gas sector (LKOH/GAZP)
    "MOEXMM": ("stock", "index", None),  # Metals & Mining sector (GMKN/CHMF/ALRS)
    "MOEXCN": ("stock", "index", None),  # Consumer sector (MGNT)
    "MOEXTL": ("stock", "index", None),  # Telecom sector (MTSS)
    "RGBI": ("stock", "index", None),    # Russian Govt Bond Index (rates -> banks)
    # Currency
    "CNYRUB_TOM": ("currency", "selt", None),
    "USD000UTSTOM": ("currency", "selt", None),  # USD/RUB spot (suspended mid-2024)
}


def resolve_instrument(ticker: str, engine: str | None, market: str | None, board: str | None):
    """Return (engine, market, board) from explicit args or the registry."""
    if engine and market:
        return engine, market, (board or None)
    if ticker in INSTRUMENT_REGISTRY:
        return INSTRUMENT_REGISTRY[ticker]
    raise ValueError(
        f"Unknown instrument {ticker!r}. Add it to INSTRUMENT_REGISTRY or pass "
        f"--engine/--market (and optional --board)."
    )


def _candles_url(engine: str, market: str, board: str | None, ticker: str) -> str:
    if board:
        return (f"{MOEX_ISS_BASE}/iss/engines/{engine}/markets/{market}"
                f"/boards/{board}/securities/{ticker}/candles.json")
    return (f"{MOEX_ISS_BASE}/iss/engines/{engine}/markets/{market}"
            f"/securities/{ticker}/candles.json")


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fetch_page(session: requests.Session, ticker: str, engine: str, market: str,
               board: str | None, timeframe: str, date_from: str, date_to: str, start: int) -> list[dict]:
    interval = INTERVAL_MAP[timeframe.upper()]
    url = _candles_url(engine, market, board, ticker)
    params = {
        "iss.meta": "off",
        "iss.only": "candles",
        "from": date_from,
        "till": date_to,
        "interval": interval,
    }
    if start > 0:
        params["start"] = start

    resp = session.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    block = data.get("candles", {})
    columns = block.get("columns", [])
    rows = block.get("data", [])
    if not columns:
        return []

    col_idx = {col: idx for idx, col in enumerate(columns)}
    # OHLC + begin/end are mandatory; volume/value are optional (indices have no volume).
    for field in ("begin", "end", "open", "high", "low", "close"):
        if field not in col_idx:
            raise ValueError(f"MOEX response missing column: {field} (have {columns})")

    candles = []
    for row in rows:
        candles.append({
            "ticker": ticker,
            "timeframe": timeframe,
            "begin": _parse_moex_dt(row[col_idx["begin"]]),
            "end": _parse_moex_dt(row[col_idx["end"]]),
            "open": _safe_float(row[col_idx["open"]]),
            "high": _safe_float(row[col_idx["high"]]),
            "low": _safe_float(row[col_idx["low"]]),
            "close": _safe_float(row[col_idx["close"]]),
            "volume": _safe_float(row[col_idx["volume"]]) if "volume" in col_idx else 0.0,
            "value": _safe_float(row[col_idx["value"]]) if "value" in col_idx else 0.0,
            "source": "moex",
        })
    return candles


def _parse_moex_dt(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse MOEX datetime: {value!r}")


def download_candles(ticker: str, timeframe: str, date_from: str, date_to: str,
                     engine: str, market: str, board: str | None) -> pd.DataFrame:
    session = requests.Session()
    session.headers["User-Agent"] = "moex-candle-predictor-download/0.2"

    all_candles: list[dict] = []
    start = 0
    while True:
        print(f"  Fetching {ticker} {timeframe} ({engine}/{market}/{board or '-'}) "
              f"from={date_from} to={date_to} start={start}...", end=" ")
        page = fetch_page(session, ticker, engine, market, board, timeframe, date_from, date_to, start)
        print(f"{len(page)} candles")
        all_candles.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE
        time.sleep(0.3)

    if not all_candles:
        raise ValueError(f"No candles returned for {ticker} {timeframe} {date_from}..{date_to}")

    df = pd.DataFrame(all_candles)
    df["begin"] = pd.to_datetime(df["begin"])
    df["end"] = pd.to_datetime(df["end"])
    df = df.drop_duplicates(subset=["begin"]).sort_values("begin").reset_index(drop=True)
    return df


def save_parquet(df: pd.DataFrame, ticker: str, timeframe: str) -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    begin_str = df["begin"].min().strftime("%Y%m%dT%H%M")
    end_str = df["begin"].max().strftime("%Y%m%dT%H%M")
    name = f"{ticker}_{timeframe}_{begin_str}_{end_str}.parquet"
    path = DATA_RAW / name
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True, help="e.g. SBER, IMOEX, RGBI, CNYRUB_TOM")
    parser.add_argument("--timeframe", default="1H", help="1H, 1D, 1M, 10M")
    parser.add_argument("--from", dest="date_from", default="2020-01-01")
    parser.add_argument("--to", dest="date_to", default=None)
    parser.add_argument("--engine", default=None, help="Override ISS engine (e.g. stock, currency, futures)")
    parser.add_argument("--market", default=None, help="Override ISS market (e.g. shares, index, selt, forts)")
    parser.add_argument("--board", default=None, help="Override ISS board (omit for indices/currency)")
    args = parser.parse_args()

    date_to = args.date_to or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ticker = args.ticker.upper() if args.ticker.isupper() or "_" not in args.ticker else args.ticker
    engine, market, board = resolve_instrument(ticker, args.engine, args.market, args.board)

    print(f"Downloading {ticker} {args.timeframe} {args.date_from} -> {date_to}  ({engine}/{market}/{board or '-'})")
    df = download_candles(ticker, args.timeframe, args.date_from, date_to, engine, market, board)
    print(f"Downloaded {len(df)} candles: {df['begin'].min()} .. {df['begin'].max()}")

    path = save_parquet(df, ticker, args.timeframe)
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
