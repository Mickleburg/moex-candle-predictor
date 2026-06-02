"""Download OHLCV candles from MOEX ISS API and save as parquet.

Usage:
    python scripts/download_candles.py --ticker SBER --timeframe 1H --from 2020-01-01 --to 2026-06-01
    python scripts/download_candles.py --ticker LKOH --timeframe 1H --from 2020-01-01 --to 2026-06-01
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
}


def fetch_page(session: requests.Session, ticker: str, timeframe: str, date_from: str, date_to: str, start: int) -> list[dict]:
    interval = INTERVAL_MAP[timeframe.upper()]
    url = (
        f"{MOEX_ISS_BASE}/iss/engines/stock/markets/shares/boards/TQBR"
        f"/securities/{ticker}/candles.json"
    )
    params = {
        "iss.meta": "off",
        "iss.only": "candles",
        "from": date_from,
        "till": date_to,
        "interval": interval,
    }
    if start > 0:
        params["start"] = start

    resp = session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    block = data.get("candles", {})
    columns = block.get("columns", [])
    rows = block.get("data", [])
    if not columns:
        return []

    col_idx = {col: idx for idx, col in enumerate(columns)}
    required = ["begin", "end", "open", "high", "low", "close", "volume", "value"]
    for field in required:
        if field not in col_idx:
            raise ValueError(f"MOEX response missing column: {field}")

    candles = []
    for row in rows:
        begin_str = row[col_idx["begin"]]
        end_str = row[col_idx["end"]]
        begin = _parse_moex_dt(begin_str)
        end = _parse_moex_dt(end_str)
        candles.append({
            "ticker": ticker,
            "timeframe": timeframe,
            "begin": begin,
            "end": end,
            "open": float(row[col_idx["open"]]),
            "high": float(row[col_idx["high"]]),
            "low": float(row[col_idx["low"]]),
            "close": float(row[col_idx["close"]]),
            "volume": float(row[col_idx["volume"]]),
            "value": float(row[col_idx["value"]]),
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


def download_candles(ticker: str, timeframe: str, date_from: str, date_to: str) -> pd.DataFrame:
    session = requests.Session()
    session.headers["User-Agent"] = "moex-candle-predictor-download/0.1"

    all_candles: list[dict] = []
    start = 0

    while True:
        print(f"  Fetching {ticker} {timeframe} from={date_from} to={date_to} start={start}...", end=" ")
        page = fetch_page(session, ticker, timeframe, date_from, date_to, start)
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
    parser.add_argument("--ticker", required=True, help="e.g. SBER, LKOH, GAZP")
    parser.add_argument("--timeframe", default="1H", help="1H, 1D, 1M, 10M")
    parser.add_argument("--from", dest="date_from", default="2020-01-01")
    parser.add_argument("--to", dest="date_to", default=None)
    args = parser.parse_args()

    date_to = args.date_to or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ticker = args.ticker.upper()

    print(f"Downloading {ticker} {args.timeframe} {args.date_from} -> {date_to}")
    df = download_candles(ticker, args.timeframe, args.date_from, date_to)
    print(f"Downloaded {len(df)} candles: {df['begin'].min()} → {df['begin'].max()}")

    path = save_parquet(df, ticker, args.timeframe)
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
