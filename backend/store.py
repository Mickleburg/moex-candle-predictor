"""File-based parquet candle store -- discovery, load, idempotent incremental merge.

Layout (unchanged from the research store): one parquet per ``(ticker, timeframe)`` in
``data/raw`` named ``{TICKER}_{TF}_{firstbegin}_{lastbegin}.parquet`` (the embedded range
is informational; discovery is by the ``{TICKER}_{TF}_*`` prefix). ``data/raw/*.parquet``
is gitignored -- candles are regenerable.

Idempotency contract
--------------------
``merge_increment`` concatenates new bars onto the existing frame, de-duplicates on
``begin`` (keeping the LAST occurrence so a re-fetched bar overwrites cleanly), sorts and
re-indexes. ``write_consolidated`` writes exactly one file per ``(ticker, timeframe)`` and
removes any stale-named siblings, so re-running ingest on unchanged data reproduces the
same single file with identical contents -- never duplicates, never grows the store.

Timezone: ``begin``/``end`` stay as the raw MSK-wall-clock value written by the
downloaders (see ``ml/src/data/load.py`` docstring). Use ``to_moscow_time`` downstream to
canonicalise; the store does not relabel, to stay byte-compatible with existing files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = _REPO_ROOT / "data" / "raw"

# Canonical column order written by scripts/download_candles.py.
CANDLE_COLUMNS = ["ticker", "timeframe", "begin", "end",
                  "open", "high", "low", "close", "volume", "value", "source"]


def store_files(ticker: str, timeframe: str, data_dir: Path = DATA_RAW) -> list[Path]:
    """All parquet files for ``(ticker, timeframe)`` (usually 0 or 1)."""
    return sorted(Path(data_dir).glob(f"{ticker}_{timeframe}_*.parquet"))


def load_ticker(ticker: str, timeframe: str,
                data_dir: Path = DATA_RAW) -> Optional[pd.DataFrame]:
    """Load the stored frame for ``(ticker, timeframe)``, or None if absent.

    Tolerant to a store that still has several historical files for the same key:
    concatenates and de-duplicates them.
    """
    files = store_files(ticker, timeframe, data_dir)
    if not files:
        return None
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["begin"] = pd.to_datetime(df["begin"])
    if "end" in df.columns:
        df["end"] = pd.to_datetime(df["end"])
    df = (df.drop_duplicates(subset=["begin"], keep="last")
            .sort_values("begin").reset_index(drop=True))
    return df


def last_begin(ticker: str, timeframe: str,
               data_dir: Path = DATA_RAW) -> Optional[pd.Timestamp]:
    """Latest stored ``begin`` for ``(ticker, timeframe)``, or None if no data."""
    df = load_ticker(ticker, timeframe, data_dir)
    if df is None or df.empty:
        return None
    return pd.Timestamp(df["begin"].max())


def merge_increment(existing: Optional[pd.DataFrame], fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge freshly-fetched bars into the existing frame, idempotently.

    De-dup keeps the LAST occurrence on ``begin`` so a re-downloaded bar replaces the
    stored one (e.g. a same-session candle that was still forming at last fetch). Output
    is sorted by ``begin`` with a clean index.
    """
    fresh = fresh.copy()
    fresh["begin"] = pd.to_datetime(fresh["begin"])
    if existing is None or existing.empty:
        merged = fresh
    else:
        existing = existing.copy()
        existing["begin"] = pd.to_datetime(existing["begin"])
        merged = pd.concat([existing, fresh], ignore_index=True)
    merged = (merged.drop_duplicates(subset=["begin"], keep="last")
                    .sort_values("begin").reset_index(drop=True))
    return merged


def write_consolidated(df: pd.DataFrame, ticker: str, timeframe: str,
                       data_dir: Path = DATA_RAW) -> Path:
    """Write exactly one parquet for ``(ticker, timeframe)``, removing stale siblings.

    The filename embeds the begin-range (matching the existing convention). Old files for
    the same key are deleted AFTER the new one is written, so a crash mid-write never
    leaves the store empty.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("begin").reset_index(drop=True)
    begin_str = pd.Timestamp(df["begin"].min()).strftime("%Y%m%dT%H%M")
    end_str = pd.Timestamp(df["begin"].max()).strftime("%Y%m%dT%H%M")
    new_path = data_dir / f"{ticker}_{timeframe}_{begin_str}_{end_str}.parquet"

    df.to_parquet(new_path, index=False, engine="pyarrow")

    for old in store_files(ticker, timeframe, data_dir):
        if old != new_path:
            old.unlink()
    return new_path
