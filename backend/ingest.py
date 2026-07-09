"""Idempotent incremental ingest -- EOD step 1 of the autonomous cycle.

For each instrument it fetches ONLY the bars at/after the last stored ``begin`` (a one-day
overlap so a still-forming session candle is refreshed, not duplicated), merges into the
local parquet store and rewrites a single consolidated file. Re-running on unchanged data
is a no-op (same file, same contents) -- proven by ``merge_increment`` de-dup +
``write_consolidated`` single-file rule.

Network resilience: each fetch is retried with backoff; a partial failure on one
instrument is recorded and does not abort the others. The integrity gate (step 3) is the
authority on whether the resulting store is fit to trade on.

Continuous futures (Brent/gas) are stitched front-month series, not plain candle
downloads, so they are refreshed only with ``--with-futures`` by delegating to
``scripts/download_futures_continuous.py`` (a rebuild, kept idempotent by its own writer).

Usage::

    python -m backend.ingest                 # incremental refresh of the candle universe
    python -m backend.ingest --with-futures   # also rebuild BR_CONT / NG_CONT
    python -m backend.ingest --backfill        # full history for any missing instrument
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from . import store
from .universe import CANDLE_INSTRUMENTS, FUTURE_INSTRUMENTS, Instrument

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPORTS = _REPO_ROOT / "data" / "reports"
_BACKFILL_START = "2020-01-01"

# fetch_fn(ticker, timeframe, date_from, date_to) -> DataFrame | None (None = no new bars)
FetchFn = Callable[[str, str, str, str], Optional[pd.DataFrame]]


@dataclass
class InstrumentResult:
    ticker: str
    timeframe: str
    status: str              # "ok" | "up_to_date" | "skipped" | "error"
    rows_before: int = 0
    rows_after: int = 0
    added: int = 0
    new_last_begin: str = ""
    detail: str = ""


# ---------------------------------------------------------------------------
# Real network fetcher (wraps scripts/download_candles.py with retries).
# ---------------------------------------------------------------------------
def _import_downloader():
    scripts_dir = str(_REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import download_candles  # noqa: E402  (scripts dir is not a package)
    return download_candles


def network_fetch(ticker: str, timeframe: str, date_from: str, date_to: str,
                  retries: int = 3, backoff: float = 2.0) -> Optional[pd.DataFrame]:
    """Fetch ``[date_from, date_to]`` candles from MOEX ISS with retry/backoff.

    Returns None when ISS has no candles in the window (e.g. the window is all
    weekend/holiday) -- that is "nothing new", not an error.
    """
    dl = _import_downloader()
    try:
        engine, market, board = dl.resolve_instrument(ticker, None, None, None)
    except ValueError:
        # Not in the scripts registry (e.g. an H9 expansion line) -> use the universe's
        # explicit ISS routing override.
        from .universe import by_key
        ins = by_key(ticker, timeframe)
        if ins is None or not ins.engine:
            raise
        engine, market, board = ins.engine, ins.market, ins.board
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return dl.download_candles(ticker, timeframe, date_from, date_to,
                                       engine, market, board)
        except ValueError as exc:
            # download_candles raises ValueError("No candles returned ...") on empty window.
            if "No candles" in str(exc):
                return None
            raise
        except Exception as exc:  # network / HTTP -> retry
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise RuntimeError(f"fetch failed for {ticker} {timeframe} after {retries} tries: {last_exc}")


# ---------------------------------------------------------------------------
# Core incremental ingest of one candle instrument.
# ---------------------------------------------------------------------------
def ingest_instrument(ins: Instrument, fetch_fn: FetchFn, today: date,
                      data_dir: Path = store.DATA_RAW,
                      backfill: bool = False) -> InstrumentResult:
    existing = store.load_ticker(ins.ticker, ins.timeframe, data_dir)
    rows_before = 0 if existing is None else len(existing)

    if existing is None or existing.empty:
        if not backfill:
            return InstrumentResult(ins.ticker, ins.timeframe, "skipped",
                                    detail="no stored history; run with --backfill to seed")
        date_from = _BACKFILL_START
    else:
        # re-fetch from the last stored day forward (one-day overlap, deduped on merge)
        date_from = pd.Timestamp(existing["begin"].max()).strftime("%Y-%m-%d")

    date_to = today.strftime("%Y-%m-%d")
    try:
        fresh = fetch_fn(ins.ticker, ins.timeframe, date_from, date_to)
    except Exception as exc:
        return InstrumentResult(ins.ticker, ins.timeframe, "error",
                                rows_before=rows_before, rows_after=rows_before,
                                detail=str(exc))

    if fresh is None or fresh.empty:
        last = "" if existing is None else str(pd.Timestamp(existing["begin"].max()))
        return InstrumentResult(ins.ticker, ins.timeframe, "up_to_date",
                                rows_before=rows_before, rows_after=rows_before,
                                new_last_begin=last)

    merged = store.merge_increment(existing, fresh)
    added = len(merged) - rows_before
    store.write_consolidated(merged, ins.ticker, ins.timeframe, data_dir)
    status = "ok" if added > 0 else "up_to_date"
    return InstrumentResult(ins.ticker, ins.timeframe, status,
                            rows_before=rows_before, rows_after=len(merged), added=added,
                            new_last_begin=str(pd.Timestamp(merged["begin"].max())))


def refresh_continuous_futures(futures: list[Instrument]) -> list[InstrumentResult]:
    """Rebuild stitched front-month futures by delegating to the existing stitcher."""
    results: list[InstrumentResult] = []
    year_to = datetime.now(timezone.utc).year
    for ins in futures:
        before = store.last_begin(ins.ticker, ins.timeframe)
        cmd = [sys.executable, str(_REPO_ROOT / "scripts" / "download_futures_continuous.py"),
               "--asset", ins.asset, "--from", "2020", "--to", str(year_to)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            results.append(InstrumentResult(ins.ticker, ins.timeframe, "error",
                                            detail=(proc.stderr or proc.stdout)[-300:]))
            continue
        after = store.last_begin(ins.ticker, ins.timeframe)
        results.append(InstrumentResult(
            ins.ticker, ins.timeframe, "ok",
            new_last_begin="" if after is None else str(after),
            detail=f"rebuilt (last {before} -> {after})"))
    return results


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def run_ingest(today: Optional[date] = None, fetch_fn: FetchFn = network_fetch,
               data_dir: Path = store.DATA_RAW, backfill: bool = False,
               with_futures: bool = False,
               instruments: Optional[list[Instrument]] = None) -> dict:
    """Run incremental ingest over the universe; return a machine-readable report."""
    today = today or datetime.now(timezone.utc).date()
    candle_set = instruments if instruments is not None else CANDLE_INSTRUMENTS

    results: list[InstrumentResult] = []
    for ins in candle_set:
        results.append(ingest_instrument(ins, fetch_fn, today, data_dir, backfill))

    if with_futures:
        results.extend(refresh_continuous_futures(FUTURE_INSTRUMENTS))

    n_err = sum(1 for r in results if r.status == "error")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_date": today.isoformat(),
        "status": "error" if n_err else "ok",
        "n_instruments": len(results),
        "n_errors": n_err,
        "n_updated": sum(1 for r in results if r.status == "ok"),
        "results": [asdict(r) for r in results],
    }
    return report


def _write_report(report: dict) -> Path:
    _REPORTS.mkdir(parents=True, exist_ok=True)
    path = _REPORTS / "ingest_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backfill", action="store_true",
                    help="full history for instruments with no stored file")
    ap.add_argument("--with-futures", action="store_true",
                    help="also rebuild continuous Brent/gas futures")
    ap.add_argument("--date", default=None, help="reference date YYYY-MM-DD (default: today UTC)")
    args = ap.parse_args(argv)

    today = (datetime.strptime(args.date, "%Y-%m-%d").date()
             if args.date else datetime.now(timezone.utc).date())
    report = run_ingest(today=today, backfill=args.backfill, with_futures=args.with_futures)
    path = _write_report(report)

    print(f"Ingest {report['status']}: {report['n_updated']} updated, "
          f"{report['n_errors']} errors of {report['n_instruments']} instruments")
    for r in report["results"]:
        if r["status"] in ("ok", "error", "skipped"):
            print(f"  [{r['status']:10}] {r['ticker']:>6} {r['timeframe']:<3} "
                  f"+{r['added']} -> {r['new_last_begin']}  {r['detail']}")
    print(f"Report: {path}")
    return 1 if report["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
