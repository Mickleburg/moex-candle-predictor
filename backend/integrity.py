"""Data-integrity gate -- EOD step 3 of the autonomous cycle.

Produces a machine-readable verdict the orchestrator reads BEFORE trading: if the store
is stale, holed or NaN-poisoned it returns HALT (with reasons) so the agent never trades
on rotten data. Checks, per instrument:

* **freshness**  -- the last stored bar is no more than ``stale_tolerance_days`` trading
  days behind the expected last trading day (RU-holiday-aware, via the shared calendar).
* **gaps**       -- every MOEX trading day between the first and last stored bar has at
  least one bar (no missing trading days).
* **values**     -- no NaN / non-positive OHLC; for shares, no NaN / non-positive volume
  (indices legitimately carry zero volume, so volume is OHLC-only there).
* **sync**       -- required instruments share the same last trading day (one series
  lagging the rest is caught even if the whole store is uniformly fresh).

A failing check on a ``required`` instrument (the names the sleeve trades + IMOEX/RGBI
context the hedge needs) -> HALT. A failure on a secondary series -> WARN (no HALT).

Verdict is written to ``data/reports/data_integrity_status.json`` and returned as a dict.
``run_gate`` exit code: 0 = OK, 1 = HALT.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import store
from .trading_calendar import MoexTradingCalendar, get_calendar
from .universe import INGEST_INSTRUMENTS, Instrument

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPORTS = _REPO_ROOT / "data" / "reports"

_OHLC = ("open", "high", "low", "close")


@dataclass
class Check:
    name: str            # "freshness" | "gaps" | "values" | "sync"
    ticker: str
    timeframe: str
    status: str          # "pass" | "fail" | "warn"
    detail: str = ""


def _bar_dates(df: pd.DataFrame) -> np.ndarray:
    """Unique day-resolution dates present in the frame (tz-stripped)."""
    begin = pd.to_datetime(df["begin"])
    if getattr(begin.dt, "tz", None) is not None:
        begin = begin.dt.tz_localize(None)
    return np.array(sorted({np.datetime64(d.date(), "D") for d in begin}),
                   dtype="datetime64[D]")


def _check_freshness(ins: Instrument, df: pd.DataFrame, ref: date,
                     cal: MoexTradingCalendar, tol: int) -> Check:
    expected_last = cal.last_trading_day_on_or_before(ref)
    actual_last = pd.Timestamp(df["begin"].max()).date()
    behind = cal.trading_days_between(actual_last, expected_last)
    sev = "fail" if ins.required else "warn"
    if behind > tol:
        return Check("freshness", ins.ticker, ins.timeframe, sev,
                     f"last bar {actual_last} is {behind} trading days behind expected "
                     f"{expected_last} (tolerance {tol})")
    return Check("freshness", ins.ticker, ins.timeframe, "pass",
                 f"last bar {actual_last}, expected {expected_last}")


def _recent_window(df: pd.DataFrame, cal: MoexTradingCalendar,
                   window_td: int) -> pd.DataFrame:
    """Slice df to the last ``window_td`` trading days (the data we trade on at EOD).

    Deep-history gaps (old holidays, corporate-action halts like GMKN/VTBR splits) are
    immutable and not actionable at EOD; scoping gap/value checks to a recent window keeps
    the gate about freshness-of-tradeable-data, not historical curation. window_td<=0 means
    'whole history'.
    """
    if window_td <= 0:
        return df
    last = pd.Timestamp(df["begin"].max()).date()
    start = cal.add_trading_days(last, -window_td)
    begin = pd.to_datetime(df["begin"])
    if getattr(begin.dt, "tz", None) is not None:
        begin = begin.dt.tz_localize(None)
    return df[begin.dt.date >= start]


def _check_gaps(ins: Instrument, df: pd.DataFrame, cal: MoexTradingCalendar) -> Check:
    present = set(_bar_dates(df).tolist())  # datetime64[D].tolist() -> datetime.date objects
    first = pd.Timestamp(df["begin"].min()).date()
    last = pd.Timestamp(df["begin"].max()).date()
    expected = cal.trading_days_index(first, last)
    missing = [d.date() for d in expected if d.date() not in present]
    sev = "fail" if ins.required else "warn"
    if missing:
        shown = ", ".join(str(m) for m in missing[:8])
        more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
        return Check("gaps", ins.ticker, ins.timeframe, sev,
                     f"{len(missing)} missing trading day(s): {shown}{more}")
    return Check("gaps", ins.ticker, ins.timeframe, "pass",
                 f"{len(expected)} trading days, no gaps")


def _check_values(ins: Instrument, df: pd.DataFrame) -> Check:
    sev = "fail" if ins.required else "warn"
    problems: list[str] = []
    # Continuous futures carry only a synthetic `close` (no OHLC/volume) by design
    # (scripts/download_futures_continuous.py); check that column instead.
    price_cols = ("close",) if ins.kind == "continuous_future" else _OHLC
    for col in price_cols:
        if col not in df.columns:
            problems.append(f"missing column {col}")
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        n_nan = int(s.isna().sum())
        n_nonpos = int((s <= 0).sum())
        if n_nan:
            problems.append(f"{col}: {n_nan} NaN")
        if n_nonpos:
            problems.append(f"{col}: {n_nonpos} <=0")
    if ins.kind == "share" and "volume" in df.columns:
        v = pd.to_numeric(df["volume"], errors="coerce")
        n_nan = int(v.isna().sum())
        n_zero = int((v <= 0).sum())
        if n_nan:
            problems.append(f"volume: {n_nan} NaN")
        if n_zero:
            problems.append(f"volume: {n_zero} zero/neg")
    if problems:
        return Check("values", ins.ticker, ins.timeframe, sev, "; ".join(problems))
    return Check("values", ins.ticker, ins.timeframe, "pass", "no NaN / non-positive values")


def run_checks(ref: Optional[date] = None, data_dir: Path = store.DATA_RAW,
               cal: Optional[MoexTradingCalendar] = None,
               instruments: Optional[list[Instrument]] = None,
               stale_tolerance_days: int = 1,
               recent_window_td: int = 60) -> dict:
    """Run all integrity checks; return a machine-readable verdict dict.

    Gaps/values are evaluated over the last ``recent_window_td`` trading days (the data
    the orchestrator trades on); freshness/sync use the latest bar. Set
    ``recent_window_td<=0`` to check the whole history.
    """
    ref = ref or datetime.now(timezone.utc).date()
    cal = cal or get_calendar()
    insts = instruments if instruments is not None else INGEST_INSTRUMENTS

    checks: list[Check] = []
    required_last: dict[str, date] = {}

    for ins in insts:
        df = store.load_ticker(ins.ticker, ins.timeframe, data_dir)
        if df is None or df.empty:
            sev = "fail" if ins.required else "warn"
            checks.append(Check("presence", ins.ticker, ins.timeframe, sev, "no data in store"))
            continue
        recent = _recent_window(df, cal, recent_window_td)
        checks.append(_check_freshness(ins, df, ref, cal, stale_tolerance_days))
        checks.append(_check_gaps(ins, recent, cal))
        checks.append(_check_values(ins, recent))
        if ins.required:
            required_last.setdefault(ins.timeframe, {})[ins.ticker] = \
                pd.Timestamp(df["begin"].max()).date()

    # cross-sync: required series of the SAME timeframe should share the last trading day
    # (comparing 1H vs 1D last bars would be apples-to-oranges).
    for tf, last_by_ticker in required_last.items():
        newest = max(last_by_ticker.values())
        for tk, last in last_by_ticker.items():
            behind = cal.trading_days_between(last, newest)
            if behind > stale_tolerance_days:
                checks.append(Check("sync", tk, tf, "fail",
                                    f"last {last} lags newest required {tf} series {newest} "
                                    f"by {behind} trading days"))

    fails = [c for c in checks if c.status == "fail"]
    warns = [c for c in checks if c.status == "warn"]
    status = "HALT" if fails else "OK"
    verdict = {
        "status": status,
        "reference_date": ref.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stale_tolerance_days": stale_tolerance_days,
        "n_checks": len(checks),
        "n_fail": len(fails),
        "n_warn": len(warns),
        "reasons": [f"{c.name}/{c.ticker}/{c.timeframe}: {c.detail}" for c in fails],
        "warnings": [f"{c.name}/{c.ticker}/{c.timeframe}: {c.detail}" for c in warns],
        "checks": [asdict(c) for c in checks],
    }
    return verdict


def _write_verdict(verdict: dict) -> Path:
    _REPORTS.mkdir(parents=True, exist_ok=True)
    path = _REPORTS / "data_integrity_status.json"
    path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_gate(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None, help="reference date YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--tolerance", type=int, default=1,
                    help="max trading days a series may lag before HALT (default 1)")
    ap.add_argument("--gap-lookback", type=int, default=60,
                    help="trading-day window for gap/value checks; <=0 = whole history (default 60)")
    args = ap.parse_args(argv)
    ref = (datetime.strptime(args.date, "%Y-%m-%d").date()
           if args.date else datetime.now(timezone.utc).date())

    verdict = run_checks(ref=ref, stale_tolerance_days=args.tolerance,
                         recent_window_td=args.gap_lookback)
    path = _write_verdict(verdict)

    print(f"Integrity {verdict['status']}: {verdict['n_fail']} fail, "
          f"{verdict['n_warn']} warn of {verdict['n_checks']} checks")
    for reason in verdict["reasons"]:
        print(f"  HALT  {reason}")
    for w in verdict["warnings"][:10]:
        print(f"  warn  {w}")
    print(f"Verdict: {path}")
    return 1 if verdict["status"] == "HALT" else 0


if __name__ == "__main__":
    raise SystemExit(run_gate())
