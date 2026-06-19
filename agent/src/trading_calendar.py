"""MOEX trading calendar — single source of truth lives in the backend block.

The orchestrator's scheduler and reconciliation need a RU-holiday-aware "is this a trading
day / next trading day" function (weekend-only logic drifts the −12/−2 dividend timing
across the May/June holiday clusters — a known correctness bug called out in the plan).

Resolution order:
  1. the canonical backend calendar, if the backend block is importable (preferred);
  2. a vendored RU-holiday fallback below (so the agent runs before backend lands).

When the backend lands, `_backend_calendar()` picks it up automatically and the fallback
goes dormant. The fallback is intentionally conservative: any date it wrongly marks as a
trading day still hits the data-integrity gate (no fresh bar -> HALT), so it fails safe.
"""

from __future__ import annotations

import datetime as _dt
from typing import Callable, Optional

# --- vendored RU non-trading days (federal holidays + standard extended blocks) ----------
# Source of truth is the backend calendar; this is a fallback for 2024-2027. Weekends are
# handled separately. Covers the correctness-critical clusters (1/9 May, 12 June, New Year).
_RU_HOLIDAYS: frozenset[str] = frozenset({
    # 2024
    "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    "2024-01-08", "2024-02-23", "2024-03-08", "2024-04-29", "2024-04-30",
    "2024-05-01", "2024-05-09", "2024-05-10", "2024-06-12", "2024-11-04",
    "2024-12-30", "2024-12-31",
    # 2025
    "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07",
    "2025-01-08", "2025-02-24", "2025-03-10", "2025-05-01", "2025-05-02",
    "2025-05-08", "2025-05-09", "2025-06-12", "2025-06-13", "2025-11-03",
    "2025-11-04", "2025-12-31",
    # 2026
    "2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07",
    "2026-01-08", "2026-02-23", "2026-03-09", "2026-05-01", "2026-05-11",
    "2026-05-12", "2026-06-12", "2026-11-04",
    # 2027
    "2027-01-01", "2027-01-04", "2027-01-05", "2027-01-06", "2027-01-07",
    "2027-01-08", "2027-02-23", "2027-03-08", "2027-05-03", "2027-05-10",
    "2027-06-14", "2027-11-04",
})


def _to_date(value) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    # accept ISO strings, optionally with a time/offset component
    return _dt.date.fromisoformat(str(value)[:10])


_backend_fn_cache: Optional[Callable[[_dt.date], bool]] = None
_backend_checked = False


def _backend_calendar() -> Optional[Callable[[_dt.date], bool]]:
    """Return the backend's is_trading_day(date) if the backend block exposes one, else None."""
    global _backend_fn_cache, _backend_checked
    if _backend_checked:
        return _backend_fn_cache
    _backend_checked = True
    for modname in ("backend.calendar", "backend.src.calendar", "backend.trading_calendar"):
        try:
            mod = __import__(modname, fromlist=["is_trading_day"])
        except Exception:  # noqa: BLE001 - backend optional / not built yet
            continue
        fn = getattr(mod, "is_trading_day", None)
        if callable(fn):
            _backend_fn_cache = fn
            break
    return _backend_fn_cache


def is_trading_day(day) -> bool:
    """True if `day` is a MOEX trading day (weekend + RU holidays excluded)."""
    d = _to_date(day)
    backend = _backend_calendar()
    if backend is not None:
        return bool(backend(d))
    if d.weekday() >= 5:           # Sat/Sun
        return False
    return d.isoformat() not in _RU_HOLIDAYS


def next_trading_day(day, inclusive: bool = False) -> _dt.date:
    """The first trading day on/after (inclusive) or strictly after `day`."""
    d = _to_date(day)
    if not inclusive:
        d += _dt.timedelta(days=1)
    while not is_trading_day(d):
        d += _dt.timedelta(days=1)
    return d


def prev_trading_day(day, inclusive: bool = False) -> _dt.date:
    """The last trading day on/before (inclusive) or strictly before `day`."""
    d = _to_date(day)
    if not inclusive:
        d -= _dt.timedelta(days=1)
    while not is_trading_day(d):
        d -= _dt.timedelta(days=1)
    return d


def add_trading_days(day, n: int) -> _dt.date:
    """`day` shifted by n trading days (n<0 = backwards). Anchor itself need not be a trading day."""
    d = _to_date(day)
    step = 1 if n >= 0 else -1
    remaining = abs(n)
    while remaining > 0:
        d += _dt.timedelta(days=step)
        if is_trading_day(d):
            remaining -= 1
    return d


def trading_days_between(start, end) -> int:
    """Count of trading days in the half-open interval (start, end] (sign follows direction)."""
    s, e = _to_date(start), _to_date(end)
    if s == e:
        return 0
    step = 1 if e > s else -1
    d, count = s, 0
    while d != e:
        d += _dt.timedelta(days=step)
        if is_trading_day(d):
            count += step
    return count


def calendar_source() -> str:
    """'backend' if the canonical calendar is wired, else 'fallback' (for health/digest)."""
    return "backend" if _backend_calendar() is not None else "fallback"
