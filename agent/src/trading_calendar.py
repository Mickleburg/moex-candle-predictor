"""MOEX trading calendar — re-export of the project-canonical backend calendar.

Single source of truth = `backend.trading_calendar` (RU-holiday-aware, overlays the real IMOEX
panel). The orchestrator's scheduler + reconciliation MUST share the exact same trading-day
definition as the ML sleeve and execution, or the −12/−2 dividend timing drifts across the
May/June holiday clusters. So this module simply re-exports the backend functions.

A tiny stdlib fallback (weekends + a maintained RU-holiday set) is kept ONLY for the degenerate
case where the backend package is not importable (e.g. an agent-only checkout); it mirrors the
backend's half-open `[start, end)` counting semantics. `calendar_source()` reports which is live.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional

_BACKEND = None
try:  # canonical calendar
    from backend import trading_calendar as _BACKEND  # type: ignore
except Exception:  # noqa: BLE001 - backend optional in an agent-only checkout
    _BACKEND = None


def calendar_source() -> str:
    """'backend' when the canonical calendar is wired, else 'fallback' (for health/digest)."""
    return "backend" if _BACKEND is not None else "fallback"


if _BACKEND is not None:
    # Re-export the canonical implementation verbatim — one source of truth.
    is_trading_day = _BACKEND.is_trading_day
    next_trading_day = _BACKEND.next_trading_day
    prev_trading_day = _BACKEND.prev_trading_day
    add_trading_days = _BACKEND.add_trading_days
    trading_days_between = _BACKEND.trading_days_between
    last_trading_day_on_or_before = _BACKEND.last_trading_day_on_or_before

else:  # ---------------------------------------------------------------- stdlib fallback
    _RU_HOLIDAYS: frozenset[str] = frozenset({
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-02-23", "2024-03-08", "2024-04-29", "2024-04-30",
        "2024-05-01", "2024-05-09", "2024-05-10", "2024-06-12", "2024-11-04",
        "2024-12-30", "2024-12-31",
        "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07",
        "2025-01-08", "2025-02-24", "2025-03-10", "2025-05-01", "2025-05-08",
        "2025-05-09", "2025-06-12", "2025-06-13", "2025-11-03", "2025-11-04",
        "2025-12-31",
        "2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07",
        "2026-01-08", "2026-02-23", "2026-03-09", "2026-05-01", "2026-05-11",
        "2026-06-12", "2026-11-04",
        "2027-01-01", "2027-01-04", "2027-01-05", "2027-01-06", "2027-01-07",
        "2027-01-08", "2027-02-23", "2027-03-08", "2027-05-03", "2027-05-10",
        "2027-06-14", "2027-11-04",
    })

    def _to_date(value) -> _dt.date:
        if isinstance(value, _dt.datetime):
            return value.date()
        if isinstance(value, _dt.date):
            return value
        return _dt.date.fromisoformat(str(value)[:10])

    def is_trading_day(day) -> bool:
        d = _to_date(day)
        return d.weekday() < 5 and d.isoformat() not in _RU_HOLIDAYS

    def next_trading_day(day) -> _dt.date:
        d = _to_date(day) + _dt.timedelta(days=1)
        while not is_trading_day(d):
            d += _dt.timedelta(days=1)
        return d

    def prev_trading_day(day) -> _dt.date:
        d = _to_date(day) - _dt.timedelta(days=1)
        while not is_trading_day(d):
            d -= _dt.timedelta(days=1)
        return d

    def last_trading_day_on_or_before(day) -> _dt.date:
        d = _to_date(day)
        return d if is_trading_day(d) else prev_trading_day(d)

    def add_trading_days(day, n: int) -> _dt.date:
        d = _to_date(day)
        if is_trading_day(d):
            step = 1 if n >= 0 else -1
            remaining = abs(n)
        elif n >= 0:
            d = next_trading_day(d)
            step, remaining = 1, n          # _days[i] is first trading day after `day`
        else:
            d = prev_trading_day(d)
            step, remaining = -1, (-n) - 1
        while remaining > 0:
            d += _dt.timedelta(days=step)
            if is_trading_day(d):
                remaining -= 1
        return d

    def trading_days_between(start, end) -> int:
        """Signed count of trading days in the half-open interval [start, end) — np.busday_count
        semantics, matching backend.trading_days_between."""
        s, e = _to_date(start), _to_date(end)
        if s == e:
            return 0
        step = 1 if e > s else -1
        d, count = s, 0
        while d != e:
            if step > 0:
                if is_trading_day(d):
                    count += 1
                d += _dt.timedelta(days=1)
            else:
                d -= _dt.timedelta(days=1)
                if is_trading_day(d):
                    count -= 1
        return count


__all__ = ["is_trading_day", "next_trading_day", "prev_trading_day", "add_trading_days",
           "trading_days_between", "last_trading_day_on_or_before", "calendar_source"]
