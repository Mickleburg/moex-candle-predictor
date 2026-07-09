"""MOEX trading-day calendar for the H9 entry/exit discipline.

The CANONICAL RU-holiday trading calendar is owned by the backend/data block (and the ML sleeve);
execution must NOT duplicate that holiday list. This class provides only the parts execution truly
needs locally — the weekend-skip policy (MOEX has no edge on weekend sessions, so we never trade
them) and trading-day arithmetic — over an INJECTED holiday set / trading-day predicate.

Wiring in production: the orchestrator passes the backend calendar in, either as
``holidays=<iterable of date>`` or as ``is_trading_day=<callable date->bool>``. Until that lands,
the default is weekday-only (Mon-Fri), which already makes the dry-run/paper reconciliation and the
weekend-skip guard correct; only multi-day holiday spans (May/June clusters) need the injected set.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    # tolerate ISO strings with a space or 'T', with or without tz suffix
    text = str(value).strip().replace(" ", "T")
    return datetime.fromisoformat(text).date()


class TradingCalendar:
    """Weekend-skip MOEX calendar with an injectable holiday set / predicate.

    Precedence for "is this a trading day?":
        1. an explicit ``is_trading_day`` callable (e.g. the backend calendar), if given;
        2. otherwise: a weekday that is not in ``holidays`` and not overridden, OR a date listed
           in ``extra_trading_days`` (MOEX occasionally schedules a working Saturday).
    """

    def __init__(
        self,
        holidays: Iterable[date | datetime | str] | None = None,
        extra_trading_days: Iterable[date | datetime | str] | None = None,
        is_trading_day: Callable[[date], bool] | None = None,
    ) -> None:
        self._holidays: set[date] = {_as_date(d) for d in (holidays or ())}
        self._extra: set[date] = {_as_date(d) for d in (extra_trading_days or ())}
        self._predicate = is_trading_day

    def is_trading_day(self, day: date | datetime | str) -> bool:
        d = _as_date(day)
        if self._predicate is not None:
            return bool(self._predicate(d))
        if d in self._extra:
            return True
        if d.weekday() >= 5:  # 5=Sat, 6=Sun
            return False
        return d not in self._holidays

    def next_trading_day(self, day: date | datetime | str, *, inclusive: bool = False) -> date:
        d = _as_date(day)
        if inclusive and self.is_trading_day(d):
            return d
        d += timedelta(days=1)
        while not self.is_trading_day(d):
            d += timedelta(days=1)
        return d

    def prev_trading_day(self, day: date | datetime | str, *, inclusive: bool = False) -> date:
        d = _as_date(day)
        if inclusive and self.is_trading_day(d):
            return d
        d -= timedelta(days=1)
        while not self.is_trading_day(d):
            d -= timedelta(days=1)
        return d

    def add_trading_days(self, day: date | datetime | str, n: int) -> date:
        """Shift by ``n`` trading days (n may be negative). The start day is not counted."""
        d = _as_date(day)
        step = 1 if n >= 0 else -1
        remaining = abs(n)
        while remaining > 0:
            d += timedelta(days=step)
            if self.is_trading_day(d):
                remaining -= 1
        return d

    def trading_days_between(self, start: date | datetime | str, end: date | datetime | str) -> int:
        """Signed count of trading days in the half-open interval [start, end).

        This mirrors ``numpy.busday_count`` (and the backend ``MoexTradingCalendar``) exactly, so the
        H9 "trading days until the anchor" — ``trading_days_between(as_of, record_date)`` — is computed
        on the SAME convention the sleeve/monitor use (``np.busday_count(as_of, rec)``). ``start`` is
        counted if it trades; ``end`` is not. Negative if ``end < start``.
        """
        a, b = _as_date(start), _as_date(end)
        if a == b:
            return 0
        sign = 1 if b > a else -1
        lo, hi = (a, b) if b > a else (b, a)
        count = 0
        d = lo
        while d < hi:
            if self.is_trading_day(d):
                count += 1
            d += timedelta(days=1)
        return sign * count


def default_trading_calendar():
    """The calendar execution uses by default: the **backend RU-holiday canon** if importable,
    else the weekday-only fallback.

    The −12/−2 discipline MUST count on the same trading-day canon as the sleeve and monitor, which
    now share ``backend.trading_calendar`` (RU holidays + the real IMOEX panel overlay). We delegate
    to that so execution never drifts on a private holiday list. The fallback (weekday-only) keeps the
    block importable in an isolated environment without backend/pandas; both share the same
    ``[start, end)`` counting convention, so they agree on every date that is a trading day.

    The backend ``MoexTradingCalendar`` exposes a compatible ``is_trading_day`` / ``trading_days_between``,
    so it is used directly (no adapter needed for the discipline path).
    """
    try:
        from backend.trading_calendar import get_calendar  # type: ignore
        return get_calendar()
    except Exception:
        return TradingCalendar()


def active_calendar_source() -> str:
    """Label for logging/README: which canon ``default_trading_calendar`` resolves to here."""
    try:
        import backend.trading_calendar  # type: ignore  # noqa: F401
        return "backend_canon"
    except Exception:
        return "weekday_fallback"
