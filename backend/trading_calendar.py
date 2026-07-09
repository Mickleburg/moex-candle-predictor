"""MOEX trading calendar -- RU-holiday-aware, shared by backend / ML / agent.

Why this exists
---------------
The live trading-day counters that time the H9 dividend sleeve's entry/exit
(``np.busday_count`` in ``ml/src/service/dividend_sleeve.py`` and
``ml/scripts/dividend_sleeve_monitor.py``) are RU-holiday-NAIVE: ``np.busday_count``
only skips weekends. Dividend record dates cluster in May-July, exactly around the
1/9 May and 12 June holidays, so a naive weekday count drifts by the number of
holidays in the window -> the sleeve enters/exits on the wrong day. This is a
correctness bug (docs/VDS_AUTONOMOUS_PLAN.md, gap analysis).

Source of truth (two regimes, matching the gap-analysis note)
-------------------------------------------------------------
* **In-panel dates** -- the actual IMOEX trading days from ``data/raw`` are ground
  truth (they already encode every real MOEX closure, including ad-hoc ones). ML
  already uses the price panel's own ``DatetimeIndex`` for historical dates; this
  calendar mirrors that by overlaying the actual trading days where available.
* **Forward dates** (beyond the panel) -- generated from weekdays minus a MAINTAINED
  list of RU public holidays (``RU_HOLIDAYS``). Update that list yearly from the
  official MOEX trading calendar / government decree.

Semantics
---------
``trading_days_between(start, end)`` mirrors ``np.busday_count`` exactly (count of
trading days in the half-open interval ``[start, end)``, negative if ``end < start``)
so it is a drop-in replacement.

No-lookahead: the calendar is a pure date function (holidays + weekday rules); it
carries no price information and is independent of ``as_of``.
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

DateLike = Union[str, date, datetime, pd.Timestamp, np.datetime64]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "raw"

# Range over which the forward (holiday + weekend) calendar is generated. Wide enough
# to cover the project history and several years of forward scheduling.
_GEN_START = np.datetime64("2015-01-01", "D")
_GEN_END = np.datetime64("2035-12-31", "D")

# ---------------------------------------------------------------------------
# Maintained RU public-holiday list (MOEX non-trading WEEKDAYS).
# Weekends are handled separately, so only list holidays that fall on a weekday
# (incl. the Monday a weekend holiday is officially observed on).
#
# ANNUAL MAINTENANCE (do this every autumn for the NEXT year, currently covered
# through 2027 -> add 2028+ here):
#   1. The RU government publishes the official non-working-days decree ~Sep/Oct.
#   2. MOEX then publishes its trading calendar (it can differ from the federal
#      calendar -- MOEX sometimes trades on a "bridge" day or adds a short session).
#      Source of truth for forward years = the MOEX trading-calendar page.
#   3. Add each non-trading WEEKDAY (incl. observed-shift Mondays) for the new year.
#   This is the ONLY forward-looking input; in-panel dates self-correct from the
#   actual IMOEX trading days (overlay below), so a stale list only mis-times dates
#   that are beyond the price panel -- check this list before each dividend season.
# ---------------------------------------------------------------------------
RU_HOLIDAYS: tuple[str, ...] = (
    # 2024 (Jan 1 = Mon) New-year week, + standard federal holidays
    "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    "2024-01-08", "2024-02-23", "2024-03-08", "2024-04-29", "2024-04-30",
    "2024-05-01", "2024-05-09", "2024-05-10", "2024-06-12", "2024-11-04",
    "2024-12-30", "2024-12-31",
    # 2025 (Jan 1 = Wed)
    "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07",
    "2025-01-08", "2025-02-23", "2025-03-08", "2025-05-01", "2025-05-08",
    "2025-05-09", "2025-06-12", "2025-06-13", "2025-11-03", "2025-11-04",
    "2025-12-31",
    # 2026 (Jan 1 = Thu) -- forward, relevant to the dividend season
    "2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07",
    "2026-01-08", "2026-02-23", "2026-03-09", "2026-05-01", "2026-05-11",
    "2026-06-12", "2026-11-04",
    # 2027 (Jan 1 = Fri) -- forward
    "2027-01-01", "2027-01-04", "2027-01-05", "2027-01-06", "2027-01-07",
    "2027-01-08", "2027-02-23", "2027-03-08", "2027-05-03", "2027-05-10",
    "2027-06-14", "2027-11-04",
)

# Last calendar year the forward holiday list is maintained through. Beyond this, forward
# dates fall back to weekends-only (federal holidays would be mis-counted) -> bump this and
# extend RU_HOLIDAYS each autumn. ``holidays_cover()`` lets callers warn on stale coverage.
RU_HOLIDAYS_THROUGH_YEAR = 2027


def holidays_cover(day: DateLike) -> bool:
    """False if ``day``'s year is past the maintained forward holiday coverage."""
    year = int(_to_d64(day).astype("datetime64[Y]").astype(int)) + 1970
    return year <= RU_HOLIDAYS_THROUGH_YEAR


def _to_d64(value: DateLike) -> np.datetime64:
    """Coerce any date-like to a day-resolution numpy datetime64 (date part only)."""
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]")
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return np.datetime64(ts.date(), "D")


class MoexTradingCalendar:
    """RU-holiday-aware MOEX trading calendar.

    Parameters
    ----------
    holidays:
        Iterable of RU public-holiday weekdays (defaults to ``RU_HOLIDAYS``).
    actual_trading_days:
        Optional ground-truth trading days (e.g. the IMOEX panel's dates). Within
        ``[min, max]`` of this set the calendar uses these days verbatim, overriding
        the generated weekday/holiday model. Outside that span the generated model
        applies. Pass ``None`` for a pure holiday+weekend calendar.
    """

    def __init__(
        self,
        holidays: Iterable[DateLike] = RU_HOLIDAYS,
        actual_trading_days: Optional[Iterable[DateLike]] = None,
    ) -> None:
        hol = np.array(sorted({_to_d64(h) for h in holidays}), dtype="datetime64[D]")
        self._holidays = hol

        all_days = np.arange(_GEN_START, _GEN_END + np.timedelta64(1, "D"),
                             dtype="datetime64[D]")
        generated = all_days[np.is_busday(all_days, holidays=hol)]

        if actual_trading_days is not None:
            actual = np.array(sorted({_to_d64(d) for d in actual_trading_days}),
                             dtype="datetime64[D]")
            if actual.size:
                lo, hi = actual[0], actual[-1]
                outside = generated[(generated < lo) | (generated > hi)]
                days = np.unique(np.concatenate([outside, actual]))
            else:
                days = generated
        else:
            days = generated

        self._days = days  # sorted, unique, day-resolution

    # -- membership -------------------------------------------------------
    def is_trading_day(self, day: DateLike) -> bool:
        """True if ``day`` is a MOEX trading day."""
        d = _to_d64(day)
        i = np.searchsorted(self._days, d)
        return bool(i < self._days.size and self._days[i] == d)

    # -- counting (np.busday_count drop-in) -------------------------------
    def trading_days_between(self, start: DateLike, end: DateLike) -> int:
        """Signed count of trading days in the half-open interval ``[start, end)``.

        Mirrors ``numpy.busday_count`` semantics exactly (negative if ``end < start``),
        so it is a drop-in replacement that additionally skips RU holidays.
        """
        s = _to_d64(start)
        e = _to_d64(end)
        # number of trading days strictly less than x
        ps = int(np.searchsorted(self._days, s, side="left"))
        pe = int(np.searchsorted(self._days, e, side="left"))
        return pe - ps

    # -- navigation -------------------------------------------------------
    def next_trading_day(self, day: DateLike) -> date:
        """First trading day strictly after ``day``."""
        d = _to_d64(day)
        i = int(np.searchsorted(self._days, d, side="right"))
        if i >= self._days.size:
            raise ValueError(f"{day!r} is beyond the calendar horizon {self._days[-1]}")
        return self._days[i].astype("datetime64[D]").astype(date)

    def prev_trading_day(self, day: DateLike) -> date:
        """Last trading day strictly before ``day``."""
        d = _to_d64(day)
        i = int(np.searchsorted(self._days, d, side="left"))
        if i <= 0:
            raise ValueError(f"{day!r} is before the calendar horizon {self._days[0]}")
        return self._days[i - 1].astype("datetime64[D]").astype(date)

    def last_trading_day_on_or_before(self, day: DateLike) -> date:
        """``day`` itself if it trades, else the most recent prior trading day."""
        d = _to_d64(day)
        if self.is_trading_day(d):
            return d.astype(date)
        return self.prev_trading_day(d)

    def add_trading_days(self, day: DateLike, n: int) -> date:
        """The trading day ``n`` trading days from ``day``.

        If ``day`` is a trading day it counts as offset 0. If it is not, offset 0
        resolves to the nearest trading day in the direction of ``n`` (next for
        ``n >= 0``, previous for ``n < 0``).
        """
        d = _to_d64(day)
        i = int(np.searchsorted(self._days, d, side="left"))
        on_day = i < self._days.size and self._days[i] == d
        if on_day:
            idx = i + n
        elif n >= 0:
            idx = i + n          # _days[i] is the first trading day after `day`
        else:
            idx = (i - 1) + (n + 1)  # _days[i-1] is the last trading day before `day`
        if idx < 0 or idx >= self._days.size:
            raise ValueError(f"offset {n} from {day!r} falls outside the calendar horizon")
        return self._days[idx].astype("datetime64[D]").astype(date)

    def trading_days_index(self, start: DateLike, end: DateLike) -> pd.DatetimeIndex:
        """All trading days in the closed interval ``[start, end]`` as a DatetimeIndex."""
        s, e = _to_d64(start), _to_d64(end)
        lo = int(np.searchsorted(self._days, s, side="left"))
        hi = int(np.searchsorted(self._days, e, side="right"))
        return pd.DatetimeIndex(self._days[lo:hi])


# ---------------------------------------------------------------------------
# Module-level default calendar (overlays the real IMOEX panel when present).
# ---------------------------------------------------------------------------
def _load_imoex_trading_days(data_dir: Path = _DEFAULT_DATA_DIR) -> Optional[np.ndarray]:
    """Read actual IMOEX trading days from the local panel (1D preferred, else 1H).

    Tolerant: returns None if no IMOEX parquet is present (pure forward calendar).
    """
    for tf in ("1D", "1H"):
        files = sorted(data_dir.glob(f"IMOEX_{tf}_*.parquet"))
        if not files:
            continue
        try:
            frames = [pd.read_parquet(f, columns=["begin"]) for f in files]
        except Exception:
            continue
        begin = pd.concat(frames, ignore_index=True)["begin"]
        begin = pd.to_datetime(begin)
        if getattr(begin.dt, "tz", None) is not None:
            begin = begin.dt.tz_localize(None)
        return np.array(sorted({np.datetime64(d.date(), "D") for d in begin}),
                       dtype="datetime64[D]")
    return None


@lru_cache(maxsize=1)
def get_calendar() -> MoexTradingCalendar:
    """Process-wide default calendar: RU holidays overlaid with the real IMOEX panel."""
    return MoexTradingCalendar(actual_trading_days=_load_imoex_trading_days())


# Convenience wrappers over the default calendar -----------------------------
def is_trading_day(day: DateLike) -> bool:
    return get_calendar().is_trading_day(day)


def trading_days_between(start: DateLike, end: DateLike) -> int:
    return get_calendar().trading_days_between(start, end)


def next_trading_day(day: DateLike) -> date:
    return get_calendar().next_trading_day(day)


def prev_trading_day(day: DateLike) -> date:
    return get_calendar().prev_trading_day(day)


def last_trading_day_on_or_before(day: DateLike) -> date:
    return get_calendar().last_trading_day_on_or_before(day)


def add_trading_days(day: DateLike, n: int) -> date:
    return get_calendar().add_trading_days(day, n)
