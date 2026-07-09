"""Task 2: the discipline calendar is the backend RU-holiday canon, not a private holiday list.

Proves execution counts trading days on the SAME canon the sleeve/monitor use, so the -12/-2 timing
cannot drift across the May-July holiday cluster where dividend record dates concentrate.
"""

from __future__ import annotations

import pytest

from execution.src.trading_calendar import (
    TradingCalendar,
    active_calendar_source,
    default_trading_calendar,
)

backend_cal = pytest.importorskip("backend.trading_calendar")


def test_default_resolves_to_backend_canon():
    assert active_calendar_source() == "backend_canon"
    cal = default_trading_calendar()
    assert not cal.is_trading_day("2026-06-12")   # Russia Day — RU holiday
    assert cal.is_trading_day("2026-06-10")        # ordinary weekday


def test_count_matches_backend_across_holiday_spans():
    cal = default_trading_calendar()
    for a, b in [("2026-06-01", "2026-06-30"), ("2026-05-01", "2026-05-20"),
                 ("2026-07-02", "2026-07-20")]:
        assert cal.trading_days_between(a, b) == backend_cal.trading_days_between(a, b), (a, b)


def test_weekday_fallback_with_ru_holidays_agrees_on_forward_dates():
    # The isolated-env fallback (weekday + injected RU_HOLIDAYS) reproduces the backend canon on
    # FORWARD dates (beyond the IMOEX panel), where backend also uses weekday+holiday generation.
    fb = TradingCalendar(holidays=backend_cal.RU_HOLIDAYS)
    for a, b in [("2027-05-01", "2027-05-20"), ("2027-06-01", "2027-06-30")]:
        assert fb.trading_days_between(a, b) == backend_cal.trading_days_between(a, b), (a, b)


def test_td_uses_busday_count_convention():
    # trading_days_between(as_of, record) == np.busday_count-style [start, end) the sleeve uses.
    cal = default_trading_calendar()
    # as_of counted, end excluded; both endpoints trade -> 12 the entry offset
    assert cal.trading_days_between("2026-07-02", "2026-07-20") == 12
    assert cal.trading_days_between("2026-07-20", "2026-07-02") == -12
