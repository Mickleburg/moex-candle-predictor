"""Trading-calendar arithmetic: weekend skip, injected holidays, signed td counting."""

from __future__ import annotations

from datetime import date

from execution.src.trading_calendar import TradingCalendar


def test_weekend_is_not_a_trading_day():
    cal = TradingCalendar()
    assert cal.is_trading_day(date(2026, 7, 20))   # Monday
    assert not cal.is_trading_day(date(2026, 7, 18))  # Saturday
    assert not cal.is_trading_day(date(2026, 7, 19))  # Sunday


def test_injected_holiday_skipped_and_overridable():
    cal = TradingCalendar(holidays=["2026-06-12"])           # Russia Day (Friday)
    assert not cal.is_trading_day("2026-06-12")
    # extra trading day can re-open a weekend (a working Saturday)
    cal2 = TradingCalendar(extra_trading_days=["2026-07-18"])
    assert cal2.is_trading_day("2026-07-18")


def test_trading_days_between_signed_and_holiday_aware():
    cal = TradingCalendar()
    # Thu 2026-07-02 -> Mon 2026-07-20 is 12 trading days (the H9 entry offset)
    assert cal.trading_days_between("2026-07-02", "2026-07-20") == 12
    # Thu 2026-07-16 -> Mon 2026-07-20 is 2 (the exit offset)
    assert cal.trading_days_between("2026-07-16", "2026-07-20") == 2
    # signed: reverse direction is negative, same magnitude
    assert cal.trading_days_between("2026-07-20", "2026-07-02") == -12
    assert cal.trading_days_between("2026-07-10", "2026-07-10") == 0


def test_holiday_shifts_td_count():
    # Inserting a holiday inside the span reduces the trading-day distance by one.
    base = TradingCalendar()
    with_holiday = TradingCalendar(holidays=["2026-07-09"])
    assert base.trading_days_between("2026-07-02", "2026-07-20") == 12
    assert with_holiday.trading_days_between("2026-07-02", "2026-07-20") == 11


def test_add_trading_days_roundtrip():
    cal = TradingCalendar()
    anchor = date(2026, 7, 20)
    entry = cal.add_trading_days(anchor, -12)
    assert cal.trading_days_between(entry, anchor) == 12
