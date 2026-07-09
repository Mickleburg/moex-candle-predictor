"""Trading calendar: RU holidays excluded, trading-day arithmetic correct."""

from __future__ import annotations

import datetime as dt

from agent.src import trading_calendar as tcal


def test_weekends_excluded():
    assert not tcal.is_trading_day("2026-06-20")   # Saturday
    assert not tcal.is_trading_day("2026-06-21")   # Sunday
    assert tcal.is_trading_day("2026-06-19")       # Friday


def test_ru_holidays_excluded():
    # the correctness-critical clusters the plan calls out
    assert not tcal.is_trading_day("2026-05-01")   # Spring/Labour
    assert not tcal.is_trading_day("2025-05-09")   # Victory Day
    assert not tcal.is_trading_day("2026-06-12")   # Russia Day
    assert not tcal.is_trading_day("2026-01-07")   # New Year block


def test_next_prev_trading_day_skip_holiday():
    # 2026-06-12 (Fri, Russia Day) -> next trading day is Mon 2026-06-15
    assert tcal.next_trading_day("2026-06-11") == dt.date(2026, 6, 15)
    assert tcal.prev_trading_day("2026-06-15") == dt.date(2026, 6, 11)


def test_add_trading_days_does_not_drift_through_holiday():
    # entering 12 trading days before a post-holiday record date must count actual trading days
    anchor = dt.date(2026, 6, 18)
    entry = tcal.add_trading_days(anchor, -12)
    # every step landed on a trading day, and the span excludes 2026-06-12
    assert tcal.is_trading_day(entry)
    assert tcal.trading_days_between(entry, anchor) == 12


def test_calendar_source_is_known():
    assert tcal.calendar_source() in ("backend", "fallback")
