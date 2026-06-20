"""Trading-calendar tests: RU holidays, np.busday_count parity, panel ground truth."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backend.trading_calendar import (MoexTradingCalendar, RU_HOLIDAYS,
                                      RU_HOLIDAYS_THROUGH_YEAR, holidays_cover)


def test_weekends_are_non_trading():
    cal = MoexTradingCalendar(actual_trading_days=None)
    assert not cal.is_trading_day("2026-06-13")  # Saturday
    assert not cal.is_trading_day("2026-06-14")  # Sunday
    assert cal.is_trading_day("2026-06-15")      # Monday


def test_ru_holidays_are_non_trading():
    """Acceptance: 1/9 May and 12 June must be flagged non-trading."""
    cal = MoexTradingCalendar(actual_trading_days=None)
    # 2026-05-01 (Fri) and 2026-06-12 (Fri) are weekday holidays.
    assert not cal.is_trading_day("2026-05-01")
    assert not cal.is_trading_day("2026-06-12")
    # 2026-05-09 (Sat) is a holiday and a weekend either way.
    assert not cal.is_trading_day("2026-05-09")
    # 2025-05-01 / 2025-06-12 likewise.
    assert not cal.is_trading_day("2025-05-01")
    assert not cal.is_trading_day("2025-06-12")


def test_busday_count_parity_with_same_holidays():
    """trading_days_between must equal np.busday_count given the same holiday set."""
    cal = MoexTradingCalendar(actual_trading_days=None)
    bdc = np.busdaycalendar(holidays=np.array(RU_HOLIDAYS, dtype="datetime64[D]"))
    pairs = [("2026-05-25", "2026-06-15"), ("2026-06-15", "2026-05-25"),
             ("2026-04-20", "2026-06-20"), ("2026-06-15", "2026-06-15")]
    for a, b in pairs:
        expected = int(np.busday_count(a, b, busdaycal=bdc))
        assert cal.trading_days_between(a, b) == expected, (a, b)


def test_holiday_window_shrinks_count():
    """A window spanning a holiday counts fewer trading days than a naive weekday count."""
    cal = MoexTradingCalendar(actual_trading_days=None)
    # 2026-06-08 (Mon) .. 2026-06-19 (Fri); 2026-06-12 is a holiday.
    naive = int(np.busday_count("2026-06-08", "2026-06-19"))
    holiday_aware = cal.trading_days_between("2026-06-08", "2026-06-19")
    assert holiday_aware == naive - 1


def test_next_prev_skip_holiday():
    cal = MoexTradingCalendar(actual_trading_days=None)
    # day before the 12 June (Fri) holiday is 11 June (Thu); next trading day is 15 June (Mon)
    assert cal.next_trading_day("2026-06-11") == date(2026, 6, 15)
    assert cal.prev_trading_day("2026-06-15") == date(2026, 6, 11)


def test_add_trading_days_roundtrip():
    cal = MoexTradingCalendar(actual_trading_days=None)
    d = cal.add_trading_days("2026-06-11", 2)   # +2 td over the 12 June holiday
    assert d == date(2026, 6, 16)               # 11(0)->15(1)->16(2)
    assert cal.add_trading_days("2026-06-16", -2) == date(2026, 6, 11)


def test_panel_ground_truth_overrides_model():
    """In-panel dates use actual trading days; 'N trading days' matches actual bars."""
    # Synthetic panel: trade Mon-Wed-Fri only over two weeks (e.g. a sparsely-traded name).
    actual = [date(2026, 3, 2), date(2026, 3, 4), date(2026, 3, 6),
              date(2026, 3, 9), date(2026, 3, 11), date(2026, 3, 13)]
    cal = MoexTradingCalendar(holidays=(), actual_trading_days=actual)
    # Tuesday is NOT a trading day in this panel even though it is a weekday.
    assert not cal.is_trading_day("2026-03-03")
    assert cal.is_trading_day("2026-03-04")
    # entry "N trading days" matches the actual bar spacing of the panel
    assert cal.trading_days_between(date(2026, 3, 2), date(2026, 3, 9)) == 3
    assert cal.next_trading_day(date(2026, 3, 4)) == date(2026, 3, 6)


def test_holiday_coverage_marker():
    """Forward holiday coverage is advertised so a stale list is observable, not silent."""
    assert holidays_cover(f"{RU_HOLIDAYS_THROUGH_YEAR}-06-12") is True
    assert holidays_cover(f"{RU_HOLIDAYS_THROUGH_YEAR + 1}-06-12") is False


def test_imoex_overlay_forward_uses_holidays():
    """Outside the actual-day span, the forward holiday model applies."""
    actual = pd.date_range("2024-01-09", "2024-12-20", freq="B").date.tolist()
    cal = MoexTradingCalendar(actual_trading_days=actual)
    # forward of the panel, the RU holiday model still skips 12 June 2026
    assert not cal.is_trading_day("2026-06-12")
    # within the panel span a plain weekday trades
    assert cal.is_trading_day("2024-03-05")
