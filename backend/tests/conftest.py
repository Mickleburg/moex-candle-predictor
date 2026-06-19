"""Shared fixtures/helpers for backend tests -- synthetic candle frames + temp store."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backend.trading_calendar import MoexTradingCalendar


def make_candles(ticker: str, timeframe: str, days, *, bars_per_day: int = 1,
                 base: float = 100.0) -> pd.DataFrame:
    """Build a valid OHLCV frame: one (or several) bars per given trading day."""
    rows = []
    for i, d in enumerate(days):
        for b in range(bars_per_day):
            hour = 10 + b
            begin = pd.Timestamp(d) + pd.Timedelta(hours=hour)
            px = base + i + 0.1 * b
            rows.append({
                "ticker": ticker, "timeframe": timeframe,
                "begin": begin, "end": begin + pd.Timedelta(hours=1),
                "open": px, "high": px + 0.5, "low": px - 0.5, "close": px + 0.2,
                "volume": 1000.0 + i, "value": (1000.0 + i) * px, "source": "test",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def trading_days():
    """A contiguous block of weekday trading days (no holidays) for tests."""
    return [d.date() for d in pd.bdate_range("2026-03-02", "2026-03-20")]


@pytest.fixture
def plain_calendar(trading_days):
    """Calendar whose ground truth is exactly the synthetic trading_days."""
    return MoexTradingCalendar(holidays=(), actual_trading_days=trading_days)
