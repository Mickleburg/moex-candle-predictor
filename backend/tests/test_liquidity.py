"""ADTV liquidity-screen tests (criterion 1 of the H9 universe-expansion spec)."""

import pandas as pd

from backend import liquidity, store
from backend.tests.conftest import make_candles


def _seed_with_turnover(tmp_path, ticker, days, value_per_day):
    """Write a 1H frame whose per-day summed ``value`` equals value_per_day."""
    df = make_candles(ticker, "1H", days, bars_per_day=4)
    # distribute the target daily turnover across the day's 4 bars
    df["value"] = value_per_day / 4.0
    store.write_consolidated(df, ticker, "1H", data_dir=tmp_path)


def test_adtv_median_and_pass(tmp_path):
    days = [d.date() for d in pd.bdate_range("2025-06-23", "2026-06-19")]
    _seed_with_turnover(tmp_path, "BIGV", days, value_per_day=500e6)
    res = liquidity.adtv("BIGV", data_dir=tmp_path)
    assert res is not None
    assert abs(res.adtv_median_rub - 500e6) < 1.0
    assert res.passed is True


def test_adtv_below_threshold_fails(tmp_path):
    days = [d.date() for d in pd.bdate_range("2025-06-23", "2026-06-19")]
    _seed_with_turnover(tmp_path, "THIN", days, value_per_day=120e6)
    res = liquidity.adtv("THIN", data_dir=tmp_path)
    assert res.passed is False
    assert res.adtv_median_rub < liquidity.ADTV_THRESHOLD_RUB


def test_adtv_uses_trailing_window_only(tmp_path):
    # old high-turnover days outside the trailing window must not lift a now-thin name
    old = [d.date() for d in pd.bdate_range("2023-01-02", "2023-06-30")]
    recent = [d.date() for d in pd.bdate_range("2025-07-01", "2026-06-19")]
    df_old = make_candles("FADE", "1H", old, bars_per_day=4); df_old["value"] = 900e6 / 4
    df_new = make_candles("FADE", "1H", recent, bars_per_day=4); df_new["value"] = 100e6 / 4
    store.write_consolidated(pd.concat([df_old, df_new], ignore_index=True),
                             "FADE", "1H", data_dir=tmp_path)
    res = liquidity.adtv("FADE", data_dir=tmp_path)
    assert res.adtv_median_rub < liquidity.ADTV_THRESHOLD_RUB  # judged on the recent window
    assert res.passed is False


def test_missing_ticker_returns_none(tmp_path):
    assert liquidity.adtv("NOPE", data_dir=tmp_path) is None
