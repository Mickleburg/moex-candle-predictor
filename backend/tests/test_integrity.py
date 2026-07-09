"""Integrity-gate tests: clean store -> OK; injected gap / NaN / staleness -> HALT."""

from datetime import date

import numpy as np
import pandas as pd

from backend import store
from backend.integrity import run_checks
from backend.universe import Instrument
from backend.tests.conftest import make_candles


def _seed(tmp_path, ticker, days, **kw):
    df = make_candles(ticker, "1D", days, **kw)
    store.write_consolidated(df, ticker, "1D", data_dir=tmp_path)
    return df


def test_clean_store_is_ok(tmp_path, trading_days, plain_calendar):
    _seed(tmp_path, "SBER", trading_days)
    ins = Instrument("SBER", "1D", "share", required=True)
    verdict = run_checks(ref=trading_days[-1], data_dir=tmp_path,
                         cal=plain_calendar, instruments=[ins])
    assert verdict["status"] == "OK", verdict["reasons"]
    assert verdict["n_fail"] == 0


def test_injected_gap_triggers_halt(tmp_path, trading_days, plain_calendar):
    # drop a middle trading day -> a hole in the panel
    holed = trading_days[:5] + trading_days[6:]
    _seed(tmp_path, "SBER", holed)
    ins = Instrument("SBER", "1D", "share", required=True)
    verdict = run_checks(ref=trading_days[-1], data_dir=tmp_path,
                         cal=plain_calendar, instruments=[ins])
    assert verdict["status"] == "HALT"
    assert any("gaps" in r and str(trading_days[5]) in r for r in verdict["reasons"])


def test_injected_nan_triggers_halt(tmp_path, trading_days, plain_calendar):
    df = make_candles("SBER", "1D", trading_days)
    df.loc[3, "close"] = np.nan
    store.write_consolidated(df, "SBER", "1D", data_dir=tmp_path)
    ins = Instrument("SBER", "1D", "share", required=True)
    verdict = run_checks(ref=trading_days[-1], data_dir=tmp_path,
                         cal=plain_calendar, instruments=[ins])
    assert verdict["status"] == "HALT"
    assert any("values" in r and "NaN" in r for r in verdict["reasons"])


def test_zero_volume_share_triggers_halt(tmp_path, trading_days, plain_calendar):
    df = make_candles("SBER", "1D", trading_days)
    df.loc[2, "volume"] = 0.0
    store.write_consolidated(df, "SBER", "1D", data_dir=tmp_path)
    ins = Instrument("SBER", "1D", "share", required=True)
    verdict = run_checks(ref=trading_days[-1], data_dir=tmp_path,
                         cal=plain_calendar, instruments=[ins])
    assert verdict["status"] == "HALT"
    assert any("volume" in r for r in verdict["reasons"])


def test_stale_store_triggers_halt(tmp_path, trading_days, plain_calendar):
    # store ends well before the reference date -> freshness fail
    _seed(tmp_path, "SBER", trading_days[:5])
    ins = Instrument("SBER", "1D", "share", required=True)
    verdict = run_checks(ref=trading_days[-1], data_dir=tmp_path,
                         cal=plain_calendar, instruments=[ins])
    assert verdict["status"] == "HALT"
    assert any("freshness" in r for r in verdict["reasons"])


def test_zero_volume_index_is_allowed(tmp_path, trading_days, plain_calendar):
    # indices legitimately carry zero volume -> must not HALT on volume
    df = make_candles("IMOEX", "1D", trading_days)
    df["volume"] = 0.0
    store.write_consolidated(df, "IMOEX", "1D", data_dir=tmp_path)
    ins = Instrument("IMOEX", "1D", "index", required=True)
    verdict = run_checks(ref=trading_days[-1], data_dir=tmp_path,
                         cal=plain_calendar, instruments=[ins])
    assert verdict["status"] == "OK", verdict["reasons"]


def test_secondary_series_failure_warns_not_halts(tmp_path, trading_days, plain_calendar):
    # a non-required series that is stale -> WARN, store still OK to trade
    _seed(tmp_path, "MOEXCN", trading_days[:5])
    ins = Instrument("MOEXCN", "1D", "index", required=False)
    verdict = run_checks(ref=trading_days[-1], data_dir=tmp_path,
                         cal=plain_calendar, instruments=[ins])
    assert verdict["status"] == "OK"
    assert verdict["n_warn"] >= 1
