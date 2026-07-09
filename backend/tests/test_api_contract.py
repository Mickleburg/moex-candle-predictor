"""Contract test: freeze the in-process API the orchestrator depends on.

If this breaks, the agent's `backend wired live in-process` integration breaks -- change
the facade deliberately, not by accident.
"""

import inspect
from datetime import date

import pandas as pd

from backend import api, store
from backend.universe import Instrument
from backend.tests.conftest import make_candles


def test_api_exports_stable_surface():
    expected = {
        "run_ingest", "check_integrity", "is_tradeable", "get_calendar",
        "is_trading_day", "trading_days_between", "next_trading_day", "prev_trading_day",
        "last_trading_day_on_or_before", "add_trading_days", "get_instrument", "figi_for",
        "lot_for", "price_step_for", "round_to_lot", "round_price", "all_verified",
        "unverified_figis",
    }
    assert expected <= set(api.__all__)
    for name in expected:
        assert hasattr(api, name), name


def test_run_ingest_signature_and_report_keys(tmp_path):
    sig = inspect.signature(api.run_ingest)
    for kw in ("today", "fetch_fn", "data_dir", "backfill", "with_futures", "instruments"):
        assert kw in sig.parameters, kw

    days = [d.date() for d in pd.bdate_range("2026-03-02", "2026-03-06")]
    full = make_candles("SBER", "1D", days)

    def fake_fetch(ticker, timeframe, date_from, date_to):
        lo, hi = pd.Timestamp(date_from), pd.Timestamp(date_to) + pd.Timedelta(days=1)
        win = full[(full["begin"] >= lo) & (full["begin"] < hi)]
        return win if not win.empty else None

    ins = Instrument("SBER", "1D", "share", required=True)
    report = api.run_ingest(today=date(2026, 3, 6), fetch_fn=fake_fetch, data_dir=tmp_path,
                            backfill=True, instruments=[ins])
    for key in ("status", "reference_date", "n_instruments", "n_errors", "n_updated", "results"):
        assert key in report, key
    assert report["status"] in ("ok", "error")


def test_check_integrity_report_keys_and_gate(tmp_path, trading_days, plain_calendar):
    df = make_candles("SBER", "1D", trading_days)
    store.write_consolidated(df, "SBER", "1D", data_dir=tmp_path)
    ins = Instrument("SBER", "1D", "share", required=True)
    verdict = api.check_integrity(ref=trading_days[-1], data_dir=tmp_path,
                                  cal=plain_calendar, instruments=[ins])
    for key in ("status", "reference_date", "n_fail", "n_warn", "reasons", "warnings", "checks"):
        assert key in verdict, key
    assert verdict["status"] in ("OK", "HALT")
    assert api.is_tradeable(verdict) is (verdict["status"] == "OK")


def test_calendar_and_instruments_callable():
    assert isinstance(api.is_trading_day("2026-06-15"), bool)
    assert isinstance(api.trading_days_between("2026-06-01", "2026-06-15"), int)
    assert api.figi_for("SBER").startswith("BBG")
    assert api.lot_for("SBER") >= 1
    assert isinstance(api.all_verified(), bool)
