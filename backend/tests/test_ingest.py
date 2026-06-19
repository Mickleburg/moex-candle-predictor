"""Ingest tests: incremental fetch + proven idempotency with an injected fetcher."""

from datetime import date

import pandas as pd

from backend import store
from backend.ingest import ingest_instrument, run_ingest
from backend.universe import Instrument
from backend.tests.conftest import make_candles


class FakeServer:
    """Stands in for MOEX ISS: holds a full series, returns the requested window."""

    def __init__(self, ticker, timeframe, all_days):
        self.full = make_candles(ticker, timeframe, all_days)
        self.calls = []

    def fetch(self, ticker, timeframe, date_from, date_to):
        self.calls.append((date_from, date_to))
        lo = pd.Timestamp(date_from)
        hi = pd.Timestamp(date_to) + pd.Timedelta(days=1)
        win = self.full[(self.full["begin"] >= lo) & (self.full["begin"] < hi)].copy()
        return win if not win.empty else None


def test_backfill_then_incremental(tmp_path):
    days = [d.date() for d in pd.bdate_range("2026-03-02", "2026-03-20")]
    server = FakeServer("SBER", "1D", days)
    ins = Instrument("SBER", "1D", "share", required=True)

    # day 1: backfill up to mid-series
    r1 = ingest_instrument(ins, server.fetch, today=date(2026, 3, 11),
                           data_dir=tmp_path, backfill=True)
    assert r1.status == "ok"
    n1 = len(store.load_ticker("SBER", "1D", tmp_path))
    assert n1 == 8                                  # Mar 2..11 business days

    # day 2: incremental to end of series adds only the new bars
    r2 = ingest_instrument(ins, server.fetch, today=date(2026, 3, 20), data_dir=tmp_path)
    assert r2.status == "ok"
    assert r2.added == len(days) - n1
    assert len(store.load_ticker("SBER", "1D", tmp_path)) == len(days)
    # incremental re-fetch started from the last stored day (one-day overlap), not history
    assert server.calls[-1][0] == "2026-03-11"


def test_rerun_is_idempotent(tmp_path):
    days = [d.date() for d in pd.bdate_range("2026-03-02", "2026-03-20")]
    server = FakeServer("SBER", "1D", days)
    ins = Instrument("SBER", "1D", "share", required=True)

    ingest_instrument(ins, server.fetch, today=date(2026, 3, 20),
                      data_dir=tmp_path, backfill=True)
    snapshot = store.load_ticker("SBER", "1D", tmp_path)
    files_before = store.store_files("SBER", "1D", tmp_path)

    # re-run on the same day: nothing added, store byte-stable, single file
    r = ingest_instrument(ins, server.fetch, today=date(2026, 3, 20), data_dir=tmp_path)
    assert r.added == 0
    assert r.status == "up_to_date"
    after = store.load_ticker("SBER", "1D", tmp_path)
    pd.testing.assert_frame_equal(snapshot, after)
    assert store.store_files("SBER", "1D", tmp_path) == files_before


def test_missing_instrument_skipped_without_backfill(tmp_path):
    ins = Instrument("ZZZZ", "1D", "share", required=True)
    server = FakeServer("ZZZZ", "1D", [date(2026, 3, 2)])
    r = ingest_instrument(ins, server.fetch, today=date(2026, 3, 20), data_dir=tmp_path)
    assert r.status == "skipped"
    assert store.store_files("ZZZZ", "1D", tmp_path) == []


def test_run_ingest_report_shape(tmp_path):
    days = [d.date() for d in pd.bdate_range("2026-03-02", "2026-03-06")]
    server = FakeServer("SBER", "1D", days)
    ins = Instrument("SBER", "1D", "share", required=True)
    report = run_ingest(today=date(2026, 3, 6), fetch_fn=server.fetch, data_dir=tmp_path,
                        backfill=True, instruments=[ins])
    assert report["status"] == "ok"
    assert report["n_instruments"] == 1
    assert report["results"][0]["ticker"] == "SBER"
