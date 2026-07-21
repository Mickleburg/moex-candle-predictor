# -*- coding: utf-8 -*-
"""Tests for the independent no-lookahead verifier and the scheduled-refresh determinism.

The verifier (`verify_dividend_feed`) is the gate the EOD refresh runs before letting a new CSV
replace the live one, so each no-lookahead invariant gets an explicit failing case. `_load_pub` is
monkeypatched to a synthetic disclosure history so the logic is tested in isolation from the live
parquets. Run: python -m pytest llm/test_dividend_refresh.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent / "scripts"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO))
import verify_dividend_feed as vfy  # noqa: E402
import build_dividend_calendar_upcoming as bld  # noqa: E402
import refresh_dividend_feed as rf  # noqa: E402


SYN_GUID = "SYN123-B-B"


@pytest.fixture
def syn_pub(monkeypatch):
    """One real board disclosure for TEST on 2026-05-01 (a reco-class title)."""
    df = pd.DataFrame({
        "event_name": ["Решения совета директоров (наблюдательного совета)"],
        "pub": [pd.Timestamp("2026-05-01 18:00:00")],
        "pseudo_guid": [SYN_GUID],
    })
    monkeypatch.setattr(vfy, "_load_pub", lambda t: df)
    return df


def _row(**over):
    rec = pd.Timestamp(over.get("record_date", "2026-07-01"))
    ex = pd.Timestamp(np.busday_offset(rec.date(), -1, roll="backward"))
    base = dict(ticker="TEST", record_date=rec.date().isoformat(), ex_date=ex.date().isoformat(),
                board_reco_date="2026-05-01", agm_date="", value="10.00", status="recommended",
                source_url="https://www.e-disclosure.ru/portal/event.aspx?EventId=" + SYN_GUID,
                as_of="2026-06-20", confidence="medium", notes="")
    base.update(over)
    return pd.DataFrame([base])


def test_clean_feed_passes(syn_pub):
    ok, _, stats = vfy.verify(_row(), pd.Timestamp("2026-06-20"))
    assert ok and stats["violations"] == 0


def test_future_board_reco_flagged(syn_pub):
    # board_reco after as_of AND with no real disclosure on that day -> no-lookahead violation
    ok, report, _ = vfy.verify(_row(board_reco_date="2026-09-01"), pd.Timestamp("2026-06-20"))
    assert not ok
    assert any("FUTURE" in l or "no matching reco-class" in l for l in report)


def test_board_reco_after_as_of_flagged(syn_pub):
    # real disclosure date (2026-05-01) but the as_of is earlier -> we couldn't have known yet
    ok, report, _ = vfy.verify(_row(), pd.Timestamp("2026-04-01"))
    assert not ok
    assert any("> as_of" in l for l in report)


def test_insufficient_lead_flagged(syn_pub):
    # record only ~5 trading days after the board reco -> can't enter 12 TD ahead
    ok, report, _ = vfy.verify(_row(record_date="2026-05-08"), pd.Timestamp("2026-06-20"))
    assert not ok
    assert any("record-12TD" in l for l in report)


def test_ex_date_inconsistent_flagged(syn_pub):
    ok, report, _ = vfy.verify(_row(ex_date="2026-07-01"), pd.Timestamp("2026-06-20"))
    assert not ok
    assert any("ex_date" in l for l in report)


@pytest.mark.parametrize("bad", ["", "n/a", "0"])
def test_bad_value_flagged(syn_pub, bad):
    ok, report, _ = vfy.verify(_row(value=bad), pd.Timestamp("2026-06-20"))
    assert not ok
    assert any("value" in l for l in report)


def test_build_is_deterministic():
    # idempotency precondition: same cache -> identical feed twice (so refresh re-runs are no-ops)
    a, _, _ = bld.feed_frames()
    b, _, _ = bld.feed_frames()
    pd.testing.assert_frame_equal(a, b)


def test_promote_realized_only_promotes_verified_rows(monkeypatch):
    # the rolling-window fix: realized events are re-verified per row, and ONLY those that pass the
    # independent no-lookahead check are promoted into history (AAA passes, BBB fails -> only AAA).
    import backend.dividends as bd
    passed = pd.DataFrame([
        dict(ticker="AAA", record_date="2026-05-04", value="10.00"),
        dict(ticker="BBB", record_date="2026-05-05", value="20.00"),
    ])
    monkeypatch.setattr(rf.vfy, "verify",
                        lambda one, as_of: (one.iloc[0]["ticker"] == "AAA", [], {}))
    captured = {}

    def fake_promote(ev, source, run_date=None):
        captured["ev"], captured["source"] = ev.copy(), source
        return {"rows_added_this_run": int(len(ev)), "source": source}

    monkeypatch.setattr(bd, "promote_events", fake_promote)
    res = rf.promote_realized(passed, pd.Timestamp("2026-07-01"))

    assert res == {"eligible": 2, "verified": 1, "added": 1}
    assert list(captured["ev"]["ticker"]) == ["AAA"]
    assert {"ticker", "date", "value"} <= set(captured["ev"].columns)   # record_date -> date
    assert captured["source"] == "e-disclosure"


def test_promote_realized_empty_is_noop(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(rf.vfy, "verify", lambda *a: called.__setitem__("n", called["n"] + 1))
    res = rf.promote_realized(pd.DataFrame(columns=["ticker", "record_date", "value"]),
                              pd.Timestamp("2026-07-01"))
    assert res == {"eligible": 0, "verified": 0, "added": 0} and called["n"] == 0
