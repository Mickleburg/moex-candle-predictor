"""P&L attribution: realized results at close, and lot-size scaling.

Guards the defect family found on 2026-08-25 while answering "did the sandbox make money?": the
July round trip opened 07-10 and closed 07-20, yet every `realized_pnl` in the store was 0.0, no
attribution row existed for the closing day at all, and marks ignored round-lot sizes. The paper
track — the forward evidence the H9 gate rests on — could not report its own result.
"""

import pytest

from agent.src import pnl




# --- realized P&L: the July-2026 round trip vanished because closing a position dropped it -------

def _pos(ticker, lots, avg, *, track="shadow", hedge=False, sleeve="s3_event", lot_note=None):
    return {"ticker": ticker, "lots": lots, "avg_price": avg, "track": track,
            "capital_state": track, "is_hedge": hedge,
            "sleeve_contributions": {} if hedge else {sleeve: 1.0}}


def _fill(ticker, side, lots, price, track="shadow"):
    coid = f"exec-20260720-{track}-{ticker}-{side}-{lots}"   # compact date: exec-YYYYMMDD-TRACK-...
    return ({"client_order_id": coid, "ticker": ticker, "side": side},
            {"client_order_id": coid, "ticker": ticker, "status": "FILLED",
             "filled_quantity_lots": lots, "avg_fill_price": price})


def test_closing_a_long_realizes_pnl_scaled_by_lot_size():
    # SNGS trades in lots of 100 — ignoring that understated every rouble figure 100-fold
    prior = [_pos("SNGS", 253, 15.35)]
    o, r = _fill("SNGS", "SELL", 253, 13.97)
    got = pnl.attribute_realized_pnl(prior, [o], [r], {"SNGS": 100})
    assert got["s3_event"] == pytest.approx(253 * 100 * (13.97 - 15.35))
    # without the lot map it would be 100x too small — the exact bug this guards
    small = pnl.attribute_realized_pnl(prior, [o], [r], {})
    assert small["s3_event"] == pytest.approx(253 * (13.97 - 15.35))


def test_closing_a_short_hedge_realizes_the_opposite_sign():
    # the July hedge was SHORT the sector index and PROFITED as it fell
    prior = [_pos("MOEXFN", -129, 8418.20, hedge=True), _pos("SBER", 208, 294.54)]
    o1, r1 = _fill("MOEXFN", "BUY", 129, 7763.76)      # buying back a short
    o2, r2 = _fill("SBER", "SELL", 208, 258.14)
    got = pnl.attribute_realized_pnl(prior, [o1, o2], [r1, r2], {})
    # hedge gain is pooled into the sleeve that carried the directional risk
    expected = 208 * (258.14 - 294.54) + 129 * (8418.20 - 7763.76)
    assert sum(got.values()) == pytest.approx(expected)
    assert got["s3_event"] == pytest.approx(expected)


def test_opening_or_adding_realizes_nothing():
    assert pnl.attribute_realized_pnl([], *_two([_fill("SBER", "BUY", 10, 300.0)]), {}) == {}
    prior = [_pos("SBER", 100, 290.0)]
    o, r = _fill("SBER", "BUY", 50, 300.0)             # adding to the same side
    assert pnl.attribute_realized_pnl(prior, [o], [r], {}) == {}


def test_partial_close_realizes_only_the_closed_lots():
    prior = [_pos("TATN", 712, 463.0)]
    o, r = _fill("TATN", "SELL", 300, 438.20)
    got = pnl.attribute_realized_pnl(prior, [o], [r], {})
    assert got["s3_event"] == pytest.approx(300 * (438.20 - 463.0))


def test_unfilled_and_foreign_track_reports_are_ignored():
    prior = [_pos("SBER", 208, 294.54)]
    o, r = _fill("SBER", "SELL", 208, 258.14)
    assert pnl.attribute_realized_pnl(prior, [o], [{**r, "status": "REJECTED"}], {}) == {}
    # a live-track fill must not realize against the shadow book
    o2, r2 = _fill("SBER", "SELL", 208, 258.14, track="live")
    assert pnl.attribute_realized_pnl(prior, [o2], [r2], {}) == {}


def test_book_marks_scale_by_lot_size():
    book = [{"ticker": "SNGS", "lots": 253, "avg_price": 15.35, "last_price": 13.97,
             "is_hedge": False, "sleeve_contributions": {"s3_event": 1.0}}]
    got = pnl.attribute_book_pnl(book, {"SNGS": 100})
    assert got["s3_event"]["unrealized"] == pytest.approx(253 * 100 * (13.97 - 15.35))
    assert got["s3_event"]["gross"] == pytest.approx(abs(253 * 100 * 13.97))


def _two(pairs):
    return [p[0] for p in pairs], [p[1] for p in pairs]
