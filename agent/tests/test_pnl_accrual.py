"""Guard: the post-fill book carries the ENTRY cost basis, so unrealized P&L accrues.

Regression for the bug where `_post_fill_book` overwrote every held position's `avg_price`
with the current market mark, making `attribute_book_pnl` compute lots*(last-avg)=0 forever —
which silently disabled the invariant #9 forward-P&L demotion gate. The existing suite could
not see this: the mock backend hands out a constant synthetic price per ticker, so last==avg
regardless of whether cost basis is preserved. Here prices actually move.
"""

from __future__ import annotations

import pytest

from agent.src import pnl
from agent.src.adapters.live import _post_fill_book


def test_hold_preserves_cost_basis_and_accrues_unrealized():
    # bought earlier at 315, held this cycle (no new order), mark has risen to 320
    current = [{"track": "live", "ticker": "SBER", "lots": 100, "avg_price": 315.0}]
    risk_book = {"net_positions": [{"ticker": "SBER", "sleeve_contributions": {"s3_event": 1.0}}]}
    book = _post_fill_book(current, submitted=[], reports=[], risk_book=risk_book,
                           prices={"SBER": 320.0})

    row = next(p for p in book if p["ticker"] == "SBER")
    assert row["avg_price"] == 315.0        # cost basis kept, NOT reset to the 320 mark
    assert row["last_price"] == 320.0

    attr = pnl.attribute_book_pnl(book)
    assert attr["s3_event"]["unrealized"] == pytest.approx(100 * (320.0 - 315.0))


def test_new_fill_weighted_averages_into_cost_basis():
    current = [{"track": "live", "ticker": "SBER", "lots": 100, "avg_price": 300.0}]
    coid = "exec-20260710-live-SBER-BUY-100"
    submitted = [{"client_order_id": coid, "ticker": "SBER", "side": "BUY", "limit_price": 320.0}]
    reports = [{"client_order_id": coid, "ticker": "SBER", "status": "FILLED",
                "filled_quantity_lots": 100, "avg_fill_price": 320.0}]
    book = _post_fill_book(current, submitted, reports, risk_book={}, prices={"SBER": 320.0})

    row = next(p for p in book if p["ticker"] == "SBER")
    assert row["lots"] == 200
    assert row["avg_price"] == pytest.approx((100 * 300.0 + 100 * 320.0) / 200)  # 310
