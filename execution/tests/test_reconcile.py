"""Reconciliation: weight -> lot sizing (round-down), delta vs current, caps, missing prices."""

from __future__ import annotations

import json
from pathlib import Path

from execution.src.config import ExecutionConfig, Mode, SanityLimits
from execution.src.reconcile import reconcile

REPO_ROOT = Path(__file__).resolve().parents[2]


def _book(positions, hedge_legs=None, as_of="2026-07-02 00:00:00+03:00"):
    return {
        "as_of": as_of,
        "net_positions": positions,
        "hedge": {"mode": "sector" if hedge_legs else "none", "legs": hedge_legs or []},
    }


def test_long_entry_lots_rounded_down(tmp_config):
    config = tmp_config(mode=Mode.DRY_RUN, capital=1_000_000.0, lot_sizes={"SBER": 10})
    book = _book([{"ticker": "SBER", "weight": 0.345, "side": "LONG"}])
    res = reconcile(book, {"SBER": 101.0}, current_lots={}, config=config)
    assert len(res.orders) == 1
    order = res.orders[0]
    # 0.345*1_000_000/101 = 3415.8 shares; /10 = 341.58 -> 341 lots (round DOWN)
    assert order.target_lots == 341
    assert order.side == "BUY"
    assert order.quantity_lots == 341
    assert order.to_order_request()["order_type"] == "LIMIT"
    assert order.to_order_request()["limit_price"] == 101.0


def test_delta_against_current_position():
    config = ExecutionConfig(mode=Mode.DRY_RUN, capital=1_000_000.0, lot_sizes={"SBER": 10})
    book = _book([{"ticker": "SBER", "weight": 0.345, "side": "LONG"}])
    # already hold 300 lots -> only buy the 41-lot delta up to 341
    res = reconcile(book, {"SBER": 101.0}, current_lots={"SBER": 300}, config=config)
    assert res.orders[0].quantity_lots == 41
    assert res.orders[0].side == "BUY"


def test_full_exit_sells_everything_when_weight_drops_to_zero():
    config = ExecutionConfig(mode=Mode.DRY_RUN, capital=1_000_000.0, lot_sizes={"SBER": 10})
    res = reconcile(_book([]), {"SBER": 101.0}, current_lots={"SBER": 341}, config=config)
    assert len(res.orders) == 1
    assert res.orders[0].side == "SELL"
    assert res.orders[0].quantity_lots == 341
    assert res.orders[0].target_lots == 0


def test_short_hedge_leg_is_a_sell():
    config = ExecutionConfig(mode=Mode.DRY_RUN, capital=1_000_000.0, lot_sizes={"MOEXOG": 1})
    book = _book([], hedge_legs=[{"instrument": "MOEXOG", "weight": -0.30}])
    res = reconcile(book, {"MOEXOG": 100.0}, current_lots={}, config=config)
    assert res.orders[0].side == "SELL"
    assert res.orders[0].is_hedge is True
    # -0.30*1_000_000/100 = -3000 shares; lot 1 -> -3000 lots
    assert res.orders[0].target_lots == -3000
    assert res.orders[0].quantity_lots == 3000


def test_sanity_notional_cap_binds():
    config = ExecutionConfig(
        mode=Mode.DRY_RUN, capital=1_000_000.0, lot_sizes={"SBER": 1},
        limits=SanityLimits(max_lots_per_name=10_000, max_notional_per_name=50_000.0),
    )
    book = _book([{"ticker": "SBER", "weight": 1.0, "side": "LONG"}])
    res = reconcile(book, {"SBER": 100.0}, current_lots={}, config=config)
    # uncapped: 1_000_000/100 = 10_000 lots; notional cap 50_000/100 = 500 lots
    assert res.orders[0].target_lots == 500
    assert "max_notional_per_name" in res.orders[0].binding


def test_missing_price_is_reported_not_guessed():
    config = ExecutionConfig(mode=Mode.DRY_RUN, capital=1_000_000.0)
    book = _book([{"ticker": "FOO", "weight": 0.2, "side": "LONG"}])
    res = reconcile(book, {}, current_lots={}, config=config)
    assert not res.orders
    assert res.skipped == [{"instrument": "FOO", "reason": "missing_or_nonpositive_price"}]


def test_zero_delta_is_a_noop():
    config = ExecutionConfig(mode=Mode.DRY_RUN, capital=1_000_000.0, lot_sizes={"SBER": 10})
    book = _book([{"ticker": "SBER", "weight": 0.345, "side": "LONG"}])
    res = reconcile(book, {"SBER": 101.0}, current_lots={"SBER": 341}, config=config)
    assert not res.orders
    assert res.noops == ["SBER"]


def _example_book_and_prices():
    book = json.loads((REPO_ROOT / "contracts" / "examples" / "risk_book.example.json").read_text("utf-8"))
    prices = json.loads((REPO_ROOT / "execution" / "examples" / "prices.example.json").read_text("utf-8"))
    return book, prices


def test_real_example_risk_book_reconciles():
    # The canonical example is all-SHADOW (is_production=false). dry-run/paper paper-trade it:
    # 3 long names (BUY) + 2 hedge legs (SELL) = 5 entry orders from a flat book.
    book, prices = _example_book_and_prices()
    config = ExecutionConfig(mode=Mode.DRY_RUN, capital=100_000_000.0)
    res = reconcile(book, prices, current_lots={}, config=config)
    sides = sorted(o.side for o in res.orders)
    assert sides == ["BUY", "BUY", "BUY", "SELL", "SELL"]
    by_name = {o.instrument: o for o in res.orders}
    assert by_name["SBER"].side == "BUY" and by_name["SBER"].quantity_lots > 0
    assert by_name["MOEXOG"].side == "SELL" and by_name["MOEXOG"].is_hedge


def test_paper_paper_trades_the_shadow_book():
    book, prices = _example_book_and_prices()
    assert book["net_positions"] == [] and book["shadow_positions"]   # guard: example is all-shadow
    res = reconcile(book, prices, current_lots={},
                    config=ExecutionConfig(mode=Mode.PAPER, capital=100_000_000.0))
    assert sorted(o.side for o in res.orders) == ["BUY", "BUY", "BUY", "SELL", "SELL"]


def test_live_ignores_shadow_book_zero_orders():
    # Prod-safety: in LIVE, an all-shadow book (gated-out sleeve) places NO real orders.
    book, prices = _example_book_and_prices()
    res = reconcile(book, prices, current_lots={},
                    config=ExecutionConfig(mode=Mode.LIVE, capital=100_000_000.0))
    assert res.orders == []
    assert res.noops == [] and res.skipped == []


def test_live_trades_only_net_positions_not_shadow():
    # A mixed book: one live name + one shadow name. LIVE trades only the live one.
    book = {
        "as_of": "2025-06-02 00:00:00+03:00",
        "net_positions": [{"ticker": "SBER", "weight": 0.2, "side": "LONG", "sector": "MOEXFN"}],
        "hedge": {"mode": "none", "legs": []},
        "shadow_positions": [{"ticker": "LKOH", "weight": 0.3, "side": "LONG", "sector": "MOEXOG"}],
        "shadow_hedge": {"mode": "none", "legs": []},
    }
    prices = {"SBER": 312.4, "LKOH": 7050.0}
    live = reconcile(book, prices, current_lots={}, config=ExecutionConfig(mode=Mode.LIVE, capital=1e8))
    assert [o.instrument for o in live.orders] == ["SBER"]
    paper = reconcile(book, prices, current_lots={}, config=ExecutionConfig(mode=Mode.PAPER, capital=1e8))
    assert sorted(o.instrument for o in paper.orders) == ["LKOH", "SBER"]
