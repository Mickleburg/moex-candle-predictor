"""Paper season replay: holdings trace must match the sleeve's -12/-2 hold window.

Reconciling a daily sequence of risk_books through the paper simulator should reproduce exactly the
position pattern the sleeve backtest assumes: flat -> one entry at the window open -> no churn while
held -> one full exit before the ex-gap -> flat. This is the paper<->sim reconciliation in miniature
(hermetic: no ML data, no network).
"""

from __future__ import annotations

from datetime import date, timedelta

from execution.src.config import Mode
from execution.src.engine import ExecutionEngine
from execution.src.trading_calendar import TradingCalendar

ANCHOR = date(2026, 7, 20)
ENTRY, EXIT = 12, 2
PRICES = {"TATN": 700.0, "MOEXOG": 8450.0}


def _build_season():
    cal = TradingCalendar()
    days, expected_held = [], {}
    d = date(2026, 7, 1)
    while d <= date(2026, 7, 17):
        if cal.is_trading_day(d):
            td = cal.trading_days_between(d, ANCHOR)
            held = EXIT < td <= ENTRY
            as_of = f"{d.isoformat()} 00:00:00+03:00"
            expected_held[as_of] = held
            if held:
                book = {"as_of": as_of,
                        "net_positions": [{"ticker": "TATN", "weight": 0.30, "side": "LONG",
                                           "sector": "MOEXOG"}],
                        "hedge": {"mode": "sector", "legs": [{"instrument": "MOEXOG", "weight": -0.30}]}}
            else:
                book = {"as_of": as_of, "net_positions": [], "hedge": {"mode": "none", "legs": []}}
            days.append({"risk_book": book, "prices": PRICES})
        d += timedelta(days=1)
    return days, expected_held


def test_season_holdings_match_discipline_window(tmp_config):
    days, expected_held = _build_season()
    engine = ExecutionEngine(tmp_config(mode=Mode.PAPER))
    out = engine.run_season(days, anchors={"TATN": ANCHOR.isoformat()})

    # Holdings each day reflect the window: long iff inside (exit, entry].
    for as_of, held in expected_held.items():
        pos = out["held_by_day"][as_of]
        if held:
            assert pos.get("TATN", 0) > 0, (as_of, pos)
            assert pos.get("MOEXOG", 0) < 0, (as_of, pos)   # short hedge leg
        else:
            assert pos.get("TATN", 0) == 0, (as_of, pos)
            assert pos.get("MOEXOG", 0) == 0, (as_of, pos)

    # Exactly one entry and one exit on the long leg (no churn while held).
    tatn_buys = sum(1 for c in out["cycles"] for o in c.submitted
                    if o["ticker"] == "TATN" and o["side"] == "BUY")
    tatn_sells = sum(1 for c in out["cycles"] for o in c.submitted
                     if o["ticker"] == "TATN" and o["side"] == "SELL")
    assert tatn_buys == 1, tatn_buys
    assert tatn_sells == 1, tatn_sells

    # Season ends flat.
    assert out["final_positions"] == {}


def test_no_orders_on_steady_hold_days(tmp_config):
    days, _ = _build_season()
    engine = ExecutionEngine(tmp_config(mode=Mode.PAPER))
    out = engine.run_season(days, anchors={"TATN": ANCHOR.isoformat()})
    # Total orders over the whole season = 2 legs x (enter + exit) = 4, not one-per-day.
    total = sum(len(c.submitted) for c in out["cycles"])
    assert total == 4, total
