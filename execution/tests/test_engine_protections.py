"""Engine safety: duplicate protection, kill-switch, weekend skip, live gating, discipline halt."""

from __future__ import annotations

import pytest

from execution.src.brokers import make_broker
from execution.src.config import LIVE_ENV_FLAG, ExecutionConfig, Mode
from execution.src.engine import ExecutionEngine

PRICES = {"SBER": 312.4, "MOEXOG": 8450.0}


def _book(weight=0.30, as_of="2026-07-02 00:00:00+03:00"):
    return {
        "as_of": as_of,
        "net_positions": [{"ticker": "SBER", "weight": weight, "side": "LONG", "sector": "MOEXFN"}],
        "hedge": {"mode": "none", "legs": []},
    }


def _empty_snapshot():
    return {"positions": []}


def test_duplicate_intent_is_not_resubmitted(tmp_config):
    engine = ExecutionEngine(tmp_config(mode=Mode.PAPER))
    first = engine.run_cycle(_book(), PRICES, current_positions=_empty_snapshot())
    assert len(first.submitted) == 1 and not first.duplicates
    # Same book + same (forced) flat positions => identical client_order_id => deduped.
    second = engine.run_cycle(_book(), PRICES, current_positions=_empty_snapshot())
    assert not second.submitted
    assert second.duplicates == first.submitted[0:1] or len(second.duplicates) == 1


def test_kill_switch_halts_and_cancels(tmp_config):
    engine = ExecutionEngine(tmp_config(mode=Mode.PAPER))
    engine.kill(reason="test")
    assert engine.halted
    res = engine.run_cycle(_book(), PRICES, current_positions=_empty_snapshot())
    assert res.halted and not res.submitted
    engine.reset_kill()
    assert not engine.halted
    res2 = engine.run_cycle(_book(), PRICES, current_positions=_empty_snapshot())
    assert res2.submitted


def test_weekend_session_is_skipped(tmp_config):
    engine = ExecutionEngine(tmp_config(mode=Mode.PAPER))
    res = engine.run_cycle(_book(as_of="2026-07-18 00:00:00+03:00"), PRICES,  # Saturday
                           current_positions=_empty_snapshot())
    assert not res.submitted
    assert any(s["reason"] == "non_trading_day" for s in res.skipped)


def test_discipline_critical_halts_cycle(tmp_config):
    engine = ExecutionEngine(tmp_config(mode=Mode.PAPER))
    # td=2 (<= exit_offset) while still long -> critical -> engine halts and places nothing.
    res = engine.run_cycle(_book(as_of="2026-07-16 00:00:00+03:00"), PRICES,
                           current_positions=_empty_snapshot(), anchors={"SBER": "2026-07-20"})
    assert res.halted and not res.submitted
    assert engine.halted


def test_live_refused_without_allow_flag(monkeypatch):
    monkeypatch.delenv(LIVE_ENV_FLAG, raising=False)
    with pytest.raises(PermissionError):
        make_broker(ExecutionConfig(mode=Mode.LIVE, allow_live=False, broker_backend="tinvest"))


def test_live_refused_without_env_flag(monkeypatch):
    monkeypatch.delenv(LIVE_ENV_FLAG, raising=False)
    # allow_live True but the runtime env flag is missing -> still refused.
    with pytest.raises(PermissionError):
        make_broker(ExecutionConfig(mode=Mode.LIVE, allow_live=True, broker_backend="tinvest"))


def test_dry_run_sends_nothing(tmp_config):
    engine = ExecutionEngine(tmp_config(mode=Mode.DRY_RUN))
    res = engine.run_cycle(_book(), PRICES, current_positions=_empty_snapshot())
    assert res.submitted                      # delta orders are produced
    assert all(r["status"] == "DRY_RUN" for r in res.reports)   # but nothing is "sent"
