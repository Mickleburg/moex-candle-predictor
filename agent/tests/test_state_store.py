"""State-store: idempotency, dedup, P&L attribution, kill-switch, restart durability."""

from __future__ import annotations

from agent.src.state_store import StateStore


def _store(tmp_path):
    return StateStore(tmp_path / "s.sqlite")


def test_begin_cycle_idempotent(tmp_path):
    s = _store(tmp_path)
    first = s.begin_cycle("2026-06-18", "eod", mode="paper", block_mode="mock", as_of="x")
    assert first["started"]
    s.finish_cycle("2026-06-18", "eod", "completed", result={"ok": True})
    again = s.begin_cycle("2026-06-18", "eod", mode="paper", block_mode="mock", as_of="x")
    assert not again["started"]
    assert again["prior"]["status"] == "completed"
    # force re-opens the slot
    forced = s.begin_cycle("2026-06-18", "eod", mode="paper", block_mode="mock", as_of="x", force=True)
    assert forced["started"]


def test_running_cycle_is_reclaimed_after_crash(tmp_path):
    s = _store(tmp_path)
    s.begin_cycle("2026-06-18", "eod", mode="paper", block_mode="mock", as_of="x")  # left 'running'
    # a fresh process can reclaim a non-terminal cycle (crash recovery)
    reclaim = s.begin_cycle("2026-06-18", "eod", mode="paper", block_mode="mock", as_of="x")
    assert reclaim["started"]


def test_positions_upsert_and_close(tmp_path):
    s = _store(tmp_path)
    s.upsert_position("SBER", 10, 300.0, 305.0, sleeve_contributions={"s3_event": 0.34})
    assert s.get_positions()[0]["ticker"] == "SBER"
    s.upsert_position("SBER", 0, 0.0, None)   # close
    assert s.get_positions() == []


def test_order_dedup(tmp_path):
    s = _store(tmp_path)
    order = {"client_order_id": "c1", "ticker": "SBER", "side": "BUY", "quantity_lots": 1,
             "order_type": "LIMIT", "limit_price": 300.0}
    assert not s.order_exists("c1")
    s.record_order(order, trade_date="2026-06-18", phase="eod", status="PLACED")
    assert s.order_exists("c1")
    assert len(s.open_orders()) == 1


def test_pnl_attribution_upsert(tmp_path):
    s = _store(tmp_path)
    s.record_pnl_attribution("2026-06-18", "s3_event", realized=0.0, unrealized=5.0, gross=100.0)
    s.record_pnl_attribution("2026-06-18", "s3_event", realized=0.0, unrealized=7.0, gross=120.0)
    rows = {r["sleeve"]: r for r in s.pnl_by_sleeve()}
    assert rows["s3_event"]["unrealized"] == 7.0   # upsert, not double-count


def test_kill_switch_persists(tmp_path):
    s = _store(tmp_path)
    assert s.kill_switch_engaged() is False
    s.set_kill_switch(True)
    s.close()
    s2 = StateStore(tmp_path / "s.sqlite")   # reopen
    assert s2.kill_switch_engaged() is True
