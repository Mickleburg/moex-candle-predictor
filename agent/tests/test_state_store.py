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


def test_orders_tagged_by_capital_state(tmp_path):
    s = _store(tmp_path)
    s.record_order({"client_order_id": "L1", "ticker": "SBER", "side": "BUY", "quantity_lots": 1,
                    "order_type": "LIMIT", "limit_price": 1.0}, trade_date="2026-06-18",
                   phase="eod", status="FILLED", capital_state="live")
    s.record_order({"client_order_id": "S1", "ticker": "LKOH", "side": "BUY", "quantity_lots": 2,
                    "order_type": "LIMIT", "limit_price": 1.0}, trade_date="2026-06-18",
                   phase="eod", status="FILLED", capital_state="shadow")
    assert [o["client_order_id"] for o in s.recent_orders("shadow")] == ["S1"]
    assert [o["client_order_id"] for o in s.recent_orders("live")] == ["L1"]
    assert len(s.recent_orders()) == 2


def test_pnl_attribution_upsert(tmp_path):
    s = _store(tmp_path)
    s.record_pnl_attribution("2026-06-18", "s3_event", realized=0.0, unrealized=5.0, gross=100.0)
    s.record_pnl_attribution("2026-06-18", "s3_event", realized=0.0, unrealized=7.0, gross=120.0)
    rows = {r["sleeve"]: r for r in s.pnl_by_sleeve()}
    assert rows["s3_event"]["unrealized"] == 7.0   # upsert, not double-count


def test_pnl_by_sleeve_returns_latest_snapshot_not_sum(tmp_path):
    # unrealized_pnl is a book mark-to-market SNAPSHOT recorded each day; the current figure is the
    # LATEST day's snapshot, NOT the sum across every day the book was held (which inflates it, 3b).
    s = _store(tmp_path)
    s.record_pnl_attribution("2026-06-17", "s3_event", realized=0.0, unrealized=-100.0, gross=1000.0)
    s.record_pnl_attribution("2026-06-18", "s3_event", realized=0.0, unrealized=-30.0, gross=900.0)
    rows = {r["sleeve"]: r for r in s.pnl_by_sleeve()}
    assert rows["s3_event"]["unrealized"] == -30.0        # latest day, not -130.0 sum
    assert rows["s3_event"]["gross"] == 900.0
    # the invariant #9 demotion gate reads the same latest snapshot, not an inflated cumulative
    assert s.forward_pnl_by_sleeve("live")["s3_event"]["forward_pnl"] == -30.0


def test_record_execution_idempotent(tmp_path):
    # a reclaimed cycle re-emits the SAME execution_report; recording it twice must not duplicate
    # the fill (3d) — the client_order_id is the idempotency key, latest report wins.
    s = _store(tmp_path)
    rep = {"client_order_id": "exec-20260618-shadow-SBER-BUY-5", "ticker": "SBER",
           "status": "FILLED", "filled_quantity_lots": 5, "avg_fill_price": 300.0, "message": "fill"}
    s.record_execution(rep)
    s.record_execution(rep)
    rows = s._conn.execute("SELECT client_order_id, message FROM executions").fetchall()
    assert len(rows) == 1
    s.record_execution({**rep, "message": "re-fill"})     # a later report updates in place
    rows = s._conn.execute("SELECT message FROM executions").fetchall()
    assert len(rows) == 1 and rows[0]["message"] == "re-fill"


def test_kill_switch_persists(tmp_path):
    s = _store(tmp_path)
    assert s.kill_switch_engaged() is False
    s.set_kill_switch(True)
    s.close()
    s2 = StateStore(tmp_path / "s.sqlite")   # reopen
    assert s2.kill_switch_engaged() is True
