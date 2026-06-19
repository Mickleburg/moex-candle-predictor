"""End-to-end orchestrator tests on mock blocks (acceptance criteria for the agent block)."""

from __future__ import annotations

import json

from agent.src.adapters import mock
from agent.tests.conftest import make_orch

TD = "2026-06-18"   # a MOEX trading day (Thu; not in the holiday clusters)


def test_full_eod_cycle_paper(tmp_path):
    orch, store, note = make_orch(tmp_path)
    out = orch.run_eod_cycle(trade_date=TD)

    assert out["status"] == "completed"
    result = out["result"]
    # contract shape
    assert result["mode"] == "paper"
    assert set(result["evaluated_tickers"]) == set(orch.config.universe)
    assert len(result["selected_orders"]) > 0
    for o in result["selected_orders"]:
        assert {"ticker", "side", "quantity_lots", "order_type", "client_order_id"} <= set(o)
        assert o["quantity_lots"] >= 1 and o["order_type"] == "LIMIT"

    # state persisted: book + orders + per-sleeve P&L + cycle result file
    assert store.get_positions(), "paper fills should leave a book"
    assert store.all_orders(), "orders should be recorded"
    assert all(o["status"] == "FILLED" for o in store.all_orders())   # paper fills immediately
    pnl = {row["sleeve"]: row for row in store.pnl_by_sleeve()}
    assert "s3_event" in pnl
    assert (tmp_path / "cycles" / f"{TD}_eod.json").exists()
    assert (tmp_path / "shadow.jsonl").exists()
    assert any("EOD digest" in s for s in note.subjects())


def test_idempotent_rerun_is_noop(tmp_path):
    orch, store, _ = make_orch(tmp_path)
    first = orch.run_eod_cycle(trade_date=TD)
    assert first["status"] == "completed"
    book_after_first = sorted((p["ticker"], p["lots"]) for p in store.get_positions())
    n_exec = len(store.open_orders())

    second = orch.run_eod_cycle(trade_date=TD)
    assert second["status"] == "skipped_idempotent"
    # nothing changed
    assert sorted((p["ticker"], p["lots"]) for p in store.get_positions()) == book_after_first
    assert len(store.open_orders()) == n_exec


def test_restart_recovers_state(tmp_path):
    orch1, store1, _ = make_orch(tmp_path)
    orch1.run_eod_cycle(trade_date=TD)
    book = sorted((p["ticker"], p["lots"]) for p in store1.get_positions())
    store1.close()

    # brand-new process/orchestrator over the SAME db file
    orch2, store2, _ = make_orch(tmp_path)
    assert sorted((p["ticker"], p["lots"]) for p in store2.get_positions()) == book
    assert store2.get_cycle(TD, "eod")["status"] == "completed"
    # re-running the same day is still idempotent after restart
    assert orch2.run_eod_cycle(trade_date=TD)["status"] == "skipped_idempotent"


def test_integrity_halt_blocks_trading(tmp_path):
    orch, store, note = make_orch(tmp_path, backend=mock.MockBackend(halt=True,
                                                                    halt_reasons=["stale bar for SBER"]))
    out = orch.run_eod_cycle(trade_date=TD)
    assert out["status"] == "halted"
    assert out["result"]["selected_orders"] == []
    assert store.get_positions() == []        # never traded
    assert store.get_cycle(TD, "eod")["status"] == "halted"
    assert any("DATA HALT" in s for s in note.subjects())


def test_regime_gate_cuts_gross(tmp_path):
    full, _, _ = make_orch(tmp_path / "a", combiner=mock.MockCombiner(exposure_scalar=1.0))
    cut, _, _ = make_orch(tmp_path / "b", combiner=mock.MockCombiner(exposure_scalar=0.4,
                                                                    regime_novel=True))
    g_full = full.run_eod_cycle(trade_date=TD)["result"]["risk_summary"]["directional_gross"]
    g_cut = cut.run_eod_cycle(trade_date=TD)["result"]["risk_summary"]["directional_gross"]
    assert g_cut < g_full
    assert abs(g_cut - 0.4 * g_full) < 1e-6


def test_kill_switch_skips_execution_but_monitors(tmp_path):
    orch, store, note = make_orch(tmp_path)
    store.set_kill_switch(True)
    out = orch.run_eod_cycle(trade_date=TD)
    assert out["status"] == "killed"
    assert out["result"]["selected_orders"] == []   # no trading
    assert store.get_positions() == []
    # monitoring still ran: a shadow-log line + risk summary were produced
    assert (tmp_path / "shadow.jsonl").exists()
    assert out["result"]["risk_summary"]["kill_switch"] is True


def test_dry_run_does_not_change_book(tmp_path):
    orch, store, _ = make_orch(tmp_path, mode="dry-run")
    out = orch.run_eod_cycle(trade_date=TD)
    assert out["status"] == "completed"
    assert out["result"]["mode"] == "dry-run"
    assert len(out["result"]["selected_orders"]) > 0   # orders computed
    assert store.get_positions() == []                 # but book untouched


def test_live_without_gate_is_forced_to_paper(tmp_path):
    orch, _, _ = make_orch(tmp_path, mode="live", enable_live=False)
    out = orch.run_eod_cycle(trade_date=TD)
    assert out["result"]["mode"] == "paper"   # paper-first invariant


def test_non_trading_day_skips(tmp_path):
    orch, _, _ = make_orch(tmp_path)
    assert orch.run_eod_cycle(trade_date="2026-06-12")["status"] == "skipped"   # Russia Day


def test_preopen_halt_cancels_open_orders(tmp_path):
    orch, store, note = make_orch(tmp_path)
    orch.run_eod_cycle(trade_date=TD)
    # simulate a resting limit order left for the next session (what a live broker leaves)
    store.record_order({"client_order_id": "resting-1", "ticker": "SBER", "side": "BUY",
                        "quantity_lots": 1, "order_type": "LIMIT", "limit_price": 300.0},
                       trade_date=TD, phase="eod", status="PLACED")
    assert store.open_orders()

    # next morning, overnight HALT -> cancel everything
    orch.adapters.backend = mock.MockBackend(halt=True, halt_reasons=["overnight gap"])
    out = orch.run_preopen(trade_date=TD)
    assert out["status"] == "halted"
    assert store.open_orders() == []
    assert any("HALT" in s for s in note.subjects())


def test_cycle_result_is_valid_json_file(tmp_path):
    orch, _, _ = make_orch(tmp_path)
    orch.run_eod_cycle(trade_date=TD)
    payload = json.loads((tmp_path / "cycles" / f"{TD}_eod.json").read_text(encoding="utf-8"))
    assert payload["risk_summary"]["block_modes"]["sleeve"] == "mock"
    assert "sleeve_pnl" in payload["risk_summary"]
