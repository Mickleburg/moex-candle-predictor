"""End-to-end orchestrator tests on mock blocks (acceptance criteria for the agent block).

Since the shadow gate landed, the H9 mock sleeve is is_production=false -> SHADOW: it is
paper-traded into the shadow track (forward-P&L measurement) with ZERO live capital. Tests
that need a LIVE book use MockSleeve(is_production=True) (a hypothetical signed-off sleeve).
"""

from __future__ import annotations

import json

from agent.src.adapters import mock
from agent.tests.conftest import make_orch

TD = "2026-06-18"   # a MOEX trading day (Thu; not in the holiday clusters)


def test_h9_default_is_shadow_paper_traded(tmp_path):
    orch, store, note = make_orch(tmp_path)              # default mock sleeve: is_production=false
    out = orch.run_eod_cycle(trade_date=TD)

    assert out["status"] == "completed"
    result = out["result"]
    assert result["mode"] == "paper"
    # paper folds the shadow book -> there ARE paper orders, but they are SHADOW capital
    assert len(result["selected_orders"]) > 0
    rs = result["risk_summary"]
    assert rs["directional_gross"] == 0.0          # ZERO live capital for an unproven sleeve
    assert rs["shadow_gross"] > 0.0
    assert any(g["sleeve"] == "s3_event" and g["capital_state"] == "shadow" for g in rs["gating"])

    # state: live track empty, shadow track holds the book
    assert store.get_positions("live") == []
    assert store.get_positions("shadow"), "shadow book should be paper-filled"
    # per-sleeve P&L is attributed to the SHADOW track, not live
    pnl = {(r["sleeve"], r["capital_state"]) for r in store.pnl_by_sleeve()}
    assert ("s3_event", "shadow") in pnl
    assert ("s3_event", "live") not in pnl
    assert (tmp_path / "shadow.jsonl").exists()
    assert any("EOD digest" in s for s in note.subjects())


def test_production_sleeve_takes_live_capital(tmp_path):
    orch, store, _ = make_orch(tmp_path, sleeve=mock.MockSleeve(is_production=True))
    out = orch.run_eod_cycle(trade_date=TD)
    rs = out["result"]["risk_summary"]
    assert rs["directional_gross"] > 0.0           # live capital deployed
    assert store.get_positions("live"), "a signed-off sleeve trades the live book"
    assert any(g["capital_state"] == "live" for g in rs["gating"])
    pnl = {(r["sleeve"], r["capital_state"]) for r in store.pnl_by_sleeve()}
    assert ("s3_event", "live") in pnl


def test_forward_pnl_gate_demotes_production_sleeve(tmp_path):
    # a signed-off sleeve whose LIVE forward P&L turned negative must be pulled back to shadow
    orch, store, _ = make_orch(tmp_path, sleeve=mock.MockSleeve(is_production=True))
    store.record_pnl_attribution("2026-06-17", "s3_event", realized=0.0, unrealized=-500.0,
                                 gross=1000.0, capital_state="live")   # prior negative forward P&L
    out = orch.run_eod_cycle(trade_date=TD)
    rs = out["result"]["risk_summary"]
    assert rs["directional_gross"] == 0.0          # demoted -> no live capital
    assert rs["shadow_gross"] > 0.0
    g = next(x for x in rs["gating"] if x["sleeve"] == "s3_event")
    assert g["capital_state"] == "shadow" and "forward_pnl" in g["reason"]
    assert store.get_positions("live") == []


def test_idempotent_rerun_is_noop(tmp_path):
    orch, store, _ = make_orch(tmp_path)
    first = orch.run_eod_cycle(trade_date=TD)
    assert first["status"] == "completed"
    book_after_first = sorted((p["ticker"], p["lots"], p["capital_state"])
                              for p in store.get_positions(None))
    second = orch.run_eod_cycle(trade_date=TD)
    assert second["status"] == "skipped_idempotent"
    assert sorted((p["ticker"], p["lots"], p["capital_state"])
                  for p in store.get_positions(None)) == book_after_first


def test_restart_recovers_state(tmp_path):
    orch1, store1, _ = make_orch(tmp_path)
    orch1.run_eod_cycle(trade_date=TD)
    book = sorted((p["ticker"], p["lots"], p["capital_state"]) for p in store1.get_positions(None))
    store1.close()

    orch2, store2, _ = make_orch(tmp_path)              # new process over the SAME db file
    assert sorted((p["ticker"], p["lots"], p["capital_state"])
                  for p in store2.get_positions(None)) == book
    assert store2.get_cycle(TD, "eod")["status"] == "completed"
    assert orch2.run_eod_cycle(trade_date=TD)["status"] == "skipped_idempotent"


def test_integrity_halt_blocks_trading(tmp_path):
    orch, store, note = make_orch(tmp_path, backend=mock.MockBackend(halt=True,
                                                                    halt_reasons=["stale bar for SBER"]))
    out = orch.run_eod_cycle(trade_date=TD)
    assert out["status"] == "halted"
    assert out["result"]["selected_orders"] == []
    assert store.get_positions(None) == []        # never traded (live or shadow)
    assert store.get_cycle(TD, "eod")["status"] == "halted"
    assert any("DATA HALT" in s for s in note.subjects())


def test_regime_gate_cuts_gross(tmp_path):
    # use a signed-off (live) sleeve so the cut shows on the LIVE directional gross
    full, _, _ = make_orch(tmp_path / "a", sleeve=mock.MockSleeve(is_production=True),
                           combiner=mock.MockCombiner(exposure_scalar=1.0))
    cut, _, _ = make_orch(tmp_path / "b", sleeve=mock.MockSleeve(is_production=True),
                          combiner=mock.MockCombiner(exposure_scalar=0.4, regime_novel=True))
    g_full = full.run_eod_cycle(trade_date=TD)["result"]["risk_summary"]["directional_gross"]
    g_cut = cut.run_eod_cycle(trade_date=TD)["result"]["risk_summary"]["directional_gross"]
    assert g_full > 0 and g_cut < g_full
    assert abs(g_cut - 0.4 * g_full) < 1e-6


def test_kill_switch_skips_execution_but_monitors(tmp_path):
    orch, store, _ = make_orch(tmp_path)
    store.set_kill_switch(True)
    out = orch.run_eod_cycle(trade_date=TD)
    assert out["status"] == "killed"
    assert out["result"]["selected_orders"] == []   # no trading at all
    assert store.get_positions(None) == []
    assert (tmp_path / "shadow.jsonl").exists()      # monitoring still ran
    assert out["result"]["risk_summary"]["kill_switch"] is True


def test_dry_run_does_not_change_book(tmp_path):
    orch, store, _ = make_orch(tmp_path, mode="dry-run")
    out = orch.run_eod_cycle(trade_date=TD)
    assert out["status"] == "completed"
    assert out["result"]["mode"] == "dry-run"
    assert len(out["result"]["selected_orders"]) > 0   # orders computed
    assert store.get_positions(None) == []             # but book untouched


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
    store.record_order({"client_order_id": "resting-1", "ticker": "SBER", "side": "BUY",
                        "quantity_lots": 1, "order_type": "LIMIT", "limit_price": 300.0},
                       trade_date=TD, phase="eod", status="PLACED")
    assert store.open_orders()

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
    assert "live_sleeve_pnl" in payload["risk_summary"]
    assert "shadow_sleeve_pnl" in payload["risk_summary"]
