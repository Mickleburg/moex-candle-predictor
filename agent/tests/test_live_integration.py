"""Integration: the orchestrator driving the REAL execution paper engine + the LLM refresh step.

These exercise the live seams (execution in-process; LLM feed-refresh subprocess) with the other
blocks still mocked, so the cycle runs offline but through real block code, not the paper mock.
"""

from __future__ import annotations

import sys

from agent.src.adapters import mock
from agent.src.adapters.live import LiveExecution
from agent.src.adapters.registry import Adapters
from agent.src.config import AgentConfig
from agent.src.orchestrator import Orchestrator
from agent.src.state_store import StateStore
from agent.tests.conftest import CapturingNotifier

TD = "2026-06-18"


def _config(tmp_path, **blocks_over):
    blocks = {
        "execution": {"mode": "live", "broker_backend": "sim", "allow_live": False,
                      "state_dir": str(tmp_path / "exstate"), "audit_dir": str(tmp_path / "exaudit")},
    }
    blocks.update(blocks_over)
    cfg = AgentConfig(mode="paper", block_mode="mock", state_db=tmp_path / "s.sqlite",
                      cycle_results_dir=tmp_path / "cyc", shadow_log=tmp_path / "sh.jsonl",
                      log_dir=tmp_path / "logs", blocks=blocks)
    cfg.ensure_dirs()
    return cfg


def _orch(cfg, *, execution=None):
    store = StateStore(cfg.state_db)
    note = CapturingNotifier()
    adapters = Adapters(
        backend=mock.MockBackend(universe=cfg.universe), sleeve=mock.MockSleeve(),
        combiner=mock.MockCombiner(hedge_mode="sector"),
        execution=execution or LiveExecution(cfg.blocks["execution"]),
        modes={"backend": "mock", "sleeve": "mock", "combiner": "mock", "execution": "live"})
    return Orchestrator(cfg, store=store, adapters=adapters, notifier=note), store, note


def test_full_cycle_with_real_execution_engine(tmp_path):
    cfg = _config(tmp_path)
    orch, store, _ = _orch(cfg)
    out = orch.run_eod_cycle(trade_date=TD)

    assert out["status"] == "completed"
    assert out["result"]["mode"] == "paper"
    # H9 is shadow: the REAL paper broker folds + fills the SHADOW book (3 longs + sector hedges),
    # but it lands in the SHADOW track with zero live capital.
    shadow = {p["ticker"] for p in store.get_positions("shadow")}
    assert {"SBER", "LKOH", "TATN"} <= shadow
    assert store.get_positions("live") == []
    assert any(p["is_hedge"] for p in store.get_positions("shadow"))
    # execution persisted its own durable state (dup-ledger / audit) in the backed-up data tree
    assert (tmp_path / "exstate").exists()
    # per-sleeve attribution recorded on the shadow track, not live
    pnl = {(r["sleeve"], r["capital_state"]) for r in store.pnl_by_sleeve()}
    assert ("s3_event", "shadow") in pnl and ("s3_event", "live") not in pnl


def test_real_execution_dedup_across_reruns(tmp_path):
    # execution's own client_order_id ledger must stop a duplicate order on a forced re-run
    cfg = _config(tmp_path)
    orch, store, _ = _orch(cfg)
    orch.run_eod_cycle(trade_date=TD)
    n_exec_first = len(store.all_orders())
    # force a second cycle for the same day -> execution sees identical target -> no-op/dup-skip
    out2 = orch.run_eod_cycle(trade_date=TD, force=True)
    assert out2["status"] == "completed"
    assert len(out2["result"]["selected_orders"]) == 0   # already at target, nothing submitted


def test_llm_refresh_runs_before_sleeve(tmp_path):
    marker = tmp_path / "llm_refreshed.txt"
    cmd = [sys.executable, "-c", f"open(r'{marker}', 'w').write('ok')"]
    cfg = _config(tmp_path, llm={"refresh_cmd": cmd})
    orch, _, _ = _orch(cfg)
    out = orch.run_eod_cycle(trade_date=TD)
    assert out["status"] == "completed"
    assert marker.exists(), "EOD step 2 should have run the configured LLM feed refresh"


def test_llm_refresh_failure_does_not_block_cycle(tmp_path):
    cmd = [sys.executable, "-c", "import sys; sys.exit(3)"]   # refresh fails
    cfg = _config(tmp_path, llm={"refresh_cmd": cmd})
    orch, store, note = _orch(cfg)
    out = orch.run_eod_cycle(trade_date=TD)
    assert out["status"] == "completed"                       # trading not blocked
    assert store.get_positions("shadow")                      # shadow book still paper-traded
    assert any("LLM feed refresh" in s for s in note.subjects())
