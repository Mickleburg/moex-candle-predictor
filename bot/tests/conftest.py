"""Fixtures: a seeded agent state.sqlite + reports, plus a BotConfig pointing at them.

The DB is seeded through the REAL ``agent.src.state_store.StateStore`` (the write owner) so the
schema the bot reads is exactly the production one — the bot itself only ever opens it read-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.src.state_store import StateStore  # noqa: E402
from bot.src.config import BotConfig  # noqa: E402


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "state.sqlite"
    store = StateStore(db_path)

    # live book: one directional name + a sector hedge
    store.upsert_position("SBER", lots=100, avg_price=300.0, last_price=315.0,
                          sleeve_contributions={"s3_event": 1.0}, capital_state="live")
    store.upsert_position("MOEXFN", lots=-40, avg_price=10000.0, last_price=10100.0,
                          is_hedge=True, capital_state="live")
    # shadow book: gated-out name (forward-shadow track)
    store.upsert_position("LKOH", lots=10, avg_price=7000.0, last_price=6800.0,
                          sleeve_contributions={"s3_event": 1.0}, capital_state="shadow")

    store.record_pnl_attribution("2026-06-19", "s3_event", realized=1500.0,
                                 unrealized=500.0, gross=31500.0, capital_state="live")
    store.record_pnl_attribution("2026-06-19", "s3_event", realized=-200.0,
                                 unrealized=-2000.0, gross=68000.0, capital_state="shadow")

    store.set_kill_switch(False)
    store.set_flag("last_cycle", {"trade_date": "2026-06-19", "phase": "eod",
                                  "status": "completed", "at": "2026-06-19T19:05:00+03:00"})

    # one completed EOD cycle with a persisted result
    store.begin_cycle("2026-06-19", "eod", mode="paper", block_mode="mock",
                      as_of="2026-06-19T19:05:00+03:00")
    result = {
        "as_of": "2026-06-19T19:05:00+03:00", "mode": "paper",
        "evaluated_tickers": ["SBER", "LKOH"],
        "selected_orders": [
            {"client_order_id": "c1", "ticker": "SBER", "side": "BUY",
             "quantity_lots": 100, "order_type": "LIMIT", "limit_price": 315.0},
        ],
        "rejected_candidates": [],
        "risk_summary": {
            "directional_gross": 31500.0, "shadow_gross": 68000.0,
            "binding_limits": ["gross"],
            "gating": [{"sleeve": "s3_event", "capital_state": "shadow",
                        "reason": "forward gate not met"}],
        },
    }
    store.finish_cycle("2026-06-19", "eod", "completed", result=result)
    store.close()
    return db_path


@pytest.fixture
def reports(tmp_path: Path) -> dict[str, Path]:
    integrity = tmp_path / "data_integrity_status.json"
    integrity.write_text(json.dumps({
        "status": "OK", "reference_date": "2026-06-19", "n_fail": 0, "n_warn": 1,
        "reasons": [], "warnings": ["freshness/MOEXTL/1D: last bar 64 td behind"],
    }), encoding="utf-8")

    gate = tmp_path / "h9_shadow_pnl.txt"
    gate.write_text(
        "H9 realized-P&L SHADOW GATE\n"
        "  FORWARD           : n= 12  net -0.0093  %pos 0.50  median -0.0071\n"
        "VERDICT: NOT MET: 12 forward events, net -0.0093 — forward does NOT confirm the edge.\n"
        "  (is_production=false until this gate is MET on accrued forward events AND sign-off.)\n",
        encoding="utf-8",
    )
    return {"integrity": integrity, "gate": gate}


@pytest.fixture
def shadow_log_file(tmp_path: Path) -> Path:
    """A forward-shadow track with two cycles (matches agent/src/pnl.append_shadow_log shape)."""
    path = tmp_path / "shadow_pnl.jsonl"
    recs = [
        {"ts": "2026-07-08T19:05:00+00:00", "trade_date": "2026-07-08",
         "as_of": "2026-07-08T19:05:00+03:00", "sleeves": ["s3_event"],
         "regime": {"exposure_scalar": 1.0, "regime_novel": False},
         "book": [], "sleeve_pnl": {}},
        {"ts": "2026-07-09T19:05:00+00:00", "trade_date": "2026-07-09",
         "as_of": "2026-07-09T19:05:00+03:00", "sleeves": ["s3_event"],
         "regime": {"exposure_scalar": 1.0, "regime_novel": False},
         "book": [{"ticker": "MTSS", "lots": 50, "last_price": 300.0, "is_hedge": False}],
         "sleeve_pnl": {"s3_event": {"unrealized": 1200.0, "gross": 15000.0}}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def bot_config(seeded_db: Path, reports: dict[str, Path], tmp_path: Path) -> BotConfig:
    return BotConfig(
        token="test-token",
        allowed_chat_ids=frozenset({111}),
        state_db=seeded_db,
        shadow_log=tmp_path / "shadow_pnl.jsonl",
        cycle_results_dir=tmp_path / "cycles",
        integrity_report=reports["integrity"],
        gate_report=reports["gate"],
        data_raw=tmp_path / "raw",
        universe=["SBER", "LKOH"],
        capital_rub=10_000_000.0,
        timeframe="1D",
    )
