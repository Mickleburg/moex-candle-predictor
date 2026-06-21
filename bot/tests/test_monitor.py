"""Monitor: each command renders the right slice, live & shadow kept separate."""

from __future__ import annotations

from pathlib import Path

from bot.src.config import BotConfig
from bot.src.datasource import make_state
from bot.src.monitor import Monitor


def _monitor(cfg: BotConfig) -> Monitor:
    return Monitor(cfg, make_state(cfg))


def test_status(bot_config: BotConfig):
    text = _monitor(bot_config).status()
    assert "mode: paper" in text
    assert "kill-switch: 🟢 off" in text
    # live and shadow gross both present and labelled
    assert "live" in text and "shadow" in text


def test_positions_split_live_shadow_with_sector(bot_config: BotConfig):
    text = _monitor(bot_config).positions()
    assert "LIVE" in text and "SHADOW" in text
    assert "SBER" in text and "MOEXFN" in text  # live (incl. hedge)
    assert "LKOH" in text                         # shadow
    assert "MOEXFN" in text and "[hedge]" in text
    # sector mapping surfaced for a directional name
    assert "MOEXFN" in text  # SBER's sector index


def test_pnl_separates_tracks(bot_config: BotConfig):
    text = _monitor(bot_config).pnl()
    assert "LIVE" in text and "SHADOW" in text
    assert "s3_event" in text


def test_gate_shows_not_met_and_is_production_false(bot_config: BotConfig):
    text = _monitor(bot_config).gate()
    assert "is_production: false" in text
    assert "NOT MET" in text
    assert "n=12" in text


def test_cycle_shows_orders_and_binding_limits(bot_config: BotConfig):
    text = _monitor(bot_config).cycle()
    assert "2026-06-19" in text
    assert "orders: 1" in text
    assert "gross" in text  # binding limit
    assert "BUY 100 SBER" in text


def test_integrity_ok(bot_config: BotConfig):
    text = _monitor(bot_config).integrity()
    assert "OK" in text
    assert "🟢" in text


def test_prices_no_data_when_store_empty(bot_config: BotConfig):
    text = _monitor(bot_config).prices(["SBER"])
    assert "SBER" in text
    assert "no data" in text  # tmp data_raw has no parquet


def test_help_lists_all_commands(bot_config: BotConfig):
    text = _monitor(bot_config).help()
    for cmd in ("/status", "/positions", "/pnl", "/prices", "/gate", "/cycle", "/integrity"):
        assert cmd in text


def test_commands_degrade_without_db(tmp_path: Path):
    cfg = BotConfig(state_db=tmp_path / "absent.sqlite", universe=["SBER"],
                    integrity_report=tmp_path / "absent.json",
                    gate_report=tmp_path / "absent.txt", data_raw=tmp_path)
    m = _monitor(cfg)
    # none of these should raise; they render empty/no-data sections
    for fn in (m.status, m.positions, m.pnl, m.gate, m.cycle, m.integrity):
        assert isinstance(fn(), str)
