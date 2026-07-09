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
    assert "Agent status" in text
    assert "mode paper" in text
    assert "🟢 kill-switch: off" in text
    # gross split: directional shown separately from hedge (no misleading >100% combined)
    assert "directional" in text and "hedge" in text
    assert "live" in text and "shadow" in text


def test_positions_split_live_shadow_with_sector(bot_config: BotConfig):
    text = _monitor(bot_config).positions()
    assert "LIVE" in text and "SHADOW" in text
    assert "SBER" in text and "MOEXFN" in text  # live (incl. hedge)
    assert "LKOH" in text                         # shadow
    assert "hedge" in text                        # hedge leg flagged in the type column
    # sector mapping surfaced for a directional name
    assert "MOEXFN" in text  # SBER's sector index


def test_pnl_separates_tracks(bot_config: BotConfig):
    text = _monitor(bot_config).pnl()
    assert "LIVE" in text and "SHADOW" in text
    assert "s3_event" in text
    # P&L carries explicit +/- signs (colour-independent profit/loss cue)
    assert "+1.5k ₽" in text    # live realized (positive)
    assert "-2.2k ₽" in text    # shadow total (negative)


def test_gate_shows_not_met_and_is_production_false(bot_config: BotConfig):
    text = _monitor(bot_config).gate()
    assert "is_production: false" in text
    assert "NOT MET" in text
    assert "n=12" in text


def test_cycle_shows_orders_and_binding_limits(bot_config: BotConfig):
    text = _monitor(bot_config).cycle()
    assert "2026-06-19" in text
    assert "orders: 1" in text
    assert "binding: gross" in text  # binding limit
    # order surfaces in the aligned table (columns are space-padded, so check cells)
    assert "BUY" in text and "SBER" in text and "100" in text


def test_shadowlog_renders_cycles(bot_config: BotConfig, shadow_log_file: Path):
    text = _monitor(bot_config).shadowlog()
    assert "2026-07-08" in text and "2026-07-09" in text
    assert "s3_event" in text
    assert "1.2k" in text        # 1200 unrealized -> money()
    assert "flat" in text        # the empty first cycle


def test_shadowlog_respects_limit(bot_config: BotConfig, shadow_log_file: Path):
    text = _monitor(bot_config).shadowlog(1)
    assert "2026-07-09" in text and "2026-07-08" not in text


def test_shadowlog_no_data(tmp_path: Path):
    cfg = BotConfig(state_db=tmp_path / "absent.sqlite",
                    shadow_log=tmp_path / "absent.jsonl", universe=["SBER"])
    text = Monitor(cfg, make_state(cfg)).shadowlog()
    assert "no shadow-log entries yet" in text


def test_integrity_ok(bot_config: BotConfig):
    text = _monitor(bot_config).integrity()
    assert "OK" in text
    assert "🟢" in text


def test_prices_no_data_when_store_empty(bot_config: BotConfig):
    text = _monitor(bot_config).prices(["SBER"])
    assert "SBER" in text
    assert "no data" in text  # tmp data_raw has no parquet


def test_help_lists_all_commands_grouped(bot_config: BotConfig):
    text = _monitor(bot_config).help()
    for cmd in ("/status", "/positions", "/pnl", "/prices", "/gate", "/shadowlog",
                "/cycle", "/integrity", "/start", "/help"):
        assert cmd in text
    # grouped by section
    assert "Monitor" in text and "Research" in text and "General" in text


def test_start_is_warm_greeting_not_help(bot_config: BotConfig):
    start = _monitor(bot_config).start()
    assert "MOEX Agent Monitor" in start
    assert "read" in start.lower() and "never trade" in start.lower()
    assert "/help" in start
    # /start is NOT the /help command list
    assert start != _monitor(bot_config).help()
    assert "/positions" not in start


def test_commands_degrade_without_db(tmp_path: Path):
    cfg = BotConfig(state_db=tmp_path / "absent.sqlite", universe=["SBER"],
                    integrity_report=tmp_path / "absent.json",
                    gate_report=tmp_path / "absent.txt",
                    shadow_log=tmp_path / "absent.jsonl", data_raw=tmp_path)
    m = _monitor(cfg)
    # none of these should raise; they render empty/no-data sections
    for fn in (m.status, m.positions, m.pnl, m.gate, m.shadowlog, m.cycle, m.integrity):
        assert isinstance(fn(), str)
