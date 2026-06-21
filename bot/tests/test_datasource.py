"""Read-only datasource: state reads, graceful degradation, report parsing."""

from __future__ import annotations

from pathlib import Path

from bot.src.datasource import (
    ReadOnlyState,
    read_gate,
    read_integrity,
    read_shadow_log,
)


def test_reads_positions_split_by_capital_state(seeded_db: Path):
    s = ReadOnlyState(seeded_db)
    assert s.available()
    live = {p["ticker"] for p in s.positions("live")}
    shadow = {p["ticker"] for p in s.positions("shadow")}
    assert live == {"SBER", "MOEXFN"}
    assert shadow == {"LKOH"}


def test_gross_marks_at_last_price(seeded_db: Path):
    s = ReadOnlyState(seeded_db)
    # SBER 100*315 + MOEXFN 40*10100
    assert s.gross("live") == 100 * 315.0 + 40 * 10100.0
    assert s.gross("shadow") == 10 * 6800.0


def test_gross_split_separates_directional_from_hedge(seeded_db: Path):
    s = ReadOnlyState(seeded_db)
    live = s.gross_split("live")
    assert live["directional"] == 100 * 315.0       # SBER only
    assert live["hedge"] == 40 * 10100.0            # MOEXFN hedge leg
    assert live["total"] == live["directional"] + live["hedge"]
    shadow = s.gross_split("shadow")
    assert shadow["directional"] == 10 * 6800.0
    assert shadow["hedge"] == 0.0


def test_pnl_and_flags_and_cycle(seeded_db: Path):
    s = ReadOnlyState(seeded_db)
    live_pnl = {r["sleeve"]: r for r in s.pnl_by_sleeve("live")}
    assert live_pnl["s3_event"]["realized"] == 1500.0
    assert s.kill_switch_engaged() is False
    assert s.last_cycle()["trade_date"] == "2026-06-19"
    cyc = s.latest_cycle("eod")
    assert cyc["status"] == "completed"
    assert cyc["result"]["selected_orders"][0]["ticker"] == "SBER"


def test_missing_db_degrades_gracefully(tmp_path: Path):
    s = ReadOnlyState(tmp_path / "nope.sqlite")
    assert s.available() is False
    assert s.positions("live") == []
    assert s.pnl_by_sleeve() == []
    assert s.kill_switch_engaged() is False
    assert s.last_cycle() is None
    assert s.latest_cycle() is None
    assert s.gross("live") == 0.0


def test_read_integrity(reports: dict[str, Path], tmp_path: Path):
    rep = read_integrity(reports["integrity"])
    assert rep["status"] == "OK"
    assert read_integrity(tmp_path / "absent.json") is None


def test_read_shadow_log_tails_and_degrades(shadow_log_file: Path, tmp_path: Path):
    recs = read_shadow_log(shadow_log_file, limit=5)
    assert [r["trade_date"] for r in recs] == ["2026-07-08", "2026-07-09"]  # newest last
    assert read_shadow_log(shadow_log_file, limit=1)[0]["trade_date"] == "2026-07-09"
    assert read_shadow_log(tmp_path / "absent.jsonl") == []


def test_read_shadow_log_skips_malformed(tmp_path: Path):
    p = tmp_path / "shadow_pnl.jsonl"
    p.write_text('{"trade_date":"2026-07-09"}\nnot json\n\n', encoding="utf-8")
    recs = read_shadow_log(p)
    assert len(recs) == 1 and recs[0]["trade_date"] == "2026-07-09"


def test_read_gate_parses_verdict_and_forward(reports: dict[str, Path], tmp_path: Path):
    g = read_gate(reports["gate"])
    assert g["found"] is True
    assert g["is_production"] is False
    assert g["met"] is False
    assert g["forward_n"] == 12
    assert g["forward_net"] == -0.0093
    # missing report => not found, but invariant is_production=false still holds
    g2 = read_gate(tmp_path / "absent.txt")
    assert g2 == {"found": False, "is_production": False}
