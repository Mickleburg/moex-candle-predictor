"""H9 -12/-2 discipline guard: window OK, early entry warn, held-into-ex-gap critical."""

from __future__ import annotations

from execution.src.config import ExecutionConfig, Mode
from execution.src.discipline import DisciplineChecker
from execution.src.trading_calendar import TradingCalendar

ANCHOR = "2026-07-20"   # Monday record/ex anchor
CONFIG = ExecutionConfig(mode=Mode.DRY_RUN, entry_offset=12, exit_offset=2)


def _checker():
    return DisciplineChecker(CONFIG, TradingCalendar())


def test_held_inside_window_is_ok():
    f = _checker().check_name("TATN", 0.30, "2026-07-02", ANCHOR)   # td=12
    assert f.td_to_anchor == 12 and f.in_window and f.severity == "ok"


def test_held_into_ex_gap_is_critical():
    f = _checker().check_name("TATN", 0.30, "2026-07-16", ANCHOR)   # td=2 <= exit_offset
    assert f.td_to_anchor == 2 and f.severity == "critical"


def test_early_entry_is_warn():
    f = _checker().check_name("TATN", 0.30, "2026-07-01", ANCHOR)   # td=13 > entry_offset
    assert f.td_to_anchor == 13 and f.severity == "warn"


def test_flat_outside_window_is_ok():
    f = _checker().check_name("TATN", 0.0, "2026-07-01", ANCHOR)
    assert f.severity == "ok" and not f.held


def test_check_book_only_flags_names_with_anchors():
    book = {
        "as_of": "2026-07-16 00:00:00+03:00",
        "net_positions": [
            {"ticker": "TATN", "weight": 0.30, "side": "LONG"},
            {"ticker": "LKOH", "weight": 0.30, "side": "LONG"},  # no anchor -> skipped
        ],
        "hedge": {"mode": "none", "legs": []},
    }
    findings = _checker().check_book(book, anchors={"TATN": ANCHOR})
    assert len(findings) == 1
    assert findings[0].instrument == "TATN"
    assert DisciplineChecker.has_critical(findings)


def test_no_anchors_no_findings():
    book = {"as_of": "2026-07-16 00:00:00+03:00",
            "net_positions": [{"ticker": "TATN", "weight": 0.3, "side": "LONG"}],
            "hedge": {"mode": "none", "legs": []}}
    assert _checker().check_book(book, anchors=None) == []
