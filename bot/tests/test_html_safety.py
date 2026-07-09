"""Guard: every formatter output is valid Telegram HTML (parse_mode="HTML").

app.py sends all replies with parse_mode="HTML", and that reply_text sits OUTSIDE the
dispatch try/except — so a message Telegram's HTML parser rejects (unknown tag like
``<chat_id>`` or an unescaped ``&``) 400s and the user silently gets NOTHING back.

Per https://core.telegram.org/bots/api#html-style only a fixed tag set is allowed and the
reserved chars ``<`` ``>`` ``&`` outside tags/entities must be escaped. These asserts run
each formatter's output through that rule so the next hand-written spec-invalid string is
caught here instead of in production.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest

from bot.src import formatters as fmt

# Tags Telegram accepts in HTML mode (plus <br>, harmless).
_TELEGRAM_TAGS = {
    "b", "strong", "i", "em", "u", "ins", "s", "strike", "del", "span",
    "tg-spoiler", "a", "code", "pre", "blockquote", "tg-emoji", "br",
}
_ENTITY = re.compile(r"&(?:amp|lt|gt|quot|apos|#\d+|#[xX][0-9a-fA-F]+);")


class _TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        self.tags.append(tag)

    def handle_startendtag(self, tag: str, attrs) -> None:  # noqa: ANN001
        self.tags.append(tag)

    def handle_endtag(self, tag: str) -> None:
        self.tags.append(tag)


def assert_telegram_html(text: str) -> None:
    """Reject unknown tags and unescaped ``&`` — the two ways Telegram's parser 400s."""
    assert "&" not in _ENTITY.sub("", text), f"unescaped '&' in: {text!r}"
    p = _TagCollector()
    p.feed(text)
    bad = sorted({t for t in p.tags if t not in _TELEGRAM_TAGS})
    assert not bad, f"non-Telegram tag(s) {bad} in: {text!r}"


def test_guard_itself_flags_the_regressions_it_exists_for():
    # the exact shapes that were broken: a literal <chat_id> tag and a raw ampersand
    with pytest.raises(AssertionError):
        assert_telegram_html("/allow <chat_id> [note]")
    with pytest.raises(AssertionError):
        assert_telegram_html("P&L by sleeve")


def test_help_is_valid_telegram_html():
    assert_telegram_html(fmt.fmt_help(is_admin=True))
    assert_telegram_html(fmt.fmt_help(is_admin=False))


def test_shadowlog_is_valid_telegram_html():
    assert_telegram_html(fmt.fmt_shadowlog([]))
    records = [{
        "trade_date": "2026-07-10", "sleeves": ["s3_event"],
        "sleeve_pnl": {"s3_event": {"unrealized": 1234.5}},
    }]
    assert_telegram_html(fmt.fmt_shadowlog(records))


def test_static_notices_are_valid_telegram_html():
    assert_telegram_html(fmt.unauthorized_text(12345))
    assert_telegram_html(fmt.admin_only_text())
    assert_telegram_html(fmt.unknown_command_text())


def test_pnl_is_valid_telegram_html():
    live = [{"sleeve": "s3_event", "realized": 1500.0, "unrealized": 500.0, "gross": 10000.0}]
    assert_telegram_html(fmt.fmt_pnl(live, []))


def test_start_is_valid_telegram_html():
    assert_telegram_html(fmt.fmt_start())


def test_status_is_valid_telegram_html():
    d = {"mode": "paper", "block_mode": "mock", "live_enabled": False, "kill_switch": False,
         "last_cycle": {"trade_date": "2026-06-19", "phase": "eod", "status": "completed"},
         "live_gross": {"directional": 31500.0, "hedge": 404000.0},
         "shadow_gross": {"directional": 68000.0, "hedge": 0.0}}
    assert_telegram_html(fmt.fmt_status(d))


def test_positions_table_is_valid_telegram_html():
    # a hostile sector string with reserved chars must survive escaping inside <pre>
    live = [{"ticker": "SBER", "sector": "A&B<x>", "lots": 100, "last_price": 315.0},
            {"ticker": "MOEXFN", "sector": "IMOEX", "lots": -40, "last_price": 10100.0,
             "is_hedge": True}]
    assert_telegram_html(fmt.fmt_positions(live, [], 10_000_000.0))
    assert_telegram_html(fmt.fmt_positions([], [], 10_000_000.0))  # empty state


def test_prices_table_is_valid_telegram_html():
    assert_telegram_html(fmt.fmt_prices([("SBER", 315.0), ("LKOH", None)], "1D"))
    assert_telegram_html(fmt.fmt_prices([], "1D"))


def test_gate_is_valid_telegram_html():
    gate = {"found": True, "is_production": False, "met": False,
            "forward_n": 12, "forward_net": -0.0093, "forward_pct_pos": 0.5}
    assert_telegram_html(fmt.fmt_gate(gate, [{"sleeve": "s3_event", "unrealized": -2200.0}]))
    assert_telegram_html(fmt.fmt_gate({"found": False, "is_production": False}, []))


def test_cycle_is_valid_telegram_html():
    cycle = {"trade_date": "2026-06-19", "status": "completed", "halt_reason": "R&D <halt>",
             "result": {"mode": "paper",
                        "selected_orders": [{"side": "BUY", "quantity_lots": 100,
                                             "ticker": "SBER", "limit_price": 315.0}],
                        "risk_summary": {"binding_limits": ["gross"],
                                         "gating": [{"sleeve": "s3_event",
                                                     "capital_state": "shadow",
                                                     "reason": "gate not met <x> & y"}]}}}
    assert_telegram_html(fmt.fmt_cycle(cycle))
    assert_telegram_html(fmt.fmt_cycle(None))


def test_integrity_is_valid_telegram_html():
    rep = {"status": "HALT", "reference_date": "2026-06-19", "n_fail": 1, "n_warn": 1,
           "reasons": ["stale <SBER> & gaps"], "warnings": ["MOEXTL behind"]}
    assert_telegram_html(fmt.fmt_integrity(rep))
    assert_telegram_html(fmt.fmt_integrity(None))


def test_users_is_valid_telegram_html():
    managed = {222: {"note": "alice & bob <x>", "added_by": 999}}
    assert_telegram_html(fmt.fmt_users({999}, {111}, managed))
