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
    live = [{"sleeve": "s3_event", "unrealized": 500.0, "gross": 10000.0}]
    assert_telegram_html(fmt.fmt_pnl(live, []))
