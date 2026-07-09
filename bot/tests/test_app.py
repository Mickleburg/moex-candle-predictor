"""App wiring: startup guards (requires python-telegram-bot; skipped if absent)."""

from __future__ import annotations

import pytest

pytest.importorskip("telegram")

from bot.src.app import build_application  # noqa: E402
from bot.src.config import BotConfig  # noqa: E402


def test_refuses_to_start_without_token(tmp_path):
    cfg = BotConfig(token=None, allowed_chat_ids=frozenset({111}),
                    allowlist_path=tmp_path / "allowlist.json")
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        build_application(cfg)


def test_refuses_to_start_with_no_access(tmp_path):
    cfg = BotConfig(token="x", admin_chat_ids=frozenset(), allowed_chat_ids=frozenset(),
                    allowlist_path=tmp_path / "allowlist.json")
    with pytest.raises(RuntimeError, match="BOT_ADMIN_CHAT_IDS"):
        build_application(cfg)


def test_builds_with_token_and_whitelist(bot_config: BotConfig):
    app = build_application(bot_config)
    assert app is not None
    assert app.handlers  # command handlers registered
