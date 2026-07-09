"""App wiring: startup guards + command-menu best-effort (requires PTB; skipped if absent)."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("telegram")

from bot.src.app import apply_command_menu, build_application  # noqa: E402
from bot.src.config import BotConfig  # noqa: E402


class _RecordingBot:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list = []

    async def set_my_commands(self, commands, scope=None):
        if self.fail:
            raise RuntimeError("network/proxy down")
        self.calls.append((scope, [c.command for c in commands]))


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


def test_command_menu_scopes_read_default_and_admin_per_chat(bot_config: BotConfig):
    from telegram import BotCommandScopeChat, BotCommandScopeDefault

    bot = _RecordingBot()
    asyncio.run(apply_command_menu(bot, bot_config))  # admin_chat_ids == {999}
    scopes = [type(scope) for scope, _ in bot.calls]
    assert BotCommandScopeDefault in scopes           # read menu for everyone
    assert BotCommandScopeChat in scopes              # admin menu scoped per admin chat
    # the admin-scoped call includes the management commands; the default one does not
    default_cmds = next(cmds for scope, cmds in bot.calls if isinstance(scope, BotCommandScopeDefault))
    admin_cmds = next(cmds for scope, cmds in bot.calls if isinstance(scope, BotCommandScopeChat))
    assert "allow" not in default_cmds and "users" not in default_cmds
    assert "allow" in admin_cmds and "deny" in admin_cmds


def test_command_menu_failure_is_swallowed(bot_config: BotConfig):
    # a network/proxy failure inside set_my_commands must NOT propagate (bot still starts)
    asyncio.run(apply_command_menu(_RecordingBot(fail=True), bot_config))
