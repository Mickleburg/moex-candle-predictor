"""Config + access-control: whitelist parsing, authorization, .env loader."""

from __future__ import annotations

from pathlib import Path

from bot.src.config import BotConfig, _maybe_load_dotenv, _parse_chat_ids


def test_parse_chat_ids_handles_separators_and_garbage():
    assert _parse_chat_ids("111, 222 ;333") == frozenset({111, 222, 333})
    assert _parse_chat_ids("111,oops,222") == frozenset({111, 222})  # garbage skipped
    assert _parse_chat_ids("") == frozenset()
    assert _parse_chat_ids(None) == frozenset()


def test_authorized_only_whitelisted():
    cfg = BotConfig(allowed_chat_ids=frozenset({111}))
    assert cfg.authorized(111) is True
    assert cfg.authorized(222) is False
    assert cfg.authorized(None) is False


def test_empty_whitelist_authorizes_nobody():
    cfg = BotConfig(allowed_chat_ids=frozenset())
    assert cfg.authorized(111) is False


def test_dotenv_loader_does_not_override_existing(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('FOO_BOT=fromfile\nBAR_BOT="quoted"\n', encoding="utf-8")
    monkeypatch.setenv("FOO_BOT", "fromenv")
    monkeypatch.delenv("BAR_BOT", raising=False)
    _maybe_load_dotenv(env_file)
    import os
    assert os.environ["FOO_BOT"] == "fromenv"   # existing value wins
    assert os.environ["BAR_BOT"] == "quoted"    # missing value filled, quotes stripped
