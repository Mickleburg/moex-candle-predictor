"""Config + access-control: whitelist parsing, authorization, .env loader."""

from __future__ import annotations

from pathlib import Path

from bot.src.config import (
    BotConfig,
    _maybe_load_dotenv,
    _parse_chat_ids,
    resolve_block_modes,
)


def test_resolve_block_modes_mirrors_agent_formula():
    # per-block override wins; unset blocks fall back to the top-level block_mode
    blocks = {"backend": {"mode": "live"}, "sleeve": {"mode": "live"},
              "combiner": {"mode": "live"}}  # execution intentionally absent
    resolved = resolve_block_modes(blocks, "mock")
    assert resolved == {"backend": "live", "sleeve": "live", "combiner": "live",
                        "execution": "mock"}


def test_resolve_block_modes_all_fallback_when_no_overrides():
    assert resolve_block_modes({}, "mock") == {
        "backend": "mock", "sleeve": "mock", "combiner": "mock", "execution": "mock"}


def test_parse_chat_ids_handles_separators_and_garbage():
    assert _parse_chat_ids("111, 222 ;333") == frozenset({111, 222, 333})
    assert _parse_chat_ids("111,oops,222") == frozenset({111, 222})  # garbage skipped
    assert _parse_chat_ids("") == frozenset()
    assert _parse_chat_ids(None) == frozenset()


def test_authorized_admin_seed_and_none(tmp_path):
    cfg = BotConfig(admin_chat_ids=frozenset({999}), allowed_chat_ids=frozenset({111}),
                    allowlist_path=tmp_path / "allowlist.json")
    assert cfg.authorized(111) is True    # env seed
    assert cfg.authorized(999) is True    # admin
    assert cfg.authorized(222) is False
    assert cfg.authorized(None) is False


def test_is_admin(tmp_path):
    cfg = BotConfig(admin_chat_ids=frozenset({999}), allowed_chat_ids=frozenset({111}),
                    allowlist_path=tmp_path / "allowlist.json")
    assert cfg.is_admin(999) is True
    assert cfg.is_admin(111) is False     # allowed but not admin
    assert cfg.is_admin(None) is False


def test_empty_config_authorizes_nobody_and_has_no_access(tmp_path):
    cfg = BotConfig(allowlist_path=tmp_path / "allowlist.json")
    assert cfg.authorized(111) is False
    assert cfg.has_any_access() is False


def test_managed_store_widens_authorized_dynamically(tmp_path):
    cfg = BotConfig(admin_chat_ids=frozenset({999}), allowlist_path=tmp_path / "allowlist.json")
    assert cfg.authorized(444) is False
    cfg.allowlist.add(444, note="x", added_by=999)
    assert cfg.authorized(444) is True    # no restart — read dynamically
    assert cfg.has_any_access() is True


def test_dotenv_loader_does_not_override_existing(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('FOO_BOT=fromfile\nBAR_BOT="quoted"\n', encoding="utf-8")
    monkeypatch.setenv("FOO_BOT", "fromenv")
    monkeypatch.delenv("BAR_BOT", raising=False)
    _maybe_load_dotenv(env_file)
    import os
    assert os.environ["FOO_BOT"] == "fromenv"   # existing value wins
    assert os.environ["BAR_BOT"] == "quoted"    # missing value filled, quotes stripped
