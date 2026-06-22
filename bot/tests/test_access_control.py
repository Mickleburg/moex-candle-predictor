"""Access-control matrix via the pure Router (acceptance a–d) + admin commands.

Covers: (a) admin can add/remove, (b) removing admin/env-seed is refused, (c) an added id is
immediately authorized, (d) a non-admin (but allowed) cannot run management commands. Plus the
unauthorized-user reply (their id, not silence).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bot.src.config import BotConfig
from bot.src.router import Router

ADMIN = 999
ALLOWED = 111      # env seed, read-only
STRANGER = 222


@pytest.fixture
def cfg(tmp_path: Path) -> BotConfig:
    return BotConfig(token="t", admin_chat_ids=frozenset({ADMIN}),
                     allowed_chat_ids=frozenset({ALLOWED}),
                     allowlist_path=tmp_path / "bot" / "allowlist.json",
                     state_db=tmp_path / "absent.sqlite",
                     integrity_report=tmp_path / "absent.json",
                     gate_report=tmp_path / "absent.txt",
                     shadow_log=tmp_path / "absent.jsonl",
                     data_raw=tmp_path, universe=["SBER"])


@pytest.fixture
def router(cfg: BotConfig) -> Router:
    return Router(cfg)


# --- (a) admin can add then remove, (c) added id is immediately authorized -----------------
def test_admin_allow_then_id_authorized_immediately(router: Router, cfg: BotConfig):
    assert cfg.authorized(STRANGER) is False
    out = router.dispatch("allow", ADMIN, [str(STRANGER), "alice"])
    assert "allowed" in out and str(STRANGER) in out
    assert cfg.authorized(STRANGER) is True               # (c) no restart
    # the newly-allowed stranger can now run a read command (not silent / not denied)
    read = router.dispatch("status", STRANGER, [])
    assert "Agent status" in read


def test_admin_deny_removes_managed_id(router: Router, cfg: BotConfig):
    router.dispatch("allow", ADMIN, [str(STRANGER)])
    assert cfg.authorized(STRANGER) is True
    out = router.dispatch("deny", ADMIN, [str(STRANGER)])
    assert "removed" in out
    assert cfg.authorized(STRANGER) is False


# --- (b) admin / env-seed cannot be removed via the bot ------------------------------------
def test_deny_refuses_admin(router: Router):
    out = router.dispatch("deny", ADMIN, [str(ADMIN)])
    assert "cannot be removed" in out


def test_deny_refuses_env_seed(router: Router):
    out = router.dispatch("deny", ADMIN, [str(ALLOWED)])
    assert "env-seed" in out


# --- (d) a non-admin (but allowed) cannot run management commands --------------------------
def test_non_admin_cannot_manage(router: Router, cfg: BotConfig):
    for cmd in ("allow", "deny", "users"):
        out = router.dispatch(cmd, ALLOWED, [str(STRANGER)])
        assert out == "⛔ Admin-only command."
    # and the managed store was NOT touched
    assert cfg.managed_ids() == set()


def test_allow_validates_integer(router: Router):
    out = router.dispatch("allow", ADMIN, ["notanint"])
    assert "not an integer" in out


def test_allow_on_admin_or_seed_is_noop_message(router: Router):
    assert "already" in router.dispatch("allow", ADMIN, [str(ADMIN)]).lower()
    assert "already" in router.dispatch("allow", ADMIN, [str(ALLOWED)]).lower()


# --- unauthorized stranger: reply with their id, never silent ------------------------------
def test_unauthorized_gets_id_reply(router: Router):
    out = router.dispatch("status", STRANGER, [])
    assert "access" in out.lower()
    assert str(STRANGER) in out


def test_unauthorized_on_admin_command_still_just_unauthorized(router: Router):
    # the auth check runs before the admin check, so a stranger never learns the command exists
    out = router.dispatch("allow", STRANGER, ["123"])
    assert str(STRANGER) in out
    assert "Admin-only" not in out


# --- /users rendering + help admin section ------------------------------------------------
def test_users_lists_tiers(router: Router):
    router.dispatch("allow", ADMIN, [str(STRANGER), "alice"])
    out = router.dispatch("users", ADMIN, [])
    assert "👑" in out and str(ADMIN) in out      # admin tier
    assert str(ALLOWED) in out                     # env seed tier
    assert str(STRANGER) in out and "alice" in out  # managed tier


def test_help_shows_admin_section_only_to_admin(router: Router):
    admin_help = router.dispatch("help", ADMIN, [])
    allowed_help = router.dispatch("help", ALLOWED, [])
    assert "/allow" in admin_help and "Admin" in admin_help
    assert "/allow" not in allowed_help
