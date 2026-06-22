"""Config env overrides — the levers a Docker/systemd deploy uses via .env (no rebuild)."""

from __future__ import annotations

from agent.src.config import load_config

_ENV_VARS = ["AGENT_MODE", "AGENT_ENABLE_LIVE", "AGENT_BACKEND_MODE", "AGENT_SLEEVE_MODE",
             "AGENT_COMBINER_MODE", "AGENT_EXECUTION_MODE", "AGENT_LLM_REFRESH_CMD"]


def _clear(monkeypatch):
    for v in _ENV_VARS:
        monkeypatch.delenv(v, raising=False)


def test_defaults_are_paper_locked(monkeypatch):
    _clear(monkeypatch)
    cfg = load_config()
    assert cfg.mode == "paper"
    assert cfg.enable_live is False and cfg.live_enabled() is False


def test_per_block_mode_env_overrides(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENT_BACKEND_MODE", "live")
    monkeypatch.setenv("AGENT_SLEEVE_MODE", "live")
    monkeypatch.setenv("AGENT_LLM_REFRESH_CMD", "python llm/scripts/refresh_dividend_feed.py")
    cfg = load_config()
    assert cfg.blocks["backend"]["mode"] == "live"
    assert cfg.blocks["sleeve"]["mode"] == "live"
    assert cfg.blocks["combiner"]["mode"] == "mock"   # committed default, untouched
    assert cfg.blocks["llm"]["refresh_cmd"] == ["python", "llm/scripts/refresh_dividend_feed.py"]


def test_live_needs_both_mode_and_flag(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENT_ENABLE_LIVE", "true")    # flag alone is not enough
    assert load_config().live_enabled() is False        # mode still paper -> paper-first holds
    monkeypatch.setenv("AGENT_MODE", "live")
    assert load_config().live_enabled() is True         # both -> live permitted
