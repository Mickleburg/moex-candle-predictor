"""Watchdog: the heartbeat stamp + the freshness probe the container healthcheck runs.

These cover the two failure modes that let the bot hang unnoticed for five days: a stamp that
could raise inside the poller's hot path, and a "healthy" verdict for a poller that stopped.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from bot.src import heartbeat


def test_touch_creates_file_and_parent_dir(tmp_path: Path):
    # data/bot/ may not exist on a fresh bind-mount — a bare Path.touch() would raise every poll
    hb = tmp_path / "bot" / "heartbeat"
    assert heartbeat.touch(hb) is True
    assert hb.exists()


def test_touch_refreshes_mtime(tmp_path: Path):
    hb = tmp_path / "heartbeat"
    heartbeat.touch(hb)
    os.utime(hb, (0, 0))
    assert heartbeat.touch(hb) is True
    assert time.time() - hb.stat().st_mtime < 60


def test_touch_returns_false_instead_of_raising_on_oserror(tmp_path: Path, monkeypatch):
    # an unwritable heartbeat is a monitoring problem, NOT a reason to kill a working poller
    def _boom(*a, **kw):
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "touch", _boom)
    assert heartbeat.touch(tmp_path / "heartbeat") is False


def test_is_fresh_true_for_new_file_false_when_stale(tmp_path: Path):
    hb = tmp_path / "heartbeat"
    heartbeat.touch(hb)
    assert heartbeat.is_fresh(hb, max_age=180) is True

    os.utime(hb, (time.time() - 3600, time.time() - 3600))  # one hour behind
    assert heartbeat.is_fresh(hb, max_age=180) is False


def test_missing_heartbeat_is_not_fresh(tmp_path: Path):
    assert heartbeat.is_fresh(tmp_path / "absent", max_age=180) is False
    assert heartbeat.age_seconds(tmp_path / "absent") is None


def test_check_and_exit_forces_exit_when_stale(tmp_path: Path, caplog):
    hb = tmp_path / "heartbeat"
    heartbeat.touch(hb)
    os.utime(hb, (time.time() - 200, time.time() - 200))  # 200s > 180s -> wedged

    calls: list[int] = []
    import logging
    with caplog.at_level(logging.CRITICAL, logger="bot.heartbeat"):
        acted = heartbeat.check_and_exit(hb, _exit=calls.append)

    assert acted is True and calls == [1]                    # os._exit(1), replaced by a stub
    assert "forcing exit for restart" in caplog.text         # the breadcrumb 07-15 lacked


def test_check_and_exit_does_nothing_when_fresh(tmp_path: Path):
    hb = tmp_path / "heartbeat"
    heartbeat.touch(hb)
    calls: list[int] = []
    assert heartbeat.check_and_exit(hb, _exit=calls.append) is False
    assert calls == []                                        # a live poller is never restarted


def test_check_and_exit_treats_missing_as_not_started_not_wedged(tmp_path: Path):
    # before the first poll (and before the post_init pre-stamp on a bad day) there is no file;
    # that must NOT be read as "wedged" or the bot would restart-loop during a slow cold start
    calls: list[int] = []
    assert heartbeat.check_and_exit(tmp_path / "absent", _exit=calls.append) is False
    assert calls == []


def test_healthcheck_cli_exit_codes(tmp_path: Path, capsys):
    hb = tmp_path / "heartbeat"

    assert heartbeat.main(["--path", str(hb)]) == 1            # never started
    assert "MISSING" in capsys.readouterr().out

    heartbeat.touch(hb)
    assert heartbeat.main(["--path", str(hb)]) == 0            # poller alive
    assert "ok" in capsys.readouterr().out

    os.utime(hb, (time.time() - 3600, time.time() - 3600))
    assert heartbeat.main(["--path", str(hb)]) == 1            # wedged
    assert "STALE" in capsys.readouterr().out


# --- the poller transport ------------------------------------------------------------------

pytest.importorskip("telegram")

from bot.src import app  # noqa: E402
from bot.src.app import build_get_updates_request  # noqa: E402
from bot.src.config import BotConfig  # noqa: E402


def _request(tmp_path: Path, **kw):
    cfg = BotConfig(token="x", admin_chat_ids=frozenset({999}), poll_timeout=30,
                    allowlist_path=tmp_path / "allowlist.json",
                    heartbeat_path=tmp_path / "heartbeat", **kw)
    return build_get_updates_request(cfg), cfg


def test_successful_round_trip_stamps_the_heartbeat(tmp_path: Path, monkeypatch):
    req, cfg = _request(tmp_path)

    async def _ok(self, *a, **kw):
        return (200, b'{"ok":true,"result":[]}')

    monkeypatch.setattr(type(req).__mro__[1], "do_request", _ok)  # patch HTTPXRequest.do_request
    assert not cfg.heartbeat_path.exists()
    out = asyncio.run(req.do_request("https://api.telegram.org/botX/getUpdates", "POST"))
    assert out == (200, b'{"ok":true,"result":[]}')
    assert heartbeat.is_fresh(cfg.heartbeat_path, max_age=60)


def test_stamp_failure_does_not_break_the_poller(tmp_path: Path, monkeypatch):
    # if the stamp raised, the exception would surface as a getUpdates error and the bot would
    # stop serving over a monitoring detail. It must degrade to "unhealthy", not "broken".
    req, _ = _request(tmp_path)

    async def _ok(self, *a, **kw):
        return (200, b"payload")

    def _boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(type(req).__mro__[1], "do_request", _ok)
    monkeypatch.setattr(Path, "touch", _boom)
    assert asyncio.run(req.do_request("u", "POST")) == (200, b"payload")


def test_failed_round_trip_leaves_heartbeat_stale(tmp_path: Path, monkeypatch):
    # the hang signature: no successful poll -> no stamp -> healthcheck goes red on its own
    req, cfg = _request(tmp_path)

    async def _fail(self, *a, **kw):
        raise TimeoutError("half-open socket")

    monkeypatch.setattr(type(req).__mro__[1], "do_request", _fail)
    with pytest.raises(TimeoutError):
        asyncio.run(req.do_request("u", "POST"))
    assert not cfg.heartbeat_path.exists()


def test_timeouts_are_bounded(tmp_path: Path):
    # the belt: no getUpdates await may block forever. read_timeout is what Bot.get_updates ADDS
    # the long-poll timeout to, so the effective ceiling is 2*poll+5 — bounded, and under the 180s
    # healthcheck threshold so one slow poll can never trip a restart.
    req, cfg = _request(tmp_path)
    timeout = req._client.timeout
    assert req.read_timeout == cfg.poll_timeout + 5
    assert (timeout.connect, timeout.read, timeout.pool) == (10, cfg.poll_timeout + 5, 10)
    assert None not in (timeout.connect, timeout.read, timeout.pool)  # None == wait forever


def test_watchdog_loop_polls_the_checker_each_tick(tmp_path: Path, monkeypatch):
    # prove the async backstop actually ticks and calls check_and_exit with the configured path;
    # a wedged getUpdates await leaves the loop alive, so this task keeps running and can act.
    _, cfg = _request(tmp_path)
    seen: list = []
    monkeypatch.setattr(app.heartbeat, "check_and_exit", lambda p: seen.append(p))

    ticks = {"n": 0}

    async def _fake_sleep(_seconds):
        ticks["n"] += 1
        if ticks["n"] >= 2:               # let one full iteration run, then stop the loop
            raise asyncio.CancelledError

    monkeypatch.setattr(app.asyncio, "sleep", _fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(app._run_watchdog(cfg, interval=0))
    assert seen == [cfg.heartbeat_path]   # checked exactly once, against the heartbeat we configured


def test_watchdog_survives_a_stray_checker_error(tmp_path: Path, monkeypatch):
    # a bug in the check must not silently kill the backstop — it logs and keeps ticking
    _, cfg = _request(tmp_path)
    ticks = {"n": 0}

    def _boom(_p):
        raise RuntimeError("unexpected")

    async def _fake_sleep(_seconds):
        ticks["n"] += 1
        if ticks["n"] >= 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(app.heartbeat, "check_and_exit", _boom)
    monkeypatch.setattr(app.asyncio, "sleep", _fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(app._run_watchdog(cfg, interval=0))
    assert ticks["n"] == 3                # kept looping through the raised errors, didn't die


def test_proxy_survives_on_the_get_updates_transport(tmp_path: Path):
    # regression guard: PTB makes get_updates_request and get_updates_proxy mutually exclusive, so
    # the proxy MUST ride on this object — silently dropping it bricks the bot on a RU VDS.
    req, _ = _request(tmp_path, proxy_url="http://2.26.136.66:25565")
    proxies = {str(getattr(getattr(tr, "_pool", None), "_proxy_url", ""))
               for tr in req._client._mounts.values()}
    assert any("2.26.136.66" in p and "25565" in p for p in proxies), proxies
