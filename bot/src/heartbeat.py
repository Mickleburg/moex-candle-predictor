"""Poller liveness heartbeat — a file whose mtime says "getUpdates came back recently".

Why a file and not an in-process check: the bot hung TWICE (2026-07-15, 2026-07-20) with the
container reported Up/healthy and the logs simply stopping mid-stream — no error spam, no crash.
The agent's notifier kept delivering digests through the same Finnish proxy, so the network and
the proxy were alive; what died was the ``getUpdates`` await itself (a half-open socket read that
never timed out). Two consequences that shape this module:

  * The event loop stays HEALTHY while the poller is wedged, so any in-process liveness probe
    (a JobQueue tick, "is the loop running") reports green and is worthless here. The heartbeat
    must be tied to the *completion of a getUpdates round-trip* — see app.build_get_updates_request.
  * The liveness signal has to survive the process boundary, because the thing that must act on it
    is Docker's healthcheck (which then lets ``restart: unless-stopped`` recreate the container).
    A file mtime in the shared ./data bind-mount is the simplest such channel — no port, no state.

Touched once per successful poll (so at most every ``poll_timeout`` seconds) and once at startup;
freshness is read back by the container healthcheck via ``python -m bot.src.heartbeat``.

Deliberately stdlib-only and dependency-free: it must NOT import PTB, the agent config, or anything
else that could fail for an unrelated reason. A healthcheck that goes red because of an unrelated
import error would restart-loop a perfectly healthy bot.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger("bot.heartbeat")

REPO_ROOT = Path(__file__).resolve().parents[2]
# Lives in the shared ./data bind-mount (gitignored via `data/bot/`) so the healthcheck running in
# the same container — and a human on the host — can both read it.
DEFAULT_HEARTBEAT = REPO_ROOT / "data" / "bot" / "heartbeat"

# Optional override. Resolved HERE, not at each call site, because the writer (the poller, via
# BotConfig) and the reader (this module's healthcheck CLI) MUST agree: if only one of them honoured
# BOT_HEARTBEAT_PATH, setting it would make the probe read a file nobody writes -> permanently
# unhealthy -> a restart loop on a perfectly healthy bot. One resolver keeps them in lockstep.
ENV_VAR = "BOT_HEARTBEAT_PATH"


def resolve_path(path: Path | str | None = None) -> Path:
    """The heartbeat path: explicit argument > $BOT_HEARTBEAT_PATH > DEFAULT_HEARTBEAT."""
    if path is not None:
        return Path(path)
    return Path(os.getenv(ENV_VAR) or DEFAULT_HEARTBEAT)

# ~5 missed long-polls at the default 30s timeout. Long enough that one slow round-trip or a proxy
# retry never restarts the bot; short enough that a wedged poller is recycled within ~3 minutes
# instead of going unnoticed for days (07-15 → 07-20 was five days of silent death).
DEFAULT_MAX_AGE = 180.0

# How often the in-process watchdog (app._run_watchdog) re-checks freshness. Matches the container
# healthcheck cadence; a stale poller is force-exited within one interval of crossing DEFAULT_MAX_AGE.
WATCHDOG_INTERVAL = 60.0


def touch(path: Path | str = DEFAULT_HEARTBEAT) -> bool:
    """Stamp the heartbeat. Best-effort: returns False instead of raising.

    Called from the poller's hot path, so it must never propagate — an unwritable heartbeat is a
    monitoring problem, not a reason to kill a bot that is otherwise serving commands fine. It
    still degrades safely: no stamp -> stale file -> healthcheck red -> restart.

    The parent mkdir is not redundant with startup: on a fresh VDS the bind-mount can come up
    without data/bot/, and a bare Path.touch() would then raise on EVERY poll, leaving the bot
    permanently "unhealthy" while working perfectly.
    """
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        return True
    except OSError:
        return False


def age_seconds(path: Path | str = DEFAULT_HEARTBEAT) -> float | None:
    """Seconds since the last successful poll, or None if the heartbeat has never been written."""
    p = Path(path)
    try:
        return max(0.0, time.time() - p.stat().st_mtime)
    except OSError:
        return None


def is_fresh(path: Path | str = DEFAULT_HEARTBEAT, max_age: float = DEFAULT_MAX_AGE) -> bool:
    """True iff a getUpdates round-trip completed within ``max_age`` seconds.

    A missing file counts as NOT fresh: either the bot has not started yet (``start_period`` covers
    that window) or it died before its first poll — both are correctly "not serving".
    """
    age = age_seconds(path)
    return age is not None and age < max_age


def check_and_exit(path: Path | str = DEFAULT_HEARTBEAT, max_age: float = DEFAULT_MAX_AGE,
                   *, _exit=os._exit) -> bool:
    """Backstop actor: if the poller is wedged (heartbeat stale), log a breadcrumb and HARD-exit so
    ``restart: unless-stopped`` recreates the container.

    This exists because ``restart: unless-stopped`` reacts to the container *exiting*, not to an
    unhealthy healthcheck (that is Swarm-only) — so an unhealthy-but-alive bot would hang forever.
    The compose healthcheck is the eye; this is the hand that actually restarts.

    Returns True if it acted (called ``_exit``), False if the poller is healthy or has not started
    yet. A heartbeat that has NEVER been written (age None) counts as "not started", NOT wedged —
    so a slow cold start can't trigger a restart loop before the first poll (``post_init`` also
    pre-stamps the file, so in practice the watchdog always sees a real timestamp).

    ``_exit`` defaults to ``os._exit`` — a hard exit that skips interpreter cleanup ON PURPOSE:
    a graceful ``Application.stop()`` could itself hang on the very HTTP client we are escaping.
    The bot is read-only with no critical in-flight state (the allowlist is written atomically),
    so there is nothing to flush. ``_exit`` is injectable purely so tests can assert the decision
    without killing the test process.
    """
    age = age_seconds(path)
    if age is None:
        return False  # never started; start_period + the post_init pre-stamp cover this window
    if age < max_age:
        return False  # poller answered recently — the healthy path, taken every tick in normal ops
    log.critical("heartbeat stale %.0fs > %.0fs — forcing exit for restart", age, max_age)
    _exit(1)
    return True  # unreachable in production (os._exit never returns); only a test stub gets here


def main(argv: list[str] | None = None) -> int:
    """Container healthcheck entry point: exit 0 = poller alive, 1 = wedged/never started."""
    parser = argparse.ArgumentParser(description="Check the bot poller heartbeat freshness.")
    parser.add_argument("--path", default=None,
                        help=f"heartbeat file (default: ${ENV_VAR} or {DEFAULT_HEARTBEAT})")
    parser.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE)
    args = parser.parse_args(argv)
    args.path = resolve_path(args.path)   # same resolution the poller uses -> they cannot diverge

    age = age_seconds(args.path)
    if age is None:
        print(f"heartbeat MISSING at {args.path} — poller has not completed a getUpdates yet")
        return 1
    if age >= args.max_age:
        print(f"heartbeat STALE — {age:.0f}s old (max {args.max_age:.0f}s); getUpdates is wedged")
        return 1
    print(f"heartbeat ok — {age:.0f}s old (max {args.max_age:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
