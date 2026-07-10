"""Long-lived scheduler daemon — the agent's heartbeat on the VDS.

In-process APScheduler (one supervised process, unified logs/state/alerts) rather than cron
or systemd-timers, because the trading-day gate and the dead-man's-switch want to live next
to the state store. TZ is Europe/Moscow throughout.

Jobs:
  * EOD cron (default 19:05 Mon-Fri), gated to MOEX trading days -> Orchestrator.run_eod_cycle
  * pre-open cron (default 09:30 Mon-Fri), gated -> Orchestrator.run_preopen
  * dead-man's-switch (interval): if the most recent due EOD never completed, alert.

APScheduler is a deploy dependency (see top-level requirements.txt); it is imported lazily so
the orchestrator core and tests never need it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Optional
from zoneinfo import ZoneInfo

from . import trading_calendar as tcal
from .config import AgentConfig
from .orchestrator import Orchestrator

log = logging.getLogger("agent.scheduler")

# A cycle that reached one of these is accounted for — the switch stays quiet.
HEALTHY_STATUSES = ("completed", "halted", "killed")
# kv key holding the last dead-man alert: {"key": "<ref>:<status>", "ts": "<iso>"}
DEADMAN_FLAG = "deadman_last_alert"


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse a stored ISO timestamp to an aware datetime (the store writes UTC). None if unusable."""
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def deadman_verdict(cycle: dict | None, ref: str, now: datetime, last_alert: dict | None,
                    repeat_hours: float, running_grace_minutes: int
                    ) -> tuple[bool, str | None, dict | None]:
    """Should the dead-man's-switch alert for the reference EOD? Pure: no I/O, no APScheduler.

    Returns (should_alert, message, flag_state_after_this_tick). `flag_state_after_this_tick` is what
    the persisted flag should become: a dict to store (ONLY after a successful send), or None to clear
    it. Suppressed ticks return the caller's `last_alert` unchanged, so the quiet window keeps running.

    - healthy (completed/halted/killed): no alert, and CLEAR the flag — a stale timestamp must never
      suppress the alert for a future failure.
    - running: younger than `running_grace_minutes` -> the cycle is working, stay quiet; older -> it
      died mid-cycle, alert (a distinct "stuck" message).
    - anything else (failed / missing / unknown): alert, deduplicated on `f"{ref}:{status}"` — the same
      incident re-alerts at most once per `repeat_hours`, a NEW incident (different ref or status)
      alerts immediately. A real dead agent therefore keeps reminding, it just stops spamming.
    """
    status = (cycle or {}).get("status") or "missing"
    if status in HEALTHY_STATUSES:
        return False, None, None

    if status == "running":
        started = _parse_ts((cycle or {}).get("started_at"))
        if started is not None and now - started < timedelta(minutes=running_grace_minutes):
            return False, None, last_alert          # in progress on a slow box, not a corpse
        since = (cycle or {}).get("started_at") or "unknown time"
        message = (f"DEAD-MAN'S-SWITCH: EOD cycle for {ref} is stuck in `running` since {since} "
                   f"(over {running_grace_minutes} min). The agent likely died mid-cycle.")
    else:
        message = f"DEAD-MAN'S-SWITCH: EOD cycle for {ref} did not complete (status={status})."

    key = f"{ref}:{status}"
    if last_alert and last_alert.get("key") == key:
        last_ts = _parse_ts(last_alert.get("ts"))    # unparseable ts -> alert rather than stay silent
        if last_ts is not None and now - last_ts < timedelta(hours=repeat_hours):
            return False, None, last_alert           # same incident, still inside the quiet window
    return True, message, {"key": key, "ts": now.isoformat()}


def deadman_tick(store, notifier, ref: str, now: datetime, *, repeat_hours: float,
                 running_grace_minutes: int) -> bool:
    """One dead-man check: read state -> verdict -> send -> persist. Returns True if an alert went out.

    Separated from the scheduler closure so the delivery/persistence rules (dedup window, clear-on-
    healthy, never-arm-the-window-on-a-failed-send) are exercised offline without APScheduler.
    """
    cycle = store.get_cycle(ref, "eod")
    last_alert = store.get_flag(DEADMAN_FLAG)
    should_alert, message, flag_state = deadman_verdict(
        cycle, ref, now, last_alert, repeat_hours=repeat_hours,
        running_grace_minutes=running_grace_minutes)

    if not should_alert:
        if flag_state != last_alert:
            store.set_flag(DEADMAN_FLAG, flag_state)     # healthy -> clear stale suppression
        return False

    status = (cycle or {}).get("status") or "missing"
    subject = ("DEAD-MAN'S-SWITCH — EOD cycle stuck" if status == "running"
               else "DEAD-MAN'S-SWITCH — missed EOD cycle")
    log.error(message)
    # Persist ONLY on a delivered alert: a failed send (network/proxy) must not arm the quiet
    # window and swallow an alert nobody ever received.
    if not notifier.send(subject, message):
        log.warning("dead-man alert not delivered — flag left untouched, will retry next tick")
        return False
    store.set_flag(DEADMAN_FLAG, flag_state)
    return True


def _setup_logging(config: AgentConfig) -> None:
    config.ensure_dirs()
    handler = RotatingFileHandler(config.log_dir / "agent.log", maxBytes=5_000_000, backupCount=10,
                                  encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(fmt)
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root = logging.getLogger("agent")
    root.setLevel(logging.INFO)
    root.handlers[:] = [handler, stream]


def _parse_hhmm(value: str) -> tuple[int, int]:
    h, m = value.split(":")
    return int(h), int(m)


def run_scheduler(config: AgentConfig) -> int:
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        print("APScheduler is not installed. Install deploy deps: pip install -r requirements.txt")
        return 2

    _setup_logging(config)
    tz = ZoneInfo(config.schedule.timezone)
    orch = Orchestrator(config)
    eod_h, eod_m = _parse_hhmm(config.schedule.eod)
    pre_h, pre_m = _parse_hhmm(config.schedule.preopen)

    def eod_job() -> None:
        today = datetime.now(tz).date().isoformat()
        if not tcal.is_trading_day(today):
            log.info("EOD skipped — %s is not a trading day", today)
            return
        log.info("EOD cycle start trade_date=%s", today)
        out = orch.run_eod_cycle(trade_date=today)
        log.info("EOD cycle done trade_date=%s status=%s", today, out.get("status"))

    def preopen_job() -> None:
        today = datetime.now(tz).date().isoformat()
        if not tcal.is_trading_day(today):
            log.info("pre-open skipped — %s is not a trading day", today)
            return
        log.info("pre-open start trade_date=%s", today)
        out = orch.run_preopen(trade_date=today)
        log.info("pre-open done trade_date=%s status=%s", today, out.get("status"))

    def deadman_job() -> None:
        """Thin wrapper: resolve the reference EOD, then hand the decision to deadman_tick."""
        now = datetime.now(tz)
        ref = _reference_eod_date(now, eod_h, eod_m)
        if ref is None:
            return
        deadman_tick(orch.store, orch.notifier, ref, now,
                     repeat_hours=config.schedule.deadman_repeat_hours,
                     running_grace_minutes=config.schedule.deadman_running_grace_minutes)

    scheduler = BlockingScheduler(timezone=tz)
    scheduler.add_job(eod_job, CronTrigger(day_of_week="mon-fri", hour=eod_h, minute=eod_m, timezone=tz),
                      id="eod", misfire_grace_time=3600, coalesce=True)
    scheduler.add_job(preopen_job, CronTrigger(day_of_week="mon-fri", hour=pre_h, minute=pre_m, timezone=tz),
                      id="preopen", misfire_grace_time=1800, coalesce=True)
    scheduler.add_job(deadman_job, IntervalTrigger(minutes=config.schedule.deadman_check_minutes, timezone=tz),
                      id="deadman", coalesce=True)

    log.info("scheduler up: tz=%s eod=%s preopen=%s mode=%s block_mode=%s calendar=%s",
             config.schedule.timezone, config.schedule.eod, config.schedule.preopen,
             config.mode, config.block_mode, tcal.calendar_source())
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("scheduler shutting down")
    return 0


def _reference_eod_date(now: datetime, eod_h: int, eod_m: int) -> str | None:
    """The most recent trading day whose EOD time has already passed (dead-man reference).

    Returns None when no due EOD exists yet (e.g. early on the first trading day after a
    holiday block before the EOD time) — avoids false alarms on weekends/holidays.
    """
    eod_today = now.replace(hour=eod_h, minute=eod_m, second=0, microsecond=0) + timedelta(minutes=15)
    today = now.date()
    if tcal.is_trading_day(today) and now >= eod_today:
        return today.isoformat()
    return tcal.prev_trading_day(today).isoformat()
