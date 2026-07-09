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
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from zoneinfo import ZoneInfo

from . import trading_calendar as tcal
from .config import AgentConfig
from .orchestrator import Orchestrator

log = logging.getLogger("agent.scheduler")


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
        ref = _reference_eod_date(datetime.now(tz), eod_h, eod_m)
        if ref is None:
            return
        cycle = orch.store.get_cycle(ref, "eod")
        if cycle is None or cycle.get("status") not in ("completed", "halted", "killed"):
            msg = (f"DEAD-MAN'S-SWITCH: EOD cycle for {ref} did not complete "
                   f"(status={cycle.get('status') if cycle else 'missing'}).")
            log.error(msg)
            orch.notifier.send("DEAD-MAN'S-SWITCH — missed EOD cycle", msg)

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
