"""Agent CLI — manual cycle runs, state inspection, and the kill-switch.

    python -m agent.src.cli run-eod [--trade-date YYYY-MM-DD] [--as-of TS] [--force]
    python -m agent.src.cli run-preopen [--trade-date YYYY-MM-DD] [--force]
    python -m agent.src.cli status
    python -m agent.src.cli kill-switch {on,off,status}
    python -m agent.src.cli init-db
    python -m agent.src.cli scheduler            # run the long-lived daemon (APScheduler)

The scheduler subcommand defers to agent.src.scheduler (needs APScheduler; see requirements).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .config import load_config
from .orchestrator import Orchestrator
from .state_store import StateStore


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def cmd_run_eod(args) -> int:
    orch = Orchestrator(load_config(args.config))
    _print(orch.run_eod_cycle(trade_date=args.trade_date, as_of=args.as_of, force=args.force))
    return 0


def cmd_run_preopen(args) -> int:
    orch = Orchestrator(load_config(args.config))
    _print(orch.run_preopen(trade_date=args.trade_date, force=args.force))
    return 0


def cmd_status(args) -> int:
    cfg = load_config(args.config)
    cfg.ensure_dirs()
    store = StateStore(cfg.state_db)
    _print({
        "mode": cfg.mode, "block_mode": cfg.block_mode, "live_enabled": cfg.live_enabled(),
        "kill_switch": store.kill_switch_engaged(),
        "last_cycle": store.get_flag("last_cycle"),
        "last_successful_cycle": store.last_successful_cycle(),
        "live_positions": store.get_positions("live"),
        "shadow_positions": store.get_positions("shadow"),
        "open_orders": store.open_orders(),
        # observability: paper-shadow trading activity (the H9 forward-shadow track), so an empty
        # live book reads as "0 live, N shadow" rather than just "nothing happening".
        "recent_shadow_orders": store.recent_orders("shadow", limit=20),
        "recent_live_orders": store.recent_orders("live", limit=20),
        "pnl_by_sleeve": store.pnl_by_sleeve(),
    })
    return 0


def cmd_kill_switch(args) -> int:
    cfg = load_config(args.config)
    cfg.ensure_dirs()
    store = StateStore(cfg.state_db)
    if args.action in ("on", "off"):
        store.set_kill_switch(args.action == "on")
    _print({"kill_switch": store.kill_switch_engaged()})
    return 0


def cmd_init_db(args) -> int:
    cfg = load_config(args.config)
    cfg.ensure_dirs()
    StateStore(cfg.state_db)
    _print({"status": "ok", "state_db": str(cfg.state_db)})
    return 0


def cmd_scheduler(args) -> int:
    from .scheduler import run_scheduler
    return run_scheduler(load_config(args.config))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent", description="MOEX V3 trading-agent orchestrator")
    p.add_argument("--config", default=None, help="path to agent_config.json")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("run-eod", help="run one EOD cycle")
    e.add_argument("--trade-date", default=None)
    e.add_argument("--as-of", default=None)
    e.add_argument("--force", action="store_true")
    e.set_defaults(func=cmd_run_eod)

    o = sub.add_parser("run-preopen", help="run one pre-open check")
    o.add_argument("--trade-date", default=None)
    o.add_argument("--force", action="store_true")
    o.set_defaults(func=cmd_run_preopen)

    s = sub.add_parser("status", help="show agent state")
    s.set_defaults(func=cmd_status)

    k = sub.add_parser("kill-switch", help="engage/release the kill-switch")
    k.add_argument("action", choices=["on", "off", "status"])
    k.set_defaults(func=cmd_kill_switch)

    d = sub.add_parser("init-db", help="create the state store")
    d.set_defaults(func=cmd_init_db)

    sc = sub.add_parser("scheduler", help="run the long-lived scheduler daemon")
    sc.set_defaults(func=cmd_scheduler)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
