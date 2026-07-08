"""Execution CLI: the orchestrator seam (`serve`) + manual dry-run / paper-season / kill helpers.

Orchestrator entry point (agent step 6) — request JSON on stdin, result JSON on stdout:

    python -m execution.src.cli serve --mode paper  < request.json  > result.json

    request  = {risk_book, positions, prices, capital, mode, trade_date, phase, [anchors]}
    result   = {orders, reports, positions, rejected, halted, is_production}

This mirrors how the agent already invokes ML/risk_manager CLIs (see
agent/src/adapters/live.py::LiveExecution): it appends `--mode <mode>`, feeds the envelope on stdin,
and parses stdout JSON. Set `blocks.execution.command` to the prefix (without `--mode`).

Manual helpers:
    python -m execution.src.cli dry-run --risk-book contracts/examples/risk_book.example.json
    python -m execution.src.cli paper-season --season execution/examples/risk_book_season.example.json
    python -m execution.src.cli kill --reason "manual stop"   # engage kill-switch (cancel all + halt)
    python -m execution.src.cli unkill                          # clear the kill-switch

Live is never reachable without the explicit gate (mode=live + allow-live + EXECUTION_ALLOW_LIVE=1).
Only `serve` writes JSON to stdout; all human text goes to stderr so stdout stays machine-parseable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ExecutionConfig, Mode
from .engine import ExecutionEngine
from .instruments import load_lot_sizes
from .trading_calendar import active_calendar_source

_MODE_MAP = {
    "dry-run": Mode.DRY_RUN, "dry_run": Mode.DRY_RUN, "dryrun": Mode.DRY_RUN,
    "paper": Mode.PAPER, "live": Mode.LIVE,
}


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_mode(name: str) -> Mode:
    try:
        return _MODE_MAP[str(name).lower()]
    except KeyError:
        raise SystemExit(f"unknown mode {name!r}; expected one of {sorted(_MODE_MAP)}")


def cmd_serve(args: argparse.Namespace) -> int:
    """Read a reconciliation request envelope, execute it, emit the result envelope as JSON."""
    raw = sys.stdin.read() if args.request == "-" else Path(args.request).read_text(encoding="utf-8")
    req = json.loads(raw.lstrip("﻿"))   # tolerate a UTF-8 BOM from shell pipes
    mode = _resolve_mode(args.mode or req.get("mode") or "paper")
    capital = float(req.get("capital", args.capital))

    config = ExecutionConfig(mode=mode, capital=capital, broker_backend=args.broker_backend,
                             lot_sizes=load_lot_sizes(), allow_live=args.allow_live)
    engine = ExecutionEngine(config)
    out = engine.reconcile_and_execute(
        risk_book=req["risk_book"], positions=req.get("positions", []),
        prices=req.get("prices", {}), anchors=req.get("anchors"),
        on_critical=args.on_critical,
    )

    payload = json.dumps(out, ensure_ascii=False)
    if args.out == "-":
        sys.stdout.write(payload + "\n")
    else:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    if not args.quiet:
        print(f"[serve] mode={mode.value} calendar={active_calendar_source()} "
              f"orders={len(out['orders'])} rejected={len(out['rejected'])} "
              f"halted={out['halted']} is_production={out['is_production']}", file=sys.stderr)
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    risk_book = _load(args.risk_book)
    prices = _load(args.prices)
    current_positions = _load(args.positions) if args.positions else None
    config = ExecutionConfig(mode=Mode.DRY_RUN, capital=args.capital, lot_sizes=load_lot_sizes())
    engine = ExecutionEngine(config)
    res = engine.run_cycle(risk_book, prices, current_positions=current_positions)

    print(f"=== execution dry-run — {res.summary_line()} ===")
    for req in res.submitted:
        print(f"  {req['side']:4} {req['ticker']:7} {req['quantity_lots']:>10} lots "
              f"@ LIMIT {req['limit_price']:>12,.4f}   id={req['client_order_id']}")
    if res.noops:
        print(f"  no-op (already at target): {', '.join(res.noops)}")
    for sk in res.skipped:
        print(f"  SKIPPED {sk['instrument']}: {sk['reason']}")
    return 0


def cmd_paper_season(args: argparse.Namespace) -> int:
    season = _load(args.season)
    days = season["days"] if isinstance(season, dict) else season
    anchors = season.get("anchors") if isinstance(season, dict) else None
    config = ExecutionConfig(mode=Mode.PAPER, broker_backend="sim", capital=args.capital,
                             lot_sizes=load_lot_sizes())
    engine = ExecutionEngine(config)
    out = engine.run_season(days, anchors=anchors)
    for res in out["cycles"]:
        print(res.summary_line())
    print(f"\nfinal positions (lots): {out['final_positions']}")
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    engine = ExecutionEngine(ExecutionConfig(mode=_resolve_mode(args.mode)))
    reports = engine.kill(reason=args.reason)
    print(f"kill-switch ENGAGED ({args.reason}); canceled {len(reports)} open order(s). "
          f"halted={engine.halted}")
    return 0


def cmd_unkill(args: argparse.Namespace) -> int:
    engine = ExecutionEngine(ExecutionConfig(mode=_resolve_mode(args.mode)))
    engine.reset_kill()
    print(f"kill-switch cleared; halted={engine.halted}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="execution", description="MOEX execution adapter (paper-first)")
    ap.add_argument("--capital", type=float, default=100_000_000.0, help="book NAV in RUB")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sv = sub.add_parser("serve", help="orchestrator seam: stdin request envelope -> stdout result JSON")
    sv.add_argument("--mode", default=None, help="dry-run | paper | live (overrides envelope.mode)")
    sv.add_argument("--request", default="-", help="request envelope JSON path, or '-' for stdin")
    sv.add_argument("--out", default="-", help="result JSON path, or '-' for stdout")
    sv.add_argument("--broker-backend", default="sim", choices=["sim", "tinvest"])
    sv.add_argument("--on-critical", default="halt", choices=["halt", "warn"],
                    help="what to do on a critical -2 discipline breach")
    sv.add_argument("--allow-live", action="store_true",
                    help="in-config half of the live gate (env EXECUTION_ALLOW_LIVE=1 is still required)")
    sv.add_argument("--quiet", action="store_true", help="suppress the stderr summary line")
    sv.set_defaults(func=cmd_serve)

    d = sub.add_parser("dry-run", help="reconcile a risk_book into delta orders, send nothing")
    d.add_argument("--risk-book", default="contracts/examples/risk_book.example.json")
    d.add_argument("--prices", default="execution/examples/prices.example.json")
    d.add_argument("--positions", default=None, help="optional portfolio_snapshot JSON")
    d.set_defaults(func=cmd_dry_run)

    s = sub.add_parser("paper-season", help="replay a season through the paper simulator")
    s.add_argument("--season", default="execution/examples/risk_book_season.example.json")
    s.set_defaults(func=cmd_paper_season)

    k = sub.add_parser("kill", help="engage the kill-switch (cancel all open orders + halt)")
    k.add_argument("--reason", default="manual")
    k.add_argument("--mode", default="paper")
    k.set_defaults(func=cmd_kill)

    u = sub.add_parser("unkill", help="clear the kill-switch")
    u.add_argument("--mode", default="paper")
    u.set_defaults(func=cmd_unkill)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
