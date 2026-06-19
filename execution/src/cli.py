"""Execution CLI: dry-run a risk_book into delta orders, or replay a paper season.

    python -m execution.src.cli dry-run --risk-book contracts/examples/risk_book.example.json \
        --prices execution/examples/prices.example.json

    python -m execution.src.cli paper-season --season execution/examples/risk_book_season.example.json

Dry-run sends nothing. Paper-season runs the internal simulator. Live is never reachable from the CLI
without the explicit gate (mode=live + allow-live + EXECUTION_ALLOW_LIVE=1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExecutionConfig, Mode
from .engine import ExecutionEngine, positions_from_snapshot


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_dry_run(args: argparse.Namespace) -> int:
    risk_book = _load(args.risk_book)
    prices = _load(args.prices)
    current = positions_from_snapshot(_load(args.positions)) if args.positions else {}
    config = ExecutionConfig(mode=Mode.DRY_RUN, capital=args.capital)
    engine = ExecutionEngine(config)
    res = engine.run_cycle(risk_book, prices, current_positions={"positions": [
        {"ticker": t, "lots": l, "avg_price": 0, "market_price": 0,
         "market_value": 0, "unrealized_pnl": 0} for t, l in current.items()]} if current else None)

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
    config = ExecutionConfig(mode=Mode.PAPER, broker_backend="sim", capital=args.capital)
    engine = ExecutionEngine(config)
    out = engine.run_season(days, anchors=anchors)
    for res in out["cycles"]:
        print(res.summary_line())
    print(f"\nfinal positions (lots): {out['final_positions']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="execution", description="MOEX execution adapter (paper-first)")
    ap.add_argument("--capital", type=float, default=100_000_000.0, help="book NAV in RUB")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dry-run", help="reconcile a risk_book into delta orders, send nothing")
    d.add_argument("--risk-book", default="contracts/examples/risk_book.example.json")
    d.add_argument("--prices", default="execution/examples/prices.example.json")
    d.add_argument("--positions", default=None, help="optional portfolio_snapshot JSON")
    d.set_defaults(func=cmd_dry_run)

    s = sub.add_parser("paper-season", help="replay a season through the paper simulator")
    s.add_argument("--season", default="execution/examples/risk_book_season.example.json")
    s.set_defaults(func=cmd_paper_season)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
