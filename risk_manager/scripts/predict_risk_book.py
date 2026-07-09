"""CLI seam: sleeve_signal JSON(s) [+ risk_analytics JSON] -> risk_book JSON.

A pure-STDLIB process boundary mirroring the ML serving CLI (predict_dividend_sleeve.py /
predict_risk_analytics.py), so the orchestrator MAY shell out instead of importing risk_manager
in-process. The combiner is pure Python (no pandas/numpy), so this script carries no heavy deps —
the H4 vol forecast + H5 regime gate arrive as a `risk_analytics` JSON file produced by the ML CLI
(omit it and the book is built without vol-targeting/regime gating, exposure_scalar defaults to 1).

    # one sleeve from a file, risk analytics from the ML CLI, book to stdout
    python risk_manager/scripts/predict_risk_book.py \
        --sleeves data/reports/sleeve_signal.json \
        --risk-analytics contracts/examples/risk_analytics.example.json --out -

    # multiple sleeves on stdin (a JSON list), default sector hedge
    cat sleeves.json | python risk_manager/scripts/predict_risk_book.py --sleeves - --out book.json

Frozen entry point the agent calls in-process: risk_manager.src.combine(sleeve_signals,
risk_analytics, config, *, sleeve_status). This CLI is the same handshake over a process boundary.
is_production stays false unless a LIVE (signed-off + gate-MET) sleeve is present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from risk_manager.src import CombinerConfig, combine  # noqa: E402


def _read_json(arg: str) -> object:
    """Read JSON from a path, or from stdin when arg == '-'."""
    text = sys.stdin.read() if arg == "-" else Path(arg).read_text(encoding="utf-8")
    return json.loads(text)


def _validate(payload: dict, name: str) -> dict:
    """Best-effort schema validation (no-op if jsonschema is absent)."""
    try:
        import jsonschema
    except ImportError:
        return payload
    schema = json.loads((REPO_ROOT / "contracts" / f"{name}.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sleeves", required=True,
                    help="path to a sleeve_signal JSON, a JSON list of them, or '-' for stdin")
    ap.add_argument("--risk-analytics", default=None,
                    help="path to a risk_analytics JSON (H4 vol + H5 regime gate); optional")
    ap.add_argument("--sleeve-status", default=None,
                    help="path to a JSON map {sleeve: {gate|forward_pnl}} from the agent's P&L attribution")
    ap.add_argument("--hedge", default="sector", choices=["sector", "market", "none"])
    ap.add_argument("--target-vol", type=float, default=0.12, help="annualized book vol target (H4)")
    ap.add_argument("--out", default="-", help="output path or '-' for stdout")
    args = ap.parse_args()

    raw = _read_json(args.sleeves)
    sleeve_signals = raw if isinstance(raw, list) else [raw]
    for s in sleeve_signals:
        _validate(s, "sleeve_signal")
    risk_analytics = _read_json(args.risk_analytics) if args.risk_analytics else None
    sleeve_status = _read_json(args.sleeve_status) if args.sleeve_status else None

    cfg = CombinerConfig(hedge_mode=args.hedge, target_book_vol_annual=args.target_vol)
    book = combine(sleeve_signals, risk_analytics, cfg, sleeve_status=sleeve_status)
    payload = _validate(book.to_dict(), "risk_book")

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out == "-":
        sys.stdout.write(text + "\n")
    else:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        live = sum(1 for g in payload["gating"] if g["capital_state"] == "live")
        shadow = len(payload["gating"]) - live
        sys.stderr.write(f"risk_book: {live} live / {shadow} shadow sleeve(s); "
                         f"directional_gross={payload['risk_scalars']['directional_gross']} -> {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
