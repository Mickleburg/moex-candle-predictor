"""Serving CLI for the H9 dividend run-up sleeve — emits a validated `sleeve_signal` JSON.

This is the PROCESS-BOUNDARY seam for the agent orchestrator (EOD step 4): instead of importing
pandas/numpy/ML internals into the stdlib-only orchestrator core, the orchestrator shells out to this
CLI and reads back a `sleeve_signal` JSON (contract `contracts/sleeve_signal.schema.json`) — exactly
how it already invokes the execution block. Past-only; is_production=false.

    python ml/scripts/predict_dividend_sleeve.py --as-of 2026-07-06 --out data/reports/sleeve_signal_dividend.json
    python ml/scripts/predict_dividend_sleeve.py --out -        # JSON to stdout (subprocess capture)

The signal is the sleeve's LONG target positions (inverse-vol sized) + a sector-hedge recommendation;
risk_manager nets sleeves, applies vol-target (H4) + regime gate (H5) + limits + the book-level hedge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from scripts.xsec_eval_harness import load_daily_panel  # noqa: E402
from src.service.dividend_sleeve import build_sleeve_signal, load_dividend_calendar  # noqa: E402

UNIVERSE = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MGNT",
            "MTSS", "SNGS", "CHMF", "ALRS", "VTBR", "MAGN", "NLMK", "PLZL"]
SCHEMA = REPO_ROOT / "contracts" / "sleeve_signal.schema.json"
DEFAULT_OUT = REPO_ROOT / "data" / "reports" / "sleeve_signal_dividend.json"
_LEGS = {"long", "short", "hedge", "flat"}


def check_contract(sig: dict) -> None:
    """Minimal structural check against contracts/sleeve_signal.schema.json (no jsonschema dep).
    Guards the orchestrator seam: a malformed signal must fail loudly here, not downstream."""
    required = json.loads(SCHEMA.read_text(encoding="utf-8"))["required"]
    missing = [k for k in required if k not in sig]
    if missing:
        raise ValueError(f"sleeve_signal missing required fields: {missing}")
    if sig["sleeve"] not in {"s1_pairs", "s2_macro", "s3_event", "s4_core"}:
        raise ValueError(f"invalid sleeve id: {sig['sleeve']}")
    if sig["is_production"] is not False:
        raise ValueError("is_production must be false (research artifact)")
    for p in sig["positions"]:
        if not {"ticker", "weight", "leg"} <= p.keys() or p["leg"] not in _LEGS:
            raise ValueError(f"invalid position: {p}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit the H9 dividend-sleeve signal as sleeve_signal JSON.")
    ap.add_argument("--as-of", default=None, help="Decision date YYYY-MM-DD (default: today, MSK).")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output path, or '-' for stdout.")
    ap.add_argument("--quiet", action="store_true", help="Suppress the human summary line.")
    args = ap.parse_args()

    as_of = (pd.Timestamp(args.as_of, tz="Europe/Moscow") if args.as_of
             else pd.Timestamp.now(tz="Europe/Moscow").normalize())
    panel = load_daily_panel(UNIVERSE)
    calendar = load_dividend_calendar()
    sig = build_sleeve_signal(panel, calendar, as_of)
    check_contract(sig)

    payload = json.dumps(sig, indent=2, ensure_ascii=False)
    if args.out == "-":
        sys.stdout.write(payload + "\n")
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        if not args.quiet:
            names = ", ".join(f"{p['ticker']}:{p['weight']:.3f}" for p in sig["positions"]) or "(none)"
            print(f"sleeve_signal {sig['sleeve']}/{sig['strategy']} as_of={sig['as_of']} "
                  f"gross_long={sig['gross_long']:.3f} positions=[{names}]  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
