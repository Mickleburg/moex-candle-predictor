"""Demo / integration: feed the H9 dividend sleeve through the risk_manager combiner.

Tries the REAL handshake first — import the ML sleeve emitter (build_sleeve_signal) and the risk
analytics (build_risk_analytics H4/H5) and run them on data/raw if present — then falls back to the
canned contract examples so the demo always runs without the ML data/deps. Prints the netted book,
the applied risk scalars, the limit audit, and the per-name risk_decision render.

    python risk_manager/scripts/demo_combine_h9.py
    python risk_manager/scripts/demo_combine_h9.py --write-example   # refresh risk_book example
    python risk_manager/scripts/demo_combine_h9.py --as-of 2022-03-01  # regime-gate stress (needs ML data)

Reads only (ml/, data/); writes only its own contract example under contracts/examples/ when asked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from risk_manager.src import CombinerConfig, combine, to_risk_decisions  # noqa: E402

EXAMPLES = REPO_ROOT / "contracts" / "examples"


def _canned() -> tuple[dict, dict, str]:
    sig = json.loads((EXAMPLES / "sleeve_signal.example.json").read_text(encoding="utf-8"))
    ra = json.loads((EXAMPLES / "risk_analytics.example.json").read_text(encoding="utf-8"))
    return sig, ra, "canned examples"


def _live(as_of_str: str | None) -> tuple[dict, dict, str]:
    """Real handshake via the ML block (read-only). Raises if ML data/deps are unavailable."""
    import pandas as pd

    ml_dir = REPO_ROOT / "ml"
    sys.path.insert(0, str(ml_dir))
    from src.features.cross_sectional import load_panels  # type: ignore  # noqa: E402
    from src.service.dividend_sleeve import (  # type: ignore  # noqa: E402
        build_sleeve_signal, load_dividend_calendar,
    )
    from src.service.risk_analytics import build_risk_analytics  # type: ignore  # noqa: E402

    universe = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MGNT",
                "MTSS", "SNGS", "CHMF", "ALRS"]
    panel, _, market = load_panels(universe, timeframe="1D")
    calendar = load_dividend_calendar()
    as_of = (pd.Timestamp(as_of_str, tz="Europe/Moscow") if as_of_str else panel.index[-1])
    sig = build_sleeve_signal(panel, calendar, as_of)
    ra = build_risk_analytics(panel, market, as_of=as_of)
    return sig, ra, f"LIVE ML handshake (as_of={as_of.date()})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None, help="decision date (live mode only)")
    ap.add_argument("--hedge", default="sector", choices=["sector", "market", "none"])
    ap.add_argument("--write-example", action="store_true",
                    help="write contracts/examples/risk_book.example.json from this run")
    ap.add_argument("--canned", action="store_true", help="force canned examples (skip ML)")
    ap.add_argument("--force-live", action="store_true",
                    help="disable the shadow gate (require_production_for_live=False) to show the live book")
    args = ap.parse_args()

    if args.canned:
        sig, ra, src = _canned()
    else:
        try:
            sig, ra, src = _live(args.as_of)
        except Exception as exc:  # noqa: BLE001 - demo: degrade to canned on any ML/data issue
            print(f"[live ML handshake unavailable: {type(exc).__name__}: {exc}] -> falling back\n")
            sig, ra, src = _canned()

    cfg = CombinerConfig(hedge_mode=args.hedge, require_production_for_live=not args.force_live)
    book = combine([sig], ra, cfg)
    d = book.to_dict()

    print(f"=== risk_manager combiner — source: {src} ===")
    print(f"as_of={d['as_of']}  sleeves={[s['sleeve'] for s in d['sleeves']]}")
    print("gating (shadow gate, invariants #9/#4):")
    for g in d.get("gating", []):
        print(f"  {g['sleeve']:9} {g['capital_state'].upper():6} gate={g['gate']:8} "
              f"is_production={g['is_production']}  ({g['reason']})")
    reg = d["risk_scalars"]
    print(f"risk scalars: vol_scalar={reg['vol_scalar']}  exposure_scalar={reg['exposure_scalar']} "
          f"(novel={reg['regime_novel']})  ->  LIVE directional_gross={reg['directional_gross']} "
          f"total_gross={reg['total_gross']}  | shadow_gross={reg.get('shadow_gross')}")
    print(f"limits: name_ok={d['limits']['name_caps_ok']} sector_ok={d['limits']['sector_caps_ok']} "
          f"gross_ok={d['limits']['gross_cap_ok']} binding={d['limits']['binding']}")
    print(f"LIVE net positions ({len(d['net_positions'])}):")
    for p in d["net_positions"]:
        print(f"  {p['ticker']:6} {p['side']:5} w={p['weight']:+.4f}  sector={p['sector']}")
    print(f"LIVE hedge ({d['hedge']['mode']}): " +
          ", ".join(f"{leg['instrument']}={leg['weight']:+.4f}" for leg in d["hedge"]["legs"]) or "—")
    if d.get("shadow_positions"):
        print(f"SHADOW positions (0 live capital, paper-only) ({len(d['shadow_positions'])}):")
        for p in d["shadow_positions"]:
            print(f"  {p['ticker']:6} {p['side']:5} w={p['weight']:+.4f}  sector={p['sector']}")
        print(f"SHADOW hedge ({d.get('shadow_hedge', {}).get('mode')}): " +
              ", ".join(f"{leg['instrument']}={leg['weight']:+.4f}"
                        for leg in d.get("shadow_hedge", {}).get("legs", [])))
    print(f"is_production={d['is_production']}")

    dec = to_risk_decisions(book)
    print(f"\nrendered {len(dec)} risk_decision objects (order_intent=null -> execution sizes to lots)")

    if args.write_example:
        out = EXAMPLES / "risk_book.example.json"
        out.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
