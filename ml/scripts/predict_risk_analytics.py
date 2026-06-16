"""CLI: emit a `risk_analytics` payload (vol forecast + regime gate + sizing) for risk_manager.

Layer-4 serving path end-to-end: daily panel -> EWMA vol (H4) + regime novelty (H5) ->
frozen risk_analytics contract (schema-validated). INFORMATION for risk_manager, not a trade.

    python ml/scripts/predict_risk_analytics.py --horizon 10 --out contracts/examples/risk_analytics.example.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from src.features.cross_sectional import load_panels  # noqa: E402
from src.service.risk_analytics import build_risk_analytics, validate_against_schema  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--out", type=str, default="contracts/examples/risk_analytics.example.json")
    args = ap.parse_args()

    panel, _, market = load_panels(timeframe="1D")
    payload = build_risk_analytics(panel, market, horizon_bars=args.horizon)
    validate_against_schema(payload)

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    reg = payload["regime"]
    top = sorted(payload["per_ticker"], key=lambda e: e["inv_vol_weight"], reverse=True)[:3]
    print(f"as_of={payload['as_of']}  horizon={args.horizon} bars  universe={len(payload['universe'])}")
    print(f"  regime: distance={reg['distance']} pct={reg['percentile']} novel={reg['novel']} "
          f"exposure_scalar={reg['exposure_scalar']}")
    print(f"  lowest-vol (highest weight): {[(e['ticker'], e['inv_vol_weight']) for e in top]}")
    print(f"  schema-valid: OK  is_production={payload['is_production']}")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
