"""CLI: produce an `aggregated_signal` for the universe at the latest available date.

Wires the V2 serving path end-to-end: daily panel -> per-ticker score -> frozen
`aggregated_signal` contract (schema-validated). Today the score is a PLACEHOLDER
(cross-sectional momentum) — price-only has no edge (see research), so this is NOT a
tradeable signal, only proof the decision->contract->risk_manager plumbing works.
When the news-fused ranker exists, swap the score source; everything downstream is unchanged.

    python ml/scripts/predict_xsec_signal.py --horizon 20 --k 3 --out data/reports/aggregated_signal.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from src.features.cross_sectional import load_panels  # noqa: E402
from src.service.cross_sectional_signal import (  # noqa: E402
    build_aggregated_signal, validate_against_schema,
)


def momentum_scores(panel, lookback: int) -> dict[str, float]:
    """Placeholder score: cross-sectionally z-scored past-return momentum at the last date."""
    past = panel.iloc[-1] / panel.iloc[-1 - lookback] - 1.0
    z = (past - past.mean()) / (past.std() + 1e-9)
    return {t: float(z[t]) for t in panel.columns if np.isfinite(z[t])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--out", type=str, default="data/reports/aggregated_signal.json")
    args = ap.parse_args()

    panel, _, _ = load_panels(timeframe="1D")
    as_of = panel.index[-1].isoformat()
    scores = momentum_scores(panel, args.lookback)

    signal = build_aggregated_signal(
        as_of=as_of,
        timeframe="1D",
        horizon_bars=args.horizon,
        scores=scores,
        k=args.k,
        model_version="placeholder_xsec_momentum_v0",
        market_neutral=True,
        is_production=False,
    )
    validate_against_schema(signal)   # raises if the contract is violated

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(signal, indent=2), encoding="utf-8")

    longs = [r["ticker"] for r in signal["rankings"] if r["leg"] == "long"]
    shorts = [r["ticker"] for r in signal["rankings"] if r["leg"] == "short"]
    print(f"as_of={as_of}  universe={len(signal['universe'])}  horizon={args.horizon} bars")
    print(f"  LONG  (top-{args.k}): {longs}")
    print(f"  SHORT (bot-{args.k}): {shorts}")
    print(f"  schema-valid: OK   is_production={signal['is_production']}")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
