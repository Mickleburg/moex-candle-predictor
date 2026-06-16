"""V2 serving: turn a cross-sectional model's per-ticker SCORES into the frozen
`aggregated_signal` contract (the decision-model output that risk_manager consumes).

This replaces the V1 per-ticker `model_registry` path (which emitted `ml_prediction`
buy/hold/sell). In V2 the decision model ranks the whole universe by predicted RELATIVE
strength; this module serialises that ranking into `contracts/aggregated_signal.schema.json`:
universe, horizon, rankings[{ticker, score, rank, percentile, leg}], market_neutral, etc.

It does NOT decide position sizes or place orders — that's risk_manager. `leg` is the
INTENDED portfolio side (top-k long / bottom-k short / middle flat); risk_manager turns it
into a beta-neutral, vol-sized portfolio.
"""

from __future__ import annotations

import math
from typing import Any


def build_aggregated_signal(
    *,
    as_of: str,
    timeframe: str,
    horizon_bars: int,
    scores: dict[str, float],
    k: int,
    model_version: str,
    expected_relative_returns: dict[str, float] | None = None,
    market_neutral: bool = True,
    is_production: bool = False,
) -> dict[str, Any]:
    """Build a dict compatible with ``aggregated_signal.schema.json``.

    `scores`: ticker -> relative-strength score (higher = expected to outperform).
    `k`: long the top-k, short the bottom-k, the rest flat. Requires len(scores) >= 2*k.
    """
    tickers = [t for t, s in scores.items() if s is not None and math.isfinite(float(s))]
    if len(tickers) < 2:
        raise ValueError("need >= 2 valid scored tickers for a cross-section")
    if k < 1 or 2 * k > len(tickers):
        raise ValueError(f"k={k} invalid for {len(tickers)} tickers (need 2*k <= n)")

    # rank 1 = strongest (highest score)
    ordered = sorted(tickers, key=lambda t: float(scores[t]), reverse=True)
    n = len(ordered)
    long_set = set(ordered[:k])
    short_set = set(ordered[-k:])

    rankings = []
    for rank, t in enumerate(ordered, start=1):
        leg = "long" if t in long_set else "short" if t in short_set else "flat"
        percentile = round((n - rank) / (n - 1), 4) if n > 1 else 0.0
        entry: dict[str, Any] = {
            "ticker": t,
            "score": round(float(scores[t]), 6),
            "rank": rank,
            "percentile": percentile,
            "leg": leg,
        }
        if expected_relative_returns and t in expected_relative_returns:
            entry["expected_relative_return"] = round(float(expected_relative_returns[t]), 6)
        rankings.append(entry)

    return {
        "as_of": as_of,
        "timeframe": timeframe,
        "universe": list(ordered),
        "horizon": {"bars": int(horizon_bars), "timeframe": timeframe},
        "rankings": rankings,
        "market_neutral": bool(market_neutral),
        "model_version": model_version,
        "is_production": bool(is_production),
    }


def validate_against_schema(signal: dict[str, Any]) -> None:
    """Validate a built signal against the frozen schema (no-op if jsonschema absent)."""
    from pathlib import Path
    import json

    repo = Path(__file__).resolve().parents[3]
    schema_path = repo / "contracts" / "aggregated_signal.schema.json"
    try:
        import jsonschema
    except ImportError:
        return
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(signal)
