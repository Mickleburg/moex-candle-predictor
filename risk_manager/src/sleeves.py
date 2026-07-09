"""Adapters that normalize heterogeneous sleeve inputs into directional positions.

The combiner accepts two sleeve forms:
  * POSITION form  (`sleeve_signal`):    target weights per ticker, signed by leg
                                         (long/short/hedge/flat). E.g. H9 build_sleeve_signal.
  * RANKING form   (`aggregated_signal`): cross-sectional rank/leg (long top-k / short bottom-k).
                                         The S1/S2 ranking sleeves (closed, but the shape is kept).

Both collapse to a `NormalizedSleeve`: directional weights on tradable names (signed) plus the
sleeve's SUGGESTED hedge legs (index instruments the risk_manager may replace with a sector hedge).
"""

from __future__ import annotations

from dataclasses import dataclass

from .sectors import is_index


@dataclass
class NormalizedSleeve:
    sleeve: str                                   # s1_pairs / s2_macro / s3_event / s4_core
    strategy: str
    directional: dict[str, float]                 # ticker -> signed weight on a tradable name
    suggested_hedge: list[dict]                   # [{instrument, weight}] the sleeve proposed
    market_neutral: bool
    gross: float                                  # sleeve self-reported gross
    is_production: bool


def _from_position_form(sig: dict) -> NormalizedSleeve:
    directional: dict[str, float] = {}
    suggested_hedge: list[dict] = []
    for p in sig.get("positions", []):
        tkr = p["ticker"]
        w = float(p["weight"])
        leg = p.get("leg", "long")
        if leg == "hedge" or is_index(tkr):
            # An index hedge the sleeve proposed; risk_manager owns the actual hedge choice.
            suggested_hedge.append({"instrument": tkr, "weight": w})
            continue
        if leg == "flat" or w == 0.0:
            continue
        directional[tkr] = directional.get(tkr, 0.0) + w
    return NormalizedSleeve(
        sleeve=sig.get("sleeve", "s3_event"),
        strategy=sig.get("strategy", sig.get("model_version", "unknown")),
        directional=directional,
        suggested_hedge=suggested_hedge,
        market_neutral=bool(sig.get("market_neutral", False)),
        gross=float(sig.get("gross", sum(abs(w) for w in directional.values()))),
        is_production=bool(sig.get("is_production", False)),
    )


def _from_ranking_form(sig: dict, name_weight: float = 0.0) -> NormalizedSleeve:
    """Convert an `aggregated_signal` ranking into directional weights.

    Long legs get +w, short legs -w, where w defaults to an equal split across the active leg
    (so the long and short books are each gross-1 and the sleeve is dollar-neutral). A ranking row
    may carry an explicit `target_weight` to override the equal split.
    """
    rankings = sig.get("rankings", [])
    longs = [r for r in rankings if r.get("leg") == "long"]
    shorts = [r for r in rankings if r.get("leg") == "short"]
    directional: dict[str, float] = {}
    if longs:
        wl = name_weight or 1.0 / len(longs)
        for r in longs:
            directional[r["ticker"]] = directional.get(r["ticker"], 0.0) + float(r.get("target_weight", wl))
    if shorts:
        ws = name_weight or 1.0 / len(shorts)
        for r in shorts:
            directional[r["ticker"]] = directional.get(r["ticker"], 0.0) - float(r.get("target_weight", ws))
    return NormalizedSleeve(
        sleeve=sig.get("sleeve", "s2_macro"),
        strategy=sig.get("model_version", "ranking"),
        directional=directional,
        suggested_hedge=[],
        market_neutral=bool(sig.get("market_neutral", True)),
        gross=sum(abs(w) for w in directional.values()),
        is_production=bool(sig.get("is_production", False)),
    )


def normalize(sig: dict) -> NormalizedSleeve:
    """Normalize a sleeve input (position-form `sleeve_signal` or ranking-form `aggregated_signal`)."""
    if "positions" in sig:
        return _from_position_form(sig)
    if "rankings" in sig:
        return _from_ranking_form(sig)
    raise ValueError("sleeve input has neither 'positions' (sleeve_signal) nor 'rankings' (aggregated_signal)")
