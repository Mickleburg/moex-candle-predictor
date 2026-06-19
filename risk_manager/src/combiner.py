"""V3 sleeve combiner + portfolio risk layer.

Pipeline (one decision step):
  1. normalize each sleeve input (position-form sleeve_signal / ranking-form aggregated_signal),
  2. NET directional weights by ticker across sleeves (one ticker -> one net position),
  3. structural LIMITS: per-name cap, then per-sector gross cap (= correlation-cluster cap),
  4. RISK SCALARS: vol-targeting (H4 per-ticker vol forecast) x regime gate (H5 exposure_scalar),
  5. re-clip name cap + apply gross cap,
  6. build the book-level HEDGE (sector-preferred for the H9 run-up; market or none also supported),
  7. emit a `risk_book` (and, via to_risk_decisions, per-name `risk_decision` for execution).

Pure-Python (no numpy/pandas) so it is a light, dependency-free portfolio layer. All inputs are
plain dicts validated against contracts/. is_production stays false until a forward gate + sign-off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .sectors import MARKET_INDEX, sector_of
from .sleeves import NormalizedSleeve, normalize

_EPS = 1e-9


@dataclass
class CombinerConfig:
    """Risk-layer knobs. Defaults match the H9 sleeve (name cap 0.34) and a sector-hedged book."""
    target_book_vol_annual: float = 0.12      # vol-targeting objective (H4)
    bars_per_year: int = 247                  # daily MOEX bars/yr (matches h9_dividend_sleeve_sim)
    max_name_weight: float = 0.34             # per-name |weight| cap
    max_sector_gross: float = 0.60            # per-sector gross cap == correlation-cluster cap
    max_gross: float = 1.0                    # directional gross cap (names only, pre-hedge)
    max_vol_leverage: float = 1.5             # cap on the vol-targeting multiplier
    hedge_mode: str = "sector"                # "sector" (preferred) | "market" | "none"
    timeframe: str = "1D"
    model_version: str = "risk_combiner_v0"


@dataclass
class RiskBook:
    as_of: str
    timeframe: str
    sleeves: list[dict]
    net_positions: list[dict]
    hedge: dict
    risk_scalars: dict
    limits: dict
    model_version: str
    is_production: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "timeframe": self.timeframe,
            "sleeves": self.sleeves,
            "net_positions": self.net_positions,
            "hedge": self.hedge,
            "risk_scalars": self.risk_scalars,
            "limits": self.limits,
            "model_version": self.model_version,
            "is_production": self.is_production,
        }


def _vol_map(risk_analytics: dict | None) -> dict[str, float]:
    if not risk_analytics:
        return {}
    out: dict[str, float] = {}
    for e in risk_analytics.get("per_ticker", []):
        v = e.get("vol_forecast")
        if e.get("valid", True) and v and math.isfinite(v) and v > 0:
            out[e["ticker"]] = float(v)
    return out


def _book_vol_per_bar(weights: dict[str, float], vols: dict[str, float]) -> float:
    """Annualization-free book vol estimate: L2 norm of (w_i * vol_i).

    The book is sector-hedged, so the common factor is removed and the residual names are treated as
    weakly correlated (zero-correlation upper-bound on the diversified part). Missing per-ticker vols
    fall back to the median of the available forecasts.
    """
    if not weights:
        return 0.0
    avail = sorted(vols[t] for t in weights if t in vols)
    if avail:
        median = avail[len(avail) // 2]
    else:
        return 0.0
    ss = 0.0
    for t, w in weights.items():
        v = vols.get(t, median)
        ss += (w * v) ** 2
    return math.sqrt(ss)


def _apply_name_cap(weights: dict[str, float], cap: float, binding: list[str]) -> dict[str, float]:
    out = {}
    for t, w in weights.items():
        if abs(w) > cap + _EPS:
            binding.append(f"name_cap:{t}")
            out[t] = math.copysign(cap, w)
        else:
            out[t] = w
    return out


def _apply_sector_cap(weights: dict[str, float], cap: float, binding: list[str]) -> dict[str, float]:
    by_sector: dict[str, float] = {}
    for t, w in weights.items():
        by_sector[sector_of(t)] = by_sector.get(sector_of(t), 0.0) + abs(w)
    scale: dict[str, float] = {}
    for sec, g in by_sector.items():
        if g > cap + _EPS:
            binding.append(f"sector_cap:{sec}")
            scale[sec] = cap / g
    if not scale:
        return dict(weights)
    return {t: w * scale.get(sector_of(t), 1.0) for t, w in weights.items()}


def _apply_gross_cap(weights: dict[str, float], cap: float, binding: list[str]) -> dict[str, float]:
    g = sum(abs(w) for w in weights.values())
    if g > cap + _EPS:
        binding.append("gross_cap")
        s = cap / g
        return {t: w * s for t, w in weights.items()}
    return dict(weights)


def _enforce_limits(weights: dict[str, float], config: "CombinerConfig",
                    binding: list[str]) -> dict[str, float]:
    """Apply name -> sector -> gross caps to a fixpoint (each cap only shrinks, so it converges).

    Must run as the FINAL step: vol-targeting can scale weights UP, which would re-break caps that
    were enforced earlier. Iterating guarantees the emitted book respects every limit.
    """
    for _ in range(8):
        before = dict(weights)
        weights = _apply_name_cap(weights, config.max_name_weight, binding)
        weights = _apply_sector_cap(weights, config.max_sector_gross, binding)
        weights = _apply_gross_cap(weights, config.max_gross, binding)
        if all(abs(weights.get(t, 0.0) - w) <= _EPS for t, w in before.items()) \
                and len(weights) == len(before):
            break
    return weights


def _build_hedge(weights: dict[str, float], mode: str) -> dict:
    """Book-level hedge against net directional exposure (neutralizes sector/market beta)."""
    if mode == "none" or not weights:
        return {"mode": "none" if mode == "none" else mode, "legs": []}
    if mode == "market":
        net = sum(weights.values())
        legs = [{"instrument": MARKET_INDEX, "weight": round(-net, 6)}] if abs(net) > _EPS else []
        return {"mode": "market", "legs": legs}
    # sector mode (preferred): short each sector index by that sector's net directional weight
    by_sector: dict[str, float] = {}
    for t, w in weights.items():
        by_sector[sector_of(t)] = by_sector.get(sector_of(t), 0.0) + w
    legs = [{"instrument": sec, "weight": round(-net, 6)}
            for sec, net in sorted(by_sector.items()) if abs(net) > _EPS]
    return {"mode": "sector", "legs": legs}


def combine(sleeve_signals: list[dict], risk_analytics: dict | None = None,
            config: CombinerConfig | None = None, as_of: str | None = None) -> RiskBook:
    """Net sleeve inputs into one book and apply the risk layer. Returns a RiskBook."""
    config = config or CombinerConfig()
    sleeves = [normalize(s) for s in sleeve_signals]
    if as_of is None:
        as_of = next((s.get("as_of") for s in sleeve_signals if s.get("as_of")), "")
    binding: list[str] = []

    # 2. net directional weights by ticker; track per-sleeve contributions
    net: dict[str, float] = {}
    contrib: dict[str, dict[str, float]] = {}
    for sl in sleeves:
        for t, w in sl.directional.items():
            net[t] = net.get(t, 0.0) + w
            contrib.setdefault(t, {})[sl.sleeve] = contrib.get(t, {}).get(sl.sleeve, 0.0) + w
    net = {t: w for t, w in net.items() if abs(w) > _EPS}

    # 3. structural shape for the vol estimate (name -> sector), no binding record yet
    shaped = _apply_sector_cap(_apply_name_cap(net, config.max_name_weight, []),
                               config.max_sector_gross, [])

    # 4. risk scalars: vol-targeting (H4) x regime gate (H5)
    vols = _vol_map(risk_analytics)
    book_vol_bar = _book_vol_per_bar(shaped, vols)
    target_bar = config.target_book_vol_annual / math.sqrt(config.bars_per_year)
    if book_vol_bar > _EPS:
        vol_scalar = min(target_bar / book_vol_bar, config.max_vol_leverage)
    else:
        vol_scalar = 1.0
    regime = (risk_analytics or {}).get("regime", {})
    exposure_scalar = float(regime.get("exposure_scalar", 1.0))
    regime_novel = bool(regime.get("novel", False))
    if vol_scalar < 1.0 - _EPS or vol_scalar > 1.0 + _EPS:
        binding.append("vol_target")
    if exposure_scalar < 1.0 - _EPS:
        binding.append("regime_gate")

    book_scalar = vol_scalar * exposure_scalar
    scaled = {t: w * book_scalar for t, w in shaped.items()}

    # 5. enforce all limits as the FINAL step (vol-target can scale up and re-break caps)
    net = _enforce_limits(scaled, config, binding)
    net = {t: w for t, w in net.items() if abs(w) > _EPS}
    directional_gross = sum(abs(w) for w in net.values())

    # 6. book-level hedge (sector-preferred)
    hedge = _build_hedge(net, config.hedge_mode)
    hedge_gross = sum(abs(leg["weight"]) for leg in hedge["legs"])

    # audit
    name_caps_ok = all(abs(w) <= config.max_name_weight + 1e-6 for w in net.values())
    sec_gross: dict[str, float] = {}
    for t, w in net.items():
        sec_gross[sector_of(t)] = sec_gross.get(sector_of(t), 0.0) + abs(w)
    sector_caps_ok = all(g <= config.max_sector_gross + 1e-6 for g in sec_gross.values())
    gross_cap_ok = directional_gross <= config.max_gross + 1e-6

    net_positions = [
        {
            "ticker": t,
            "weight": round(w, 6),
            "side": "LONG" if w > 0 else "SHORT",
            "sector": sector_of(t),
            "sleeve_contributions": {k: round(v, 6) for k, v in contrib.get(t, {}).items()},
        }
        for t, w in sorted(net.items(), key=lambda kv: -abs(kv[1]))
    ]

    risk_scalars = {
        "target_book_vol_annual": config.target_book_vol_annual,
        "book_vol_estimate_annual": round(book_vol_bar * math.sqrt(config.bars_per_year), 6),
        "vol_scalar": round(vol_scalar, 6),
        "exposure_scalar": round(exposure_scalar, 6),
        "regime_novel": regime_novel,
        "directional_gross": round(directional_gross, 6),
        "total_gross": round(directional_gross + hedge_gross, 6),
    }
    limits = {
        "max_name_weight": config.max_name_weight,
        "max_sector_gross": config.max_sector_gross,
        "max_gross": config.max_gross,
        "name_caps_ok": bool(name_caps_ok),
        "sector_caps_ok": bool(sector_caps_ok),
        "gross_cap_ok": bool(gross_cap_ok),
        "binding": sorted(set(binding)),
    }
    is_production = bool(sleeves) and all(sl.is_production for sl in sleeves)

    return RiskBook(
        as_of=str(as_of),
        timeframe=config.timeframe,
        sleeves=[{"sleeve": sl.sleeve, "strategy": sl.strategy, "gross": round(sl.gross, 6)}
                 for sl in sleeves],
        net_positions=net_positions,
        hedge=hedge,
        risk_scalars=risk_scalars,
        limits=limits,
        model_version=config.model_version,
        is_production=is_production,
    )


def to_risk_decisions(book: RiskBook) -> list[dict]:
    """Render per-instrument `risk_decision` objects (valid vs risk_decision.schema.json).

    Weights are handed to execution, which sizes them to integer lots given capital/price — so
    order_intent is null here (the combiner does not know capital or price). Hedge legs are emitted
    as decisions too, so execution receives the complete book.
    """
    checks_ok = {
        "name_cap_ok": book.limits["name_caps_ok"],
        "sector_cap_ok": book.limits["sector_caps_ok"],
        "gross_cap_ok": book.limits["gross_cap_ok"],
        "regime_ok": not book.risk_scalars["regime_novel"],
    }
    decisions: list[dict] = []
    for p in book.net_positions:
        side = "BUY" if p["side"] == "LONG" else "SELL"
        decisions.append({
            "ticker": p["ticker"],
            "requested_action": side,
            "approved_action": side,
            "position_side": p["side"],
            "approved": True,
            "position_context": {
                "target_weight": p["weight"],
                "sector": p.get("sector"),
                "sleeve_contributions": p.get("sleeve_contributions", {}),
                "is_hedge": False,
            },
            "risk_checks": dict(checks_ok),
            "order_intent": None,
            "blocked_reason": None,
        })
    for leg in book.hedge["legs"]:
        side = "BUY" if leg["weight"] > 0 else "SELL"
        decisions.append({
            "ticker": leg["instrument"],
            "requested_action": side,
            "approved_action": side,
            "position_side": "LONG" if leg["weight"] > 0 else "SHORT",
            "approved": True,
            "position_context": {
                "target_weight": leg["weight"],
                "hedge_mode": book.hedge["mode"],
                "is_hedge": True,
            },
            "risk_checks": dict(checks_ok),
            "order_intent": None,
            "blocked_reason": None,
        })
    return decisions
