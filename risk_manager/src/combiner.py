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
from dataclasses import dataclass, field
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
    # Shadow gate (invariants #9 + #4): a sleeve gets LIVE capital only if it passed its gate.
    require_production_for_live: bool = True   # is_production=false sleeve -> shadow (0 live capital)
    # Correlation cap by sleeve-id (future readiness, S1/S2/S4): cap any one sleeve's directional
    # gross so a single edge-source can't dominate the book. None = off (one sleeve today).
    max_sleeve_gross: float | None = None


@dataclass
class RiskBook:
    as_of: str
    timeframe: str
    sleeves: list[dict]
    net_positions: list[dict]                 # LIVE book (real capital) — execution sizes these
    hedge: dict                               # LIVE hedge
    risk_scalars: dict
    limits: dict
    model_version: str
    gating: list[dict] = field(default_factory=list)            # per-sleeve live/shadow decision
    shadow_positions: list[dict] = field(default_factory=list)  # gated-out sleeves' intended book (0 live)
    shadow_hedge: dict = field(default_factory=lambda: {"mode": "none", "legs": []})
    is_production: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "timeframe": self.timeframe,
            "sleeves": self.sleeves,
            "gating": self.gating,
            "net_positions": self.net_positions,
            "hedge": self.hedge,
            "shadow_positions": self.shadow_positions,
            "shadow_hedge": self.shadow_hedge,
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


def _shadow_gate(sleeve: NormalizedSleeve, status: dict | None,
                 config: CombinerConfig) -> tuple[str, str, str]:
    """Decide a sleeve's capital state (invariants #9 + #4). Returns (state, gate, reason).

    A sleeve gets LIVE capital only if it passed its OWN gate: (1) the block signed it off
    (`is_production=true`) AND (2) its forward shadow gate is MET — no negative forward-P&L
    attribution. `status` (per-sleeve, from the agent state-store) may carry an explicit
    {"gate": "MET"|"NOT_MET"} or a {"forward_pnl": float}. Without sign-off OR with a NOT-MET
    forward gate the sleeve is SHADOW-only (0 live capital), tracked for attribution but not risked.
    """
    if not config.require_production_for_live:
        return ("live", "DISABLED", "gating disabled")
    if not sleeve.is_production:
        return ("shadow", "NOT_MET", "is_production=false")
    if status:
        if str(status.get("gate", "")).upper() == "NOT_MET":
            return ("shadow", "NOT_MET", str(status.get("reason", "forward gate not met")))
        fpnl = status.get("forward_pnl")
        if fpnl is not None and float(fpnl) < 0:
            return ("shadow", "NOT_MET", f"forward_pnl={fpnl}<0")
    return ("live", "MET", "production + forward gate met")


def shadow_gate_status(sleeve_signal: dict, status: dict | None = None,
                       config: CombinerConfig | None = None) -> dict:
    """Public helper: classify a single sleeve_signal as live/shadow (testable, reusable)."""
    sl = normalize(sleeve_signal)
    state, gate, reason = _shadow_gate(sl, status, config or CombinerConfig())
    return {"sleeve": sl.sleeve, "strategy": sl.strategy, "capital_state": state,
            "is_production": sl.is_production, "gate": gate, "reason": reason}


def _net_sleeves(sleeves: list[NormalizedSleeve], config: CombinerConfig,
                 binding: list[str]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Net directional weights by ticker across sleeves, with an optional per-sleeve gross cap
    (the correlation cap by sleeve-id: no single edge-source may dominate)."""
    net: dict[str, float] = {}
    contrib: dict[str, dict[str, float]] = {}
    for sl in sleeves:
        dirw = sl.directional
        if config.max_sleeve_gross is not None:
            g = sum(abs(w) for w in dirw.values())
            if g > config.max_sleeve_gross + _EPS:
                binding.append(f"sleeve_cap:{sl.sleeve}")
                s = config.max_sleeve_gross / g
                dirw = {t: w * s for t, w in dirw.items()}
        for t, w in dirw.items():
            net[t] = net.get(t, 0.0) + w
            contrib.setdefault(t, {})[sl.sleeve] = contrib.get(t, {}).get(sl.sleeve, 0.0) + w
    return {t: w for t, w in net.items() if abs(w) > _EPS}, contrib


def _size_and_hedge(net: dict[str, float], contrib: dict[str, dict[str, float]],
                    risk_analytics: dict | None, config: CombinerConfig, binding: list[str],
                    exposure_scalar: float) -> tuple[list[dict], dict, dict]:
    """vol-target (H4) x regime gate (H5) x limits, then build the hedge. Returns (positions, hedge, meta)."""
    shaped = _apply_sector_cap(_apply_name_cap(net, config.max_name_weight, []),
                               config.max_sector_gross, [])
    vols = _vol_map(risk_analytics)
    book_vol_bar = _book_vol_per_bar(shaped, vols)
    target_bar = config.target_book_vol_annual / math.sqrt(config.bars_per_year)
    vol_scalar = min(target_bar / book_vol_bar, config.max_vol_leverage) if book_vol_bar > _EPS else 1.0
    if vol_scalar < 1.0 - _EPS or vol_scalar > 1.0 + _EPS:
        binding.append("vol_target")
    if exposure_scalar < 1.0 - _EPS and shaped:
        binding.append("regime_gate")

    scaled = {t: w * vol_scalar * exposure_scalar for t, w in shaped.items()}
    final = {t: w for t, w in _enforce_limits(scaled, config, binding).items() if abs(w) > _EPS}
    directional_gross = sum(abs(w) for w in final.values())

    hedge = _build_hedge(final, config.hedge_mode)
    hedge_gross = sum(abs(leg["weight"]) for leg in hedge["legs"])

    sec_gross: dict[str, float] = {}
    for t, w in final.items():
        sec_gross[sector_of(t)] = sec_gross.get(sector_of(t), 0.0) + abs(w)
    positions = [
        {"ticker": t, "weight": round(w, 6), "side": "LONG" if w > 0 else "SHORT",
         "sector": sector_of(t),
         "sleeve_contributions": {k: round(v, 6) for k, v in contrib.get(t, {}).items()}}
        for t, w in sorted(final.items(), key=lambda kv: -abs(kv[1]))
    ]
    meta = {
        "vol_scalar": vol_scalar,
        "book_vol_estimate_annual": round(book_vol_bar * math.sqrt(config.bars_per_year), 6),
        "directional_gross": directional_gross,
        "hedge_gross": hedge_gross,
        "name_caps_ok": all(abs(w) <= config.max_name_weight + 1e-6 for w in final.values()),
        "sector_caps_ok": all(g <= config.max_sector_gross + 1e-6 for g in sec_gross.values()),
        "gross_cap_ok": directional_gross <= config.max_gross + 1e-6,
    }
    return positions, hedge, meta


def combine(sleeve_signals: list[dict], risk_analytics: dict | None = None,
            config: CombinerConfig | None = None, as_of: str | None = None,
            *, sleeve_status: dict[str, dict] | None = None) -> RiskBook:
    """Net sleeve inputs into one book and apply the risk layer + shadow gate. Returns a RiskBook.

    Frozen entry point (the agent calls this in-process via LiveCombiner):
        combine(sleeve_signals: list[dict], risk_analytics: dict|None, config: CombinerConfig|None,
                as_of: str|None, *, sleeve_status: dict|None) -> RiskBook

    Shadow gate (invariants #9/#4): each sleeve is classified live vs shadow. LIVE sleeves are netted
    into `net_positions` (real capital, execution sizes these). SHADOW sleeves (e.g. H9 while
    is_production=false / forward gate NOT MET) go to `shadow_positions` with ZERO live capital — so
    execution, reading net_positions, places no real orders for an unproven edge. `sleeve_status` (per
    sleeve, from the agent's P&L attribution) can force a production sleeve back to shadow on a
    negative forward gate.
    """
    config = config or CombinerConfig()
    sleeves = [normalize(s) for s in sleeve_signals]
    if as_of is None:
        as_of = next((s.get("as_of") for s in sleeve_signals if s.get("as_of")), "")

    regime = (risk_analytics or {}).get("regime", {})
    exposure_scalar = float(regime.get("exposure_scalar", 1.0))
    regime_novel = bool(regime.get("novel", False))

    # gate each sleeve -> live / shadow
    gating: list[dict] = []
    live_sleeves: list[NormalizedSleeve] = []
    shadow_sleeves: list[NormalizedSleeve] = []
    for sl in sleeves:
        status = (sleeve_status or {}).get(sl.sleeve)
        state, gate, reason = _shadow_gate(sl, status, config)
        gating.append({"sleeve": sl.sleeve, "strategy": sl.strategy, "capital_state": state,
                       "is_production": sl.is_production, "gate": gate, "reason": reason})
        (live_sleeves if state == "live" else shadow_sleeves).append(sl)

    binding: list[str] = []
    net_l, contrib_l = _net_sleeves(live_sleeves, config, binding)
    net_positions, hedge, lm = _size_and_hedge(net_l, contrib_l, risk_analytics, config,
                                               binding, exposure_scalar)

    shadow_binding: list[str] = []
    net_s, contrib_s = _net_sleeves(shadow_sleeves, config, shadow_binding)
    shadow_positions, shadow_hedge, sm = _size_and_hedge(net_s, contrib_s, risk_analytics, config,
                                                         shadow_binding, exposure_scalar)

    risk_scalars = {
        "target_book_vol_annual": config.target_book_vol_annual,
        "book_vol_estimate_annual": lm["book_vol_estimate_annual"],
        "vol_scalar": round(lm["vol_scalar"], 6),
        "exposure_scalar": round(exposure_scalar, 6),
        "regime_novel": regime_novel,
        "directional_gross": round(lm["directional_gross"], 6),
        "total_gross": round(lm["directional_gross"] + lm["hedge_gross"], 6),
        "shadow_gross": round(sm["directional_gross"], 6),
        "shadow_total_gross": round(sm["directional_gross"] + sm["hedge_gross"], 6),
    }
    limits = {
        "max_name_weight": config.max_name_weight,
        "max_sector_gross": config.max_sector_gross,
        "max_gross": config.max_gross,
        "name_caps_ok": bool(lm["name_caps_ok"]),
        "sector_caps_ok": bool(lm["sector_caps_ok"]),
        "gross_cap_ok": bool(lm["gross_cap_ok"]),
        "binding": sorted(set(binding)),
    }
    # The LIVE book is production only if there IS a live sleeve and all live sleeves are production.
    is_production = bool(live_sleeves) and all(sl.is_production for sl in live_sleeves)

    return RiskBook(
        as_of=str(as_of),
        timeframe=config.timeframe,
        sleeves=[{"sleeve": sl.sleeve, "strategy": sl.strategy, "gross": round(sl.gross, 6)}
                 for sl in sleeves],
        gating=gating,
        net_positions=net_positions,
        hedge=hedge,
        shadow_positions=shadow_positions,
        shadow_hedge=shadow_hedge,
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
