"""Reconciliation: target book (risk_book) vs current positions -> delta LIMIT orders.

The risk_manager's `risk_book` is split by a SHADOW GATE (invariants #9/#4): proven LIVE sleeves sit in
`net_positions` (+ live `hedge`); gated-out, paper-only sleeves sit in `shadow_positions` (+
`shadow_hedge`) with ZERO live capital. Execution honours that split by MODE:

  * dry-run / paper -> effective book = net_positions ∪ shadow_positions (+ hedge ∪ shadow_hedge).
    Paper-trades the shadow book so the forward-shadow track accrues through the real execution path.
  * live            -> net_positions + live hedge ONLY -> an all-shadow book places ZERO real orders.

Each weight becomes a target lot count at the given book capital and reference price, rounded DOWN to
MOEX round lots, per-name sanity-capped, and diffed against current holdings. Only non-zero diffs
become orders, priced at the reference close (so a paper replay reproduces the close-to-close sleeve
backtest).

Sizing: target_notional = weight * capital ; shares = target_notional / price ;
lots = trunc(shares / lot_size) (round toward zero = "round down" in lot magnitude).
This is pure/no-side-effect; protections, dedupe, and submission live in the engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from .config import ExecutionConfig, Mode


@dataclass
class Target:
    """One row of the target book after flattening + netting names + hedge legs."""

    instrument: str
    weight: float           # signed final book fraction (negative = short)
    is_hedge: bool
    sector: str | None = None
    sleeve_contributions: dict | None = None


@dataclass
class DeltaOrder:
    """An actionable delta plus the sizing provenance behind it."""

    instrument: str
    side: str               # BUY | SELL
    quantity_lots: int      # > 0
    limit_price: float
    client_order_id: str
    target_lots: int        # signed target after caps
    current_lots: int       # signed current
    is_hedge: bool
    target_weight: float
    binding: list[str] = field(default_factory=list)  # sanity caps that bound the size

    def to_order_request(self) -> dict:
        """A pure `order_request` contract dict (contracts/order_request.schema.json)."""
        return {
            "ticker": self.instrument,
            "side": self.side,
            "quantity_lots": int(self.quantity_lots),
            "order_type": "LIMIT",
            "limit_price": float(self.limit_price),
            "client_order_id": self.client_order_id,
        }


@dataclass
class ReconcileResult:
    as_of: str
    orders: list[DeltaOrder] = field(default_factory=list)
    noops: list[str] = field(default_factory=list)     # instruments already at target
    skipped: list[dict] = field(default_factory=list)  # instrument + reason (e.g. missing price)

    def order_requests(self) -> list[dict]:
        return [o.to_order_request() for o in self.orders]


def _hedge_legs(hedge: dict | None) -> list[dict]:
    """Legs of a hedge block, or [] when absent or mode='none'."""
    hedge = hedge or {}
    if hedge.get("mode") in (None, "none"):
        return []
    return list(hedge.get("legs", []))


def book_targets(risk_book: dict, include_shadow: bool = False) -> list[Target]:
    """Flatten + net the effective book into one signed target list.

    ``include_shadow`` (paper/dry-run) folds `shadow_positions` and `shadow_hedge` into the book so it
    is paper-traded; live (``include_shadow=False``) sees only `net_positions` + live `hedge`. Names
    and hedge legs are merged by instrument (weights summed) so a ticker that appears in both the live
    and shadow books nets to a single target order.
    """
    pos_rows = list(risk_book.get("net_positions", []))
    hedge_legs = _hedge_legs(risk_book.get("hedge"))
    if include_shadow:
        pos_rows += list(risk_book.get("shadow_positions", []))
        hedge_legs += _hedge_legs(risk_book.get("shadow_hedge"))

    # Merge names by ticker (sum weights; merge sleeve_contributions; keep first-seen sector/order).
    names: dict[str, Target] = {}
    for p in pos_rows:
        t = p["ticker"]
        tgt = names.get(t)
        if tgt is None:
            names[t] = Target(instrument=t, weight=float(p["weight"]), is_hedge=False,
                              sector=p.get("sector"),
                              sleeve_contributions=dict(p.get("sleeve_contributions") or {}))
        else:
            tgt.weight += float(p["weight"])
            for k, v in (p.get("sleeve_contributions") or {}).items():
                tgt.sleeve_contributions[k] = tgt.sleeve_contributions.get(k, 0.0) + float(v)

    # Merge hedge legs by instrument (sum weights).
    hedges: dict[str, Target] = {}
    for leg in hedge_legs:
        inst = leg["instrument"]
        tgt = hedges.get(inst)
        if tgt is None:
            hedges[inst] = Target(instrument=inst, weight=float(leg["weight"]), is_hedge=True,
                                  sector=inst, sleeve_contributions={})
        else:
            tgt.weight += float(leg["weight"])

    return list(names.values()) + list(hedges.values())


def _as_of_tag(as_of: str) -> str:
    """Compact YYYYMMDD tag for client_order_id from a possibly space/tz-bearing timestamp."""
    text = str(as_of).strip().replace(" ", "T")
    try:
        return datetime.fromisoformat(text).strftime("%Y%m%d")
    except ValueError:
        return "".join(ch for ch in text[:10] if ch.isdigit())


def _capped_target_lots(
    weight: float, price: float, lot_size: int, config: ExecutionConfig,
) -> tuple[int, list[str]]:
    """Signed target lots after lot-rounding (toward zero) and per-name sanity caps."""
    binding: list[str] = []
    raw_shares = weight * config.capital / price
    lots = math.trunc(raw_shares / lot_size)            # round DOWN in magnitude
    sign = -1 if lots < 0 else 1
    mag = abs(lots)
    # cap: max lots per name
    if mag > config.limits.max_lots_per_name:
        mag = config.limits.max_lots_per_name
        binding.append("max_lots_per_name")
    # cap: max notional per name
    max_lots_by_notional = math.trunc(config.limits.max_notional_per_name / (price * lot_size))
    if mag > max_lots_by_notional:
        mag = max_lots_by_notional
        binding.append("max_notional_per_name")
    return sign * mag, binding


def reconcile(
    risk_book: dict,
    prices: dict[str, float],
    current_lots: dict[str, int] | None = None,
    config: ExecutionConfig | None = None,
) -> ReconcileResult:
    """Diff the target book against current holdings into delta LIMIT orders.

    ``prices`` maps every instrument in the book (names + hedge proxies) to a reference close.
    A missing/non-positive price means we cannot size that leg -> it is reported in ``skipped``,
    never silently dropped or guessed.
    """
    config = config or ExecutionConfig()
    current_lots = dict(current_lots or {})
    as_of = str(risk_book.get("as_of", ""))
    tag = _as_of_tag(as_of)
    result = ReconcileResult(as_of=as_of)

    # Live trades only proven (net) sleeves; dry-run/paper also paper-trade the shadow book.
    include_shadow = config.mode is not Mode.LIVE
    seen: set[str] = set()
    for tgt in book_targets(risk_book, include_shadow):
        seen.add(tgt.instrument)
        price = prices.get(tgt.instrument)
        if price is None or not (price > 0):
            result.skipped.append({"instrument": tgt.instrument, "reason": "missing_or_nonpositive_price"})
            continue
        lot = config.lot_size(tgt.instrument)
        target_lots, binding = _capped_target_lots(tgt.weight, price, lot, config)
        cur = int(current_lots.get(tgt.instrument, 0))
        order = _delta_order(tgt.instrument, target_lots, cur, price, tag,
                             is_hedge=tgt.is_hedge, target_weight=tgt.weight, binding=binding)
        _place(result, tgt.instrument, order)

    # Flatten anything we still hold that has dropped out of the target book entirely (target -> 0).
    # This is how an EXIT happens: the sleeve removes the name from net_positions, so reconciliation
    # must sell the residual. Shorts (hedge legs) are covered with a BUY.
    for inst, cur in current_lots.items():
        if inst in seen or int(cur) == 0:
            continue
        price = prices.get(inst)
        if price is None or not (price > 0):
            result.skipped.append({"instrument": inst, "reason": "missing_price_for_exit"})
            continue
        order = _delta_order(inst, 0, int(cur), price, tag,
                             is_hedge=False, target_weight=0.0, binding=[])
        _place(result, inst, order)

    return result


def _delta_order(instrument: str, target_lots: int, cur: int, price: float, tag: str, *,
                 is_hedge: bool, target_weight: float, binding: list[str]) -> DeltaOrder | None:
    delta = target_lots - cur
    if delta == 0:
        return None
    side = "BUY" if delta > 0 else "SELL"
    qty = abs(delta)
    coid = f"exec-{tag}-{instrument}-{side}-{qty}"
    return DeltaOrder(
        instrument=instrument, side=side, quantity_lots=qty,
        limit_price=float(price), client_order_id=coid,
        target_lots=target_lots, current_lots=cur, is_hedge=is_hedge,
        target_weight=target_weight, binding=binding,
    )


def _place(result: ReconcileResult, instrument: str, order: DeltaOrder | None) -> None:
    if order is None:
        result.noops.append(instrument)
    else:
        result.orders.append(order)
