"""Reconciliation: target book (risk_book) vs current positions -> delta LIMIT orders.

The risk_manager's `risk_book` is split by a SHADOW GATE (invariants #9/#4): proven LIVE sleeves sit in
`net_positions` (+ live `hedge`); gated-out, paper-only sleeves sit in `shadow_positions` (+
`shadow_hedge`) with ZERO live capital. Execution honours that split by MODE:

  * dry-run / paper -> paper-trade BOTH tracks so the forward-shadow track accrues through the real
    execution path — but the live and shadow books are reconciled SEPARATELY (never netted), and
    orders are tagged by track (the track is in the client_order_id). A shadow short must never
    collapse a live long on the same ticker, or the shadow P&L accrual is contaminated.
  * live            -> the live track (net_positions + live hedge) ONLY -> an all-shadow book places
    ZERO real orders.

Each weight becomes a target lot count at the given book capital and reference price, rounded DOWN to
MOEX round lots, per-name sanity-capped, and diffed against that TRACK's current holdings. Only
non-zero diffs become orders, priced at the reference close (so a paper replay reproduces the
close-to-close sleeve backtest).

Sizing: target_notional = weight * capital ; shares = target_notional / price ;
lots = trunc(shares / lot_size) (round toward zero = "round down" in lot magnitude).
This is pure/no-side-effect; protections, dedupe, and submission live in the engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from .config import ExecutionConfig, Mode


TRACKS = ("live", "shadow")


@dataclass
class Target:
    """One netted row of a SINGLE track's target book (names + hedge legs merged by instrument)."""

    instrument: str
    weight: float           # signed final book fraction (negative = short)
    is_hedge: bool
    sector: str | None = None
    sleeve_contributions: dict | None = None
    track: str = "live"     # "live" (net_positions) | "shadow" (shadow_positions) — NEVER netted across


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
    track: str = "live"
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


def track_targets(risk_book: dict, track: str) -> list[Target]:
    """One track's netted targets: its names + its hedge legs, merged by instrument.

    Merging within a track (2a) prevents a ticker that is BOTH a book-name AND a hedge-leg from being
    diffed twice against the same current position. Tracks are handled independently (2b): the shadow
    book is NEVER netted against the live book, so a shadow short cannot collapse a live long.
    """
    if track == "live":
        pos_rows, hedge = risk_book.get("net_positions", []), risk_book.get("hedge")
    else:
        pos_rows, hedge = risk_book.get("shadow_positions", []), risk_book.get("shadow_hedge")

    merged: dict[str, Target] = {}
    for p in pos_rows:
        inst = p["ticker"]
        tgt = merged.get(inst)
        if tgt is None:
            merged[inst] = Target(instrument=inst, weight=float(p["weight"]), is_hedge=False,
                                  sector=p.get("sector"),
                                  sleeve_contributions=dict(p.get("sleeve_contributions") or {}),
                                  track=track)
        else:
            tgt.weight += float(p["weight"])
            tgt.is_hedge = False   # a real position on this ticker (even if also a hedge leg)
            for k, v in (p.get("sleeve_contributions") or {}).items():
                tgt.sleeve_contributions[k] = (tgt.sleeve_contributions or {}).get(k, 0.0) + float(v)
    for leg in _hedge_legs(hedge):
        inst = leg["instrument"]
        tgt = merged.get(inst)
        if tgt is None:
            merged[inst] = Target(instrument=inst, weight=float(leg["weight"]), is_hedge=True,
                                  sector=inst, sleeve_contributions={}, track=track)
        else:
            tgt.weight += float(leg["weight"])   # 2a: name + hedge on same ticker -> ONE target
    return list(merged.values())


def book_targets(risk_book: dict, include_shadow: bool = False) -> list[Target]:
    """Targets across tracks kept SEPARATE: the live track always; the shadow track if include_shadow.

    Tracks are concatenated, NOT merged — a ticker present in both yields two Targets (one per track),
    so the live and shadow books reconcile independently.
    """
    targets = track_targets(risk_book, "live")
    if include_shadow:
        targets += track_targets(risk_book, "shadow")
    return targets


def _normalize_current(current_lots: dict | None) -> dict[str, dict[str, int]]:
    """Normalize current positions to {track: {ticker: lots}}.

    Accepts either a per-track nested dict, or a flat {ticker: lots} (treated as the live track,
    e.g. the paper simulator's positions()).
    """
    base: dict[str, dict[str, int]] = {"live": {}, "shadow": {}}
    if not current_lots:
        return base
    if all(isinstance(v, dict) for v in current_lots.values()):
        for t in TRACKS:
            base[t] = {k: int(v) for k, v in (current_lots.get(t) or {}).items()}
        return base
    base["live"] = {k: int(v) for k, v in current_lots.items()}
    return base


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
    current_lots: dict | None = None,
    config: ExecutionConfig | None = None,
) -> ReconcileResult:
    """Diff the target book against current holdings into delta LIMIT orders, PER TRACK.

    The live track (``net_positions`` + ``hedge``) and, in dry-run/paper, the shadow track
    (``shadow_positions`` + ``shadow_hedge``) are reconciled INDEPENDENTLY against their own current
    holdings — never netted against each other (2b). ``current_lots`` may be a per-track
    ``{track: {ticker: lots}}`` map or a flat ``{ticker: lots}`` (= live track).

    ``prices`` must map every instrument in the book — names AND hedge proxies (MOEXFN…) — to a
    reference close. A missing/non-positive price is reported in ``skipped`` (so a held-but-unpriced
    leg can't silently become a stuck position, 2c), never guessed.
    """
    config = config or ExecutionConfig()
    current_by_track = _normalize_current(current_lots)
    as_of = str(risk_book.get("as_of", ""))
    tag = _as_of_tag(as_of)
    result = ReconcileResult(as_of=as_of)

    # Live trades only proven (net) sleeves; dry-run/paper also paper-trade the shadow book.
    include_shadow = config.mode is not Mode.LIVE
    tracks = TRACKS if include_shadow else ("live",)

    for track in tracks:
        track_current = current_by_track.get(track, {})
        seen: set[str] = set()
        for tgt in track_targets(risk_book, track):
            seen.add(tgt.instrument)
            price = prices.get(tgt.instrument)
            if price is None or not (price > 0):
                result.skipped.append({"instrument": tgt.instrument, "track": track,
                                       "reason": "missing_or_nonpositive_price"})
                continue
            lot = config.lot_size(tgt.instrument)
            target_lots, binding = _capped_target_lots(tgt.weight, price, lot, config)
            cur = int(track_current.get(tgt.instrument, 0))
            order = _delta_order(tgt.instrument, target_lots, cur, price, tag, track=track,
                                 is_hedge=tgt.is_hedge, target_weight=tgt.weight, binding=binding)
            _place(result, tgt.instrument, order)

        # Flatten anything still held in THIS track that dropped out of its target book (target -> 0).
        # An EXIT (name/hedge removed from the book) sells the residual (shorts covered with a BUY).
        for inst, cur in track_current.items():
            if inst in seen or int(cur) == 0:
                continue
            price = prices.get(inst)
            if price is None or not (price > 0):
                # 2c: a held leg (incl. a hedge index) with no price can't be flattened -> surface it
                # loudly rather than leave a stuck position.
                result.skipped.append({"instrument": inst, "track": track,
                                       "reason": "missing_price_for_exit"})
                continue
            order = _delta_order(inst, 0, int(cur), price, tag, track=track,
                                 is_hedge=False, target_weight=0.0, binding=[])
            _place(result, inst, order)

    return result


def _delta_order(instrument: str, target_lots: int, cur: int, price: float, tag: str, *,
                 track: str, is_hedge: bool, target_weight: float, binding: list[str]) -> DeltaOrder | None:
    delta = target_lots - cur
    if delta == 0:
        return None
    side = "BUY" if delta > 0 else "SELL"
    qty = abs(delta)
    coid = f"exec-{tag}-{track}-{instrument}-{side}-{qty}"   # track in the id -> live/shadow never collide
    return DeltaOrder(
        instrument=instrument, side=side, quantity_lots=qty,
        limit_price=float(price), client_order_id=coid,
        target_lots=target_lots, current_lots=cur, is_hedge=is_hedge,
        target_weight=target_weight, track=track, binding=binding,
    )


def _place(result: ReconcileResult, instrument: str, order: DeltaOrder | None) -> None:
    if order is None:
        result.noops.append(instrument)
    else:
        result.orders.append(order)
