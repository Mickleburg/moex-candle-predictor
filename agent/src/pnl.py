"""Per-SLEEVE P&L attribution.

The V3 monitoring invariant: P&L is attributed BY SLEEVE so a sleeve that stops earning on
forward data is visible and its book weight can be zeroed (risk_manager owns the zeroing;
the agent surfaces the attribution). With one live sleeve (s3_event dividend run-up) plus its
book-level hedge, this attributes each name's mark to the sleeve(s) that put it there, and
distributes the hedge legs' marks across sleeves by their directional gross share.

Marks here are book mark-to-market (lots * (last_price − avg_price)). REALIZED per-event
run-up accrual (the shadow gate that lifts is_production) is owned by the ML chat's
h9_shadow_pnl; the agent writes a shadow-log line each cycle for that consumer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def attribute_book_pnl(positions: list[dict]) -> dict[str, dict[str, float]]:
    """Attribute current-book unrealized P&L + gross to sleeves.

    `positions` rows carry lots, avg_price, last_price, is_hedge, sleeve_contributions.
    Returns {sleeve: {"unrealized": float, "gross": float}}.
    """
    by_sleeve: dict[str, dict[str, float]] = {}
    hedge_unrealized = 0.0
    sleeve_dir_gross: dict[str, float] = {}

    def _bump(sleeve: str, unreal: float, gross: float) -> None:
        d = by_sleeve.setdefault(sleeve, {"unrealized": 0.0, "gross": 0.0})
        d["unrealized"] += unreal
        d["gross"] += gross

    for p in positions:
        lots = int(p.get("lots", 0))
        last = p.get("last_price")
        avg = float(p.get("avg_price", 0.0))
        if last is None:
            last = avg
        unreal = lots * (float(last) - avg)
        gross = abs(lots * float(last))
        contributions = p.get("sleeve_contributions") or {}
        if isinstance(contributions, str):
            contributions = json.loads(contributions)

        if p.get("is_hedge") or not contributions:
            if p.get("is_hedge"):
                hedge_unrealized += unreal
                continue
            _bump("unattributed", unreal, gross)
            continue

        total = sum(abs(float(w)) for w in contributions.values()) or 1.0
        for sleeve, w in contributions.items():
            share = abs(float(w)) / total
            _bump(sleeve, unreal * share, gross * share)
            sleeve_dir_gross[sleeve] = sleeve_dir_gross.get(sleeve, 0.0) + gross * share

    # distribute hedge marks across sleeves by directional gross share
    total_dir = sum(sleeve_dir_gross.values())
    if hedge_unrealized and total_dir > 0:
        for sleeve, g in sleeve_dir_gross.items():
            by_sleeve[sleeve]["unrealized"] += hedge_unrealized * (g / total_dir)
    elif hedge_unrealized:
        _bump("hedge", hedge_unrealized, 0.0)

    return by_sleeve


def append_shadow_log(path: Path | str, *, trade_date: str, as_of: str, risk_book: dict,
                      positions: list[dict], sleeve_pnl: dict[str, dict[str, float]]) -> None:
    """Append one no-lookahead shadow-log line for the realized-P&L gate consumer (ML chat)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trade_date": trade_date,
        "as_of": as_of,
        "sleeves": risk_book.get("sleeves", []),
        "regime": {
            "exposure_scalar": risk_book.get("risk_scalars", {}).get("exposure_scalar"),
            "regime_novel": risk_book.get("risk_scalars", {}).get("regime_novel"),
        },
        "book": [{"ticker": p["ticker"], "lots": p["lots"], "last_price": p.get("last_price"),
                  "is_hedge": bool(p.get("is_hedge"))} for p in positions],
        "sleeve_pnl": sleeve_pnl,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
