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
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _lot(lot_sizes: dict[str, int] | None, ticker: str) -> int:
    """Shares per round lot. Positions are stored in LOTS (reconcile sizes with
    `lots = trunc(shares / lot_size)`), so every rouble figure must scale by this or names with
    lot != 1 are silently understated — SNGS (lot 100) was reported at 1/100th of its real gross."""
    try:
        return int((lot_sizes or {}).get(str(ticker).upper(), 1) or 1)
    except (TypeError, ValueError):
        return 1


def attribute_book_pnl(positions: list[dict],
                       lot_sizes: dict[str, int] | None = None) -> dict[str, dict[str, float]]:
    """Attribute current-book unrealized P&L + gross to sleeves.

    `positions` rows carry lots, avg_price, last_price, is_hedge, sleeve_contributions.
    `lot_sizes` maps ticker -> shares per lot (omit only when every instrument is lot=1).
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
        shares = lots * _lot(lot_sizes, p.get("ticker", ""))
        unreal = shares * (float(last) - avg)
        gross = abs(shares * float(last))
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


def _track_of(client_order_id: str) -> str:
    """Track embedded in an execution client_order_id (exec-DATE-TRACK-TICKER-SIDE-QTY)."""
    parts = str(client_order_id).split("-")
    return parts[2] if len(parts) >= 6 and parts[0] == "exec" else "live"


def attribute_realized_pnl(prior_positions: list[dict], orders: list[dict], reports: list[dict],
                           lot_sizes: dict[str, int] | None = None) -> dict[str, float]:
    """Realized P&L per sleeve from fills that REDUCE or CLOSE a position held BEFORE this cycle.

    Why this exists: the book only ever carried UNREALIZED marks, and `_resulting_book` drops a
    position the moment it reaches zero lots. So when the July-2026 round trip closed on 07-20 its
    result simply vanished — `realized_pnl` stayed 0.0 for every row and no attribution row was
    written for the closing day at all. The paper track could not answer "did this make money"
    from its own books; the figure had to be reconstructed by hand from fills.

    `prior_positions` MUST be the book as it stood BEFORE this cycle's fills were applied (that is
    where the cost basis and the sleeve split live). Adds to an existing side realize nothing.
    Hedge legs are pooled and spread over the sleeves by their share of directional realized
    notional — the same convention `attribute_book_pnl` uses for hedge marks.
    """
    prior = {(str(p.get("track") or p.get("capital_state") or "live"), p["ticker"]): p
             for p in prior_positions}
    order_by_id = {o["client_order_id"]: o for o in orders}

    by_sleeve: dict[str, float] = {}
    hedge_realized = 0.0
    sleeve_notional: dict[str, float] = {}

    for rep in reports:
        if rep.get("status") != "FILLED":
            continue
        order = order_by_id.get(rep.get("client_order_id"))
        if order is None:                      # report for the other track's order
            continue
        pos = prior.get((_track_of(rep["client_order_id"]), rep["ticker"]))
        if not pos:
            continue
        held = int(pos.get("lots", 0))
        if held == 0:
            continue
        filled = int(rep.get("filled_quantity_lots") or 0)
        signed = filled if order.get("side") == "BUY" else -filled
        if signed == 0 or (held > 0) == (signed > 0):
            continue                            # opening or adding — nothing realized yet

        closed_lots = min(abs(signed), abs(held))
        shares = closed_lots * _lot(lot_sizes, rep["ticker"])
        price = rep.get("avg_fill_price")
        if price is None:
            continue
        avg = float(pos.get("avg_price", 0.0) or 0.0)
        direction = 1 if held > 0 else -1       # short legs profit when the exit price is LOWER
        realized = shares * (float(price) - avg) * direction
        notional = abs(shares * float(price))

        contributions = pos.get("sleeve_contributions") or {}
        if isinstance(contributions, str):
            contributions = json.loads(contributions)

        if pos.get("is_hedge"):
            hedge_realized += realized
            continue
        if not contributions:
            by_sleeve["unattributed"] = by_sleeve.get("unattributed", 0.0) + realized
            continue
        total = sum(abs(float(w)) for w in contributions.values()) or 1.0
        for sleeve, w in contributions.items():
            share = abs(float(w)) / total
            by_sleeve[sleeve] = by_sleeve.get(sleeve, 0.0) + realized * share
            sleeve_notional[sleeve] = sleeve_notional.get(sleeve, 0.0) + notional * share

    total_notional = sum(sleeve_notional.values())
    if hedge_realized and total_notional > 0:
        for sleeve, n in sleeve_notional.items():
            by_sleeve[sleeve] += hedge_realized * (n / total_notional)
    elif hedge_realized:
        by_sleeve["hedge"] = by_sleeve.get("hedge", 0.0) + hedge_realized
    return by_sleeve


def append_shadow_log(path: Path | str, *, trade_date: str, as_of: str, risk_book: dict,
                      positions: list[dict], sleeve_pnl: dict[str, dict[str, float]]) -> None:
    """Write one no-lookahead shadow-log line for the realized-P&L gate consumer (ML chat).

    IDEMPOTENT by trade_date: if a line for this trade_date already exists it is REPLACED in
    place (not duplicated). A cycle that fails on validate/alert has its 'failed' slot reclaimed
    by begin_cycle and the EOD body re-runs, so a plain append would emit a second line for the
    same day — which import_forward_snapshot._check_shadow_track rejects as a duplicate trade_date,
    blocking the snapshot import. Matches the order-ledger's order_exists dedup discipline.
    """
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
    line = json.dumps(record, ensure_ascii=False)

    # Read existing lines, replace the one for this trade_date in place (preserving chronological
    # position so as_of stays monotonic), else append. Rewrite atomically via a temp file + replace.
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out_lines: list[str] = []
    replaced = False
    for raw in existing:
        if not raw.strip():
            continue
        try:
            if json.loads(raw).get("trade_date") == trade_date:
                out_lines.append(line)
                replaced = True
                continue
        except ValueError:
            pass  # keep malformed lines untouched rather than dropping evidence
        out_lines.append(raw)
    if not replaced:
        out_lines.append(line)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)
