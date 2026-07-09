"""Instrument reference data (lot sizes + FIGI) — backend-first, with a safe fallback.

Execution is NOT the source of truth for instrument metadata; the backend/data block owns it and
serves it from ``backend.instruments`` (built into ``config/instruments.json`` by
``scripts/build_instrument_metadata.py``). This module adapts that API to the small maps execution
needs and falls back to in-repo defaults when backend (or its generated JSON) is unavailable, so the
block keeps working in an isolated environment.

Backend surface consumed (see ``backend/instruments.py``):
``load_instruments() -> {ticker: {lot, figi, min_price_step, decimals, figi_verified, ...}}``,
``all_verified()`` (live gate: every FIGI validated against a T-Invest dump).
"""

from __future__ import annotations

from .config import DEFAULT_LOT_SIZES


def _backend_instruments() -> dict[str, dict] | None:
    """``{ticker: metadata}`` from backend, or None if backend/JSON is unavailable."""
    try:
        from backend.instruments import load_instruments  # type: ignore
        return load_instruments()
    except Exception:
        return None


def load_lot_sizes() -> dict[str, int]:
    """MOEX round-lot sizes (shares per lot), backend-first overlaid on the defaults.

    Backend values override the defaults; the defaults fill any gaps (e.g. the index hedge proxies
    the backend universe may not list).
    """
    lots = dict(DEFAULT_LOT_SIZES)
    insts = _backend_instruments()
    if insts:
        for ticker, meta in insts.items():
            if meta.get("lot"):
                lots[str(ticker).upper()] = int(meta["lot"])
    return lots


def load_figi_map() -> dict[str, str]:
    """ticker -> FIGI for the live T-Invest path, backend-first; empty if unavailable.

    The live adapter refuses to trade a ticker without a FIGI rather than guess, so an empty map
    simply means live must be handed the mapping explicitly until backend serves it.
    """
    insts = _backend_instruments()
    if not insts:
        return {}
    return {str(t).upper(): str(m["figi"]) for t, m in insts.items() if m.get("figi")}


def figis_all_verified() -> bool:
    """True only when backend reports every FIGI verified against a T-Invest dump.

    Used as an EXTRA live gate (live must not place real orders against curated/unverified FIGIs).
    Unknown/unavailable -> False (fail closed for live).
    """
    try:
        from backend.instruments import all_verified  # type: ignore
        return bool(all_verified())
    except Exception:
        return False
