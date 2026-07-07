"""Instrument metadata accessor -- FIGI / round-lot / price-step for execution & agent.

Loads the shared ``config/instruments.json`` (built by
``scripts/build_instrument_metadata.py``) and exposes typed lookups so execution and the
orchestrator stop relying on placeholder defaults. Import surface is intentionally small
and stable:

    from backend.instruments import (
        get_instrument, figi_for, lot_for, round_to_lot, round_price, all_verified,
    )

* ``figi_for(ticker)``   -- T-Invest FIGI (raises if unknown). Check ``all_verified()``
  before live: curated FIGIs must be validated against a T-Invest dump first.
* ``lot_for(ticker)``    -- exchange round-lot (shares per lot).
* ``round_to_lot(ticker, qty)`` -- floor a share quantity to a whole number of lots.
* ``round_price(ticker, price)`` -- snap a price to the instrument's MINSTEP grid.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PATH = _REPO_ROOT / "config" / "instruments.json"


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found -- run scripts/build_instrument_metadata.py to generate it")
    return json.loads(p.read_text(encoding="utf-8"))


def load_instruments(path: Path | str = _DEFAULT_PATH) -> dict[str, dict]:
    """Return ``{ticker: metadata}`` for the whole universe."""
    return _load(str(path))["instruments"]


def get_instrument(ticker: str, path: Path | str = _DEFAULT_PATH) -> dict:
    """Full metadata dict for one ticker (raises KeyError if unknown)."""
    insts = load_instruments(path)
    tk = ticker.upper()
    if tk not in insts:
        raise KeyError(f"unknown instrument {ticker!r} (not in {path})")
    return insts[tk]


def figi_for(ticker: str, path: Path | str = _DEFAULT_PATH) -> str:
    figi = get_instrument(ticker, path).get("figi")
    if not figi:
        raise ValueError(f"no FIGI for {ticker!r}")
    return figi


def lot_for(ticker: str, path: Path | str = _DEFAULT_PATH) -> int:
    return int(get_instrument(ticker, path)["lot"])


def price_step_for(ticker: str, path: Path | str = _DEFAULT_PATH) -> float:
    return float(get_instrument(ticker, path)["min_price_step"])


def round_to_lot(ticker: str, quantity: float, path: Path | str = _DEFAULT_PATH) -> int:
    """Floor ``quantity`` shares to a whole number of lots (never over-orders)."""
    lot = lot_for(ticker, path)
    # floor the MAGNITUDE toward zero so a signed target never over-orders either leg — plain
    # ``quantity // lot`` floors toward -inf, which OVER-orders a negative (short/hedge) target.
    n_lots = int(abs(quantity) // lot)
    if quantity < 0:
        n_lots = -n_lots
    return n_lots * lot


def round_price(ticker: str, price: float, path: Path | str = _DEFAULT_PATH) -> float:
    """Snap ``price`` to the instrument's MINSTEP grid (and its decimal precision)."""
    inst = get_instrument(ticker, path)
    step = float(inst["min_price_step"])
    decimals = int(inst.get("decimals", 2))
    if step <= 0:
        return round(price, decimals)
    return round(round(price / step) * step, decimals)


def all_verified(path: Path | str = _DEFAULT_PATH) -> bool:
    """True only when the universe is non-empty AND every FIGI is validated (live gate).

    Recomputes from the per-name ``figi_verified`` truth instead of trusting the cached top-level
    ``all_figis_verified`` flag: the two can drift (a hand-edit, a partial/expanded build, or an
    empty instrument map where ``all([])`` is vacuously True) and this gates REAL money — so the
    money gate must agree with ``unverified_figis()``, never diverge from it.
    """
    insts = load_instruments(path)
    return bool(insts) and not unverified_figis(path)


def unverified_figis(path: Path | str = _DEFAULT_PATH) -> list[str]:
    """Tickers whose FIGI is still curated/unverified (must clear before live trading)."""
    return [tk for tk, v in load_instruments(path).items() if not v.get("figi_verified")]
