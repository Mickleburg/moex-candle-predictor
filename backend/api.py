"""Stable in-process entry point for the orchestrator (agent calls backend live, no HTTP).

This module is the FROZEN contract surface between the backend/data block and the agent.
The orchestrator imports ONLY from here; internal modules (ingest/integrity/store/...) may
refactor freely as long as these signatures and the documented return-dict keys hold. A
contract test (backend/tests/test_api_contract.py) locks the shape so it does not drift.

Daily-cycle usage (docs/VDS_AUTONOMOUS_PLAN.md)::

    from backend import api

    # EOD step 1 -- refresh the store (idempotent)
    ingest = api.run_ingest(with_futures=True)          # -> dict, ingest["status"] in {ok,error}

    # EOD step 3 -- gate BEFORE trading
    verdict = api.check_integrity()                      # -> dict, verdict["status"] in {OK,HALT}
    if not api.is_tradeable(verdict):
        halt(verdict["reasons"])                         # do not trade on rotten data

    # trading-day timing (RU-holiday aware)
    if api.is_trading_day(today):
        enter = api.add_trading_days(record_date, -12)

    # order sizing / routing metadata
    figi = api.figi_for("SBER"); qty = api.round_to_lot("SBER", raw_qty)
"""

from __future__ import annotations

from datetime import date
from typing import Optional

# -- ingest (EOD step 1) -----------------------------------------------------
from .ingest import run_ingest  # (today=None, fetch_fn=..., data_dir=..., backfill=False,
#                                  with_futures=False, instruments=None) -> report dict
#   report keys: status('ok'|'error'), reference_date, n_instruments, n_errors,
#                n_updated, results[]

# -- integrity (EOD step 3) --------------------------------------------------
from .integrity import run_checks as check_integrity  # (ref=None, data_dir=...,
#   cal=None, instruments=None, stale_tolerance_days=1, recent_window_td=60) -> verdict dict
#   verdict keys: status('OK'|'HALT'), reference_date, generated_at, n_checks,
#                 n_fail, n_warn, reasons[], warnings[], checks[]

# -- trading calendar (RU-holiday aware) -------------------------------------
from .trading_calendar import (
    get_calendar,
    is_trading_day,
    trading_days_between,
    next_trading_day,
    prev_trading_day,
    last_trading_day_on_or_before,
    add_trading_days,
)

# -- instrument metadata (execution / sizing) --------------------------------
from .instruments import (
    get_instrument,
    figi_for,
    lot_for,
    price_step_for,
    round_to_lot,
    round_price,
    all_verified,
    unverified_figis,
)


def is_tradeable(verdict: dict) -> bool:
    """True iff the integrity gate passed (orchestrator's go/no-go for the cycle)."""
    return verdict.get("status") == "OK"


__all__ = [
    # ingest
    "run_ingest",
    # integrity
    "check_integrity", "is_tradeable",
    # calendar
    "get_calendar", "is_trading_day", "trading_days_between", "next_trading_day",
    "prev_trading_day", "last_trading_day_on_or_before", "add_trading_days",
    # instruments
    "get_instrument", "figi_for", "lot_for", "price_step_for", "round_to_lot",
    "round_price", "all_verified", "unverified_figis",
]
