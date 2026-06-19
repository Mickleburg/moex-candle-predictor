"""Execution configuration: modes, lot sizes, sanity limits, live-gating.

Three escalating modes (`Mode`): DRY_RUN (print, send nothing) -> PAPER (sandbox / internal
simulator) -> LIVE. Live is refused unless ALL of: mode==LIVE, `allow_live=True`, and the
environment variable EXECUTION_ALLOW_LIVE=="1" (see brokers.make_broker). is_production stays
false on every artifact until the forward-shadow gate + team sign-off.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

EXEC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXEC_ROOT.parent

# Env var that must equal "1" to even consider live trading (belt-and-suspenders over allow_live).
LIVE_ENV_FLAG = "EXECUTION_ALLOW_LIVE"


class Mode(str, Enum):
    """How seriously orders are taken. String-valued so it round-trips through JSON/CLI."""

    DRY_RUN = "dry-run"   # reconcile + print delta orders, never contact a broker
    PAPER = "paper"       # execute against a sandbox / internal simulator (no real money)
    LIVE = "live"         # real orders — gated, off by default


# MOEX round-lot sizes (shares per lot) for the H9 universe + the sector/market hedge instruments.
# These are sensible defaults to be CONFIRMED/overridden from backend instrument metadata (MOEX ISS
# securities table) once the backend block serves it — execution must not be the source of truth for
# instrument reference data. Index instruments (MOEX*) are not directly lot-traded; in live the hedge
# is worked via index futures/ETF — lot=1 here is a paper placeholder (see README).
DEFAULT_LOT_SIZES: dict[str, int] = {
    "SBER": 10, "GAZP": 10, "LKOH": 1, "GMKN": 1, "ROSN": 1, "NVTK": 1,
    "TATN": 1, "MGNT": 1, "MTSS": 10, "SNGS": 100, "CHMF": 1, "ALRS": 10,
    # sector / market hedge proxies (paper placeholders, lot=1)
    "MOEXOG": 1, "MOEXFN": 1, "MOEXMM": 1, "MOEXCN": 1, "MOEXTL": 1, "IMOEX": 1,
}

DEFAULT_LOT = 1  # fallback lot size for any ticker missing from the map


@dataclass(frozen=True)
class SanityLimits:
    """Hard caps applied per name AFTER lot rounding, independent of what the book asks for.

    Defaults sit below the sleeve's per-name capacity (~130-190 M RUB whole-book; see
    ml/scripts/h9_capacity.py) so a malformed book can never blow through liquidity.
    """

    max_lots_per_name: int = 5_000_000
    max_notional_per_name: float = 60_000_000.0  # RUB


@dataclass
class ExecutionConfig:
    """Everything the engine needs that is not part of a single risk_book."""

    mode: Mode = Mode.DRY_RUN
    broker_backend: str = "sim"            # "sim" (internal paper) | "tinvest" (sandbox/live)
    capital: float = 100_000_000.0         # book NAV in RUB; weight*capital = target notional
    lot_sizes: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_LOT_SIZES))
    limits: SanityLimits = field(default_factory=SanityLimits)
    entry_offset: int = 12                 # H9: enter ~this many trading days before the anchor
    exit_offset: int = 2                   # H9: exit ~this many trading days before the anchor
    allow_live: bool = False               # must be True (and env flag set) to permit LIVE
    audit_dir: Path = EXEC_ROOT / "var" / "audit"
    state_dir: Path = EXEC_ROOT / "var" / "state"
    is_production: bool = False            # propagated onto every artifact; flips only after sign-off

    def lot_size(self, ticker: str) -> int:
        return int(self.lot_sizes.get(ticker, DEFAULT_LOT))

    def live_enabled(self) -> bool:
        """Live is permitted only with the explicit in-config flag AND the runtime env flag."""
        return self.allow_live and os.environ.get(LIVE_ENV_FLAG) == "1"
