"""Adapter interfaces + shared result types for the block seams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class IntegrityStatus:
    """Result of the backend data-integrity gate (cycle step 3). status=HALT blocks trading."""
    status: str            # "OK" | "HALT"
    as_of: str
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "OK"


@dataclass
class ExecutionResult:
    """What the execution block reports back after reconciling the target book (cycle step 6)."""
    orders: list[dict] = field(default_factory=list)        # order_request contract objects
    reports: list[dict] = field(default_factory=list)       # execution_report contract objects
    # Book after fills: list of {ticker, lots, avg_price, last_price, is_hedge, sleeve_contributions}.
    positions: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)      # [{ticker, reason}]


class BackendAdapter(Protocol):
    def run_ingest(self, as_of: str) -> dict:
        """Cycle step 1: pull today's candles + market context. Returns a status dict."""
        ...

    def integrity_gate(self, as_of: str) -> IntegrityStatus:
        """Cycle step 3: freshness / gaps / NaN gate. HALT -> the agent must not trade."""
        ...

    def latest_prices(self, universe: list[str], as_of: str) -> dict[str, float]:
        """Last close per instrument (names + hedge indices) for sizing and marks."""
        ...


class SleeveAdapter(Protocol):
    def build_sleeve(self, as_of: str) -> dict:
        """Cycle step 4: the H9 dividend run-up sleeve_signal for `as_of` (past-only)."""
        ...


class CombinerAdapter(Protocol):
    def combine(self, sleeve_signals: list[dict], as_of: str,
                *, sleeve_status: dict[str, dict] | None = None) -> dict:
        """Cycle step 5: net sleeves -> risk_book (vol-target x regime gate x limits x hedge x
        shadow gate). `sleeve_status` (per-sleeve, from the agent's LIVE forward-P&L attribution)
        closes invariant #9: a production sleeve with negative forward P&L is demoted to shadow."""
        ...


class ExecutionAdapter(Protocol):
    def reconcile_and_execute(self, *, risk_book: dict, positions: list[dict],
                              prices: dict[str, float], capital: float, mode: str,
                              trade_date: str, phase: str) -> ExecutionResult:
        """Cycle step 6: target book vs current positions -> limit orders -> fills (paper)."""
        ...
