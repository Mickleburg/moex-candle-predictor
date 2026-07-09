"""Internal paper brokers: a no-op dry-run and a deterministic fill simulator.

PaperBroker fills every accepted LIMIT order fully at its limit price and books the signed lots.
That close-to-close assumption is exactly the one the H9 sleeve backtest makes
(`ml/scripts/h9_dividend_sleeve_sim.py` holds weight.shift(1) and earns close-to-close), so a paper
season replay reconciles against the sim. No money moves; positions/cash live in memory (or are
snapshotted by the engine's state store).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import BrokerAdapter, execution_report


class DryRunBroker(BrokerAdapter):
    """Sends nothing. Every order comes back DRY_RUN with zero fill."""

    name = "dry-run"

    def place_order(self, order: dict) -> dict:
        self._require_limit(order)
        return execution_report(order["client_order_id"], order["ticker"], "DRY_RUN",
                                message="dry-run: not sent")

    def cancel(self, client_order_id: str) -> dict:
        return execution_report(client_order_id, "", "CANCELED", message="dry-run: nothing to cancel")

    def cancel_all(self) -> list[dict]:
        return []


@dataclass
class PaperBroker(BrokerAdapter):
    """Deterministic simulator: LIMIT orders fill in full at the limit price."""

    name: str = "paper-sim"
    _positions: dict[str, int] = field(default_factory=dict)
    _cash: float = 0.0
    _open: dict[str, dict] = field(default_factory=dict)   # client_order_id -> order (transient)
    _seq: int = 0

    def place_order(self, order: dict) -> dict:
        self._require_limit(order)
        ticker = order["ticker"]
        qty = int(order["quantity_lots"])
        price = float(order["limit_price"])
        signed = qty if order["side"] == "BUY" else -qty
        # MOEX has no fractional lots; the engine has already lot-rounded. Fill fully at the limit.
        self._positions[ticker] = self._positions.get(ticker, 0) + signed
        self._cash -= signed * price                       # buys consume cash, sells return it (lot-priced)
        self._seq += 1
        exch_id = f"paper-{self._seq:06d}"
        return execution_report(order["client_order_id"], ticker, "FILLED",
                                exchange_order_id=exch_id, filled_quantity_lots=qty,
                                avg_fill_price=price, message="paper fill at limit")

    def cancel(self, client_order_id: str) -> dict:
        self._open.pop(client_order_id, None)
        return execution_report(client_order_id, "", "CANCELED", message="paper cancel")

    def cancel_all(self) -> list[dict]:
        reports = [execution_report(coid, o.get("ticker", ""), "CANCELED", message="paper cancel-all")
                   for coid, o in self._open.items()]
        self._open.clear()
        return reports

    def positions(self) -> dict[str, int]:
        return {t: l for t, l in self._positions.items() if l != 0}

    @property
    def cash(self) -> float:
        return self._cash
