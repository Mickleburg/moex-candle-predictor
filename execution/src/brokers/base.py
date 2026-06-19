"""Broker adapter interface + execution_report helpers shared by all backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

VALID_STATUSES = {"DRY_RUN", "PLACED", "REJECTED", "FILLED", "CANCELED"}


def execution_report(
    client_order_id: str,
    ticker: str,
    status: str,
    *,
    exchange_order_id: str | None = None,
    filled_quantity_lots: int = 0,
    avg_fill_price: float | None = None,
    message: str = "",
) -> dict:
    """Build an `execution_report` contract dict (contracts/execution_report.schema.json)."""
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid execution status: {status}")
    return {
        "client_order_id": client_order_id,
        "ticker": ticker,
        "status": status,
        "exchange_order_id": exchange_order_id,
        "filled_quantity_lots": int(filled_quantity_lots),
        "avg_fill_price": avg_fill_price,
        "message": message,
    }


class BrokerAdapter(ABC):
    """Minimal surface the engine needs. Implementations must accept ONLY LIMIT orders."""

    name: str = "base"

    @abstractmethod
    def place_order(self, order: dict) -> dict:
        """Submit one `order_request`; return an `execution_report`."""

    @abstractmethod
    def cancel(self, client_order_id: str) -> dict:
        """Cancel a single open order by client_order_id; return an `execution_report`."""

    @abstractmethod
    def cancel_all(self) -> list[dict]:
        """Cancel every open order (used by the kill-switch); return one report per order."""

    def positions(self) -> dict[str, int]:
        """Current signed lot positions by ticker (best effort; default: unknown -> empty)."""
        return {}

    @staticmethod
    def _require_limit(order: dict) -> None:
        if order.get("order_type") != "LIMIT":
            raise ValueError("execution accepts LIMIT orders only")
        if order.get("limit_price") is None:
            raise ValueError("LIMIT order requires a limit_price")
