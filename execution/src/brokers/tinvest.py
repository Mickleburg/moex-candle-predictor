"""T-Invest (Tinkoff Invest) broker adapter — sandbox-first, live gated.

Why T-Invest (recorded per task): it ships a first-class SANDBOX that mirrors the production order
API, so paper trading is the real order path with fake money — the cleanest paper->live promotion of
the candidates (Finam TradeAPI / ALOR OpenAPI / QUIK connector). One REST gateway, token auth.

This adapter speaks the public REST gateway with the stdlib only (no extra deps), so the rest of the
block stays importable without network/credentials. It is intentionally thin and UNTESTED against the
wire here — it constructs lazily and refuses loudly without a token, and the engine/factory refuse
LIVE unless explicitly enabled. FIGI resolution (ticker -> instrument id) is a TODO to wire from the
backend instrument table; without it, pass a figi map in.

Secrets: TINVEST_TOKEN / TINVEST_ACCOUNT_ID come from the environment / .env (never git).
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field

from .base import BrokerAdapter, execution_report

REST_BASE = "https://invest-public-api.tinkoff.ru/rest"
# Sandbox and production share the request schema; only the service prefix + token's environment differ.
SANDBOX_SERVICE = "tinkoff.public.invest.api.contract.v1.SandboxService"
ORDERS_SERVICE = "tinkoff.public.invest.api.contract.v1.OrdersService"


@dataclass
class TInvestBroker(BrokerAdapter):
    """LIMIT-only adapter over the T-Invest REST gateway. Sandbox by default."""

    name: str = "tinvest"
    token: str | None = None
    account_id: str | None = None
    sandbox: bool = True
    figi_by_ticker: dict[str, str] = field(default_factory=dict)
    _open: dict[str, str] = field(default_factory=dict)   # client_order_id -> exchange order id

    def __post_init__(self) -> None:
        self.token = self.token or os.environ.get("TINVEST_TOKEN")
        self.account_id = self.account_id or os.environ.get("TINVEST_ACCOUNT_ID")
        if not self.token:
            raise RuntimeError(
                "TInvestBroker requires a token (TINVEST_TOKEN env / .env). Refusing to construct "
                "without credentials — there is no anonymous order path.")

    # --- REST plumbing -----------------------------------------------------------------
    def _post(self, service: str, method: str, body: dict) -> dict:
        url = f"{REST_BASE}/{service}/{method}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=15) as resp:   # noqa: S310 - fixed trusted host
            return json.loads(resp.read().decode("utf-8"))

    def _figi(self, ticker: str) -> str:
        figi = self.figi_by_ticker.get(ticker)
        if not figi:
            raise RuntimeError(f"no FIGI mapping for {ticker}; pass figi_by_ticker (from backend "
                               "instrument metadata)")
        return figi

    def _orders_service(self) -> str:
        # Sandbox routes orders through SandboxService; production through OrdersService.
        return SANDBOX_SERVICE if self.sandbox else ORDERS_SERVICE

    # --- BrokerAdapter -----------------------------------------------------------------
    def place_order(self, order: dict) -> dict:
        self._require_limit(order)
        method = "PostSandboxOrder" if self.sandbox else "PostOrder"
        body = {
            "figi": self._figi(order["ticker"]),
            "quantity": str(order["quantity_lots"]),
            "price": _to_quotation(_snap_price(order["ticker"], float(order["limit_price"]))),
            "direction": "ORDER_DIRECTION_BUY" if order["side"] == "BUY" else "ORDER_DIRECTION_SELL",
            "accountId": self.account_id,
            "orderType": "ORDER_TYPE_LIMIT",
            "orderId": order["client_order_id"],   # idempotency key honored by the API
        }
        resp = self._post(self._orders_service(), method, body)
        exch_id = resp.get("orderId")
        self._open[order["client_order_id"]] = exch_id
        return execution_report(order["client_order_id"], order["ticker"], "PLACED",
                                exchange_order_id=exch_id, message="submitted to T-Invest "
                                f"({'sandbox' if self.sandbox else 'LIVE'})")

    def cancel(self, client_order_id: str) -> dict:
        method = "CancelSandboxOrder" if self.sandbox else "CancelOrder"
        exch_id = self._open.get(client_order_id, client_order_id)
        self._post(self._orders_service(), method, {"accountId": self.account_id, "orderId": exch_id})
        self._open.pop(client_order_id, None)
        return execution_report(client_order_id, "", "CANCELED", exchange_order_id=exch_id,
                                message="canceled at T-Invest")

    def cancel_all(self) -> list[dict]:
        return [self.cancel(coid) for coid in list(self._open)]


def _snap_price(ticker: str, price: float) -> float:
    """Snap a limit price to the instrument's MINSTEP grid via backend metadata (no-op if absent).

    MOEX rejects limit prices off the price step; backend owns the per-instrument step.
    """
    try:
        from backend.instruments import round_price  # type: ignore
        return float(round_price(ticker, price))
    except Exception:
        return price


def _to_quotation(price: float) -> dict:
    """T-Invest prices are {units, nano}; nano is the 1e-9 fractional part."""
    units = int(price)
    nano = int(round((price - units) * 1_000_000_000))
    return {"units": str(units), "nano": nano}
