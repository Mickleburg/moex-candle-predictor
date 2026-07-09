"""T-Invest (Tinkoff Invest) broker adapter — sandbox-first, live gated.

Why T-Invest (recorded per task): it ships a first-class SANDBOX that mirrors the production order
API, so paper trading is the real order path with fake money — the cleanest paper->live promotion of
the candidates (Finam TradeAPI / ALOR OpenAPI / QUIK connector). One REST gateway, token auth.

This adapter speaks the public REST gateway with the stdlib only (no extra deps), so the rest of the
block stays importable without network/credentials. It constructs lazily and refuses loudly without a
token, and the engine/factory refuse LIVE unless explicitly enabled (EXECUTION_ALLOW_LIVE=1) AND every
FIGI is verified. FIGI resolution (ticker -> instrument id) comes from the backend instrument table
(`backend.instruments`), injected via `figi_by_ticker`.

Prices: in LIVE the limit price is sourced from the broker's REAL-TIME quote
(`price_from_quote=True` -> GetOrderBook / GetLastPrices) — T-Invest serves these on the brokerage
account with NO separate paid market-data subscription. In paper/sandbox we keep the deterministic
reference-close limit so a paper run reconciles against the sleeve sim.

Secrets: TINVEST_TOKEN / TINVEST_ACCOUNT_ID come from the environment / .env (never git).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

from .base import BrokerAdapter, execution_report

REST_BASE = "https://invest-public-api.tinkoff.ru/rest"
_SVC_PREFIX = "tinkoff.public.invest.api.contract.v1"
# T-Invest requires the order idempotency key to be a UUID. Our client_order_ids are human-readable
# (exec-DATE-TICKER-SIDE-QTY), so we map each to a DETERMINISTIC uuid5 — same client_order_id -> same
# UUID, which keeps the broker-side idempotency (a re-sent order is deduped by the API).
_OID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def _wire_order_id(client_order_id: str) -> str:
    return str(uuid.uuid5(_OID_NAMESPACE, client_order_id))

# Wire executionReportStatus -> our execution_report status enum.
_WIRE_STATUS = {
    "EXECUTION_REPORT_STATUS_FILL": "FILLED",
    "EXECUTION_REPORT_STATUS_PARTIALLYFILL": "PLACED",
    "EXECUTION_REPORT_STATUS_NEW": "PLACED",
    "EXECUTION_REPORT_STATUS_REJECTED": "REJECTED",
    "EXECUTION_REPORT_STATUS_CANCELLED": "CANCELED",
    "EXECUTION_REPORT_STATUS_UNSPECIFIED": "PLACED",
}


class TInvestError(RuntimeError):
    """A T-Invest REST error, carrying the HTTP status + response body for diagnosis."""


@dataclass
class TInvestBroker(BrokerAdapter):
    """LIMIT-only adapter over the T-Invest REST gateway. Sandbox by default."""

    name: str = "tinvest"
    token: str | None = None
    account_id: str | None = None
    sandbox: bool = True
    figi_by_ticker: dict[str, str] = field(default_factory=dict)
    price_from_quote: bool = False        # live: price the limit off the broker's real-time quote
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
        url = f"{REST_BASE}/{_SVC_PREFIX}.{service}/{method}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:   # noqa: S310 - fixed trusted host
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise TInvestError(f"{service}/{method} -> HTTP {exc.code}: {detail}") from None

    def _instrument_id(self, ticker: str) -> str:
        """T-Invest instrumentId — the FIGI works as an instrumentId on every order/market-data call."""
        figi = self.figi_by_ticker.get(ticker)
        if not figi:
            raise RuntimeError(f"no FIGI mapping for {ticker}; pass figi_by_ticker (from backend "
                               "instrument metadata)")
        return figi

    def _orders_service(self) -> str:
        return "SandboxService" if self.sandbox else "OrdersService"

    # --- sandbox lifecycle (sandbox only) ----------------------------------------------
    def open_sandbox_account(self) -> str:
        """Open a fresh sandbox account and adopt it; returns the account id."""
        self.account_id = self._post("SandboxService", "OpenSandboxAccount", {})["accountId"]
        return self.account_id

    def get_sandbox_accounts(self) -> list[dict]:
        return self._post("SandboxService", "GetSandboxAccounts", {}).get("accounts", [])

    def sandbox_pay_in(self, units: int, currency: str = "rub") -> dict:
        return self._post("SandboxService", "SandboxPayIn", {
            "accountId": self.account_id,
            "amount": {"currency": currency, "units": str(int(units)), "nano": 0}})

    def close_sandbox_account(self, account_id: str | None = None) -> dict:
        return self._post("SandboxService", "CloseSandboxAccount",
                          {"accountId": account_id or self.account_id})

    # --- market data (quotes; live limit prices come from here, no paid subscription) ----
    def last_price(self, ticker: str) -> float:
        resp = self._post("MarketDataService", "GetLastPrices",
                          {"instrumentId": [self._instrument_id(ticker)]})
        prices = resp.get("lastPrices") or []
        if not prices:
            raise TInvestError(f"no last price for {ticker}")
        return _money_to_float(prices[0]["price"])

    def order_book(self, ticker: str, depth: int = 1) -> dict:
        return self._post("MarketDataService", "GetOrderBook",
                          {"instrumentId": self._instrument_id(ticker), "depth": depth})

    def quote_limit_price(self, ticker: str, side: str, marketable: bool = False) -> float:
        """A limit price drawn from the live order book. marketable=True crosses (best opposite side),
        else it joins the touch (passive). Falls back to last/close price if the book is empty."""
        ob = self.order_book(ticker, depth=1)
        asks, bids = ob.get("asks") or [], ob.get("bids") or []
        if marketable:
            book = asks if side == "BUY" else bids
        else:
            book = bids if side == "BUY" else asks
        if book:
            return _money_to_float(book[0]["price"])
        for key in ("lastPrice", "closePrice"):
            if ob.get(key):
                return _money_to_float(ob[key])
        return self.last_price(ticker)

    # --- orders ------------------------------------------------------------------------
    def place_order(self, order: dict) -> dict:
        self._require_limit(order)
        coid = order["client_order_id"]
        # Idempotency layer 1: never re-send a client_order_id we already placed this session.
        if coid in self._open:
            return execution_report(coid, order["ticker"], "PLACED", exchange_order_id=self._open[coid],
                                    message="duplicate intent — already submitted (idempotent, not re-sent)")
        price = float(order["limit_price"])
        if self.price_from_quote:        # live: price off the broker's real-time quote, not the EOD close
            price = self.quote_limit_price(order["ticker"], order["side"], marketable=False)
        price = _snap_price(order["ticker"], price)
        method = "PostSandboxOrder" if self.sandbox else "PostOrder"
        body = {
            "accountId": self.account_id,
            "instrumentId": self._instrument_id(order["ticker"]),
            "quantity": str(int(order["quantity_lots"])),
            "price": _to_quotation(price),
            "direction": "ORDER_DIRECTION_BUY" if order["side"] == "BUY" else "ORDER_DIRECTION_SELL",
            "orderType": "ORDER_TYPE_LIMIT",
            "orderId": _wire_order_id(coid),   # deterministic UUID idempotency key (API requirement)
        }
        try:
            resp = self._post(self._orders_service(), method, body)
        except TInvestError as exc:
            # Idempotency layer 2: the API itself dedups by orderId (code 30057). A duplicate is NOT a
            # failure — the original order stands; we just did not create a second one.
            if "duplicate" in str(exc).lower() or "30057" in str(exc):
                return execution_report(coid, order["ticker"], "PLACED",
                                        message="duplicate at broker (idempotent, not re-created)")
            raise
        return self._report(coid, order["ticker"], resp)

    def order_state(self, client_order_id: str, ticker: str = "") -> dict:
        method = "GetSandboxOrderState" if self.sandbox else "GetOrderState"
        exch_id = self._open.get(client_order_id, client_order_id)
        resp = self._post(self._orders_service(), method,
                          {"accountId": self.account_id, "orderId": exch_id})
        return self._report(client_order_id, ticker, resp)

    def get_orders(self) -> list[dict]:
        method = "GetSandboxOrders" if self.sandbox else "GetOrders"
        return self._post(self._orders_service(), method,
                          {"accountId": self.account_id}).get("orders", [])

    def cancel(self, client_order_id: str) -> dict:
        method = "CancelSandboxOrder" if self.sandbox else "CancelOrder"
        exch_id = self._open.get(client_order_id, client_order_id)
        self._post(self._orders_service(), method,
                   {"accountId": self.account_id, "orderId": exch_id})
        self._open.pop(client_order_id, None)
        return execution_report(client_order_id, "", "CANCELED", exchange_order_id=exch_id,
                                message="canceled at T-Invest")

    def cancel_all(self) -> list[dict]:
        return [self.cancel(coid) for coid in list(self._open)]

    # --- response mapping --------------------------------------------------------------
    def _report(self, client_order_id: str, ticker: str, resp: dict) -> dict:
        exch_id = resp.get("orderId")
        if exch_id:
            self._open[client_order_id] = exch_id
        wire = resp.get("executionReportStatus", "EXECUTION_REPORT_STATUS_UNSPECIFIED")
        status = _WIRE_STATUS.get(wire, "PLACED")
        filled = int(resp.get("lotsExecuted") or 0)
        avg = None
        if filled > 0 and resp.get("executedOrderPrice"):
            v = _money_to_float(resp["executedOrderPrice"])
            avg = v if v > 0 else None
        if status == "CANCELED":
            self._open.pop(client_order_id, None)
        return execution_report(client_order_id, ticker, status, exchange_order_id=exch_id,
                                filled_quantity_lots=filled, avg_fill_price=avg,
                                message=f"{'sandbox' if self.sandbox else 'LIVE'}: {wire}")


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


def _money_to_float(q: dict) -> float:
    """{units, nano} (MoneyValue / Quotation) -> float. units may be a string."""
    return int(q.get("units", 0)) + int(q.get("nano", 0)) / 1_000_000_000
