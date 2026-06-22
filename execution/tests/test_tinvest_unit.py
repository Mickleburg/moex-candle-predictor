"""Offline unit tests for the T-Invest adapter (no network).

Covers the pure wire-mapping helpers + the no-network paths of the adapter: response -> contract
execution_report, deterministic UUID idempotency key, quotation round-trip, the cached-duplicate
short-circuit, and the no-token refusal. The networked sandbox e2e lives in test_tinvest_sandbox.py
(opt-in).
"""

from __future__ import annotations

import uuid

import pytest

from execution.src.brokers.tinvest import (
    TInvestBroker,
    _money_to_float,
    _to_quotation,
    _wire_order_id,
)


@pytest.fixture
def broker(monkeypatch):
    monkeypatch.setenv("TINVEST_TOKEN", "dummy-token-for-offline-tests")
    return TInvestBroker(sandbox=True, account_id="acc-1", figi_by_ticker={"SBER": "BBG004730N88"})


def test_requires_a_token(monkeypatch):
    monkeypatch.delenv("TINVEST_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        TInvestBroker(sandbox=True)


def test_wire_order_id_is_a_deterministic_uuid():
    a = _wire_order_id("exec-20250602-SBER-BUY-10")
    b = _wire_order_id("exec-20250602-SBER-BUY-10")
    c = _wire_order_id("exec-20250602-SBER-BUY-11")
    assert a == b and a != c
    uuid.UUID(a)   # parses as a valid UUID (T-Invest requires this)


def test_quotation_roundtrip():
    q = _to_quotation(304.86)
    assert q == {"units": "304", "nano": 860000000}
    assert abs(_money_to_float(q) - 304.86) < 1e-9
    assert _money_to_float({"units": "11800", "nano": 0}) == 11800.0


def test_report_maps_wire_status_to_contract(broker):
    fill = broker._report("exec-x", "SBER", {
        "orderId": "ex-1", "executionReportStatus": "EXECUTION_REPORT_STATUS_FILL",
        "lotsExecuted": 2, "executedOrderPrice": {"units": "304", "nano": 860000000}})
    assert fill["status"] == "FILLED" and fill["filled_quantity_lots"] == 2
    assert abs(fill["avg_fill_price"] - 304.86) < 1e-9
    assert fill["exchange_order_id"] == "ex-1"

    new = broker._report("exec-y", "SBER", {
        "orderId": "ex-2", "executionReportStatus": "EXECUTION_REPORT_STATUS_NEW", "lotsExecuted": 0})
    assert new["status"] == "PLACED" and new["filled_quantity_lots"] == 0 and new["avg_fill_price"] is None

    rej = broker._report("exec-z", "SBER", {
        "orderId": "ex-3", "executionReportStatus": "EXECUTION_REPORT_STATUS_REJECTED"})
    assert rej["status"] == "REJECTED"


def test_cached_duplicate_short_circuits_without_network(broker):
    # Pre-seed the open ledger; place_order must return the cached report and NOT hit the network.
    broker._open["exec-dup"] = "ex-existing"
    rep = broker.place_order({"ticker": "SBER", "side": "BUY", "quantity_lots": 1,
                              "order_type": "LIMIT", "limit_price": 300.0, "client_order_id": "exec-dup"})
    assert rep["status"] == "PLACED"
    assert rep["exchange_order_id"] == "ex-existing"
    assert "idempotent" in rep["message"]


def test_place_order_requires_limit(broker):
    with pytest.raises(ValueError):
        broker.place_order({"ticker": "SBER", "side": "BUY", "quantity_lots": 1,
                            "order_type": "MARKET", "limit_price": None, "client_order_id": "x"})
