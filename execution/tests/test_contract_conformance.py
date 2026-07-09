"""Every order/report execution emits must validate against the shared JSON contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution.src.brokers.base import execution_report
from execution.src.brokers.paper import DryRunBroker, PaperBroker
from execution.src.config import ExecutionConfig, Mode
from execution.src.reconcile import reconcile

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = REPO_ROOT / "contracts"

jsonschema = pytest.importorskip("jsonschema")
from jsonschema import Draft202012Validator  # noqa: E402


def _validator(name: str) -> "Draft202012Validator":
    schema = json.loads((CONTRACTS / f"{name}.schema.json").read_text("utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_generated_orders_conform_to_order_request():
    book = json.loads((CONTRACTS / "examples" / "risk_book.example.json").read_text("utf-8"))
    prices = json.loads((REPO_ROOT / "execution" / "examples" / "prices.example.json").read_text("utf-8"))
    res = reconcile(book, prices, current_lots={},
                    config=ExecutionConfig(mode=Mode.DRY_RUN, capital=100_000_000.0))
    validator = _validator("order_request")
    assert res.orders
    for order in res.orders:
        validator.validate(order.to_order_request())


def test_paper_and_dry_run_reports_conform_to_execution_report():
    validator = _validator("execution_report")
    order = {"ticker": "SBER", "side": "BUY", "quantity_lots": 3, "order_type": "LIMIT",
             "limit_price": 312.4, "client_order_id": "exec-20260702-SBER-BUY-3"}

    paper = PaperBroker().place_order(order)
    assert paper["status"] == "FILLED"
    validator.validate(paper)

    dry = DryRunBroker().place_order(order)
    assert dry["status"] == "DRY_RUN"
    validator.validate(dry)

    # cancel + a synthetic rejection are also valid reports
    validator.validate(PaperBroker().cancel("x"))
    validator.validate(execution_report("y", "SBER", "REJECTED", message="duplicate"))
