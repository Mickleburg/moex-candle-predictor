"""The orchestrator seam: engine.reconcile_and_execute + the `serve` CLI envelope.

Asserts the exact contract agent/src/adapters/live.py::LiveExecution expects: a request envelope
{risk_book, positions, prices, capital, mode} -> {orders, reports, positions, rejected}, with orders
validating as order_request, reports as execution_report, and the resulting book reflecting fills.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from execution.src.cli import main
from execution.src.config import Mode
from execution.src.engine import ExecutionEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = REPO_ROOT / "contracts"
REQUEST = REPO_ROOT / "execution" / "examples" / "serve_request.example.json"

jsonschema = pytest.importorskip("jsonschema")
from jsonschema import Draft202012Validator  # noqa: E402


def _validator(name: str):
    schema = json.loads((CONTRACTS / f"{name}.schema.json").read_text("utf-8"))
    return Draft202012Validator(schema)


def test_reconcile_and_execute_envelope(tmp_config):
    req = json.loads(REQUEST.read_text("utf-8"))
    engine = ExecutionEngine(tmp_config(mode=Mode.PAPER, capital=req["capital"]))
    out = engine.reconcile_and_execute(
        risk_book=req["risk_book"], positions=req["positions"], prices=req["prices"])

    assert set(out) >= {"orders", "reports", "positions", "rejected"}
    # 3 long names (BUY) + 2 hedge legs (SELL) from a flat book
    assert len(out["orders"]) == 5
    assert {o["side"] for o in out["orders"]} == {"BUY", "SELL"}

    order_v, report_v = _validator("order_request"), _validator("execution_report")
    for o in out["orders"]:
        order_v.validate(o)
    for r in out["reports"]:
        report_v.validate(r)
        assert r["status"] == "FILLED"        # paper sim fills

    # resulting book reflects fills: longs positive, hedge legs negative, enriched fields present
    by = {p["ticker"]: p for p in out["positions"]}
    assert by["SBER"]["lots"] > 0 and by["SBER"]["is_hedge"] is False
    assert by["MOEXOG"]["lots"] < 0 and by["MOEXOG"]["is_hedge"] is True
    assert by["SBER"]["sleeve_contributions"] == {"s3_event": 0.34}
    assert by["SBER"]["last_price"] == req["prices"]["SBER"]
    assert out["is_production"] is False


def test_serve_cli_stdin_stdout(monkeypatch, capsys, tmp_path):
    # Redirect runtime state/audit to tmp so the dedupe ledger doesn't touch the repo's var/.
    monkeypatch.setenv("EXECUTION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("EXECUTION_AUDIT_DIR", str(tmp_path / "audit"))
    raw = REQUEST.read_text("utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO(raw))
    rc = main(["serve", "--mode", "paper", "--quiet"])
    assert rc == 0
    captured = capsys.readouterr()
    out = json.loads(captured.out)           # stdout must be pure JSON (agent does json.loads on it)
    assert len(out["orders"]) == 5
    assert all(r["status"] == "FILLED" for r in out["reports"])
    assert captured.err == ""                # --quiet -> no banner leaked to stderr


def test_exit_flattens_when_name_drops_from_book(tmp_config):
    # Current book holds SBER; the new book is empty -> reconcile must SELL it all.
    engine = ExecutionEngine(tmp_config(mode=Mode.PAPER, capital=100_000_000.0))
    out = engine.reconcile_and_execute(
        risk_book={"as_of": "2025-06-10 00:00:00+03:00", "net_positions": [],
                   "hedge": {"mode": "none", "legs": []}},
        positions=[{"ticker": "SBER", "lots": 1000, "avg_price": 300.0}],
        prices={"SBER": 312.4})
    assert len(out["orders"]) == 1
    assert out["orders"][0]["side"] == "SELL" and out["orders"][0]["quantity_lots"] == 1000
    assert out["positions"] == []            # flat after the exit


def test_missing_price_surfaces_in_rejected(tmp_config):
    engine = ExecutionEngine(tmp_config(mode=Mode.DRY_RUN, capital=100_000_000.0))
    out = engine.reconcile_and_execute(
        risk_book={"as_of": "2025-06-02 00:00:00+03:00",
                   "net_positions": [{"ticker": "SBER", "weight": 0.3, "side": "LONG"}],
                   "hedge": {"mode": "none", "legs": []}},
        positions=[], prices={})            # no price for SBER
    assert {"ticker": "SBER", "track": "live", "reason": "missing_or_nonpositive_price"} in out["rejected"]
    assert out["orders"] == []
