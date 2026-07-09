"""Every JSON crossing a block seam validates against its contract."""

from __future__ import annotations

from agent.src import contracts
from agent.src.adapters import mock


def test_canned_examples_validate():
    import json
    from pathlib import Path
    ex = Path(contracts.REPO_ROOT) / "contracts" / "examples"
    for name in ("sleeve_signal", "risk_book", "agent_cycle_result", "order_request",
                 "execution_report"):
        payload = json.loads((ex / f"{name}.example.json").read_text(encoding="utf-8"))
        contracts.validate(payload, name)


def test_mock_pipeline_emits_valid_contracts():
    as_of = "2026-06-18T19:05:00+03:00"
    sig = mock.MockSleeve().build_sleeve(as_of)
    contracts.validate(sig, "sleeve_signal")
    assert contracts.is_research_artifact(sig)   # is_production=false invariant

    book = mock.MockCombiner(hedge_mode="sector").combine([sig], as_of)
    contracts.validate(book, "risk_book")
    assert contracts.is_research_artifact(book)

    prices = mock.MockBackend().latest_prices(["SBER", "LKOH", "TATN"], as_of)
    er = mock.PaperBrokerExecution().reconcile_and_execute(
        risk_book=book, positions=[], prices=prices, capital=10_000_000.0, mode="paper",
        trade_date="2026-06-18", phase="eod")
    for order in er.orders:
        contracts.validate(order, "order_request")
    for rep in er.reports:
        contracts.validate(rep, "execution_report")
    assert er.orders and er.reports


def test_invalid_payload_raises():
    import pytest
    with pytest.raises(contracts.ContractError):
        contracts.validate({"sleeve": "s3_event"}, "sleeve_signal")   # missing required keys
