"""Validate demo architecture contracts and examples.

The script intentionally stays lightweight. It always checks JSON syntax and
cross-contract consistency. If ``jsonschema`` is installed, it also validates
each example against the matching schema.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_DIR = REPO_ROOT / "contracts"
EXAMPLES_DIR = CONTRACTS_DIR / "examples"
CONFIG_DIR = REPO_ROOT / "config"

CONTRACTS = [
    "candle_batch",
    "market_snapshot",
    "portfolio_snapshot",
    "ml_prediction",
    "llm_analysis",
    "aggregated_signal",
    "risk_decision",
    "order_request",
    "execution_report",
    "agent_cycle_result",
]

PROBABILITY_KEYS = {"buy", "hold", "sell"}
SUPPORTED_ACTIONS = {"BUY", "SELL", "HOLD", "BUY_MORE", "SELL_PARTIAL", "SELL_ALL"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_optional_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError:
        print("PyYAML is not installed; using a minimal supported_tickers.yaml fallback parser.")
        return parse_supported_tickers_fallback(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_supported_tickers_fallback(path: Path) -> dict[str, Any]:
    tickers: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("- ticker:"):
            current = {"ticker": line.split(":", 1)[1].strip()}
            tickers.append(current)
        elif current is not None and ":" in line:
            key, value = line.split(":", 1)
            value = value.strip()
            if value.lower() in {"true", "false"}:
                current[key] = value.lower() == "true"
            elif value.startswith("[") and value.endswith("]"):
                current[key] = [item.strip().strip('"') for item in value[1:-1].split(",") if item.strip()]
            else:
                current[key] = value
    return {"tickers": tickers}


def validate_jsonschema_if_available(schemas: dict[str, dict[str, Any]], examples: dict[str, dict[str, Any]]) -> None:
    try:
        import jsonschema
        from jsonschema import Draft202012Validator, RefResolver
    except ImportError:
        print("jsonschema is not installed; skipped schema-vs-example validation.")
        print("Install with: pip install jsonschema")
        return

    store = {f"{name}.schema.json": schema for name, schema in schemas.items()}
    for name, schema in schemas.items():
        Draft202012Validator.check_schema(schema)
        resolver = RefResolver(base_uri=f"file:///{CONTRACTS_DIR.as_posix()}/", referrer=schema, store=store)
        validator = jsonschema.Draft202012Validator(schema, resolver=resolver)
        validator.validate(examples[name])


def validate_schema_shapes(schemas: dict[str, dict[str, Any]]) -> None:
    for name, schema in schemas.items():
        if schema.get("type") != "object":
            raise AssertionError(f"{name}.schema.json must describe an object")
        if not schema.get("required"):
            raise AssertionError(f"{name}.schema.json must define required fields")
        if not schema.get("properties"):
            raise AssertionError(f"{name}.schema.json must define properties")

    for name, field in [
        ("ml_prediction", "probabilities"),
        ("llm_analysis", "probabilities"),
        ("aggregated_signal", "combined_probabilities"),
    ]:
        proba_schema = schemas[name]["properties"][field]
        keys = set(proba_schema["properties"])
        if keys != PROBABILITY_KEYS:
            raise AssertionError(f"{name}.{field} schema keys mismatch: {sorted(keys)}")

    if set(schemas["aggregated_signal"]["properties"]["raw_decision"].get("enum", [])) != {"BUY", "HOLD", "SELL"}:
        raise AssertionError("aggregated_signal.raw_decision must enumerate BUY/HOLD/SELL")
    for field in ["requested_action", "approved_action"]:
        enum = set(schemas["risk_decision"]["properties"][field].get("enum", []))
        if not SUPPORTED_ACTIONS.issubset(enum):
            raise AssertionError(f"risk_decision.{field} missing supported action enum values")
    if set(schemas["order_request"]["properties"]["side"].get("enum", [])) != {"BUY", "SELL"}:
        raise AssertionError("order_request.side must enumerate BUY/SELL")


def assert_probability_vector(payload: dict[str, Any], field: str) -> None:
    proba = payload[field]
    if set(proba) != PROBABILITY_KEYS:
        raise AssertionError(f"{field} has keys {sorted(proba)}, expected {sorted(PROBABILITY_KEYS)}")
    total = sum(float(proba[key]) for key in PROBABILITY_KEYS)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError(f"{field} sums to {total}, expected 1.0")


def validate_cross_contracts(examples: dict[str, dict[str, Any]]) -> None:
    candle = examples["candle_batch"]
    market = examples["market_snapshot"]
    ml = examples["ml_prediction"]
    llm = examples["llm_analysis"]
    signal = examples["aggregated_signal"]
    portfolio = examples["portfolio_snapshot"]
    risk = examples["risk_decision"]
    order = examples["order_request"]
    execution = examples["execution_report"]
    agent = examples["agent_cycle_result"]

    for name, payload, field in [
        ("ml_prediction", ml, "probabilities"),
        ("llm_analysis", llm, "probabilities"),
        ("aggregated_signal", signal, "combined_probabilities"),
    ]:
        assert_probability_vector(payload, field)
        print(f"{name}: probability vector OK")

    for payload_name, payload in [
        ("candle_batch", candle),
        ("market_snapshot", market),
        ("ml_prediction", ml),
        ("llm_analysis", llm),
        ("aggregated_signal", signal),
    ]:
        if payload["ticker"] != "SBER":
            raise AssertionError(f"{payload_name} ticker mismatch")
        if payload["timeframe"] != "1H":
            raise AssertionError(f"{payload_name} timeframe mismatch")

    if set(ml["probabilities"]) != set(llm["probabilities"]):
        raise AssertionError("ML and LLM probabilities use different keys")
    if signal["raw_decision"] != risk["requested_action"]:
        raise AssertionError("aggregated_signal.raw_decision must match risk_decision.requested_action")
    if risk["requested_action"] not in SUPPORTED_ACTIONS or risk["approved_action"] not in SUPPORTED_ACTIONS:
        raise AssertionError("risk_decision action is not in supported action set")
    if risk["approved"]:
        if risk["order_intent"] is None:
            raise AssertionError("approved risk_decision must include order_intent")
        intent = risk["order_intent"]
        for key in ["side", "quantity_lots", "order_type", "limit_price"]:
            if intent[key] != order[key]:
                raise AssertionError(f"risk order_intent.{key} does not match order_request.{key}")
    if order["client_order_id"] != execution["client_order_id"]:
        raise AssertionError("execution_report client_order_id must match order_request")
    if order not in agent["selected_orders"]:
        raise AssertionError("agent_cycle_result.selected_orders must include order_request example")
    if "SBER" not in agent["evaluated_tickers"] or "GAZP" not in agent["evaluated_tickers"] or "LKOH" not in agent["evaluated_tickers"]:
        raise AssertionError("agent_cycle_result must include demo tickers")
    if not any(position["ticker"] == "SBER" for position in portfolio["positions"]):
        raise AssertionError("portfolio_snapshot must include SBER position")


def validate_supported_tickers() -> None:
    data = load_optional_yaml(CONFIG_DIR / "supported_tickers.yaml")
    tickers = data.get("tickers", [])
    by_ticker = {item.get("ticker"): item for item in tickers}
    for ticker in ["SBER", "GAZP", "LKOH"]:
        item = by_ticker.get(ticker)
        if item is None:
            raise AssertionError(f"supported_tickers.yaml missing {ticker}")
        for key in ["ticker", "name", "market", "board", "enabled", "timeframes"]:
            if key not in item:
                raise AssertionError(f"{ticker} missing {key}")
        if "1H" not in item["timeframes"]:
            raise AssertionError(f"{ticker} missing 1H timeframe")


def main() -> int:
    schemas: dict[str, dict[str, Any]] = {}
    examples: dict[str, dict[str, Any]] = {}
    for name in CONTRACTS:
        schema_path = CONTRACTS_DIR / f"{name}.schema.json"
        example_path = EXAMPLES_DIR / f"{name}.example.json"
        if not schema_path.exists():
            raise FileNotFoundError(schema_path)
        if not example_path.exists():
            raise FileNotFoundError(example_path)
        schemas[name] = load_json(schema_path)
        examples[name] = load_json(example_path)
    validate_schema_shapes(schemas)
    validate_jsonschema_if_available(schemas, examples)
    validate_cross_contracts(examples)
    validate_supported_tickers()
    print("All contract checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
