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
    "feature_bundle",
    "aggregated_signal",
    "risk_decision",
    "order_request",
    "execution_report",
    "agent_cycle_result",
    "risk_analytics",
    # V3 sleeve combiner (risk_manager): position-form sleeve input + combined book output.
    "sleeve_signal",
    "risk_book",
]

PROBABILITY_KEYS = {"buy", "hold", "sell"}
SUPPORTED_ACTIONS = {"BUY", "SELL", "HOLD", "BUY_MORE", "SELL_PARTIAL", "SELL_ALL"}
LEG_VALUES = {"long", "short", "flat"}
SLEEVE_VALUES = {"s1_pairs", "s2_macro", "s3_event", "s4_core"}
HEDGE_MODES = {"sector", "market", "none"}


def parse_dt(value: str):
    from datetime import datetime

    return datetime.fromisoformat(value)


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

    # ml_prediction keeps the per-ticker buy/hold/sell forecast vector (V1 contract, still served).
    proba_schema = schemas["ml_prediction"]["properties"]["probabilities"]
    if set(proba_schema["properties"]) != PROBABILITY_KEYS:
        raise AssertionError("ml_prediction.probabilities schema keys mismatch")

    # V2: llm_analysis emits NEWS FEATURES, not a buy/hold/sell vector.
    llm_props = schemas["llm_analysis"]["properties"]
    if "probabilities" in llm_props:
        raise AssertionError("V2: llm_analysis must NOT carry buy/hold/sell probabilities (features only)")
    if "features" not in llm_props:
        raise AssertionError("V2: llm_analysis must define a 'features' object")

    # V2: aggregated_signal is a cross-sectional ranking, not a late-fusion vector.
    agg_props = schemas["aggregated_signal"]["properties"]
    for dead in ("combined_probabilities", "raw_decision", "components"):
        if dead in agg_props:
            raise AssertionError(f"V2: aggregated_signal must NOT carry late-fusion field '{dead}'")
    if "rankings" not in agg_props:
        raise AssertionError("V2: aggregated_signal must define a 'rankings' array")
    leg_enum = set(agg_props["rankings"]["items"]["properties"]["leg"].get("enum", []))
    if leg_enum != LEG_VALUES:
        raise AssertionError(f"aggregated_signal.rankings[].leg must enumerate {sorted(LEG_VALUES)}")

    # V2: feature_bundle carries the early-fusion [quant + news] matrix.
    fb_props = schemas["feature_bundle"]["properties"]
    for required_field in ("universe", "feature_spec", "entries"):
        if required_field not in fb_props:
            raise AssertionError(f"feature_bundle must define '{required_field}'")

    for field in ["requested_action", "approved_action"]:
        enum = set(schemas["risk_decision"]["properties"][field].get("enum", []))
        if not SUPPORTED_ACTIONS.issubset(enum):
            raise AssertionError(f"risk_decision.{field} missing supported action enum values")
    if set(schemas["order_request"]["properties"]["side"].get("enum", [])) != {"BUY", "SELL"}:
        raise AssertionError("order_request.side must enumerate BUY/SELL")

    # V3: sleeve_signal is a POSITION-form sleeve input (target weights + sleeve tag), not a ranking.
    ss_props = schemas["sleeve_signal"]["properties"]
    if "positions" not in ss_props:
        raise AssertionError("V3: sleeve_signal must define a 'positions' array")
    if set(ss_props["sleeve"].get("enum", [])) != SLEEVE_VALUES:
        raise AssertionError(f"sleeve_signal.sleeve must enumerate {sorted(SLEEVE_VALUES)}")

    # V3: risk_book is the combiner output (netted book + risk scalars + limits + hedge).
    rb_props = schemas["risk_book"]["properties"]
    for required_field in ("net_positions", "hedge", "risk_scalars", "limits"):
        if required_field not in rb_props:
            raise AssertionError(f"risk_book must define '{required_field}'")
    if set(rb_props["hedge"]["properties"]["mode"].get("enum", [])) != HEDGE_MODES:
        raise AssertionError(f"risk_book.hedge.mode must enumerate {sorted(HEDGE_MODES)}")


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
    bundle = examples["feature_bundle"]
    signal = examples["aggregated_signal"]
    portfolio = examples["portfolio_snapshot"]
    risk = examples["risk_decision"]
    order = examples["order_request"]
    execution = examples["execution_report"]
    agent = examples["agent_cycle_result"]

    # Only ml_prediction keeps the per-ticker buy/hold/sell vector (V1 contract, still served).
    assert_probability_vector(ml, "probabilities")
    print("ml_prediction: probability vector OK")

    # Per-ticker payloads still carry ticker/timeframe.
    for payload_name, payload in [
        ("candle_batch", candle),
        ("market_snapshot", market),
        ("ml_prediction", ml),
        ("llm_analysis", llm),
    ]:
        if payload["ticker"] != "SBER":
            raise AssertionError(f"{payload_name} ticker mismatch")
        if payload["timeframe"] != "1H":
            raise AssertionError(f"{payload_name} timeframe mismatch")

    # V2: llm_analysis is news features, no-lookahead by publish time (published_at <= as_of).
    as_of = parse_dt(llm["as_of"])
    for src in llm.get("sources", []):
        if parse_dt(src["published_at"]) > as_of:
            raise AssertionError("llm_analysis source published_at is after as_of (lookahead)")
    if "sentiment" not in llm["features"]:
        raise AssertionError("llm_analysis.features must include sentiment")
    print("llm_analysis: news features + no-lookahead OK")

    # V2: feature_bundle entries align with universe and feature_spec (early fusion matrix).
    spec = bundle["feature_spec"]
    bundle_tickers = [e["ticker"] for e in bundle["entries"]]
    if set(bundle_tickers) != set(bundle["universe"]):
        raise AssertionError("feature_bundle entries must cover exactly the universe")
    for entry in bundle["entries"]:
        if len(entry["quant"]) != len(spec["quant_features"]):
            raise AssertionError(f"feature_bundle {entry['ticker']} quant length != quant_features")
        if len(entry["news"]) != len(spec["news_features"]):
            raise AssertionError(f"feature_bundle {entry['ticker']} news length != news_features")
    print("feature_bundle: universe + feature alignment OK")

    # V2: aggregated_signal is a cross-sectional ranking over the universe.
    rank_tickers = [r["ticker"] for r in signal["rankings"]]
    if set(rank_tickers) != set(signal["universe"]):
        raise AssertionError("aggregated_signal rankings must cover exactly the universe")
    ranks = sorted(r["rank"] for r in signal["rankings"])
    if ranks != list(range(1, len(ranks) + 1)):
        raise AssertionError("aggregated_signal ranks must be a 1..N permutation")
    for r in signal["rankings"]:
        if r["leg"] not in LEG_VALUES:
            raise AssertionError(f"aggregated_signal leg invalid: {r['leg']}")
    by_ticker = {r["ticker"]: r for r in signal["rankings"]}
    if "SBER" not in by_ticker:
        raise AssertionError("aggregated_signal universe must include SBER (demo)")
    # Linkage: SBER is the top long leg -> risk_decision opens a LONG (BUY) on SBER.
    sber_leg = by_ticker["SBER"]["leg"]
    if sber_leg == "long":
        if risk.get("position_side") != "LONG":
            raise AssertionError("SBER long leg must map to risk_decision.position_side LONG")
        if risk["requested_action"] not in {"BUY", "BUY_MORE"}:
            raise AssertionError("SBER long leg must map to a BUY-side risk request")
    print("aggregated_signal: cross-sectional ranking + risk linkage OK")

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


def validate_v3_sleeve_contracts(examples: dict[str, dict[str, Any]]) -> None:
    """V3 risk_manager combiner: sleeve_signal (input) + risk_book (combined output) examples."""
    sleeve = examples["sleeve_signal"]
    book = examples["risk_book"]

    # sleeve_signal: research artifact, position-form, sleeve tag in enum.
    if sleeve["is_production"] is not False:
        raise AssertionError("sleeve_signal example must keep is_production=false")
    if sleeve["sleeve"] not in SLEEVE_VALUES:
        raise AssertionError(f"sleeve_signal.sleeve invalid: {sleeve['sleeve']}")
    for p in sleeve["positions"]:
        if p["leg"] not in (LEG_VALUES | {"hedge"}):
            raise AssertionError(f"sleeve_signal position leg invalid: {p['leg']}")
    print("sleeve_signal: position-form sleeve + is_production OK")

    # risk_book: research artifact; limits respected; total_gross = directional + hedge; sides consistent.
    if book["is_production"] is not False:
        raise AssertionError("risk_book example must keep is_production=false")
    lim = book["limits"]
    if not (lim["name_caps_ok"] and lim["sector_caps_ok"] and lim["gross_cap_ok"]):
        raise AssertionError("risk_book example must respect every limit (name/sector/gross)")
    for p in book["net_positions"]:
        if abs(p["weight"]) > lim["max_name_weight"] + 1e-6:
            raise AssertionError(f"risk_book {p['ticker']} exceeds max_name_weight")
        if (p["weight"] > 0) != (p["side"] == "LONG"):
            raise AssertionError(f"risk_book {p['ticker']} side/sign mismatch")
    sec_gross: dict[str, float] = {}
    for p in book["net_positions"]:
        sec_gross[p["sector"]] = sec_gross.get(p["sector"], 0.0) + abs(p["weight"])
    if any(g > lim["max_sector_gross"] + 1e-6 for g in sec_gross.values()):
        raise AssertionError("risk_book per-sector gross exceeds max_sector_gross")
    rs = book["risk_scalars"]
    hedge_gross = sum(abs(leg["weight"]) for leg in book["hedge"]["legs"])
    if not math.isclose(rs["total_gross"], rs["directional_gross"] + hedge_gross, abs_tol=1e-4):
        raise AssertionError("risk_book total_gross != directional_gross + hedge gross")
    if book["hedge"]["mode"] not in HEDGE_MODES:
        raise AssertionError(f"risk_book hedge.mode invalid: {book['hedge']['mode']}")
    # Linkage: the book's sleeve provenance includes the sleeve_signal's sleeve.
    if sleeve["sleeve"] not in {s["sleeve"] for s in book["sleeves"]}:
        raise AssertionError("risk_book.sleeves must record the sleeve_signal provenance")
    print("risk_book: netting + limits + hedge + sleeve linkage OK")


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


def validate_optional_generated_ml_prediction(schema: dict[str, Any]) -> None:
    path = REPO_ROOT / "data" / "reports" / "ml_prediction_example.json"
    if not path.exists():
        return
    payload = load_json(path)
    for field in schema["required"]:
        if field not in payload:
            raise AssertionError(f"generated ml_prediction missing {field}")
    assert_probability_vector(payload, "probabilities")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise AssertionError("generated ml_prediction diagnostics must be an object")
    print("generated ml_prediction example: probability vector OK")


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
    validate_v3_sleeve_contracts(examples)
    validate_supported_tickers()
    validate_optional_generated_ml_prediction(schemas["ml_prediction"])
    print("All contract checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
