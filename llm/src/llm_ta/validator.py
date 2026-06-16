"""Validate llm_analysis payloads against the frozen V2 contract.

V2 = NEWS FEATURES (no buy/hold/sell). Beyond JSON-schema validation we enforce the
no-lookahead invariant: every sources[].published_at must be <= as_of.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

LLM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = LLM_ROOT / "schemas" / "llm_analysis.schema.json"

REQUIRED_TOP = ("as_of", "ticker", "timeframe", "features", "model_version", "is_production")
REQUIRED_FEATURES = ("sentiment", "impact_score", "novelty", "event_type", "news_count")


class LLMAnalysisValidationError(ValueError):
    pass


def load_schema(schema_path: Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise LLMAnalysisValidationError(f"{schema_path} must contain a JSON object")
    return schema


def parse_strict_json(raw_output: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise LLMAnalysisValidationError(f"LLM output is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMAnalysisValidationError("LLM output must be a JSON object")
    return payload


def validate_analysis(payload: dict[str, Any], schema_path: Path = DEFAULT_SCHEMA_PATH) -> None:
    schema = load_schema(schema_path)
    try:
        import jsonschema
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError:
        _validate_without_jsonschema(payload)
    else:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
        if errors:
            raise LLMAnalysisValidationError("; ".join(error.message for error in errors))
    _validate_no_lookahead(payload)


def _validate_without_jsonschema(payload: dict[str, Any]) -> None:
    missing = set(REQUIRED_TOP) - set(payload)
    if missing:
        raise LLMAnalysisValidationError(f"missing fields: {sorted(missing)}")
    for field in ("as_of", "ticker", "timeframe", "model_version"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise LLMAnalysisValidationError(f"{field} must be a non-empty string")
    if not isinstance(payload["is_production"], bool):
        raise LLMAnalysisValidationError("is_production must be a boolean")
    features = payload["features"]
    if not isinstance(features, dict):
        raise LLMAnalysisValidationError("features must be an object")
    missing_f = set(REQUIRED_FEATURES) - set(features)
    if missing_f:
        raise LLMAnalysisValidationError(f"features missing: {sorted(missing_f)}")
    if "probabilities" in payload or "probabilities" in features:
        raise LLMAnalysisValidationError("V2: llm_analysis must not carry buy/hold/sell probabilities")
    if not _in_range(features["sentiment"], -1, 1):
        raise LLMAnalysisValidationError("features.sentiment must be in [-1, 1]")
    for f in ("impact_score", "novelty"):
        if not _in_range(features[f], 0, 1):
            raise LLMAnalysisValidationError(f"features.{f} must be in [0, 1]")
    if not isinstance(features["event_type"], str) or not features["event_type"]:
        raise LLMAnalysisValidationError("features.event_type must be a non-empty string")
    if not isinstance(features["news_count"], int) or features["news_count"] < 0:
        raise LLMAnalysisValidationError("features.news_count must be a non-negative integer")


def _in_range(value: Any, lo: float, hi: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and lo <= value <= hi


def _parse_dt(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


def _validate_no_lookahead(payload: dict[str, Any]) -> None:
    """Every contributing source must be published at or before as_of."""
    as_of_raw = payload.get("as_of")
    if not isinstance(as_of_raw, str):
        raise LLMAnalysisValidationError("as_of must be a string")
    try:
        as_of = _parse_dt(as_of_raw)
    except ValueError as exc:
        raise LLMAnalysisValidationError(f"as_of is not ISO-8601: {exc}") from exc
    for src in payload.get("sources", []) or []:
        pub = src.get("published_at")
        if pub is None:
            raise LLMAnalysisValidationError("source missing published_at")
        try:
            pub_dt = _parse_dt(pub)
        except ValueError as exc:
            raise LLMAnalysisValidationError(f"source published_at is not ISO-8601: {exc}") from exc
        if pub_dt > as_of:
            raise LLMAnalysisValidationError(
                f"lookahead: source published_at {pub} is after as_of {as_of_raw}")
