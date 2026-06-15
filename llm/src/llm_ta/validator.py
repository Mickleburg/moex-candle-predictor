from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


LLM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = LLM_ROOT / "schemas" / "llm_analysis.schema.json"
PROBABILITY_KEYS = ("buy", "hold", "sell")
PROBABILITY_SUM_TOLERANCE = 0.02
TECHNICAL_VIEW_VALUES = {
    "strongly_bullish",
    "moderately_bullish",
    "neutral",
    "moderately_bearish",
    "strongly_bearish",
}


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
        validator = jsonschema.Draft202012Validator(schema, format_checker=FormatChecker())
        errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
        if errors:
            message = "; ".join(error.message for error in errors)
            raise LLMAnalysisValidationError(message)
    _validate_probability_sum(payload)


def _validate_without_jsonschema(payload: dict[str, Any]) -> None:
    required = {
        "ticker",
        "timeframe",
        "as_of",
        "technical_view",
        "probabilities",
        "confidence",
        "key_reasons",
        "risk_notes",
    }
    unknown = set(payload) - required
    if unknown:
        raise LLMAnalysisValidationError(f"unknown fields: {sorted(unknown)}")
    missing = required - set(payload)
    if missing:
        raise LLMAnalysisValidationError(f"missing fields: {sorted(missing)}")

    for field in ("ticker", "timeframe", "as_of"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise LLMAnalysisValidationError(f"{field} must be a non-empty string")

    if payload["technical_view"] not in TECHNICAL_VIEW_VALUES:
        raise LLMAnalysisValidationError("technical_view has an unsupported value")

    probabilities = payload["probabilities"]
    if not isinstance(probabilities, dict):
        raise LLMAnalysisValidationError("probabilities must be an object")
    if set(probabilities) != set(PROBABILITY_KEYS):
        raise LLMAnalysisValidationError("probabilities must contain buy, hold, sell only")
    for key in PROBABILITY_KEYS:
        if not _is_number_between_zero_and_one(probabilities[key]):
            raise LLMAnalysisValidationError(f"probabilities.{key} must be a number between 0 and 1")

    if not _is_number_between_zero_and_one(payload["confidence"]):
        raise LLMAnalysisValidationError("confidence must be a number between 0 and 1")

    for field in ("key_reasons", "risk_notes"):
        if not isinstance(payload[field], list) or not all(isinstance(item, str) for item in payload[field]):
            raise LLMAnalysisValidationError(f"{field} must be an array of strings")


def _is_number_between_zero_and_one(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


def _validate_probability_sum(payload: dict[str, Any]) -> None:
    probabilities = payload.get("probabilities")
    if not isinstance(probabilities, dict):
        raise LLMAnalysisValidationError("probabilities must be an object")
    total = sum(float(probabilities[key]) for key in PROBABILITY_KEYS)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=PROBABILITY_SUM_TOLERANCE):
        raise LLMAnalysisValidationError(f"probabilities sum to {total}, expected approximately 1.0")
