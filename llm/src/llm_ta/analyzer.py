from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .providers import BaseLLMProvider
from .validator import LLMAnalysisValidationError, parse_strict_json, validate_analysis


LLM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = LLM_ROOT / "prompts" / "technical_analysis_prompt.txt"
DEFAULT_SCHEMA_PATH = LLM_ROOT / "schemas" / "llm_analysis.schema.json"


def build_prompt(request_payload: dict[str, Any], prompt_path: Path = DEFAULT_PROMPT_PATH) -> str:
    template = prompt_path.read_text(encoding="utf-8")
    snapshot_json = json.dumps(request_payload, ensure_ascii=False, indent=2, sort_keys=True)
    return template.replace("{{TECHNICAL_SNAPSHOT_JSON}}", snapshot_json)


def fallback_analysis(request_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": str(request_payload.get("ticker", "")),
        "timeframe": str(request_payload.get("timeframe", "")),
        "as_of": str(request_payload.get("as_of", "")),
        "technical_view": "neutral",
        "probabilities": {
            "buy": 0.0,
            "hold": 1.0,
            "sell": 0.0,
        },
        "confidence": 0.0,
        "key_reasons": ["LLM output was invalid"],
        "risk_notes": ["fallback response used"],
    }


class TechnicalAnalysisService:
    def __init__(
        self,
        provider: BaseLLMProvider,
        prompt_path: Path = DEFAULT_PROMPT_PATH,
        schema_path: Path = DEFAULT_SCHEMA_PATH,
    ) -> None:
        self.provider = provider
        self.prompt_path = prompt_path
        self.schema_path = schema_path

    def analyze(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        try:
            prompt = build_prompt(request_payload, self.prompt_path)
            raw_output = self.provider.generate(prompt=prompt, request_payload=request_payload)
            analysis = parse_strict_json(raw_output)
            validate_analysis(analysis, self.schema_path)
            return analysis
        except Exception:
            fallback = fallback_analysis(request_payload)
            try:
                validate_analysis(fallback, self.schema_path)
            except LLMAnalysisValidationError:
                # If required request fields are absent, still return a safe object
                # rather than exposing an invalid provider response.
                pass
            return fallback
