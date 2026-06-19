"""Orchestrates V2 news-feature extraction into the frozen llm_analysis contract.

Always computes the deterministic baseline (features.py). If an LLM provider is
supplied, it refines sentiment/novelty/event_type from the window's disclosure titles;
any provider error falls back to the baseline. Output is always schema-valid and
no-lookahead clean. Research artefact => is_production=false.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from . import features as feat
from .providers import BaseLLMProvider
from .validator import DEFAULT_SCHEMA_PATH, parse_strict_json, validate_analysis

LLM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = LLM_ROOT / "prompts" / "news_features_prompt.txt"


def _coerce_float(value: Any, lo: float, hi: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(lo, min(hi, float(value)))


class NewsFeatureService:
    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        prompt_path: Path = DEFAULT_PROMPT_PATH,
        schema_path: Path = DEFAULT_SCHEMA_PATH,
        window_hours: int = feat.DEFAULT_WINDOW_HOURS,
    ) -> None:
        self.provider = provider
        self.prompt_path = prompt_path
        self.schema_path = schema_path
        self.window_hours = window_hours

    def analyze(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        ticker = str(request_payload["ticker"])
        as_of = feat.to_msk(dt.datetime.fromisoformat(str(request_payload["as_of"])))
        timeframe = str(request_payload.get("timeframe", "1H"))
        window_hours = int(request_payload.get("window_hours", self.window_hours))

        disclosures = feat.load_disclosures(ticker)
        analysis = feat.build_analysis(
            ticker=ticker, as_of=as_of, timeframe=timeframe,
            window_hours=window_hours, disclosures=disclosures)

        if self.provider is not None:
            window = [d for d in disclosures
                      if as_of - dt.timedelta(hours=window_hours) <= d.pub_date <= as_of]
            if window:
                analysis = self._refine_with_llm(analysis, window)

        validate_analysis(analysis, self.schema_path)
        return analysis

    def _refine_with_llm(self, analysis: dict[str, Any], window: list[feat.Disclosure]) -> dict[str, Any]:
        try:
            titles = "\n".join(f"- [{d.pub_date.date()}] {d.event_name}" for d in window[-40:])
            template = self.prompt_path.read_text(encoding="utf-8")
            prompt = template.replace("{{TICKER}}", analysis["ticker"]).replace("{{DISCLOSURES}}", titles)
            raw = self.provider.generate(prompt=prompt, request_payload=analysis)
            parsed = parse_strict_json(raw)
        except Exception:
            return analysis  # baseline already valid

        merged = dict(analysis["features"])
        s = _coerce_float(parsed.get("sentiment"), -1.0, 1.0)
        if s is not None:
            merged["sentiment"] = round(s, 4)
        n = _coerce_float(parsed.get("novelty"), 0.0, 1.0)
        if n is not None:
            merged["novelty"] = round(n, 4)
        i = _coerce_float(parsed.get("impact_score"), 0.0, 1.0)
        if i is not None:
            merged["impact_score"] = round(i, 4)
        if isinstance(parsed.get("event_type"), str) and parsed["event_type"].strip():
            merged["event_type"] = parsed["event_type"].strip()

        refined = dict(analysis)
        refined["features"] = merged
        refined["model_version"] = f"edisc_llm_{getattr(self.provider, 'model', self.provider.name)}"
        refined["is_production"] = False  # research artefact until forward gate + sign-off
        return refined
