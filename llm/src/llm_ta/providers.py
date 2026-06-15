from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class BaseLLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, request_payload: dict[str, Any]) -> str:
        """Return raw model output as text."""


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_near_resistance(last_close: float | None, resistance: float | None) -> bool:
    if last_close is None or resistance is None or last_close <= 0:
        return False
    distance = (resistance - last_close) / last_close
    return 0 <= distance <= 0.02


def _build_mock_analysis(request_payload: dict[str, Any]) -> dict[str, Any]:
    snapshot = request_payload.get("technical_snapshot", {})
    if not isinstance(snapshot, dict):
        snapshot = {}

    trend_regime = str(snapshot.get("trend_regime", "")).lower()
    volatility_regime = str(snapshot.get("volatility_regime", "")).lower()
    last_close = _as_float(snapshot.get("last_close"))
    ema_20 = _as_float(snapshot.get("ema_20"))
    volume_zscore = _as_float(snapshot.get("volume_zscore"))
    nearest_resistance = _as_float(snapshot.get("nearest_resistance"))

    if trend_regime == "up" and last_close is not None and ema_20 is not None and last_close > ema_20:
        technical_view = "moderately_bullish"
        probabilities = {"buy": 0.42, "hold": 0.38, "sell": 0.20}
        confidence = 0.55
        key_reasons = ["trend regime is up", "price above EMA20"]
    elif trend_regime == "down" and last_close is not None and ema_20 is not None and last_close < ema_20:
        technical_view = "moderately_bearish"
        probabilities = {"buy": 0.20, "hold": 0.38, "sell": 0.42}
        confidence = 0.55
        key_reasons = ["trend regime is down", "price below EMA20"]
    else:
        technical_view = "neutral"
        probabilities = {"buy": 0.20, "hold": 0.60, "sell": 0.20}
        confidence = 0.35
        key_reasons = ["mixed technical conditions"]

    risk_notes: list[str] = []
    if volume_zscore is not None and volume_zscore > 1.0:
        key_reasons.append("volume above average")
    if _is_near_resistance(last_close, nearest_resistance):
        risk_notes.append("near resistance")
    if volatility_regime == "high":
        risk_notes.append("high volatility regime")
    if not risk_notes:
        risk_notes.append("no major technical risk flags")

    return {
        "ticker": str(request_payload.get("ticker", "")),
        "timeframe": str(request_payload.get("timeframe", "")),
        "as_of": str(request_payload.get("as_of", "")),
        "technical_view": technical_view,
        "probabilities": probabilities,
        "confidence": confidence,
        "key_reasons": key_reasons,
        "risk_notes": risk_notes,
    }


class MockProvider(BaseLLMProvider):
    name = "mock"

    def generate(self, prompt: str, request_payload: dict[str, Any]) -> str:
        del prompt
        return json.dumps(_build_mock_analysis(request_payload), ensure_ascii=False)


@dataclass(frozen=True)
class OpenAICompatibleProvider(BaseLLMProvider):
    """Adapter for local OpenAI-compatible endpoints such as vLLM or Ollama /v1."""

    base_url: str
    model: str
    api_key: str = ""
    timeout_seconds: float = 30.0
    name: str = "openai-compatible"

    @classmethod
    def from_env(cls) -> "OpenAICompatibleProvider":
        return cls(
            base_url=os.environ.get("LLM_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"),
            model=os.environ.get("LLM_MODEL", "llama3.1"),
            api_key=os.environ.get("LLM_OPENAI_API_KEY", ""),
            timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "30")),
        )

    def generate(self, prompt: str, request_payload: dict[str, Any]) -> str:
        del request_payload
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only strict JSON. Do not trade or make final trading decisions.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"openai-compatible provider failed: {exc}") from exc

        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("openai-compatible provider returned an unexpected response shape") from exc


def provider_from_name(name: str) -> BaseLLMProvider:
    normalized = name.strip().lower()
    if normalized == "mock":
        return MockProvider()
    if normalized in {"openai-compatible", "openai_compatible", "local"}:
        return OpenAICompatibleProvider.from_env()
    raise ValueError(f"unknown LLM provider: {name}")
