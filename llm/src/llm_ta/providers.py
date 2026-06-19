"""LLM providers for the V2 news-feature block.

Provider-agnostic: the deterministic baseline (features.py) needs no provider. When a
model is wanted we speak the OpenAI-compatible chat API, so the same adapter serves:
  * Positive LLM (internal vLLM gateway) — Gemma / Qwen, structured output, our default.
  * DeepSeek (cheap, planned), local Ollama/vLLM (free), Groq/OpenRouter — same code.

Provider choice is configuration (base_url / model / sampling), never a code change.
vLLM specifics honoured (from the gateway docs):
  * NEVER temperature=0 for Qwen/Gemma (loops) — per-family sampling presets below.
  * structured output via response_format=json_schema (more reliable than json_object).
  * extra_body (e.g. chat_template_kwargs.enable_thinking) is unfolded to top-level.
  * 429/5xx/timeout -> retry with exponential backoff + jitter (fixed-window RPM/TPM).
We do not hard-wire Claude (cost); the emitted contract is identical regardless of model.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import requests

log = logging.getLogger("llm_ta.providers")

POSITIVE_BASE_URL = "https://api-llm.ml.ptsecurity.ru/v1"

NEWS_SYSTEM_PROMPT = (
    "Ты извлекаешь СТРУКТУРИРОВАННЫЕ НОВОСТНЫЕ ПРИЗНАКИ из заголовков корпоративных "
    "раскрытий российского эмитента. Верни СТРОГО JSON-объект и ничего больше. "
    "Не давай торговых рекомендаций (никаких buy/hold/sell)."
)

# Coarse event classes (mirror features._EVENT_RULES) for structured output.
EVENT_TYPES = [
    "dividend", "earnings", "meeting", "m_and_a", "listing", "rating",
    "distress", "sanctions", "guidance", "price_impact", "management", "other", "none",
]

NEWS_FEATURE_SCHEMA: dict[str, Any] = {
    "name": "news_features",
    "schema": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "number", "minimum": -1, "maximum": 1},
            "impact_score": {"type": "number", "minimum": 0, "maximum": 1},
            "novelty": {"type": "number", "minimum": 0, "maximum": 1},
            "event_type": {"type": "string", "enum": EVENT_TYPES},
        },
        "required": ["sentiment", "impact_score", "novelty", "event_type"],
        "additionalProperties": False,
    },
}

# per-family sampling presets (gateway recommendations; never temperature=0)
GEMMA_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
QWEN_INSTRUCT_SAMPLING = {"temperature": 0.7, "top_p": 0.80, "top_k": 20,
                          "min_p": 0.0, "presence_penalty": 1.5}
QWEN_THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                          "min_p": 0.0, "presence_penalty": 1.5}

_RETRY_DELAYS = (5, 10, 20, 40)


class BaseLLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, request_payload: dict[str, Any]) -> str:
        """Return raw model output as text (expected to be strict JSON)."""


@dataclass
class OpenAICompatibleProvider(BaseLLMProvider):
    """Adapter for any OpenAI-compatible /chat/completions endpoint."""

    base_url: str
    model: str
    api_key: str = ""
    timeout_seconds: float = 120.0
    max_tokens: int = 1024
    sampling: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    json_schema: dict[str, Any] | None = None
    max_retries: int = 5
    no_proxy: bool = False  # internal gateways must bypass the corporate proxy
    name: str = "openai-compatible"

    @classmethod
    def from_env(cls) -> "OpenAICompatibleProvider":
        return cls(
            base_url=os.environ.get("LLM_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"),
            model=os.environ.get("LLM_MODEL", "qwen2.5"),
            api_key=os.environ.get("LLM_OPENAI_API_KEY", ""),
            timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "120")),
        )

    def _build_body(self, prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": NEWS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
        }
        body.update(self.sampling)
        if self.extra_body:
            body.update(self.extra_body)  # vLLM expects e.g. chat_template_kwargs top-level
        if self.json_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": self.json_schema}
        else:
            body["response_format"] = {"type": "json_object"}
        return body

    def generate(self, prompt: str, request_payload: dict[str, Any]) -> str:
        del request_payload
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = self._build_body(prompt)
        # internal gateways are reached directly; bypass any env HTTP(S)_PROXY
        proxies = {"http": None, "https": None} if self.no_proxy else None
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, headers=headers, json=body,
                                  timeout=self.timeout_seconds, proxies=proxies)
                if r.status_code == 429 or r.status_code >= 500:
                    delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)] + random.uniform(0, 1.5)
                    log.warning(f"[{self.model}] HTTP {r.status_code} -> retry in {delay:.1f}s "
                                f"({attempt + 1}/{self.max_retries})")
                    last_error = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    time.sleep(delay)
                    continue
                if r.status_code == 401:
                    raise RuntimeError("401 Unauthorized — check POSITIVE_LLM_API_KEY")
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                content = r.json()["choices"][0]["message"]["content"]
                if content is None:  # refusal / context overflow -> transient, retry
                    delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)] + random.uniform(0, 1.5)
                    last_error = RuntimeError("content=None")
                    time.sleep(delay)
                    continue
                return str(content)
            except (requests.Timeout, requests.ConnectionError) as exc:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)] + random.uniform(0, 1.5)
                log.warning(f"[{self.model}] {type(exc).__name__} -> retry in {delay:.1f}s")
                last_error = exc
                time.sleep(delay)
            except RuntimeError:
                raise
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"unexpected response shape: {exc}") from exc

        raise RuntimeError(f"[{self.model}] no response after {self.max_retries} retries; "
                           f"last error: {last_error}")


def _positive(model: str, sampling: dict[str, Any], *, max_tokens: int = 1024,
              extra_body: dict[str, Any] | None = None) -> OpenAICompatibleProvider:
    key = (os.environ.get("POSITIVE_LLM_API_KEY")
           or os.environ.get("API_KEY")
           or os.environ.get("LLM_OPENAI_API_KEY", ""))
    return OpenAICompatibleProvider(
        base_url=os.environ.get("LLM_OPENAI_BASE_URL", POSITIVE_BASE_URL),
        model=model, api_key=key, max_tokens=max_tokens,
        sampling=dict(sampling), extra_body=dict(extra_body or {}),
        json_schema=NEWS_FEATURE_SCHEMA, no_proxy=True, name=f"positive:{model}",
    )


def provider_from_name(name: str) -> BaseLLMProvider | None:
    """Return a provider, or None to use the deterministic baseline (no LLM)."""
    n = name.strip().lower()
    if n in {"baseline", "none", "mock", "deterministic", ""}:
        return None
    # Positive LLM internal gateway (default for real LLM use)
    if n in {"positive", "positive-chat", "gemma"}:
        return _positive("positive-llm-chat", GEMMA_SAMPLING)
    if n in {"positive-qwen", "qwen36", "positive-qwen36"}:
        return _positive("positive-llm-qwen36", QWEN_INSTRUCT_SAMPLING,
                         extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    if n in {"positive-strong", "experimental", "qwen397"}:
        return _positive("positive-llm-experimental", QWEN_THINKING_SAMPLING, max_tokens=4096)
    # DeepSeek (planned cheap fallback)
    if n == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("LLM_OPENAI_API_KEY", ""))
        return OpenAICompatibleProvider(
            base_url=os.environ.get("LLM_OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
            api_key=key, sampling={"temperature": 0.3}, name="deepseek")
    # generic local / OpenAI-compatible (Ollama, vLLM, Groq, OpenRouter)
    if n in {"openai-compatible", "openai_compatible", "local", "ollama"}:
        p = OpenAICompatibleProvider.from_env()
        p.sampling = {"temperature": 0.7}
        return p
    raise ValueError(f"unknown LLM provider: {name}")
