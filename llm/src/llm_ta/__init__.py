"""LLM technical-analysis block.

This package returns structured technical analysis only. It never executes
orders and never makes the final trading decision.
"""

from .analyzer import TechnicalAnalysisService, build_prompt, fallback_analysis
from .providers import BaseLLMProvider, MockProvider, OpenAICompatibleProvider, provider_from_name
from .validator import LLMAnalysisValidationError, parse_strict_json, validate_analysis

__all__ = [
    "BaseLLMProvider",
    "LLMAnalysisValidationError",
    "MockProvider",
    "OpenAICompatibleProvider",
    "TechnicalAnalysisService",
    "build_prompt",
    "fallback_analysis",
    "parse_strict_json",
    "provider_from_name",
    "validate_analysis",
]
