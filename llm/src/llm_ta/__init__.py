"""LLM news-feature block (V2).

Extracts structured NEWS FEATURES (sentiment, impact, novelty, event_type, ...) for a
ticker at a decision time, for early fusion into the cross-sectional decision model.
It never returns buy/hold/sell and never makes the trading decision.
"""

from . import features
from .analyzer import NewsFeatureService
from .providers import BaseLLMProvider, OpenAICompatibleProvider, provider_from_name
from .validator import LLMAnalysisValidationError, parse_strict_json, validate_analysis

__all__ = [
    "BaseLLMProvider",
    "LLMAnalysisValidationError",
    "NewsFeatureService",
    "OpenAICompatibleProvider",
    "features",
    "parse_strict_json",
    "provider_from_name",
    "validate_analysis",
]
