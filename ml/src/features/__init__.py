"""Feature engineering utilities."""

from .indicators import (
    compute_all_indicators,
    compute_ema,
    compute_ema_distance,
    compute_rolling_volatility,
    compute_time_features,
    compute_volume_ratio,
    compute_atr,
    compute_candle_features,
    compute_returns,
)
from .tokenizer import CandleTokenizer
from .windows import (
    build_inference_window,
    build_sequence_windows,
    build_tabular_windows,
    build_token_windows,
    resolve_feature_columns,
)

__all__ = [
    # Indicators
    "compute_returns",
    "compute_atr",
    "compute_rolling_volatility",
    "compute_volume_ratio",
    "compute_ema",
    "compute_candle_features",
    "compute_ema_distance",
    "compute_time_features",
    "compute_all_indicators",
    # Tokenizer
    "CandleTokenizer",
    # Windows
    "build_tabular_windows",
    "build_sequence_windows",
    "build_token_windows",
    "build_inference_window",
    "resolve_feature_columns",
]
