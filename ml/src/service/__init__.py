"""Inference service."""

from .api import app
from .contracts import CandleBatch, build_ml_prediction_response, candle_batch_to_dataframe, load_candle_batch_json
from .model_registry import TickerModelRouter, predict_candle_batch, resolve_artifact_dir
from .predictor import CandlePredictor
from .research_artifact import load_research_artifact, predict_with_artifact
from .schemas import Candle, ErrorResponse, HealthResponse, PredictRequest, PredictResponse

__all__ = [
    # API
    "app",
    # Predictor
    "CandlePredictor",
    # JSON contracts
    "CandleBatch",
    "load_candle_batch_json",
    "candle_batch_to_dataframe",
    "build_ml_prediction_response",
    "load_research_artifact",
    "predict_with_artifact",
    # Per-ticker routing
    "TickerModelRouter",
    "predict_candle_batch",
    "resolve_artifact_dir",
    # Schemas
    "Candle",
    "PredictRequest",
    "PredictResponse",
    "HealthResponse",
    "ErrorResponse",
]
