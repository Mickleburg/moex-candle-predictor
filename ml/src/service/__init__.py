"""Inference service."""

from .api import app
from .contracts import CandleBatch, build_ml_prediction_response, candle_batch_to_dataframe, load_candle_batch_json
from .predictor import CandlePredictor
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
    # Schemas
    "Candle",
    "PredictRequest",
    "PredictResponse",
    "HealthResponse",
    "ErrorResponse",
]
