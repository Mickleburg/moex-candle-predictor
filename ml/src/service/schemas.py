"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Candle(BaseModel):
    """Single candle data."""
    
    begin: datetime = Field(..., description="Candle start time")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: float = Field(..., description="Volume")
    ticker: str = Field(default="SBER", description="Ticker symbol")
    timeframe: str = Field(default="1H", description="Timeframe")


class PredictRequest(BaseModel):
    """Request for prediction."""
    
    candles: list[Candle] = Field(
        ...,
        description="List of recent candles (must have at least L=32 candles)"
    )
    model_version: Optional[str] = Field(
        None,
        description="Optional model version to use (default: latest)"
    )


class PredictResponse(BaseModel):
    """Response with prediction results."""
    
    predicted_token: int = Field(..., description="Predicted token class (0 to K-1)")
    probabilities: list[float] = Field(..., description="Class probabilities")
    action: str = Field(..., description="Trading action: buy, sell, or hold")
    confidence: float = Field(..., description="Confidence of prediction (max probability)")
    model_version: str = Field(..., description="Model version used")
    ticker: str = Field(..., description="Ticker symbol")
    timeframe: str = Field(..., description="Timeframe")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    n_candles_used: int = Field(..., description="Number of candles used for prediction")
    diagnostics: Optional[dict] = Field(None, description="Optional diagnostic information")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Model version")
    timestamp: datetime = Field(..., description="Current time")


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
