"""FastAPI service for candle prediction inference."""

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from .predictor import CandlePredictor
from .schemas import Candle, ErrorResponse, HealthResponse, PredictRequest, PredictResponse


# Global predictor instance
predictor: Optional[CandlePredictor] = None


def get_predictor() -> CandlePredictor:
    """Get or initialize predictor.
    
    Returns:
        CandlePredictor instance.
        
    Raises:
        HTTPException: If predictor not loaded.
    """
    global predictor
    
    if predictor is None:
        predictor = CandlePredictor()
        predictor.load()
    
    return predictor


# Create FastAPI app
app = FastAPI(
    title="Candle Prediction API",
    description="ML service for candle prediction inference",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        get_predictor()
        print("Model loaded successfully on startup")
    except Exception as e:
        print(f"Failed to load model on startup: {e}")
        # Don't raise - allow service to start, will fail on first request


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.
    
    Returns:
        HealthResponse with service status.
    """
    global predictor
    
    model_loaded = predictor is not None and predictor.is_loaded
    model_version = None
    
    if model_loaded and predictor.metadata:
        model_version = predictor.metadata.get("artifact_version")
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_version,
        timestamp=datetime.utcnow()
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Make prediction on candles.
    
    Args:
        request: PredictRequest with candle data.
        
    Returns:
        PredictResponse with prediction results.
        
    Raises:
        HTTPException: If prediction fails.
    """
    try:
        # Get predictor
        pred = get_predictor()
        
        # Convert Pydantic models to dicts
        candles = [c.model_dump() for c in request.candles]
        
        # Make prediction
        result = pred.predict(candles, model_version=request.model_version)
        
        return PredictResponse(**result)
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifacts not found: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler.
    
    Args:
        request: Request object.
        exc: Exception.
        
    Returns:
        JSONResponse with error details.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
