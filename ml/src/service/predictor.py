"""Prediction class with preprocessing for inference."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..features import CandleTokenizer, build_inference_window, compute_all_indicators
from ..models import LGBMClassifier
from ..utils.io import load_joblib, read_json


class CandlePredictor:
    """Predictor for candle predictions with preprocessing.
    
    Loads model, tokenizer, and metadata from artifacts.
    Performs the same feature engineering as training.
    """
    
    def __init__(self, artifacts_dir: Optional[str] = None):
        """Initialize predictor.
        
        Args:
            artifacts_dir: Path to artifacts directory.
        """
        if artifacts_dir is None:
            self.artifacts_dir = Path(__file__).resolve().parents[2] / "artifacts"
        else:
            path = Path(artifacts_dir)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            self.artifacts_dir = path
        
        self.model: Optional[LGBMClassifier] = None
        self.tokenizer: Optional[CandleTokenizer] = None
        self.metadata: Optional[dict] = None
        
        self.is_loaded = False
    
    def load(self) -> None:
        """Load model, tokenizer, and metadata from artifacts."""
        if self.is_loaded:
            return
        
        # Load metadata
        metadata_path = self.artifacts_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        self.metadata = read_json(metadata_path)
        
        # Load model
        model_path = self.artifacts_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = LGBMClassifier.load(model_path)
        
        # Load tokenizer
        tokenizer_path = self.artifacts_dir / "tokenizer.pkl"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        self.tokenizer = CandleTokenizer.load(tokenizer_path)
        
        self.is_loaded = True
        print(f"Predictor loaded: model={self.model.__class__.__name__}, version={self.metadata.get('artifact_version')}")
    
    def _candles_to_dataframe(self, candles: list) -> pd.DataFrame:
        """Convert list of candle objects to DataFrame.
        
        Args:
            candles: List of candle dictionaries or objects.
            
        Returns:
            DataFrame with normalized column names.
        """
        # Convert to list of dicts if needed
        if hasattr(candles[0], 'model_dump'):
            # Pydantic model
            candle_dicts = [c.model_dump() for c in candles]
        elif hasattr(candles[0], '__dict__'):
            # Dataclass or object
            candle_dicts = [c.__dict__ for c in candles]
        else:
            # Already dicts
            candle_dicts = candles
        
        df = pd.DataFrame(candle_dicts)
        
        # Normalize column names
        column_mapping = {
            "open": "open",
            "Open": "open",
            "high": "high",
            "High": "high",
            "low": "low",
            "Low": "low",
            "close": "close",
            "Close": "close",
            "volume": "volume",
            "Volume": "volume",
            "begin": "begin",
            "Begin": "begin",
            "datetime": "begin",
            "time": "begin",
            "ticker": "ticker",
            "Ticker": "ticker",
            "timeframe": "timeframe",
            "Timeframe": "timeframe",
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure datetime
        if "begin" in df.columns:
            df["begin"] = pd.to_datetime(df["begin"])

        # Keep inference schema aligned with training raw data.
        if "value" not in df.columns and {"close", "volume"}.issubset(df.columns):
            df["value"] = df["close"] * df["volume"]
        
        return df
    
    def _compute_action(self, prediction: int, n_classes: int) -> str:
        """Convert prediction to trading action.
        
        Args:
            prediction: Predicted class.
            n_classes: Number of classes.
            
        Returns:
            Action string: "buy", "sell", or "hold".
        """
        mid_point = n_classes // 2
        
        if prediction > mid_point:
            return "buy"
        elif prediction < mid_point:
            return "sell"
        else:
            return "hold"
    
    def predict(
        self,
        candles: list,
        model_version: Optional[str] = None
    ) -> dict:
        """Make prediction on candles.
        
        Args:
            candles: List of candle data (dicts or Pydantic models).
            model_version: Optional model version (currently ignored, uses loaded model).
            
        Returns:
            Dictionary with prediction results.
            
        Raises:
            ValueError: If insufficient candles or predictor not loaded.
        """
        if not self.is_loaded:
            raise ValueError("Predictor not loaded. Call load() first.")
        
        # Convert to DataFrame
        df = self._candles_to_dataframe(candles)
        
        # Check minimum candles
        window_size = self.metadata.get("L", 32)
        if len(df) < window_size:
            raise ValueError(
                f"Insufficient candles: {len(df)} provided, need at least {window_size}"
            )
        
        # Compute features (same as training)
        features_df = compute_all_indicators(df)
        
        # Transform with fitted tokenizer
        tokens = self.tokenizer.transform(features_df)
        
        # Build inference window
        X = build_inference_window(features_df, tokens, window_size=window_size)
        
        # Flatten for tabular model
        X_flat = X.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(X_flat)[0]
        probabilities = self.model.predict_proba(X_flat)[0]
        
        # Compute action
        n_classes = self.metadata.get("K", 7)
        action = self._compute_action(prediction, n_classes)
        confidence = float(probabilities.max())
        
        # Get ticker and timeframe from last candle
        ticker = df["ticker"].iloc[-1] if "ticker" in df.columns else "SBER"
        timeframe = df["timeframe"].iloc[-1] if "timeframe" in df.columns else "1H"
        
        return {
            "predicted_token": int(prediction),
            "probabilities": probabilities.tolist(),
            "action": action,
            "confidence": confidence,
            "model_version": self.metadata.get("artifact_version", "unknown"),
            "ticker": ticker,
            "timeframe": timeframe,
            "timestamp": datetime.utcnow().isoformat(),
            "n_candles_used": len(df),
            "diagnostics": {
                "window_size": window_size,
                "n_classes": n_classes,
                "horizon": self.metadata.get("horizon", 3),
            }
        }


if __name__ == "__main__":
    # Example usage with mock data
    from datetime import datetime, timedelta
    
    predictor = CandlePredictor()
    
    # Generate mock candles
    n_candles = 50
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    base_price = 250.0
    
    candles = []
    for i in range(n_candles):
        price = base_price + np.random.randn() * 0.5
        candles.append({
            "begin": base_time + timedelta(hours=i),
            "open": price,
            "high": price + abs(np.random.randn() * 0.2),
            "low": price - abs(np.random.randn() * 0.2),
            "close": price + np.random.randn() * 0.1,
            "volume": np.random.randint(100000, 500000),
            "ticker": "SBER",
            "timeframe": "1H"
        })
    
    print(f"Generated {len(candles)} mock candles")
    
    # Note: This will fail if artifacts don't exist
    # predictor.load()
    # result = predictor.predict(candles)
    # print(result)
