"""Online evaluation for live predictions."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.io import write_parquet


def save_prediction(
    prediction: int,
    probabilities: np.ndarray,
    timestamp: datetime,
    model_version: str,
    output_path: str
) -> None:
    """Save a single prediction to file.
    
    Args:
        prediction: Predicted class label.
        probabilities: Probability distribution.
        timestamp: Prediction timestamp.
        model_version: Model version identifier.
        output_path: Path to save prediction (CSV or Parquet).
    """
    # Create record
    record = {
        "timestamp": timestamp,
        "prediction": prediction,
        "model_version": model_version,
    }
    
    # Add probabilities as separate columns
    for i, prob in enumerate(probabilities):
        record[f"prob_class_{i}"] = prob
    
    df = pd.DataFrame([record])
    
    # Append to file or create new
    output_path = Path(output_path)
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    write_parquet(df, output_path)
    print(f"Prediction saved to {output_path}")


def load_predictions(output_path: str) -> pd.DataFrame:
    """Load saved predictions from file.
    
    Args:
        output_path: Path to predictions file.
        
    Returns:
        DataFrame with predictions.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {output_path}")
    
    df = pd.read_parquet(output_path)
    return df


def evaluate_online_predictions(
    predictions_path: str,
    actual_data: pd.DataFrame,
    horizon: int = 3
) -> dict:
    """Evaluate online predictions against actual outcomes.
    
    Args:
        predictions_path: Path to saved predictions file.
        actual_data: DataFrame with actual candle data.
        horizon: Prediction horizon (number of candles ahead).
        
    Returns:
        Dictionary with online evaluation metrics.
    """
    # Load predictions
    pred_df = load_predictions(predictions_path)
    
    if len(pred_df) == 0:
        return {"error": "No predictions found"}
    
    # Merge predictions with actual data
    # Align by timestamp (prediction time vs actual outcome time)
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    actual_data["begin"] = pd.to_datetime(actual_data["begin"])
    
    # Find actual outcome for each prediction
    results = []
    for _, pred_row in pred_df.iterrows():
        pred_time = pred_row["timestamp"]
        pred_class = pred_row["prediction"]
        
        # Find actual candle at horizon
        future_time = pred_time + pd.Timedelta(hours=horizon)
        actual_row = actual_data[actual_data["begin"] == future_time]
        
        if len(actual_row) == 0:
            continue
        
        actual_close = actual_row["close"].iloc[0]
        pred_close = actual_data.loc[actual_data["begin"] == pred_time, "close"].iloc[0]
        
        # Calculate actual return
        actual_return = (actual_close - pred_close) / pred_close
        
        # Determine actual class (simplified: positive return = up)
        actual_class = 1 if actual_return > 0 else 0
        
        results.append({
            "prediction": pred_class,
            "actual_class": actual_class,
            "actual_return": actual_return,
        })
    
    if len(results) == 0:
        return {"error": "No matching predictions found"}
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    accuracy = (results_df["prediction"] == results_df["actual_class"]).mean()
    
    # Calculate PnL (simple: buy if prediction > threshold)
    buy_mask = results_df["prediction"] > 3  # Assuming 7 classes, buy if > 3
    strategy_returns = np.where(buy_mask, results_df["actual_return"], 0)
    total_pnl = strategy_returns.sum()
    
    # Hit rate
    profitable_trades = strategy_returns > 0
    hit_rate = profitable_trades.mean() if len(profitable_trades) > 0 else 0.0
    
    return {
        "n_predictions": len(results_df),
        "accuracy": accuracy,
        "total_pnl": total_pnl,
        "hit_rate": hit_rate,
        "n_profitable_trades": profitable_trades.sum(),
    }


def save_online_report(
    metrics: dict,
    output_path: str
) -> None:
    """Save online evaluation report.
    
    Args:
        metrics: Online evaluation metrics.
        output_path: Path to save report (JSON format).
    """
    import json
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Online report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate mock predictions
    n_predictions = 10
    n_classes = 7
    
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    
    for i in range(n_predictions):
        prediction = np.random.randint(0, n_classes)
        probabilities = np.random.rand(n_classes)
        probabilities = probabilities / probabilities.sum()
        timestamp = base_time + timedelta(hours=i)
        
        save_prediction(
            prediction,
            probabilities,
            timestamp,
            "model_v1",
            "../../data/predictions/online_predictions.parquet"
        )
    
    print(f"Saved {n_predictions} mock predictions")
