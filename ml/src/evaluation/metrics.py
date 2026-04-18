"""Evaluation metrics for classification and trading."""

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "macro"
) -> dict:
    """Compute classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (optional, for log_loss).
        average: Averaging method for F1/Precision/Recall ('macro', 'weighted', etc.).
        
    Returns:
        Dictionary with metrics.
    """
    metrics = {}
    
    # Filter invalid labels (-1)
    valid_mask = y_true != -1
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_true_valid) == 0:
        return {"error": "No valid labels"}
    
    # Accuracy
    metrics["accuracy"] = accuracy_score(y_true_valid, y_pred_valid)
    
    # F1 score
    metrics["macro_f1"] = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)
    
    # Precision
    metrics["macro_precision"] = precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
    metrics["weighted_precision"] = precision_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)
    
    # Recall
    metrics["macro_recall"] = recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
    metrics["weighted_recall"] = recall_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)
    
    # Log loss (if probabilities provided)
    if y_proba is not None:
        y_proba_valid = y_proba[valid_mask]
        try:
            metrics["log_loss"] = log_loss(y_true_valid, y_proba_valid)
        except ValueError:
            metrics["log_loss"] = float("nan")
    
    # Confusion matrix (as list of lists for JSON serialization)
    cm = confusion_matrix(y_true_valid, y_pred_valid)
    metrics["confusion_matrix"] = cm.tolist()
    
    return metrics


def compute_trading_metrics(
    returns: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """Compute simple trading metrics.
    
    Args:
        returns: Array of returns.
        predictions: Predicted class labels (0=down, K-1=up).
        threshold: Probability threshold for trading (if using probabilities).
        
    Returns:
        Dictionary with trading metrics.
    """
    metrics = {}
    
    # Simple strategy: buy when prediction indicates up (class > K//2)
    # This is a placeholder - proper implementation needs actual price data
    n_classes = len(np.unique(predictions))
    buy_threshold = n_classes // 2
    
    # Generate trading signals
    signals = np.where(predictions > buy_threshold, 1, 0)
    
    # Calculate strategy returns (placeholder)
    strategy_returns = signals * returns
    
    # Total PnL
    metrics["total_pnl"] = strategy_returns.sum()
    
    # Win rate
    wins = strategy_returns > 0
    metrics["win_rate"] = wins.mean() if len(wins) > 0 else 0.0
    
    # Sharpe ratio (simplified, annualization not applied)
    if len(strategy_returns) > 0 and strategy_returns.std() > 0:
        metrics["sharpe_ratio"] = strategy_returns.mean() / strategy_returns.std()
    else:
        metrics["sharpe_ratio"] = 0.0
    
    # Max drawdown (simplified)
    cumulative = np.cumsum(strategy_returns)
    if len(cumulative) > 0:
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        metrics["max_drawdown"] = drawdown.max() if len(drawdown) > 0 else 0.0
    else:
        metrics["max_drawdown"] = 0.0
    
    return metrics


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    np.random.seed(42)
    n_samples = 100
    n_classes = 7
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_proba = np.random.rand(n_samples, n_classes)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    # Classification metrics
    cls_metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    print("Classification metrics:")
    for k, v in cls_metrics.items():
        print(f"  {k}: {v}")
    
    # Trading metrics
    returns = np.random.randn(n_samples) * 0.01
    trading_metrics = compute_trading_metrics(returns, y_pred)
    print("\nTrading metrics:")
    for k, v in trading_metrics.items():
        print(f"  {k}: {v}")
