"""Evaluation and metrics."""

from .backtest import backtest_strategy, predictions_to_signals, save_backtest_report
from .metrics import compute_classification_metrics, compute_trading_metrics
from .online_eval import (
    evaluate_online_predictions,
    load_predictions,
    save_online_report,
    save_prediction,
)

__all__ = [
    # Metrics
    "compute_classification_metrics",
    "compute_trading_metrics",
    # Backtest
    "predictions_to_signals",
    "backtest_strategy",
    "save_backtest_report",
    # Online evaluation
    "save_prediction",
    "load_predictions",
    "evaluate_online_predictions",
    "save_online_report",
]
