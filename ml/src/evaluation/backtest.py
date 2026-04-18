"""Backtesting utilities for trading strategies."""

from typing import Optional

import numpy as np
import pandas as pd


def predictions_to_signals(
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    buy_threshold: float = 0.5,
    sell_threshold: float = 0.5,
    n_classes: int = 7
) -> np.ndarray:
    """Convert model predictions to trading signals.
    
    Simple strategy:
    - If prediction indicates up (class > n_classes // 2): buy (1)
    - If prediction indicates down (class < n_classes // 2): sell (-1)
    - Otherwise: hold (0)
    
    Args:
        predictions: Predicted class labels.
        probabilities: Predicted probabilities (optional, for confidence-based signals).
        buy_threshold: Threshold for buy signal (if using probabilities).
        sell_threshold: Threshold for sell signal (if using probabilities).
        n_classes: Number of classes.
        
    Returns:
        Array of signals: 1 (buy), -1 (sell), 0 (hold).
    """
    signals = np.zeros(len(predictions), dtype=int)
    
    # Simple threshold based on class label
    mid_point = n_classes // 2
    
    # Buy signal for up predictions
    buy_mask = predictions > mid_point
    signals[buy_mask] = 1
    
    # Sell signal for down predictions
    sell_mask = predictions < mid_point
    signals[sell_mask] = -1
    
    return signals


def backtest_strategy(
    df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    initial_capital: float = 100000.0,
    commission: float = 0.0005,
    position_size: float = 1.0,
    buy_threshold: float = 0.5,
    sell_threshold: float = 0.5,
    n_classes: int = 7
) -> dict:
    """Run backtest on predictions.
    
    Args:
        df: DataFrame with candle data (must have 'close' column).
        predictions: Predicted class labels.
        probabilities: Predicted probabilities (optional).
        initial_capital: Starting capital.
        commission: Transaction cost per trade (e.g., 0.0005 = 0.05%).
        position_size: Fraction of capital to use per trade (0-1).
        buy_threshold: Threshold for buy signal.
        sell_threshold: Threshold for sell signal.
        n_classes: Number of classes.
        
    Returns:
        Dictionary with backtest results and equity curve.
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must have 'close' column")
    
    # Ensure predictions match data length
    if len(predictions) != len(df):
        raise ValueError(f"Predictions length {len(predictions)} != DataFrame length {len(df)}")
    
    # Convert predictions to signals
    signals = predictions_to_signals(
        predictions, probabilities, buy_threshold, sell_threshold, n_classes
    )
    
    # Calculate returns
    close_prices = df["close"].values
    returns = np.diff(close_prices) / close_prices[:-1]
    
    # Pad returns to match signals length (first return is NaN)
    returns = np.concatenate([[0.0], returns])
    
    # Calculate strategy returns (signal * return - commission)
    # Apply commission on trade execution
    trade_costs = np.abs(np.diff(signals, prepend=0)) * commission
    strategy_returns = (signals * returns) - trade_costs
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + strategy_returns)
    
    # Calculate equity curve
    equity = initial_capital * cumulative_returns
    
    # Calculate metrics
    total_pnl = equity[-1] - initial_capital
    total_return = total_pnl / initial_capital
    
    # Sharpe ratio (simplified, not annualized)
    if len(strategy_returns) > 1 and strategy_returns.std() > 0:
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std()
    else:
        sharpe_ratio = 0.0
    
    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max
    max_drawdown = drawdown.max()
    
    # Number of trades
    n_trades = np.sum(np.abs(np.diff(signals, prepend=0)) > 0)
    
    # Win rate
    profitable_trades = strategy_returns > 0
    win_rate = profitable_trades.mean() if len(profitable_trades) > 0 else 0.0
    
    # Hit rate (non-zero return trades)
    non_zero_returns = strategy_returns != 0
    if non_zero_returns.sum() > 0:
        hit_rate = (strategy_returns[non_zero_returns] > 0).mean()
    else:
        hit_rate = 0.0
    
    return {
        "total_pnl": total_pnl,
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "hit_rate": hit_rate,
        "final_equity": equity[-1],
        "equity_curve": equity.tolist(),
        "signals": signals.tolist(),
        "strategy_returns": strategy_returns.tolist(),
    }


def save_backtest_report(
    results: dict,
    output_path: str
) -> None:
    """Save backtest report to file.
    
    Args:
        results: Backtest results dictionary.
        output_path: Path to save report (JSON format).
    """
    import json
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create summary without large arrays
    summary = {
        "total_pnl": results["total_pnl"],
        "total_return": results["total_return"],
        "sharpe_ratio": results["sharpe_ratio"],
        "max_drawdown": results["max_drawdown"],
        "n_trades": results["n_trades"],
        "win_rate": results["win_rate"],
        "hit_rate": results["hit_rate"],
        "final_equity": results["final_equity"],
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Backtest report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    n_samples = 100
    n_classes = 7
    
    # Generate mock data
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="1H")
    close_prices = 250 + np.cumsum(np.random.randn(n_samples) * 0.5)
    
    df = pd.DataFrame({
        "close": close_prices,
        "begin": dates,
    })
    
    # Generate mock predictions
    predictions = np.random.randint(0, n_classes, n_samples)
    probabilities = np.random.rand(n_samples, n_classes)
    probabilities = probabilities / probabilities.sum(axis=1, keepaxes=True)
    
    # Run backtest
    results = backtest_strategy(
        df,
        predictions,
        probabilities,
        initial_capital=100000.0,
        commission=0.0005,
        n_classes=n_classes
    )
    
    print("Backtest results:")
    for k, v in results.items():
        if k not in ["equity_curve", "signals", "strategy_returns"]:
            print(f"  {k}: {v}")
