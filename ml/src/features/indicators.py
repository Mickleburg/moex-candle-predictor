"""Technical indicators for candle data."""

from typing import Optional

import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame, periods: Optional[list[int]] = None) -> pd.DataFrame:
    """Compute close-to-close returns for multiple periods.
    
    Args:
        df: DataFrame with 'close' column.
        periods: List of periods for return calculation.
        
    Returns:
        DataFrame with return columns (e.g., 'return_1', 'return_3').
    """
    df = df.copy()

    if periods is None:
        periods = [1]
    
    if "close" not in df.columns:
        raise ValueError("DataFrame must have 'close' column")
    
    for period in periods:
        col_name = f"return_{period}"
        df[col_name] = df["close"].pct_change(period)
    
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR).
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        period: ATR period.
        
    Returns:
        Series with ATR values.
    """
    df = df.copy()
    
    required_cols = ["high", "low", "close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # True Range components
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def compute_rolling_volatility(
    df: pd.DataFrame,
    window: int = 20,
    return_col: str = "return_1"
) -> pd.Series:
    """Compute rolling volatility of returns.
    
    Args:
        df: DataFrame with return column.
        window: Rolling window size.
        return_col: Name of return column to use.
        
    Returns:
        Series with rolling volatility (std of returns).
    """
    df = df.copy()
    
    if return_col not in df.columns:
        raise ValueError(f"Return column '{return_col}' not found")
    
    volatility = df[return_col].rolling(window=window).std()
    
    return volatility


def compute_volume_ratio(
    df: pd.DataFrame,
    window: int = 20
) -> pd.Series:
    """Compute volume ratio to rolling mean.
    
    Args:
        df: DataFrame with 'volume' column.
        window: Rolling window size.
        
    Returns:
        Series with volume ratio (current / rolling_mean).
    """
    df = df.copy()
    
    if "volume" not in df.columns:
        raise ValueError("DataFrame must have 'volume' column")
    
    rolling_mean = df["volume"].shift(1).rolling(window=window).mean()
    volume_ratio = df["volume"] / rolling_mean
    
    return volume_ratio


def compute_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute Exponential Moving Average.
    
    Args:
        df: DataFrame with 'close' column.
        period: EMA period.
        
    Returns:
        Series with EMA values.
    """
    df = df.copy()
    
    if "close" not in df.columns:
        raise ValueError("DataFrame must have 'close' column")
    
    ema = df["close"].ewm(span=period, adjust=False).mean()
    
    return ema


def compute_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute candle body, range, and wick ratios.
    
    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns.
        
    Returns:
        DataFrame with additional candle feature columns.
    """
    df = df.copy()
    
    required_cols = ["open", "high", "low", "close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Candle body (absolute difference between open and close)
    df["body"] = (df["close"] - df["open"]).abs()
    
    # Candle range (high - low)
    df["range"] = df["high"] - df["low"]
    
    # Upper wick (distance from high to max(open, close))
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    
    # Lower wick (distance from min(open, close) to low)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    
    # Body ratio (body / range)
    df["body_ratio"] = np.where(
        df["range"] > 0,
        df["body"] / df["range"],
        0.0
    )
    
    # Upper wick ratio
    df["upper_wick_ratio"] = np.where(
        df["range"] > 0,
        df["upper_wick"] / df["range"],
        0.0
    )
    
    # Lower wick ratio
    df["lower_wick_ratio"] = np.where(
        df["range"] > 0,
        df["lower_wick"] / df["range"],
        0.0
    )
    
    return df


def compute_ema_distance(
    df: pd.DataFrame,
    periods: list[int] = [9, 20, 50]
) -> pd.DataFrame:
    """Compute distance of price from EMAs.
    
    Args:
        df: DataFrame with 'close' column.
        periods: List of EMA periods.
        
    Returns:
        DataFrame with EMA distance columns (e.g., 'ema_9_distance').
    """
    df = df.copy()
    
    if "close" not in df.columns:
        raise ValueError("DataFrame must have 'close' column")
    
    for period in periods:
        ema = compute_ema(df, period)
        col_name = f"ema_{period}_distance"
        df[col_name] = (df["close"] - ema) / ema
    
    return df


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute time-based features from timestamp.
    
    Args:
        df: DataFrame with 'begin' column (datetime).
        
    Returns:
        DataFrame with time feature columns.
    """
    df = df.copy()
    
    if "begin" not in df.columns:
        raise ValueError("DataFrame must have 'begin' column")
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df["begin"]):
        df["begin"] = pd.to_datetime(df["begin"])
    
    # Hour of day (0-23)
    df["hour"] = df["begin"].dt.hour
    
    # Day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = df["begin"].dt.dayofweek
    
    # Month (1-12)
    df["month"] = df["begin"].dt.month
    
    return df


def compute_all_indicators(
    df: pd.DataFrame,
    return_periods: list[int] = [1, 3, 5],
    atr_period: int = 14,
    volatility_window: int = 20,
    volume_window: int = 20,
    ema_periods: list[int] = [9, 20, 50]
) -> pd.DataFrame:
    """Compute all technical indicators.
    
    Args:
        df: DataFrame with OHLCV columns.
        return_periods: Periods for return calculation.
        atr_period: ATR period.
        volatility_window: Rolling volatility window.
        volume_window: Volume ratio window.
        ema_periods: EMA periods.
        
    Returns:
        DataFrame with all indicator columns.
    """
    df = df.copy()
    
    # Compute returns first (needed for volatility)
    df = compute_returns(df, periods=return_periods)
    
    # Compute ATR
    df["atr"] = compute_atr(df, period=atr_period)
    
    # Compute rolling volatility
    df["rolling_volatility"] = compute_rolling_volatility(
        df, window=volatility_window, return_col=f"return_{return_periods[0]}"
    )
    
    # Compute volume ratio
    df["volume_ratio"] = compute_volume_ratio(df, window=volume_window)
    
    # Compute candle features
    df = compute_candle_features(df)
    
    # Compute EMA distances
    df = compute_ema_distance(df, periods=ema_periods)
    
    # Compute time features
    df = compute_time_features(df)
    
    return df


if __name__ == "__main__":
    # Example usage with mock data
    from ..data.fixtures import generate_mock_candles
    
    # Generate mock data
    mock_df = generate_mock_candles(n=100, ticker="SBER", timeframe="1H", seed=42)
    print(f"Generated {len(mock_df)} mock candles")
    print(mock_df.head())
    
    # Compute all indicators
    indicators_df = compute_all_indicators(mock_df)
    print(f"\nIndicators computed, shape: {indicators_df.shape}")
    print(f"Columns: {indicators_df.columns.tolist()}")
    print(indicators_df.head(10))
