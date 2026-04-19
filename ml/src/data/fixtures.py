"""Mock data generator for smoke tests and development."""

from typing import Optional

import numpy as np
import pandas as pd


def generate_mock_candles(
    n: int = 100,
    ticker: str = "SBER",
    timeframe: str = "1H",
    start_price: float = 250.0,
    volatility: float = 0.01,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """Generate mock candle data for testing.
    
    Args:
        n: Number of candles to generate.
        ticker: Ticker symbol.
        timeframe: Timeframe string.
        start_price: Starting price.
        volatility: Price volatility (standard deviation of returns).
        seed: Random seed for reproducibility.
        
    Returns:
        DataFrame with mock candle data.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate timestamps
    start_time = pd.Timestamp("2024-01-01", tz="UTC")
    if timeframe == "1H":
        freq = "1h"
    elif timeframe == "1D":
        freq = "1D"
    else:
        freq = "1h"  # Default to hourly
    
    timestamps = pd.date_range(start=start_time, periods=n, freq=freq)
    
    # Generate price movements using geometric Brownian motion
    returns = np.random.normal(0, volatility, n)
    prices = start_price * np.cumprod(1 + returns)
    
    # Generate OHLC from prices
    high_noise = np.random.uniform(0, volatility, n) * prices
    low_noise = np.random.uniform(0, volatility, n) * prices
    
    opens = prices
    closes = prices * (1 + returns)
    highs = np.maximum(opens, closes) + high_noise
    lows = np.minimum(opens, closes) - low_noise
    
    # Generate volume
    base_volume = 1000000
    volume_noise = np.random.lognormal(0, 0.5, n)
    volumes = base_volume * volume_noise
    
    # Generate value (price * volume)
    values = closes * volumes
    
    df = pd.DataFrame({
        "ticker": ticker,
        "timeframe": timeframe,
        "begin": timestamps,
        "end": timestamps + pd.Timedelta(hours=1) if timeframe == "1H" else timestamps + pd.Timedelta(days=1),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "value": values,
        "source": "mock"
    })
    
    return df
