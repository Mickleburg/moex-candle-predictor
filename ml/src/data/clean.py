"""Validate and clean candle data."""

import warnings
from typing import Optional

import pandas as pd


# Required columns for candle data
REQUIRED_COLUMNS = [
    "ticker",
    "timeframe",
    "begin",
    "end",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "value",
    "source",
]


def validate_candles(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Validate candle data integrity.
    
    Args:
        df: DataFrame with candle data.
        strict: If True, raise exceptions on validation errors.
               If False, return warnings and drop invalid rows.
        
    Returns:
        DataFrame with validation warnings/issues reported.
        
    Raises:
        ValueError: If required columns are missing and strict=True.
    """
    issues = []
    
    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        msg = f"Missing required columns: {missing_cols}"
        if strict:
            raise ValueError(msg)
        issues.append(msg)
        warnings.warn(msg)
    
    if not issues:
        print("Validation passed: All required columns present")
    
    return df


def clean_candles(
    df: pd.DataFrame,
    drop_invalid: bool = True,
    timezone: Optional[str] = None
) -> pd.DataFrame:
    """Clean and preprocess candle data.
    
    Args:
        df: DataFrame with candle data.
        drop_invalid: If True, drop invalid candles. If False, keep them with flags.
        timezone: Target timezone for timestamps. If None, convert to UTC and remove tz.
                  Use 'UTC' to keep UTC, or None to make timezone-naive.
        
    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()
    
    # Normalize timestamp
    df = normalize_timestamp(df, timezone)
    
    # Sort by time
    df = sort_by_time(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Handle invalid candles
    df = handle_invalid_candles(df, drop_invalid)
    
    # Handle missing values
    df = handle_missing_values(df, drop_invalid)
    
    return df.reset_index(drop=True)


def normalize_timestamp(df: pd.DataFrame, timezone: Optional[str] = None) -> pd.DataFrame:
    """Normalize timestamp column to consistent format.
    
    Args:
        df: DataFrame with candle data.
        timezone: Target timezone. If None, convert to timezone-naive.
        
    Returns:
        DataFrame with normalized timestamps.
    """
    df = df.copy()
    
    if "begin" not in df.columns:
        return df
    
    # Convert to datetime if not already
    df["begin"] = pd.to_datetime(df["begin"])
    
    if "end" in df.columns:
        df["end"] = pd.to_datetime(df["end"])
    
    # Handle timezone
    if timezone is not None:
        # Convert to specified timezone
        if df["begin"].dt.tz is not None:
            df["begin"] = df["begin"].dt.tz_convert(timezone)
        else:
            df["begin"] = df["begin"].dt.tz_localize("UTC").dt.tz_convert(timezone)
        
        if "end" in df.columns:
            if df["end"].dt.tz is not None:
                df["end"] = df["end"].dt.tz_convert(timezone)
            else:
                df["end"] = df["end"].dt.tz_localize("UTC").dt.tz_convert(timezone)
    else:
        # Make timezone-naive (convert to UTC first if tz-aware)
        if df["begin"].dt.tz is not None:
            df["begin"] = df["begin"].dt.tz_convert("UTC").dt.tz_localize(None)
        
        if "end" in df.columns and df["end"].dt.tz is not None:
            df["end"] = df["end"].dt.tz_convert("UTC").dt.tz_localize(None)
    
    return df


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by timestamp.
    
    Args:
        df: DataFrame with candle data.
        
    Returns:
        Sorted DataFrame.
    """
    df = df.copy()
    
    if "begin" in df.columns:
        df = df.sort_values("begin")
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate candles based on timestamp.
    
    Args:
        df: DataFrame with candle data.
        
    Returns:
        DataFrame with duplicates removed.
    """
    df = df.copy()
    
    if "begin" in df.columns:
        initial_len = len(df)
        df = df.drop_duplicates(subset=["begin"], keep="last")
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"Removed {dropped} duplicate candles")
    
    return df


def handle_invalid_candles(df: pd.DataFrame, drop_invalid: bool = True) -> pd.DataFrame:
    """Handle invalid candles (OHLC violations, non-positive prices).
    
    Args:
        df: DataFrame with candle data.
        drop_invalid: If True, drop invalid candles.
        
    Returns:
        DataFrame with invalid candles handled.
    """
    df = df.copy()
    
    # Check for required OHLC columns
    ohlc_cols = ["open", "high", "low", "close"]
    missing_ohlc = [col for col in ohlc_cols if col not in df.columns]
    if missing_ohlc:
        print(f"Warning: Missing OHLC columns {missing_ohlc}, skipping validation")
        return df
    
    # Flag invalid candles
    invalid_mask = pd.Series(False, index=df.index)
    
    # High < Low violation
    high_low_invalid = df["high"] < df["low"]
    if high_low_invalid.any():
        count = high_low_invalid.sum()
        msg = f"Found {count} candles with high < low"
        warnings.warn(msg)
        invalid_mask |= high_low_invalid
    
    # Non-positive prices
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        non_positive = df[col] <= 0
        if non_positive.any():
            count = non_positive.sum()
            msg = f"Found {count} candles with non-positive {col}"
            warnings.warn(msg)
            invalid_mask |= non_positive
    
    # High not >= max(open, close)
    high_invalid = df["high"] < df[["open", "close"]].max(axis=1)
    if high_invalid.any():
        count = high_invalid.sum()
        msg = f"Found {count} candles with high < max(open, close)"
        warnings.warn(msg)
        invalid_mask |= high_invalid
    
    # Low not <= min(open, close)
    low_invalid = df["low"] > df[["open", "close"]].min(axis=1)
    if low_invalid.any():
        count = low_invalid.sum()
        msg = f"Found {count} candles with low > min(open, close)"
        warnings.warn(msg)
        invalid_mask |= low_invalid
    
    # Drop or flag invalid candles
    if invalid_mask.any():
        count = invalid_mask.sum()
        if drop_invalid:
            df = df[~invalid_mask]
            print(f"Dropped {count} invalid candles")
        else:
            df["invalid"] = invalid_mask
            print(f"Flagged {count} invalid candles")
    
    return df


def handle_missing_values(df: pd.DataFrame, drop_invalid: bool = True) -> pd.DataFrame:
    """Handle missing values in OHLCV columns.
    
    Args:
        df: DataFrame with candle data.
        drop_invalid: If True, drop rows with missing OHLCV values.
        
    Returns:
        DataFrame with missing values handled.
    """
    df = df.copy()
    
    # Check for missing values in critical columns
    critical_cols = ["open", "high", "low", "close", "volume"]
    available_cols = [col for col in critical_cols if col in df.columns]
    
    if not available_cols:
        return df
    
    missing_mask = df[available_cols].isna().any(axis=1)
    
    if missing_mask.any():
        count = missing_mask.sum()
        if drop_invalid:
            df = df[~missing_mask]
            print(f"Dropped {count} candles with missing OHLCV values")
        else:
            print(f"Warning: {count} candles have missing OHLCV values")
    
    return df


if __name__ == "__main__":
    # Example usage with mock data
    from .fixtures import generate_mock_candles
    
    # Generate mock data with some invalid candles
    mock_df = generate_mock_candles(n=100, ticker="SBER", timeframe="1H", seed=42)
    
    # Introduce some invalid candles for testing
    mock_df.loc[5, "high"] = mock_df.loc[5, "low"] - 1  # high < low
    mock_df.loc[10, "open"] = -1  # negative price
    mock_df.loc[15, "close"] = None  # missing value
    
    print("Original data:", len(mock_df))
    print(mock_df.head(10))
    
    # Validate
    validated = validate_candles(mock_df, strict=False)
    
    # Clean
    cleaned = clean_candles(mock_df, drop_invalid=True)
    print(f"\nCleaned data: {len(cleaned)} candles")
    print(cleaned.head())
