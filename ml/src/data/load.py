"""Load candle data from raw storage."""

from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils.io import read_csv, read_parquet


# Column name mapping for normalization
COLUMN_MAP = {
    "timestamp": "begin",
    "datetime": "begin",
    "time": "begin",
    "Date": "begin",
    "date": "begin",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vol": "volume",
    "tic": "ticker",
    "symbol": "ticker",
    "tf": "timeframe",
    "interval": "timeframe",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to standard schema.
    
    Args:
        df: DataFrame with potentially non-standard column names.
        
    Returns:
        DataFrame with normalized column names.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns=COLUMN_MAP)
    return df


def load_candles(
    path: str | Path,
    ticker: Optional[str] = None,
    timeframe: Optional[str] = None,
    format: Optional[str] = None
) -> pd.DataFrame:
    """Load candles from file or directory.
    
    Args:
        path: Path to file or directory containing candle data.
        ticker: Optional ticker filter.
        timeframe: Optional timeframe filter.
        format: File format ('parquet' or 'csv'). If None, inferred from extension.
        
    Returns:
        DataFrame with normalized candle schema.
        
    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If format is unsupported or data is empty.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Determine format if not specified
    if format is None:
        if path.is_file():
            format = "parquet" if path.suffix == ".parquet" else "csv"
        else:
            # For directories, look for parquet files first
            parquet_files = list(path.glob("*.parquet"))
            format = "parquet" if parquet_files else "csv"
    
    # Load data
    if path.is_file():
        df = _load_single_file(path, format)
    else:
        df = _load_directory(path, format)
    
    if df.empty:
        raise ValueError(f"No data found in {path}")
    
    # Normalize column names
    df = normalize_columns(df)
    
    # Apply filters
    if ticker is not None:
        if "ticker" in df.columns:
            df = df[df["ticker"] == ticker]
        else:
            raise ValueError("Ticker filter requested but 'ticker' column not found")
    
    if timeframe is not None:
        if "timeframe" in df.columns:
            df = df[df["timeframe"] == timeframe]
        else:
            raise ValueError("Timeframe filter requested but 'timeframe' column not found")
    
    return df.reset_index(drop=True)


def _load_single_file(path: Path, format: str) -> pd.DataFrame:
    """Load a single file.
    
    Args:
        path: Path to file.
        format: File format ('parquet' or 'csv').
        
    Returns:
        DataFrame with candle data.
    """
    if format == "parquet":
        return read_parquet(path)
    elif format == "csv":
        return read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_directory(path: Path, format: str) -> pd.DataFrame:
    """Load all files from directory.
    
    Args:
        path: Path to directory.
        format: File format ('parquet' or 'csv').
        
    Returns:
        Concatenated DataFrame with candle data.
    """
    if format == "parquet":
        files = list(path.glob("*.parquet"))
    elif format == "csv":
        files = list(path.glob("*.csv"))
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    if not files:
        raise ValueError(f"No {format} files found in {path}")
    
    dfs = []
    for file in files:
        try:
            if format == "parquet":
                df = read_parquet(file)
            else:
                df = read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")
    
    if not dfs:
        raise ValueError(f"Failed to load any files from {path}")
    
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    # Example usage with mock data
    from .fixtures import generate_mock_candles
    
    # Generate mock data
    mock_df = generate_mock_candles(n=100, ticker="SBER", timeframe="1H", seed=42)
    print(f"Generated {len(mock_df)} mock candles")
    print(mock_df.head())
    
    # Test column normalization
    test_df = mock_df.rename(columns={"begin": "timestamp", "open": "o"})
    normalized = normalize_columns(test_df)
    print("\nNormalized columns:", normalized.columns.tolist())
