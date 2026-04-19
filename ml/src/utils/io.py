"""I/O utilities for data and artifacts."""

import json
import pickle
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read Parquet file.
    
    Args:
        path: Path to Parquet file.
        
    Returns:
        DataFrame with data.
        
    Raises:
        FileNotFoundError: If file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write DataFrame to Parquet file.
    
    Args:
        df: DataFrame to write.
        path: Output path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read CSV file.
    
    Args:
        path: Path to CSV file.
        **kwargs: Additional arguments for pd.read_csv.
        
    Returns:
        DataFrame with data.
        
    Raises:
        FileNotFoundError: If file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Write DataFrame to CSV file.
    
    Args:
        df: DataFrame to write.
        path: Output path.
        **kwargs: Additional arguments for df.to_csv.
    """
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, **kwargs)


def read_json(path: str | Path) -> dict:
    """Read JSON file.
    
    Args:
        path: Path to JSON file.
        
    Returns:
        Dictionary with JSON data.
        
    Raises:
        FileNotFoundError: If file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict, path: str | Path) -> None:
    """Write dictionary to JSON file.
    
    Args:
        data: Dictionary to write.
        path: Output path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_json(data: dict, path: str | Path) -> None:
    """Backward-compatible alias for JSON artifact saving."""
    write_json(data, path)


def save_joblib(obj: Any, path: str | Path) -> None:
    """Save object using joblib.
    
    Args:
        obj: Object to save.
        path: Output path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path, compress=3)


def load_joblib(path: str | Path) -> Any:
    """Load object using joblib.
    
    Args:
        path: Path to joblib file.
        
    Returns:
        Loaded object.
        
    Raises:
        FileNotFoundError: If file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return joblib.load(path)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save object using pickle.
    
    Args:
        obj: Object to save.
        path: Output path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str | Path) -> Any:
    """Load object using pickle.
    
    Args:
        path: Path to pickle file.
        
    Returns:
        Loaded object.
        
    Raises:
        FileNotFoundError: If file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path: str | Path) -> None:
    """Create directory if it does not exist.
    
    Args:
        path: Directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
