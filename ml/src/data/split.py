"""Time-based data splitting for time series."""

from dataclasses import dataclass
from typing import Generator, Tuple

import pandas as pd


@dataclass(frozen=True)
class WalkForwardRange:
    """Index ranges for one expanding-window validation fold."""

    fold_id: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int

    @property
    def train_len(self) -> int:
        return self.train_end - self.train_start

    @property
    def val_len(self) -> int:
        return self.val_end - self.val_start


def walk_forward_ranges(
    n_rows: int,
    *,
    n_splits: int = 4,
    initial_train_size: int | None = None,
    val_size: int | None = None,
    gap: int = 0,
    min_train_size: int = 1000,
) -> list[WalkForwardRange]:
    """Return expanding train/validation index ranges.

    Fold ``i`` uses train ``[0:train_end)`` and validation
    ``[train_end + gap:val_end)``. The function returns index ranges only,
    so callers can build strict within-fold labels/windows without touching
    future folds.
    """

    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if gap < 0:
        raise ValueError("gap must be >= 0")

    initial_train_size = int(initial_train_size if initial_train_size is not None else max(min_train_size, n_rows // 2))
    if initial_train_size < min_train_size:
        raise ValueError(f"initial_train_size must be >= min_train_size ({min_train_size})")
    if initial_train_size + gap >= n_rows:
        raise ValueError("initial_train_size + gap leaves no validation rows")

    available = n_rows - initial_train_size - gap
    val_size = int(val_size if val_size is not None else available // n_splits)
    if val_size < 1:
        raise ValueError("val_size must be >= 1")

    folds: list[WalkForwardRange] = []
    for fold_idx in range(n_splits):
        train_end = initial_train_size + fold_idx * val_size
        val_start = train_end + gap
        val_end = val_start + val_size
        if val_end > n_rows:
            break
        fold = WalkForwardRange(
            fold_id=fold_idx + 1,
            train_start=0,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
        )
        _validate_walk_forward_range(fold, n_rows)
        folds.append(fold)

    if not folds:
        raise ValueError("No walk-forward folds could be generated")
    return folds


def _validate_walk_forward_range(fold: WalkForwardRange, n_rows: int) -> None:
    if not (0 <= fold.train_start < fold.train_end <= fold.val_start < fold.val_end <= n_rows):
        raise ValueError(f"Invalid walk-forward fold: {fold}")


def time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_train_size: int = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by time into train/val/test sets.
    
    Args:
        df: DataFrame sorted by time (with 'begin' column).
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        min_train_size: Minimum number of samples in training set.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
        
    Raises:
        ValueError: If ratios don't sum to 1 or data is insufficient.
    """
    df = df.copy()
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    if len(df) < min_train_size:
        raise ValueError(
            f"Insufficient data: {len(df)} samples, need at least {min_train_size}"
        )
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Ensure train has minimum size
    if train_end < min_train_size:
        train_end = min_train_size
        val_end = train_end + int((n - train_end) * val_ratio / (val_ratio + test_ratio))
    
    # Split data
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Validate no leakage
    if "begin" in df.columns:
        max_train_time = train_df["begin"].max()
        min_val_time = val_df["begin"].min()
        min_test_time = test_df["begin"].min()
        
        if max_train_time >= min_val_time:
            raise ValueError("Time leakage: train end >= val start")
        if val_df["begin"].max() >= min_test_time:
            raise ValueError("Time leakage: val end >= test start")
    
    print(f"Time split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


def time_split_indices(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_train_size: int = 1000
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """Get train/val/test indices by time.
    
    Args:
        df: DataFrame sorted by time.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        min_train_size: Minimum number of samples in training set.
        
    Returns:
        Tuple of (train_idx, val_idx, test_idx) as pandas Index objects.
    """
    train_df, val_df, test_df = time_split(
        df, train_ratio, val_ratio, test_ratio, min_train_size
    )
    
    return train_df.index, val_df.index, test_df.index


class WalkForwardSplit:
    """Walk-forward cross-validation generator for time series.
    
    Yields expanding window splits for time series validation.
    Ensures no data leakage by maintaining chronological order.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_size: int = 1000,
        test_size: int = 200,
        gap: int = 0
    ):
        """Initialize walk-forward splitter.
        
        Args:
            n_splits: Number of splits to generate.
            train_size: Minimum size of training window.
            test_size: Size of validation window.
            gap: Gap between train and test (to avoid leakage).
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
    
    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """Generate train/test splits.
        
        Args:
            df: DataFrame sorted by time.
            
        Yields:
            Tuple of (train_df, test_df) for each split.
        """
        n = len(df)
        
        if n < self.train_size + self.test_size + self.gap:
            raise ValueError(
                f"Insufficient data: need at least {self.train_size + self.test_size + self.gap} samples"
            )
        
        for i in range(self.n_splits):
            # Calculate split indices
            train_end = self.train_size + i * self.test_size
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            
            # Check if we have enough data
            if test_end > n:
                print(f"Stopping at split {i+1}/{self.n_splits}: insufficient data")
                break
            
            # Split data
            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            # Validate no leakage
            if "begin" in df.columns:
                max_train_time = train_df["begin"].max()
                min_test_time = test_df["begin"].min()
                if max_train_time >= min_test_time:
                    raise ValueError(f"Time leakage in split {i+1}")
            
            print(f"Split {i+1}: train={len(train_df)}, test={len(test_df)}")
            yield train_df, test_df
    
    def split_indices(self, df: pd.DataFrame) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
        """Generate train/test indices.
        
        Args:
            df: DataFrame sorted by time.
            
        Yields:
            Tuple of (train_idx, test_idx) for each split.
        """
        for train_df, test_df in self.split(df):
            yield train_df.index, test_df.index


if __name__ == "__main__":
    # Example usage with mock data
    from .fixtures import generate_mock_candles
    
    # Generate mock data
    mock_df = generate_mock_candles(n=1000, ticker="SBER", timeframe="1H", seed=42)
    print(f"Generated {len(mock_df)} mock candles")
    
    # Simple time split
    train, val, test = time_split(mock_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print(f"\nSimple split: train={len(train)}, val={len(val)}, test={len(test)}")
    
    # Walk-forward split
    print("\nWalk-forward split:")
    splitter = WalkForwardSplit(n_splits=3, train_size=200, test_size=100, gap=10)
    for i, (tr, te) in enumerate(splitter.split(mock_df)):
        print(f"  Fold {i+1}: train={len(tr)}, test={len(te)}")
