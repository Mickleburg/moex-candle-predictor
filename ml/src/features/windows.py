"""Sliding window creation for supervised learning."""

from typing import Optional

import numpy as np
import pandas as pd


def resolve_feature_columns(
    features_df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None
) -> list[str]:
    """Resolve model input feature order from a feature dataframe.

    If ``feature_cols`` is provided, it is treated as the source of truth and
    validated against the dataframe. Otherwise the order is derived from the
    dataframe's numeric columns, excluding non-feature identifiers.
    """
    df = features_df.copy()

    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    exclude_cols = ["begin", "end", "ticker", "timeframe", "source"]
    resolved = [col for col in feature_cols if col not in exclude_cols]
    missing = [col for col in resolved if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    return resolved


def build_tabular_windows(
    features_df: pd.DataFrame,
    tokens: np.ndarray,
    window_size: int = 32,
    horizon: int = 3,
    feature_cols: Optional[list[str]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Build tabular windows for tree-based models (LightGBM).

    Flattens a historical feature window into a single feature vector per sample.
    Target is the token at the last candle of the historical window. The token
    itself already encodes the future return over ``horizon`` candles.

    Args:
        features_df: DataFrame with feature columns.
        tokens: Array of token IDs (same length as features_df).
        window_size: Number of historical tokens to include (L).
        horizon: Prediction horizon h encoded inside each target token.
        feature_cols: List of feature columns to include. If None, uses all numeric columns.

    Returns:
        Tuple of (X, y) where:
        - X shape: (n_samples, n_features) - flattened features per window
        - y shape: (n_samples,) - target token IDs
    """
    df = features_df.copy()
    
    feature_cols = resolve_feature_columns(df, feature_cols)
    
    # Extract feature matrix
    feature_matrix = df[feature_cols].values
    
    n_samples = len(df)
    min_length = window_size
    
    if n_samples < min_length:
        raise ValueError(
            f"Insufficient data: {n_samples} samples, need at least {min_length} "
            f"(window_size={window_size})"
        )
    
    X_list = []
    y_list = []
    
    # Create sliding windows
    for i in range(n_samples - min_length + 1):
        # Features: window_size steps of historical data only.
        window_features = feature_matrix[i:i + window_size].flatten()
        X_sample = window_features
        
        # Target token for the current decision point; the token value already
        # contains the future-return horizon.
        target_idx = i + window_size - 1
        y_sample = tokens[target_idx]
        
        # Skip if target is invalid (-1)
        if y_sample == -1:
            continue
        
        X_list.append(X_sample)
        y_list.append(y_sample)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Tabular windows: X shape={X.shape}, y shape={y.shape}")
    
    return X, y


def build_sequence_windows(
    features_df: pd.DataFrame,
    tokens: np.ndarray,
    window_size: int = 32,
    horizon: int = 3,
    feature_cols: Optional[list[str]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Build sequence windows for RNN models (LSTM/GRU).

    Creates 3D tensor of sequences for sequence models. Target is the token at
    the last candle of the historical window. The token itself already encodes
    the future return over ``horizon`` candles.

    Args:
        features_df: DataFrame with feature columns.
        tokens: Array of token IDs (same length as features_df).
        window_size: Number of historical tokens to include (L).
        horizon: Prediction horizon h encoded inside each target token.
        feature_cols: List of feature columns to include. If None, uses all numeric columns.

    Returns:
        Tuple of (X, y) where:
        - X shape: (n_samples, window_size, n_features) - 3D sequence tensor
        - y shape: (n_samples,) - target token IDs
    """
    df = features_df.copy()
    
    feature_cols = resolve_feature_columns(df, feature_cols)
    
    # Extract feature matrix
    feature_matrix = df[feature_cols].values
    
    n_samples = len(df)
    min_length = window_size
    
    if n_samples < min_length:
        raise ValueError(
            f"Insufficient data: {n_samples} samples, need at least {min_length} "
            f"(window_size={window_size})"
        )
    
    X_list = []
    y_list = []
    
    # Create sliding windows
    for i in range(n_samples - min_length + 1):
        # Features: window_size steps of historical data only.
        window_features = feature_matrix[i:i + window_size]
        X_sample = window_features
        
        # Target token for the current decision point; the token value already
        # contains the future-return horizon.
        target_idx = i + window_size - 1
        y_sample = tokens[target_idx]
        
        # Skip if target is invalid (-1)
        if y_sample == -1:
            continue
        
        X_list.append(X_sample)
        y_list.append(y_sample)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Sequence windows: X shape={X.shape}, y shape={y.shape}")
    
    return X, y


def build_token_windows(
    tokens: np.ndarray,
    window_size: int = 32,
    horizon: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Build token-only windows for sequence baselines such as Markov models.

    Args:
        tokens: Array of token IDs.
        window_size: Number of historical realized tokens to include.
        horizon: Prediction horizon h encoded inside each target token. A token
            for candle ``t`` becomes observable only after ``t + h``.

    Returns:
        Tuple of (X, y), where X contains token context windows.
    """
    n_samples = len(tokens)
    min_length = window_size + horizon

    if n_samples < min_length:
        raise ValueError(
            f"Insufficient data: {n_samples} samples, need at least {min_length} "
            f"(window_size={window_size}, horizon={horizon})"
        )

    X_list = []
    y_list = []

    for target_idx in range(window_size + horizon - 1, n_samples):
        context_end = target_idx - horizon + 1
        context_start = context_end - window_size
        window_tokens = tokens[context_start:context_end]
        y_sample = tokens[target_idx]

        if y_sample == -1 or np.any(window_tokens == -1):
            continue

        X_list.append(window_tokens)
        y_list.append(y_sample)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"Token windows: X shape={X.shape}, y shape={y.shape}")

    return X, y


def build_inference_window(
    features_df: pd.DataFrame,
    window_size: int = 32,
    feature_cols: Optional[list[str]] = None
) -> np.ndarray:
    """Build single window for inference.
    
    Takes the last window_size samples from data for prediction.
    
    Args:
        features_df: DataFrame with feature columns.
        window_size: Number of historical tokens to include (L).
        feature_cols: List of feature columns to include.
        
    Returns:
        X shape: (1, window_size, n_features) for sequence models
        or (1, flattened_features) for tabular models
        
    Note: For tabular models, flatten the result manually.
    """
    df = features_df.copy()
    
    if len(df) < window_size:
        raise ValueError(
            f"Insufficient data for inference: {len(df)} samples, need {window_size}"
        )
    
    # Take last window_size samples
    df_window = df.iloc[-window_size:].copy()
    feature_cols = resolve_feature_columns(df, feature_cols)
    
    # Extract feature matrix
    feature_matrix = df_window[feature_cols].values

    # Add batch dimension
    X = feature_matrix[np.newaxis, ...]
    
    print(f"Inference window: X shape={X.shape}")
    
    return X


if __name__ == "__main__":
    # Example usage with mock data
    from ..data.fixtures import generate_mock_candles
    from .indicators import compute_all_indicators
    from .tokenizer import CandleTokenizer
    
    # Generate mock data
    mock_df = generate_mock_candles(n=200, ticker="SBER", timeframe="1H", seed=42)
    
    # Compute indicators
    indicators_df = compute_all_indicators(mock_df)
    
    # Fit tokenizer
    tokenizer = CandleTokenizer(n_bins=7, horizon=3, random_state=42)
    tokenizer.fit(indicators_df)
    
    # Transform to tokens
    print(f"Data shape: {indicators_df.shape}")
    tokens = tokenizer.transform(indicators_df)
    print(f"Tokens shape: {tokens.shape}")
    
    # Build tabular windows
    X_tab, y_tab = build_tabular_windows(
        indicators_df, tokens, window_size=32, horizon=3
    )
    print(f"\nTabular: X={X_tab.shape}, y={y_tab.shape}")
    
    # Build sequence windows
    X_seq, y_seq = build_sequence_windows(
        indicators_df, tokens, window_size=32, horizon=3
    )
    print(f"Sequence: X={X_seq.shape}, y={y_seq.shape}")
    
    # Build inference window
    X_inf = build_inference_window(
        indicators_df, window_size=32
    )
    print(f"Inference: X={X_inf.shape}")
