"""Quantile-based tokenizer for candle returns."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.io import load_pickle, save_pickle


class CandleTokenizer:
    """Tokenizer for candle returns using quantile binning.
    
    Tokenization strategy:
    - Compute normalized return: return_h / ATR (where return_h is close-to-close return over horizon h)
    - Use ATR for normalization (more robust than rolling volatility)
    - Fit quantile bins on train data only
    - Transform returns to token IDs [0, K-1]
    """
    
    def __init__(
        self,
        n_bins: int = 7,
        horizon: int = 3,
        random_state: int = 42
    ):
        """Initialize tokenizer.
        
        Args:
            n_bins: Number of quantile bins (K).
            horizon: Prediction horizon h (number of candles ahead).
            random_state: Random seed for reproducibility.
        """
        self.n_bins = n_bins
        self.horizon = horizon
        self.random_state = random_state
        
        # Fitted parameters
        self.bin_edges_: np.ndarray = None
        self.is_fitted_ = False
        
        # Metadata
        self.metadata_ = {
            "n_bins": n_bins,
            "horizon": horizon,
            "normalization": "atr",  # Using ATR for normalization
            "bin_edges": None,
        }
    
    def _compute_normalized_returns(self, df: pd.DataFrame) -> np.ndarray:
        """Compute normalized returns: return_h / ATR.
        
        Args:
            df: DataFrame with 'close' and 'atr' columns.
            
        Returns:
            Array of normalized returns.
        """
        # Compute future return over horizon h
        future_close = df["close"].shift(-self.horizon)
        current_close = df["close"]
        return_h = (future_close - current_close) / current_close
        
        # Normalize by ATR (handle division by zero)
        atr = df["atr"].values
        
        # Use small epsilon to avoid division by zero
        epsilon = 1e-8
        atr_safe = np.where(np.abs(atr) < epsilon, epsilon, atr)
        
        normalized_return = return_h / atr_safe
        
        return normalized_return.values
    
    def fit(self, df: pd.DataFrame) -> "CandleTokenizer":
        """Fit tokenizer on training data.
        
        Computes quantile bin edges from normalized returns.
        
        Args:
            df: DataFrame with 'close' and 'atr' columns.
            
        Returns:
            Self (fitted tokenizer).
            
        Raises:
            ValueError: If insufficient data or missing columns.
        """
        df = df.copy()
        
        # Validate columns
        required_cols = ["close", "atr"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Need at least n_bins * 10 samples for reliable quantile estimation
        min_samples = self.n_bins * 10
        if len(df) < min_samples:
            raise ValueError(
                f"Insufficient data: {len(df)} samples, need at least {min_samples}"
            )
        
        # Compute normalized returns
        normalized_returns = self._compute_normalized_returns(df)
        
        # Remove NaN values (from horizon shift at end of data)
        normalized_returns = normalized_returns[~np.isnan(normalized_returns)]
        
        if len(normalized_returns) < min_samples:
            raise ValueError(
                f"Insufficient valid returns: {len(normalized_returns)}, need at least {min_samples}"
            )
        
        # Compute quantile bin edges
        # Use (n_bins + 1) edges to create n_bins intervals
        self.bin_edges_ = np.quantile(
            normalized_returns,
            np.linspace(0, 1, self.n_bins + 1),
            method="linear"
        )
        
        # Ensure edges are unique (handle constant values)
        if len(np.unique(self.bin_edges_)) < 2:
            # Fallback: use min/max with uniform spacing
            self.bin_edges_ = np.linspace(
                normalized_returns.min(),
                normalized_returns.max(),
                self.n_bins + 1
            )
        
        # Update metadata
        self.metadata_["bin_edges"] = self.bin_edges_.tolist()
        self.is_fitted_ = True
        
        print(f"Tokenizer fitted: n_bins={self.n_bins}, horizon={self.horizon}")
        print(f"Bin edges: {self.bin_edges_}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform DataFrame to token IDs.
        
        Args:
            df: DataFrame with 'close' and 'atr' columns.
            
        Returns:
            Array of token IDs (integers in [0, n_bins-1]).
            
        Raises:
            ValueError: If tokenizer not fitted or missing columns.
        """
        if not self.is_fitted_:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        
        df = df.copy()
        
        # Validate columns
        required_cols = ["close", "atr"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Compute normalized returns
        normalized_returns = self._compute_normalized_returns(df)
        
        # Digitize using bin edges
        # np.digitize returns bin index in [1, n_bins+1], so subtract 1
        tokens = np.digitize(normalized_returns, self.bin_edges_) - 1
        
        # Clip to valid range [0, n_bins-1]
        # Values below min edge get bin 0, above max edge get bin n_bins-1
        tokens = np.clip(tokens, 0, self.n_bins - 1)
        
        # Set NaN to -1 (invalid token)
        tokens = np.where(np.isnan(normalized_returns), -1, tokens)
        
        return tokens
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit tokenizer and transform data.
        
        Args:
            df: DataFrame with 'close' and 'atr' columns.
            
        Returns:
            Array of token IDs.
        """
        self.fit(df)
        return self.transform(df)
    
    def save(self, path: str | Path) -> None:
        """Save tokenizer to file.
        
        Args:
            path: Path to save tokenizer.
        """
        path = Path(path)
        
        save_data = {
            "n_bins": self.n_bins,
            "horizon": self.horizon,
            "random_state": self.random_state,
            "bin_edges_": self.bin_edges_,
            "metadata_": self.metadata_,
            "is_fitted_": self.is_fitted_,
        }
        
        save_pickle(save_data, path)
        print(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> "CandleTokenizer":
        """Load tokenizer from file.
        
        Args:
            path: Path to load tokenizer from.
            
        Returns:
            Loaded tokenizer instance.
        """
        path = Path(path)
        
        save_data = load_pickle(path)
        
        tokenizer = cls(
            n_bins=save_data["n_bins"],
            horizon=save_data["horizon"],
            random_state=save_data["random_state"]
        )
        
        tokenizer.bin_edges_ = save_data["bin_edges_"]
        tokenizer.metadata_ = save_data["metadata_"]
        tokenizer.is_fitted_ = save_data["is_fitted_"]
        
        print(f"Tokenizer loaded from {path}")
        
        return tokenizer
    
    def get_metadata(self) -> dict:
        """Get tokenizer metadata.
        
        Returns:
            Dictionary with tokenizer parameters and bin edges.
        """
        if not self.is_fitted_:
            return {
                "n_bins": self.n_bins,
                "horizon": self.horizon,
                "random_state": self.random_state,
                "is_fitted": False,
            }
        
        return {
            "n_bins": self.n_bins,
            "horizon": self.horizon,
            "random_state": self.random_state,
            "is_fitted": True,
            "bin_edges": self.bin_edges_.tolist(),
            "normalization": self.metadata_["normalization"],
        }


if __name__ == "__main__":
    # Example usage with mock data
    from ..data.fixtures import generate_mock_candles
    from .indicators import compute_all_indicators
    
    # Generate mock data
    mock_df = generate_mock_candles(n=200, ticker="SBER", timeframe="1H", seed=42)
    
    # Compute indicators (need ATR)
    indicators_df = compute_all_indicators(mock_df)
    
    # Split into train/test
    train_df = indicators_df.iloc[:150]
    test_df = indicators_df.iloc[150:]
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Fit on train
    tokenizer = CandleTokenizer(n_bins=7, horizon=3, random_state=42)
    tokenizer.fit(train_df)
    
    # Transform train
    train_tokens = tokenizer.transform(train_df)
    print(f"\nTrain tokens shape: {train_tokens.shape}")
    print(f"Train tokens (first 20): {train_tokens[:20]}")
    print(f"Train token distribution: {np.bincount(train_tokens[train_tokens >= 0])}")
    
    # Transform test (uses fitted bin edges)
    test_tokens = tokenizer.transform(test_df)
    print(f"\nTest tokens shape: {test_tokens.shape}")
    print(f"Test tokens: {test_tokens[:20]}")
    
    # Get metadata
    metadata = tokenizer.get_metadata()
    print(f"\nMetadata: {metadata}")
