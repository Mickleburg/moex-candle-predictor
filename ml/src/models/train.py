"""Training pipeline orchestrator."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..data import clean_candles, load_candles, time_split
from ..evaluation.metrics import compute_classification_metrics
from ..features import (
    CandleTokenizer,
    build_tabular_windows,
    compute_all_indicators,
    resolve_feature_columns,
)
from ..models import LGBMClassifier, LogisticRegressionBaseline, MajorityClassifier, MarkovClassifier
from ..utils.config import load_all_configs
from ..utils.io import ensure_dir, save_json, save_pickle


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed.
    """
    np.random.seed(seed)


def load_and_prepare_data(data_config: dict, features_config: dict) -> pd.DataFrame:
    """Load and prepare raw candle data.
    
    Args:
        data_config: Data configuration dictionary.
        features_config: Features configuration dictionary (for window size check).
        
    Returns:
        Cleaned DataFrame with candles.
    """
    # Resolve path (relative to ml/ directory)
    raw_path = Path(__file__).parent.parent.parent / data_config["raw_data_path"]
    
    # Load data
    print(f"Loading data from {raw_path}")
    df = load_candles(
        raw_path,
        ticker=data_config.get("tickers", ["SBER"])[0],
        timeframe=data_config.get("timeframes", ["1H"])[0],
        format=data_config.get("data_format", "parquet")
    )
    
    print(f"Loaded {len(df)} candles")
    
    # Clean data
    df = clean_candles(df, drop_invalid=True)
    print(f"After cleaning: {len(df)} candles")
    
    # Validate minimum data for window building
    window_size = features_config.get("window_size", 32)
    horizon = features_config.get("horizon", 3)
    min_required = window_size + horizon + 50  # buffer for rolling windows
    
    if len(df) < min_required:
        raise ValueError(
            f"Insufficient data after cleaning: {len(df)} candles, "
            f"need at least {min_required} for window_size={window_size}, horizon={horizon}"
        )
    
    return df


def compute_features_and_tokens(
    df: pd.DataFrame,
    features_config: dict,
    tokenizer: Optional[CandleTokenizer] = None
) -> tuple[pd.DataFrame, np.ndarray, Optional[CandleTokenizer]]:
    """Compute features and tokens.
    
    Args:
        df: DataFrame with cleaned candles.
        features_config: Features configuration dictionary.
        tokenizer: Fitted tokenizer (if None, will fit on df).
        
    Returns:
        Tuple of (features_df, tokens, tokenizer).
    """
    # Compute technical indicators
    print("Computing technical indicators...")
    features_df = compute_all_indicators(df)
    print(f"Features computed: {features_df.shape}")
    
    # Fit or transform tokenizer
    n_bins = features_config.get("num_classes", 7)
    horizon = features_config.get("horizon", 3)
    random_state = 42
    
    if tokenizer is None:
        print(f"Fitting tokenizer (n_bins={n_bins}, horizon={horizon})...")
        tokenizer = CandleTokenizer(
            n_bins=n_bins,
            horizon=horizon,
            random_state=random_state
        )
        tokens = tokenizer.fit_transform(features_df)
    else:
        print("Transforming with fitted tokenizer...")
        tokens = tokenizer.transform(features_df)
    
    print(f"Tokens computed: {tokens.shape}")
    
    return features_df, tokens, tokenizer


def build_training_windows(
    features_df: pd.DataFrame,
    tokens: np.ndarray,
    features_config: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Build training windows.
    
    Args:
        features_df: DataFrame with features.
        tokens: Token array.
        features_config: Features configuration.
        
    Returns:
        Tuple of (X, y) for training.
    """
    window_size = features_config.get("window_size", 32)
    horizon = features_config.get("horizon", 3)
    
    print(f"Building windows (window_size={window_size}, horizon={horizon})...")
    X, y = build_tabular_windows(
        features_df,
        tokens,
        window_size=window_size,
        horizon=horizon
    )
    
    print(f"Windows built: X shape={X.shape}, y shape={y.shape}")
    
    return X, y


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_config: dict,
    features_config: dict
):
    """Train model based on configuration.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        train_config: Training configuration.
        features_config: Features configuration.
        
    Returns:
        Trained model instance.
    """
    model_type = train_config.get("model_type", "lgbm")
    n_classes = features_config.get("num_classes", 7)
    random_state = train_config.get("random_state", 42)
    
    print(f"Training model: {model_type}")
    
    if model_type == "majority":
        model = MajorityClassifier()
        model.fit(X_train, y_train)
    elif model_type == "markov":
        model = MarkovClassifier(n_classes=n_classes)
        model.fit(X_train, y_train)
    elif model_type == "logistic":
        model = LogisticRegressionBaseline(
            n_classes=n_classes,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
    elif model_type == "lgbm":
        lgbm_params = dict(train_config.get("lgbm_params", {}))
        # Wrapper already hardcodes multiclass objective and class count.
        lgbm_params.pop("objective", None)
        lgbm_params.pop("num_class", None)

        model = LGBMClassifier(
            n_classes=n_classes,
            random_state=random_state,
            **lgbm_params
        )
        model.fit(X_train, y_train, X_val, y_val)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str
) -> dict:
    """Evaluate model on data split.
    
    Args:
        model: Trained model.
        X: Features.
        y: True labels.
        split_name: Name of split (e.g., "val", "test").
        
    Returns:
        Dictionary with metrics.
    """
    print(f"Evaluating on {split_name}...")
    
    y_pred = model.predict(X)
    
    # Get probabilities if available
    try:
        y_proba = model.predict_proba(X)
    except (AttributeError, NotImplementedError):
        y_proba = None
    
    metrics = compute_classification_metrics(y, y_pred, y_proba)
    metrics["split"] = split_name
    metrics["n_samples"] = len(y)
    
    print(f"{split_name} metrics: {metrics}")
    
    return metrics


def save_artifacts(
    model,
    tokenizer: CandleTokenizer,
    metadata: dict,
    train_config: dict
) -> None:
    """Save model, tokenizer, and metadata.
    
    Args:
        model: Trained model.
        tokenizer: Fitted tokenizer.
        metadata: Metadata dictionary.
        train_config: Training configuration.
    """
    # Resolve artifacts directory
    artifacts_dir = Path(__file__).parent.parent.parent / train_config["artifacts_dir"]
    ensure_dir(artifacts_dir)
    
    # Save model
    model_path = artifacts_dir / "model.pkl"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save tokenizer
    tokenizer_path = artifacts_dir / "tokenizer.pkl"
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Save metadata
    metadata_path = artifacts_dir / "metadata.json"
    save_json(metadata, metadata_path)
    print(f"Metadata saved to {metadata_path}")


def build_metadata(
    data_config: dict,
    features_config: dict,
    train_config: dict,
    train_df: pd.DataFrame,
    train_features_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: CandleTokenizer,
    model,
    val_metrics: dict,
    test_metrics: dict
) -> dict:
    """Build metadata dictionary.
    
    Args:
        data_config: Data configuration.
        features_config: Features configuration.
        train_config: Training configuration.
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        tokenizer: Fitted tokenizer.
        model: Trained model.
        val_metrics: Validation metrics.
        test_metrics: Test metrics.
        
    Returns:
        Metadata dictionary.
    """
    # Time periods
    train_period = {
        "start": train_df["begin"].min().isoformat(),
        "end": train_df["begin"].max().isoformat(),
        "n_samples": len(train_df)
    }
    val_period = {
        "start": val_df["begin"].min().isoformat(),
        "end": val_df["begin"].max().isoformat(),
        "n_samples": len(val_df)
    }
    test_period = {
        "start": test_df["begin"].min().isoformat(),
        "end": test_df["begin"].max().isoformat(),
        "n_samples": len(test_df)
    }
    
    # Feature set used by the model as input.
    feature_set = resolve_feature_columns(train_features_df)
    
    # Tokenizer settings
    tokenizer_metadata = tokenizer.get_metadata()
    
    # Model class
    model_class = model.__class__.__name__
    
    # Artifact version (timestamp)
    artifact_version = datetime.utcnow().isoformat()
    
    metadata = {
        "ticker": data_config.get("tickers", ["SBER"])[0],
        "timeframe": data_config.get("timeframes", ["1H"])[0],
        "horizon": features_config.get("horizon", 3),
        "K": features_config.get("num_classes", 7),
        "L": features_config.get("window_size", 32),
        "feature_set": feature_set,
        "tokenizer": tokenizer_metadata,
        "model_class": model_class,
        "train_period": train_period,
        "val_period": val_period,
        "test_period": test_period,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "artifact_version": artifact_version,
        "training_config": train_config,
    }
    
    return metadata


def train_pipeline(config_dir: str = "configs") -> dict:
    """Run complete training pipeline.
    
    Args:
        config_dir: Path to configs directory.
        
    Returns:
        Dictionary with training results and metadata.
    """
    # Load configs
    print("Loading configurations...")
    configs = load_all_configs(config_dir)
    data_config = configs["data"]
    features_config = configs["features"]
    train_config = configs["train"]
    eval_config = configs["eval"]
    
    # Set random seed
    random_state = train_config.get("random_state", 42)
    set_random_seed(random_state)
    print(f"Random seed set to {random_state}")
    
    # Load and prepare data
    df = load_and_prepare_data(data_config, features_config)
    
    # Time split
    print("\nSplitting data by time...")
    train_df, val_df, test_df = time_split(
        df,
        train_ratio=data_config["train_ratio"],
        val_ratio=data_config["val_ratio"],
        test_ratio=data_config["test_ratio"],
        min_train_size=data_config["min_train_size"]
    )
    
    # Compute features and fit tokenizer on train only
    print("\nProcessing training data...")
    train_features, train_tokens, tokenizer = compute_features_and_tokens(
        train_df, features_config, tokenizer=None
    )
    
    # Build training windows
    X_train, y_train = build_training_windows(train_features, train_tokens, features_config)
    
    # Process validation data (use fitted tokenizer)
    print("\nProcessing validation data...")
    val_features, val_tokens, _ = compute_features_and_tokens(
        val_df, features_config, tokenizer=tokenizer
    )
    X_val, y_val = build_training_windows(val_features, val_tokens, features_config)
    
    # Process test data (use fitted tokenizer)
    print("\nProcessing test data...")
    test_features, test_tokens, _ = compute_features_and_tokens(
        test_df, features_config, tokenizer=tokenizer
    )
    X_test, y_test = build_training_windows(test_features, test_tokens, features_config)
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train, X_val, y_val, train_config, features_config)
    
    # Evaluate on validation
    val_metrics = evaluate_model(model, X_val, y_val, "val")
    
    # Evaluate on test
    test_metrics = evaluate_model(model, X_test, y_test, "test")
    
    # Build metadata
    print("\nBuilding metadata...")
    metadata = build_metadata(
        data_config,
        features_config,
        train_config,
        train_df,
        train_features,
        val_df,
        test_df,
        tokenizer,
        model,
        val_metrics,
        test_metrics
    )
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, tokenizer, metadata, train_config)
    
    print("\nTraining pipeline completed successfully!")
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "metadata": metadata,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    }


def main():
    """CLI entry point for training pipeline."""
    parser = argparse.ArgumentParser(description="Train candle prediction model")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Path to configs directory (relative to ml/)"
    )
    
    args = parser.parse_args()
    
    try:
        results = train_pipeline(config_dir=args.config_dir)
        print("\nTraining completed successfully!")
        print(f"Test macro-F1: {results['test_metrics'].get('macro_f1', 'N/A')}")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
