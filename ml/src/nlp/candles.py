"""Candle preparation helpers for the candle-as-language approach."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


ACTION_LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}


@dataclass(frozen=True)
class SentenceSamples:
    """Sentence windows and aligned supervised labels."""

    sentences: list[str]
    token_lists: list[list[str]]
    y: np.ndarray
    future_returns: np.ndarray
    target_indices: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.y))


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def candle_shape_matrix(df: pd.DataFrame, variant: str = "ohlc") -> tuple[np.ndarray, list[str]]:
    """Build normalized candle shape features.

    The papers normalize OHLC by the candle open. The open/open value is always
    one, so this implementation keeps the informative open-relative offsets.
    """

    _require_columns(df, ["open", "high", "low", "close"])
    variant = variant.lower()

    open_ = df["open"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()

    safe_open = np.where(np.abs(open_) < 1e-12, np.nan, open_)
    high_rel = high / safe_open - 1.0
    low_rel = low / safe_open - 1.0
    close_rel = close / safe_open - 1.0
    body = (close - open_) / safe_open
    upper_shadow = (high - np.maximum(open_, close)) / safe_open
    lower_shadow = (np.minimum(open_, close) - low) / safe_open
    candle_range = (high - low) / safe_open

    if variant == "ohlc":
        columns = ["high_to_open", "low_to_open", "close_to_open"]
        matrix = np.column_stack([high_rel, low_rel, close_rel])
    elif variant == "shape":
        columns = ["body", "upper_shadow", "lower_shadow", "range"]
        matrix = np.column_stack([body, upper_shadow, lower_shadow, candle_range])
    elif variant == "ohlc_shape":
        columns = [
            "high_to_open",
            "low_to_open",
            "close_to_open",
            "body",
            "upper_shadow",
            "lower_shadow",
            "range",
        ]
        matrix = np.column_stack(
            [high_rel, low_rel, close_rel, body, upper_shadow, lower_shadow, candle_range]
        )
    else:
        raise ValueError(f"Unknown candle shape variant: {variant}")

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix.astype(float), columns


def make_action_labels(
    df: pd.DataFrame,
    horizon: int,
    commission: float = 0.0005,
    min_return: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create BUY/HOLD/SELL labels from future return."""

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    _require_columns(df, ["close"])

    close = df["close"].astype(float)
    future_returns = (close.shift(-horizon) / close - 1.0).to_numpy()
    threshold = max(float(min_return), 2.0 * float(commission))

    labels = np.full(len(df), -1, dtype=int)
    valid = ~np.isnan(future_returns)
    labels[valid] = 1
    labels[future_returns > threshold] = 2
    labels[future_returns < -threshold] = 0

    return labels, future_returns.astype(float), threshold


def split_ranges(
    n_rows: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, tuple[int, int]]:
    """Return chronological train/validation/test ranges as half-open indices."""

    if n_rows < 100:
        raise ValueError(f"Need at least 100 rows, got {n_rows}")
    if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0:
        raise ValueError("train_ratio and val_ratio must be in (0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_rows),
    }


def make_sentence_samples(
    word_tokens: list[str],
    labels: np.ndarray,
    future_returns: np.ndarray,
    split_start: int,
    split_end: int,
    window_size: int,
    horizon: int,
) -> SentenceSamples:
    """Create sentence windows for one chronological split."""

    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if split_start < 0 or split_end > len(word_tokens) or split_start >= split_end:
        raise ValueError("Invalid split range")

    sentences: list[str] = []
    token_lists: list[list[str]] = []
    y_values: list[int] = []
    returns: list[float] = []
    indices: list[int] = []

    first_target = split_start + window_size - 1
    last_target_exclusive = split_end - horizon
    for target_idx in range(first_target, last_target_exclusive):
        label = int(labels[target_idx])
        if label < 0:
            continue
        start = target_idx - window_size + 1
        tokens = word_tokens[start : target_idx + 1]
        if len(tokens) != window_size or any(token is None for token in tokens):
            continue
        sentences.append(" ".join(tokens))
        token_lists.append(list(tokens))
        y_values.append(label)
        returns.append(float(future_returns[target_idx]))
        indices.append(target_idx)

    return SentenceSamples(
        sentences=sentences,
        token_lists=token_lists,
        y=np.asarray(y_values, dtype=int),
        future_returns=np.asarray(returns, dtype=float),
        target_indices=np.asarray(indices, dtype=int),
    )


def label_distribution(y: np.ndarray) -> dict[str, Any]:
    """Return JSON-friendly class distribution."""

    result: dict[str, Any] = {}
    if len(y) == 0:
        return result

    values, counts = np.unique(y, return_counts=True)
    total = float(len(y))
    for value, count in zip(values, counts):
        name = ACTION_LABELS.get(int(value), str(int(value)))
        result[name] = {
            "count": int(count),
            "share": float(count / total),
        }
    return result
