"""Research-only action target definitions for candle-language experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .candles import ACTION_LABELS, make_action_labels


@dataclass(frozen=True)
class ActionTargetSpec:
    """Serializable action-target specification."""

    mode: str
    horizon: int = 1
    commission: float = 0.0005
    min_return: float = 0.0
    vol_window: int = 16
    vol_k: float = 1.0
    barrier_horizon: int = 3
    barrier_vol_window: int = 16
    barrier_up_k: float = 1.0
    barrier_down_k: float = 1.0
    return_threshold_mult: float = 1.0
    buy_threshold_mult: float = 1.0
    sell_threshold_mult: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        mode = self.mode.lower()
        if mode == "return_threshold":
            return f"return_threshold:h{self.horizon}:m{self.return_threshold_mult:g}"
        if mode == "volatility_adjusted_return":
            return f"vol_adj:h{self.horizon}:w{self.vol_window}:k{self.vol_k:g}"
        if mode == "triple_barrier":
            return (
                f"triple_barrier:h{self.barrier_horizon}:w{self.barrier_vol_window}:"
                f"up{self.barrier_up_k:g}:down{self.barrier_down_k:g}"
            )
        if mode == "neutral_zone_return":
            return (
                f"neutral_zone:h{self.horizon}:buy{self.buy_threshold_mult:g}:"
                f"sell{self.sell_threshold_mult:g}"
            )
        return mode

    @property
    def effective_horizon(self) -> int:
        return int(self.barrier_horizon if self.mode.lower() == "triple_barrier" else self.horizon)


@dataclass(frozen=True)
class ActionTargetResult:
    """Action labels plus diagnostics for one target definition."""

    labels: np.ndarray
    future_returns: np.ndarray
    effective_horizon: int
    threshold: float | np.ndarray
    metadata: dict[str, Any]


def make_research_action_targets(df: pd.DataFrame, spec: ActionTargetSpec) -> ActionTargetResult:
    """Build action labels for research without leaking targets into features."""

    mode = spec.mode.lower()
    if mode == "return_threshold":
        labels, future_returns, threshold = _return_threshold(df, spec)
        return ActionTargetResult(
            labels=labels,
            future_returns=future_returns,
            effective_horizon=spec.horizon,
            threshold=float(threshold),
            metadata={
                "mode": mode,
                "base_threshold": float(2.0 * float(spec.commission)),
                "return_threshold_mult": float(spec.return_threshold_mult),
                "threshold": float(threshold),
            },
        )
    if mode == "volatility_adjusted_return":
        return _volatility_adjusted_return(df, spec)
    if mode == "triple_barrier":
        return _triple_barrier(df, spec)
    if mode == "neutral_zone_return":
        return _neutral_zone_return(df, spec)
    raise ValueError(f"Unsupported target mode: {spec.mode}")


def _return_threshold(df: pd.DataFrame, spec: ActionTargetSpec) -> tuple[np.ndarray, np.ndarray, float]:
    close = df["close"].astype(float)
    future_returns = (close.shift(-spec.horizon) / close - 1.0).to_numpy(dtype=float)
    threshold = max(float(spec.min_return), 2.0 * float(spec.commission) * float(spec.return_threshold_mult))
    labels = np.full(len(df), -1, dtype=int)
    valid = np.isfinite(future_returns)
    labels[valid] = 1
    labels[valid & (future_returns > threshold)] = 2
    labels[valid & (future_returns < -threshold)] = 0
    return labels, future_returns.astype(float), float(threshold)


def past_return_volatility(df: pd.DataFrame, window: int) -> np.ndarray:
    """Past/current rolling close-return volatility known at candle t."""

    if window < 2:
        raise ValueError("volatility window must be >= 2")
    close = df["close"].astype(float)
    returns = close.pct_change()
    vol = returns.rolling(window=window, min_periods=max(2, min(window, 4))).std()
    return np.nan_to_num(vol.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def target_analysis(labels: np.ndarray, future_returns: np.ndarray, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return compact class-balance and target-shape diagnostics."""

    labels = np.asarray(labels, dtype=int)
    future_returns = np.asarray(future_returns, dtype=float)
    valid = labels >= 0
    result: dict[str, Any] = {
        "n_valid": int(np.count_nonzero(valid)),
        "n_invalid": int(np.count_nonzero(~valid)),
        "class_distribution": _label_distribution(labels[valid]),
    }
    if np.count_nonzero(valid) > 1:
        result["label_change_rate"] = float(np.mean(labels[valid][1:] != labels[valid][:-1]))
    else:
        result["label_change_rate"] = 0.0
    per_label: dict[str, Any] = {}
    for label_id, label_name in ACTION_LABELS.items():
        mask = valid & (labels == label_id) & np.isfinite(future_returns)
        values = future_returns[mask]
        per_label[label_name] = {
            "count": int(len(values)),
            "mean_future_return": float(values.mean()) if len(values) else 0.0,
            "median_future_return": float(np.median(values)) if len(values) else 0.0,
            "mean_abs_future_return": float(np.abs(values).mean()) if len(values) else 0.0,
        }
    result["per_label"] = per_label
    if metadata:
        result["metadata"] = metadata
    return result


def _volatility_adjusted_return(df: pd.DataFrame, spec: ActionTargetSpec) -> ActionTargetResult:
    close = df["close"].astype(float)
    future_returns = (close.shift(-spec.horizon) / close - 1.0).to_numpy(dtype=float)
    base_threshold = max(float(spec.min_return), 2.0 * float(spec.commission))
    past_vol = past_return_volatility(df, spec.vol_window)
    thresholds = np.maximum(base_threshold, float(spec.vol_k) * past_vol)
    labels = np.full(len(df), -1, dtype=int)
    valid = np.isfinite(future_returns)
    labels[valid] = 1
    labels[valid & (future_returns > thresholds)] = 2
    labels[valid & (future_returns < -thresholds)] = 0
    return ActionTargetResult(
        labels=labels,
        future_returns=future_returns,
        effective_horizon=spec.horizon,
        threshold=thresholds,
        metadata={
            "mode": "volatility_adjusted_return",
            "base_threshold": float(base_threshold),
            "vol_window": int(spec.vol_window),
            "vol_k": float(spec.vol_k),
            "mean_threshold": float(np.mean(thresholds[np.isfinite(thresholds)])),
            "median_threshold": float(np.median(thresholds[np.isfinite(thresholds)])),
        },
    )


def _neutral_zone_return(df: pd.DataFrame, spec: ActionTargetSpec) -> ActionTargetResult:
    close = df["close"].astype(float)
    future_returns = (close.shift(-spec.horizon) / close - 1.0).to_numpy(dtype=float)
    base_threshold = max(float(spec.min_return), 2.0 * float(spec.commission))
    buy_threshold = base_threshold * float(spec.buy_threshold_mult)
    sell_threshold = base_threshold * float(spec.sell_threshold_mult)
    labels = np.full(len(df), -1, dtype=int)
    valid = np.isfinite(future_returns)
    labels[valid] = 1
    labels[valid & (future_returns > buy_threshold)] = 2
    labels[valid & (future_returns < -sell_threshold)] = 0
    return ActionTargetResult(
        labels=labels,
        future_returns=future_returns,
        effective_horizon=spec.horizon,
        threshold={"buy": float(buy_threshold), "sell": float(sell_threshold)},
        metadata={
            "mode": "neutral_zone_return",
            "base_threshold": float(base_threshold),
            "buy_threshold": float(buy_threshold),
            "sell_threshold": float(sell_threshold),
        },
    )


def _triple_barrier(df: pd.DataFrame, spec: ActionTargetSpec) -> ActionTargetResult:
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    horizon = int(spec.barrier_horizon)
    base_threshold = max(float(spec.min_return), 2.0 * float(spec.commission))
    past_vol = past_return_volatility(df, spec.barrier_vol_window)
    upper_returns = np.maximum(base_threshold, float(spec.barrier_up_k) * past_vol)
    lower_returns = np.maximum(base_threshold, float(spec.barrier_down_k) * past_vol)
    labels = np.full(len(df), -1, dtype=int)
    future_returns = np.full(len(df), np.nan, dtype=float)
    ambiguous = 0
    no_touch = 0
    for idx in range(0, len(df) - horizon):
        if close[idx] <= 0 or not np.isfinite(close[idx]):
            continue
        future_returns[idx] = close[idx + horizon] / close[idx] - 1.0
        upper = close[idx] * (1.0 + upper_returns[idx])
        lower = close[idx] * (1.0 - lower_returns[idx])
        labels[idx] = 1
        for step in range(1, horizon + 1):
            hit_up = high[idx + step] >= upper
            hit_down = low[idx + step] <= lower
            if hit_up and hit_down:
                labels[idx] = 1
                ambiguous += 1
                break
            if hit_up:
                labels[idx] = 2
                break
            if hit_down:
                labels[idx] = 0
                break
        if labels[idx] == 1:
            no_touch += 1
    valid = labels >= 0
    return ActionTargetResult(
        labels=labels,
        future_returns=future_returns,
        effective_horizon=horizon,
        threshold={"upper": upper_returns, "lower": lower_returns},
        metadata={
            "mode": "triple_barrier",
            "base_threshold": float(base_threshold),
            "barrier_horizon": int(horizon),
            "barrier_vol_window": int(spec.barrier_vol_window),
            "barrier_up_k": float(spec.barrier_up_k),
            "barrier_down_k": float(spec.barrier_down_k),
            "ambiguous_samples": int(ambiguous),
            "ambiguous_share": float(ambiguous / max(1, np.count_nonzero(valid))),
            "no_touch_samples": int(no_touch),
            "no_touch_share": float(no_touch / max(1, np.count_nonzero(valid))),
        },
    )


def _label_distribution(labels: np.ndarray) -> dict[str, Any]:
    if len(labels) == 0:
        return {}
    values, counts = np.unique(labels, return_counts=True)
    total = float(len(labels))
    return {
        ACTION_LABELS.get(int(value), str(int(value))): {"count": int(count), "share": float(count / total)}
        for value, count in zip(values, counts)
    }
