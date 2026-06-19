"""Leakage-safe LM-derived features for downstream action classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .word_lm import NGramBackoffLanguageModel


@dataclass(frozen=True)
class LMActionFeatures:
    """Feature matrix and column names for action samples."""

    X: np.ndarray
    names: list[str]


def make_lm_action_features(
    *,
    word_ids: Sequence[int],
    target_indices: Sequence[int],
    context_size: int,
    model: NGramBackoffLanguageModel,
    distance_matrix: np.ndarray | None = None,
    include_probabilities: bool = False,
    include_topn: int = 3,
    beam_horizon: int = 3,
    beam_width: int = 3,
) -> LMActionFeatures:
    """Build next-word LM features using only words known at each target index.

    The function does not accept future target words by design. For a sample
    ending at ``target_idx=t``, the LM context is
    ``words[t-context_size+1]..words[t]``.
    """

    if context_size < 1:
        raise ValueError("context_size must be >= 1")
    if model.n_words_ is None:
        raise ValueError("Language model is not fitted")

    word_array = np.asarray(word_ids, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    n_words = int(model.n_words_)
    scalar_rows: list[list[float]] = []
    proba_rows: list[np.ndarray] = []

    for target_idx in target_indices:
        start = int(target_idx) - context_size + 1
        if start < 0:
            raise ValueError("LM context would start before the word sequence")
        context = word_array[start : int(target_idx) + 1]
        if len(context) != context_size:
            raise ValueError("LM context length mismatch")
        if np.any(context < 0):
            raise ValueError("LM context contains unassigned word IDs")

        proba = model.next_proba(context)
        if not np.all(np.isfinite(proba)) or not np.isclose(proba.sum(), 1.0):
            raise ValueError("Invalid LM probability distribution")
        current_word = int(context[-1])
        order = np.argsort(proba)[::-1]
        top = order[: max(2, include_topn)]
        top1 = int(top[0])
        top2 = int(top[1]) if len(top) > 1 else top1
        top3 = order[: min(3, n_words)]
        top3_mass = float(proba[top3].sum())
        entropy = float(-np.sum(proba[proba > 0] * np.log(proba[proba > 0])))
        expected_distance = 0.0
        if distance_matrix is not None and distance_matrix.size and 0 <= current_word < distance_matrix.shape[0]:
            expected_distance = float(np.dot(proba, distance_matrix[current_word, :n_words]))

        beam = model.beam_search(context, max(1, beam_horizon), beam_width=beam_width)
        beam_best = beam[0].log_probability if beam else 0.0
        beam_second = beam[1].log_probability if len(beam) > 1 else beam_best
        mean_step_entropy = _rollout_mean_entropy(model, context, max(1, beam_horizon))

        row = [
            float(proba[top1]),
            float(proba[top2]),
            top3_mass,
            entropy,
            float(proba[top1] - proba[top2]),
            float(top1 / max(1, n_words - 1)),
            expected_distance,
            float(proba[current_word]) if 0 <= current_word < n_words else 0.0,
            float(beam_best),
            float(beam_second),
            float(beam_best - beam_second),
            mean_step_entropy,
        ]
        for rank in range(include_topn):
            word = int(order[rank]) if rank < len(order) else 0
            row.extend([float(word / max(1, n_words - 1)), float(proba[word])])
        scalar_rows.append(row)
        if include_probabilities:
            proba_rows.append(proba.astype(float))

    names = [
        "lm_top1_prob",
        "lm_top2_prob",
        "lm_top3_mass",
        "lm_entropy",
        "lm_margin_top1_top2",
        "lm_predicted_next_word_id_norm",
        "lm_expected_centroid_distance_from_current",
        "lm_self_transition_probability",
        "lm_beam_best_logprob",
        "lm_beam_second_logprob",
        "lm_beam_margin",
        "lm_mean_step_entropy",
    ]
    for rank in range(include_topn):
        names.extend([f"lm_top{rank + 1}_word_id_norm", f"lm_top{rank + 1}_prob"])

    X = np.asarray(scalar_rows, dtype=float)
    if include_probabilities:
        X = np.hstack([X, np.vstack(proba_rows)])
        names.extend([f"lm_word_proba_{idx}" for idx in range(n_words)])
    if not np.all(np.isfinite(X)):
        raise ValueError("LM action features contain non-finite values")
    return LMActionFeatures(X=X, names=names)


def _rollout_mean_entropy(model: NGramBackoffLanguageModel, context: np.ndarray, horizon: int) -> float:
    entropies = []
    running = [int(item) for item in context]
    for _ in range(horizon):
        proba = model.next_proba(running)
        positive = proba[proba > 0]
        entropies.append(float(-np.sum(positive * np.log(positive))))
        running.append(int(np.argmax(proba)))
    return float(np.mean(entropies)) if entropies else 0.0


def make_continuous_past_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build leakage-safe continuous features known at the close of candle t."""

    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    safe_open = open_.where(open_.abs() > 1e-12, np.nan)
    safe_close = close.where(close.abs() > 1e-12, np.nan)
    candle_range = (high - low).where((high - low).abs() > 1e-12, np.nan)
    ret1 = close.pct_change()

    features: dict[str, pd.Series] = {}
    for period in (1, 3, 6, 12, 24):
        features[f"ret_{period}"] = close.pct_change(period)
    for window in (8, 16, 32):
        features[f"vol_{window}"] = ret1.rolling(window=window, min_periods=max(2, min(4, window))).std()
        features[f"range_mean_{window}"] = ((high - low) / safe_close).rolling(
            window=window, min_periods=max(2, min(4, window))
        ).mean()
        ema = close.ewm(span=window, adjust=False).mean()
        features[f"ema_distance_{window}"] = (close - ema) / ema.where(ema.abs() > 1e-12, np.nan)

    body_signed = (close - open_) / safe_open
    features["body_signed"] = body_signed
    features["body_abs"] = body_signed.abs()
    features["range_to_open"] = (high - low) / safe_open
    features["upper_shadow"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / safe_open
    features["lower_shadow"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / safe_open
    features["close_position_in_candle"] = (close - low) / candle_range

    prev_volume_mean = volume.shift(1).rolling(window=20, min_periods=4).mean()
    prev_volume_std = volume.shift(1).rolling(window=20, min_periods=4).std()
    features["volume_ratio_20"] = volume / prev_volume_mean.where(prev_volume_mean.abs() > 1e-12, np.nan)
    features["volume_z_20"] = (volume - prev_volume_mean) / prev_volume_std.where(prev_volume_std.abs() > 1e-12, np.nan)

    if "begin" in df.columns:
        begin = pd.to_datetime(df["begin"])
        hour = begin.dt.hour.astype(float)
        dow = begin.dt.dayofweek.astype(float)
        features["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        features["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        features["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        features["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
        hour_gap = begin.diff().dt.total_seconds().div(3600.0)
        features["large_time_gap_flag"] = (hour_gap > 3.0).astype(float)
    else:
        zeros = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        features["hour_sin"] = zeros
        features["hour_cos"] = zeros
        features["dow_sin"] = zeros
        features["dow_cos"] = zeros
        features["large_time_gap_flag"] = zeros

    names = list(features)
    matrix = pd.DataFrame(features, index=df.index)[names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = matrix.to_numpy(dtype=float)
    if not np.all(np.isfinite(X)):
        raise ValueError("Continuous past features contain non-finite values")
    return X, names


def make_lag_sequence_features(df: pd.DataFrame, n_lags: int = 10) -> tuple[np.ndarray, list[str]]:
    """Past-only lag features that capture price trajectory shape, not just snapshot.

    The current continuous_regime feature set has cumulative returns (ret_3, ret_6...)
    but loses trajectory shape: a 3h cumulative return of +0.5% could be three equal
    small ups or one big move followed by reversals. These lags expose the path.

    Features added (all past-only, no lookahead at row i):
      - lag_ret_{k}:     individual 1h return k steps back, for k=2..n_lags
                         (k=1 already covered by ret_1)
      - lag_body_{k}:    signed candle body k steps back: (close-open)/open, k=1..5
      - lag_vol_ratio_{k}: volume k steps back relative to 20-bar mean, k=1..5
      - ret_day_{d}:     ~d-day return using session length (7h/day), d=1..3
      - day_range:       (max_high - min_low) of last 7 candles / close
      - close_in_day_range: close position within last 7 candles' range [0, 1]
      - up_streak:       consecutive positive 1h returns ending now (0..5)
      - down_streak:     consecutive negative 1h returns ending now (0..5)
    """
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    safe_open = open_.where(open_.abs() > 1e-12, np.nan)

    features: dict[str, pd.Series] = {}

    # Individual 1h lag returns: what did price do k hours ago?
    for k in range(2, n_lags + 1):
        features[f"lag_ret_{k}"] = close.shift(k - 1).pct_change()

    # Lag candle bodies (direction + magnitude k steps ago)
    for k in range(1, 6):
        body = (close.shift(k) - open_.shift(k)) / safe_open.shift(k)
        features[f"lag_body_{k}"] = body

    # Lag volume ratios
    vol_mean = volume.rolling(window=20, min_periods=4).mean()
    for k in range(1, 6):
        features[f"lag_vol_ratio_{k}"] = volume.shift(k) / vol_mean.shift(k).where(
            vol_mean.shift(k).abs() > 1e-12, np.nan
        )

    # Multi-day returns (MOEX session ~7h, so 7 candles ≈ 1 trading day)
    SESSION = 7
    for d in range(1, 4):
        features[f"ret_day_{d}"] = close.pct_change(d * SESSION)

    # Close position within last-session range
    last_high = high.rolling(window=SESSION, min_periods=2).max()
    last_low = low.rolling(window=SESSION, min_periods=2).min()
    day_range = (last_high - last_low) / close.where(close.abs() > 1e-12, np.nan)
    features["day_range"] = day_range
    rng = (last_high - last_low).where((last_high - last_low).abs() > 1e-12, np.nan)
    features["close_in_day_range"] = (close - last_low) / rng

    # Streak features (how many consecutive up/down hours ending now)
    ret1 = close.pct_change()
    ret1_arr = ret1.to_numpy()
    up = np.zeros(len(df), dtype=float)
    dn = np.zeros(len(df), dtype=float)
    for i in range(1, len(df)):
        max_streak = 5
        u, d_ = 0, 0
        for lag in range(1, max_streak + 1):
            if i - lag < 0:
                break
            if ret1_arr[i - lag + 1] > 0:
                u += 1
            else:
                break
        for lag in range(1, max_streak + 1):
            if i - lag < 0:
                break
            if ret1_arr[i - lag + 1] < 0:
                d_ += 1
            else:
                break
        up[i] = float(u)
        dn[i] = float(d_)
    features["up_streak"] = pd.Series(up, index=df.index)
    features["down_streak"] = pd.Series(dn, index=df.index)

    names = list(features)
    matrix = pd.DataFrame(features, index=df.index)[names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = matrix.to_numpy(dtype=float)
    if not np.all(np.isfinite(X)):
        raise ValueError("Lag sequence features contain non-finite values")
    return X, names


def standardize_by_train(
    feature_matrix: np.ndarray,
    train_indices: Sequence[int],
    target_indices: Sequence[int],
) -> np.ndarray:
    """Select target rows and standardize them using train target rows only."""

    X = np.asarray(feature_matrix, dtype=float)
    train_indices = np.asarray(train_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    train = X[train_indices]
    target = X[target_indices]
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std = np.where(std < 1e-12, 1.0, std)
    result = (target - mean) / std
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(result)):
        raise ValueError("Standardized continuous features contain non-finite values")
    return result
