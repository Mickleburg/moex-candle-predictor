"""Nested walk-forward threshold selection for LM-derived action features."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.data.split import WalkForwardRange
from src.nlp import (
    ClassifierSpec,
    VectorizerSpec,
    candle_shape_matrix,
    label_distribution,
    make_action_labels,
    make_lm_action_features,
    make_sentence_samples,
)
from src.nlp.classifiers import build_classifier, classifier_requires_dense, maybe_dense
from src.nlp.clustering import CandleClusterer
from src.nlp.vectorizers import build_vectorizer
from src.nlp.word_forecast import clusterer_distance_matrix
from src.nlp.word_lm import NGramBackoffLanguageModel
from src.utils.io import ensure_dir

from sber_action_lm_features_walk_forward import (
    VocabularyConfig,
    _append,
    _past_regime_arrays,
    _proper_action_probabilities,
    _quantile_thresholds,
    _regime_rows,
    action_metrics,
    build_cluster_spec,
    build_folds,
    calibration_diagnostics,
    load_sber_frame,
    parse_list,
    parse_vocab_configs,
)


@dataclass(frozen=True)
class NestedRange:
    outer_fold: WalkForwardRange
    inner_train_start: int
    inner_train_end: int
    calibration_start: int
    calibration_end: int


@dataclass(frozen=True)
class ThresholdDecision:
    mode: str
    calibration_method: str
    temperature: float
    buy_threshold: float | None
    sell_threshold: float | None
    calibration_score: float
    calibration_metrics: dict[str, Any]
    regime_thresholds: dict[str, dict[str, float]] | None = None
    is_oracle: bool = False


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_class_weights(value: str) -> list[str | None]:
    result: list[str | None] = []
    for item in parse_list(value):
        if item.lower() == "none":
            result.append(None)
        elif item.lower() == "balanced":
            result.append("balanced")
        else:
            raise ValueError(f"Unsupported class weight: {item}")
    return result


def nested_range(fold: WalkForwardRange, calibration_size: int) -> NestedRange:
    if calibration_size < 1:
        raise ValueError("calibration_size must be >= 1")
    calibration_start = fold.train_end - calibration_size
    if calibration_start <= fold.train_start:
        raise ValueError("calibration_size leaves no inner training rows")
    if not (fold.train_start < calibration_start < fold.train_end <= fold.val_start):
        raise ValueError("Invalid nested calibration range")
    return NestedRange(
        outer_fold=fold,
        inner_train_start=fold.train_start,
        inner_train_end=calibration_start,
        calibration_start=calibration_start,
        calibration_end=fold.train_end,
    )


def fit_words_for_nested(
    shape_matrix: np.ndarray,
    ranges: NestedRange,
    vocab_config: VocabularyConfig,
    *,
    random_state: int,
) -> tuple[np.ndarray, CandleClusterer]:
    clusterer = CandleClusterer(build_cluster_spec(vocab_config), random_state=random_state)
    clusterer.fit(shape_matrix[ranges.inner_train_start : ranges.inner_train_end])
    word_ids = np.full(shape_matrix.shape[0], -1, dtype=int)
    word_ids[ranges.inner_train_start : ranges.inner_train_end] = clusterer.train_labels_
    word_ids[ranges.calibration_start : ranges.calibration_end] = clusterer.predict(
        shape_matrix[ranges.calibration_start : ranges.calibration_end]
    )
    word_ids[ranges.outer_fold.val_start : ranges.outer_fold.val_end] = clusterer.predict(
        shape_matrix[ranges.outer_fold.val_start : ranges.outer_fold.val_end]
    )
    return word_ids, clusterer


def build_nested_samples(
    word_tokens: list[str],
    labels: np.ndarray,
    future_returns: np.ndarray,
    ranges: NestedRange,
    *,
    window_size: int,
    horizon: int,
    lm_context_size: int,
) -> dict[str, Any]:
    fold = ranges.outer_fold
    samples = {
        "inner_train": make_sentence_samples(
            word_tokens,
            labels,
            future_returns,
            ranges.inner_train_start,
            ranges.inner_train_end,
            window_size,
            horizon,
        ),
        "calibration": make_sentence_samples(
            word_tokens,
            labels,
            future_returns,
            ranges.calibration_start,
            ranges.calibration_end,
            window_size,
            horizon,
        ),
        "outer_val": make_sentence_samples(
            word_tokens,
            labels,
            future_returns,
            fold.val_start,
            fold.val_end,
            window_size,
            horizon,
        ),
    }
    _validate_nested_samples(samples, ranges, window_size, horizon, lm_context_size)
    return samples


def run_nested_fold_vocab_horizon(
    df: pd.DataFrame,
    shape_matrix: np.ndarray,
    labels: np.ndarray,
    future_returns: np.ndarray,
    ranges: NestedRange,
    vocab_config: VocabularyConfig,
    *,
    feature_sets: list[str],
    classifiers: list[str],
    class_weights: list[str | None],
    context_size: int,
    action_window_size: int,
    action_horizon: int,
    lm_order: int,
    lm_alpha: float,
    lm_forecast_horizon: int,
    threshold_grid: list[float],
    temperature_grid: list[float],
    threshold_modes: list[str],
    selection_metric: str,
    min_regime_calibration_samples: int,
    random_state: int,
) -> list[dict[str, Any]]:
    word_ids, clusterer = fit_words_for_nested(shape_matrix, ranges, vocab_config, random_state=random_state)
    word_tokens = clusterer.labels_to_words(word_ids)
    samples = build_nested_samples(
        word_tokens,
        labels,
        future_returns,
        ranges,
        window_size=action_window_size,
        horizon=action_horizon,
        lm_context_size=context_size,
    )
    vectorizer = build_vectorizer(
        VectorizerSpec(
            "cooccurrence_svd",
            {"embedding_dim": 24, "context_window": 2, "pool": ("mean", "std", "last"), "include_histogram": True},
        ),
        random_state=random_state,
    )
    X_base_train = vectorizer.fit_transform(samples["inner_train"].sentences, samples["inner_train"].token_lists)
    X_base_calib = vectorizer.transform(samples["calibration"].sentences, samples["calibration"].token_lists)
    X_base_val = vectorizer.transform(samples["outer_val"].sentences, samples["outer_val"].token_lists)

    lm = NGramBackoffLanguageModel(order=lm_order, alpha=lm_alpha).fit(
        word_ids,
        train_start=ranges.inner_train_start,
        train_end=ranges.inner_train_end,
        n_words=clusterer.n_words_,
    )
    distance_matrix = clusterer_distance_matrix(clusterer)
    lm_train = make_lm_action_features(
        word_ids=word_ids,
        target_indices=samples["inner_train"].target_indices,
        context_size=context_size,
        model=lm,
        distance_matrix=distance_matrix,
        include_probabilities=True,
        beam_horizon=lm_forecast_horizon,
    )
    lm_calib = make_lm_action_features(
        word_ids=word_ids,
        target_indices=samples["calibration"].target_indices,
        context_size=context_size,
        model=lm,
        distance_matrix=distance_matrix,
        include_probabilities=True,
        beam_horizon=lm_forecast_horizon,
    )
    lm_val = make_lm_action_features(
        word_ids=word_ids,
        target_indices=samples["outer_val"].target_indices,
        context_size=context_size,
        model=lm,
        distance_matrix=distance_matrix,
        include_probabilities=True,
        beam_horizon=lm_forecast_horizon,
    )

    regime_train = build_regime_labels(df, samples["inner_train"].target_indices, samples["inner_train"].target_indices, lm_train.X, lm_train.X)
    regime_calib = build_regime_labels(df, samples["inner_train"].target_indices, samples["calibration"].target_indices, lm_train.X, lm_calib.X)
    regime_val = build_regime_labels(df, samples["inner_train"].target_indices, samples["outer_val"].target_indices, lm_train.X, lm_val.X)

    rows = []
    for feature_set in feature_sets:
        X_train, X_calib, X_val = build_feature_matrices(
            feature_set,
            X_base_train,
            X_base_calib,
            X_base_val,
            lm_train.X,
            lm_calib.X,
            lm_val.X,
        )
        for class_weight in class_weights:
            for classifier_name in classifiers:
                classifier = build_classifier(
                    ClassifierSpec(classifier_name, {"max_iter": 1000, "class_weight": class_weight}),
                    random_state=random_state,
                )
                fit_train = X_train
                fit_calib = X_calib
                fit_val = X_val
                if classifier_requires_dense(ClassifierSpec(classifier_name)):
                    fit_train = maybe_dense(fit_train)
                    fit_calib = maybe_dense(fit_calib)
                    fit_val = maybe_dense(fit_val)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    classifier.fit(fit_train, samples["inner_train"].y)
                calib_proba = _require_proba(_proper_action_probabilities(classifier, fit_calib))
                val_proba = _require_proba(_proper_action_probabilities(classifier, fit_val))
                rows.extend(
                    evaluate_threshold_modes(
                        ranges,
                        samples,
                        vocab_config=vocab_config,
                        feature_set=feature_set,
                        classifier_name=classifier_name,
                        class_weight=class_weight,
                        action_horizon=action_horizon,
                        calib_proba=calib_proba,
                        val_proba=val_proba,
                        threshold_grid=threshold_grid,
                        temperature_grid=temperature_grid,
                        threshold_modes=threshold_modes,
                        selection_metric=selection_metric,
                        min_regime_calibration_samples=min_regime_calibration_samples,
                        regime_calib=regime_calib,
                        regime_val=regime_val,
                        random_state=random_state,
                    )
                )
    return rows


def evaluate_threshold_modes(
    ranges: NestedRange,
    samples: dict[str, Any],
    *,
    vocab_config: VocabularyConfig,
    feature_set: str,
    classifier_name: str,
    class_weight: str | None,
    action_horizon: int,
    calib_proba: np.ndarray,
    val_proba: np.ndarray,
    threshold_grid: list[float],
    temperature_grid: list[float],
    threshold_modes: list[str],
    selection_metric: str,
    min_regime_calibration_samples: int,
    regime_calib: dict[str, np.ndarray],
    regime_val: dict[str, np.ndarray],
    random_state: int,
) -> list[dict[str, Any]]:
    rows = []
    argmax_decision = ThresholdDecision(
        mode="argmax",
        calibration_method="none",
        temperature=1.0,
        buy_threshold=None,
        sell_threshold=None,
        calibration_score=_score_metrics(samples["calibration"].y, np.argmax(calib_proba, axis=1))[selection_metric],
        calibration_metrics=_score_metrics(samples["calibration"].y, np.argmax(calib_proba, axis=1)),
    )
    rows.append(
        result_row(
            ranges,
            samples["outer_val"].y,
            val_proba,
            argmax_decision,
            vocab_config=vocab_config,
            feature_set=feature_set,
            classifier_name=classifier_name,
            class_weight=class_weight,
            action_horizon=action_horizon,
            random_state=random_state,
            regime_labels=regime_val,
        )
    )

    if "global" in threshold_modes:
        global_decision = select_global_thresholds(
            samples["calibration"].y,
            calib_proba,
            threshold_grid,
            temperature_grid,
            selection_metric=selection_metric,
            mode="global",
        )
        rows.append(
            result_row(
                ranges,
                samples["outer_val"].y,
                val_proba,
                global_decision,
                vocab_config=vocab_config,
                feature_set=feature_set,
                classifier_name=classifier_name,
                class_weight=class_weight,
                action_horizon=action_horizon,
                random_state=random_state,
                regime_labels=regime_val,
            )
        )

    for mode in threshold_modes:
        if mode not in {"regime_volatility", "regime_trend"}:
            continue
        regime_name = "volatility" if mode == "regime_volatility" else "trend"
        regime_decision = select_regime_thresholds(
            samples["calibration"].y,
            calib_proba,
            regime_calib[regime_name],
            threshold_grid,
            temperature_grid,
            selection_metric=selection_metric,
            mode=mode,
            min_regime_calibration_samples=min_regime_calibration_samples,
        )
        rows.append(
            result_row(
                ranges,
                samples["outer_val"].y,
                val_proba,
                regime_decision,
                vocab_config=vocab_config,
                feature_set=feature_set,
                classifier_name=classifier_name,
                class_weight=class_weight,
                action_horizon=action_horizon,
                random_state=random_state,
                regime_labels=regime_val,
                active_regime=regime_name,
            )
        )

    oracle_decision = select_global_thresholds(
        samples["outer_val"].y,
        val_proba,
        threshold_grid,
        temperature_grid,
        selection_metric=selection_metric,
        mode="oracle_global",
        is_oracle=True,
    )
    rows.append(
        result_row(
            ranges,
            samples["outer_val"].y,
            val_proba,
            oracle_decision,
            vocab_config=vocab_config,
            feature_set=feature_set,
            classifier_name=classifier_name,
            class_weight=class_weight,
            action_horizon=action_horizon,
            random_state=random_state,
            regime_labels=regime_val,
        )
    )
    return rows


def select_global_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold_grid: list[float],
    temperature_grid: list[float],
    *,
    selection_metric: str,
    mode: str,
    is_oracle: bool = False,
) -> ThresholdDecision:
    candidates = []
    for temperature in temperature_grid:
        calibrated = apply_temperature(proba, temperature)
        for buy_threshold in threshold_grid:
            for sell_threshold in threshold_grid:
                pred = threshold_predictions(calibrated, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
                metrics = _score_metrics(y_true, pred)
                candidates.append(
                    {
                        "temperature": float(temperature),
                        "buy_threshold": float(buy_threshold),
                        "sell_threshold": float(sell_threshold),
                        "metrics": metrics,
                    }
                )
    best = max(candidates, key=lambda row: (row["metrics"][selection_metric], row["metrics"]["buy_f1"] + row["metrics"]["sell_f1"]))
    return ThresholdDecision(
        mode=mode,
        calibration_method="temperature_scaling",
        temperature=best["temperature"],
        buy_threshold=best["buy_threshold"],
        sell_threshold=best["sell_threshold"],
        calibration_score=float(best["metrics"][selection_metric]),
        calibration_metrics=best["metrics"],
        is_oracle=is_oracle,
    )


def select_regime_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    regimes: np.ndarray,
    threshold_grid: list[float],
    temperature_grid: list[float],
    *,
    selection_metric: str,
    mode: str,
    min_regime_calibration_samples: int,
) -> ThresholdDecision:
    global_decision = select_global_thresholds(
        y_true,
        proba,
        threshold_grid,
        temperature_grid,
        selection_metric=selection_metric,
        mode="global_fallback",
    )
    calibrated = apply_temperature(proba, global_decision.temperature)
    regime_thresholds: dict[str, dict[str, float]] = {}
    pred = np.full(len(y_true), 1, dtype=int)
    for regime in sorted({str(item) for item in regimes}):
        mask = regimes == regime
        if np.count_nonzero(mask) < min_regime_calibration_samples:
            regime_thresholds[regime] = {
                "buy_threshold": float(global_decision.buy_threshold or 0.0),
                "sell_threshold": float(global_decision.sell_threshold or 0.0),
                "n_calibration_samples": int(np.count_nonzero(mask)),
                "fallback": True,
            }
        else:
            local = select_global_thresholds(
                y_true[mask],
                calibrated[mask],
                threshold_grid,
                [1.0],
                selection_metric=selection_metric,
                mode=f"{mode}:{regime}",
            )
            regime_thresholds[regime] = {
                "buy_threshold": float(local.buy_threshold or 0.0),
                "sell_threshold": float(local.sell_threshold or 0.0),
                "n_calibration_samples": int(np.count_nonzero(mask)),
                "fallback": False,
            }
        row = regime_thresholds[regime]
        pred[mask] = threshold_predictions(
            calibrated[mask],
            buy_threshold=row["buy_threshold"],
            sell_threshold=row["sell_threshold"],
        )
    metrics = _score_metrics(y_true, pred)
    return ThresholdDecision(
        mode=mode,
        calibration_method="temperature_scaling",
        temperature=global_decision.temperature,
        buy_threshold=global_decision.buy_threshold,
        sell_threshold=global_decision.sell_threshold,
        calibration_score=float(metrics[selection_metric]),
        calibration_metrics=metrics,
        regime_thresholds=regime_thresholds,
    )


def result_row(
    ranges: NestedRange,
    y_true: np.ndarray,
    val_proba: np.ndarray,
    decision: ThresholdDecision,
    *,
    vocab_config: VocabularyConfig,
    feature_set: str,
    classifier_name: str,
    class_weight: str | None,
    action_horizon: int,
    random_state: int,
    regime_labels: dict[str, np.ndarray],
    active_regime: str | None = None,
) -> dict[str, Any]:
    calibrated = apply_temperature(val_proba, decision.temperature)
    if decision.mode == "argmax":
        pred = np.argmax(calibrated, axis=1)
    elif decision.regime_thresholds and active_regime:
        pred = np.full(len(y_true), 1, dtype=int)
        for regime, thresholds in decision.regime_thresholds.items():
            mask = regime_labels[active_regime] == regime
            pred[mask] = threshold_predictions(
                calibrated[mask],
                buy_threshold=thresholds["buy_threshold"],
                sell_threshold=thresholds["sell_threshold"],
            )
    else:
        pred = threshold_predictions(
            calibrated,
            buy_threshold=float(decision.buy_threshold or 0.0),
            sell_threshold=float(decision.sell_threshold or 0.0),
        )
    metrics = action_metrics(y_true, pred)
    metrics["action_rate"] = float(np.mean(np.isin(pred, [0, 2])))
    metrics["hold_rate"] = float(np.mean(pred == 1))
    return {
        "outer_fold_id": int(ranges.outer_fold.fold_id),
        "random_state": int(random_state),
        "inner_train_start": int(ranges.inner_train_start),
        "inner_train_end": int(ranges.inner_train_end),
        "calibration_start": int(ranges.calibration_start),
        "calibration_end": int(ranges.calibration_end),
        "outer_val_start": int(ranges.outer_fold.val_start),
        "outer_val_end": int(ranges.outer_fold.val_end),
        "vocabulary": vocab_config.label,
        "feature_set": feature_set,
        "classifier": classifier_name,
        "class_weight": "none" if class_weight is None else str(class_weight),
        "action_horizon": int(action_horizon),
        "calibration_method": decision.calibration_method,
        "temperature": float(decision.temperature),
        "threshold_mode": decision.mode,
        "selected_buy_threshold": decision.buy_threshold,
        "selected_sell_threshold": decision.sell_threshold,
        "selection_metric_on_calibration": float(decision.calibration_score),
        "is_oracle": bool(decision.is_oracle),
        "metrics": metrics,
        "prediction_distribution": label_distribution(pred),
        "true_distribution": label_distribution(y_true),
        "calibration_metrics": decision.calibration_metrics,
        "outer_validation_metrics": metrics,
        "calibration_reliability_table": calibration_diagnostics(y_true, pred, calibrated),
        "regime_thresholds_if_any": decision.regime_thresholds,
        "regime_results": regime_rows_for_predictions(regime_labels, y_true, pred, calibrated, decision),
    }


def build_feature_matrices(
    feature_set: str,
    base_train: Any,
    base_calib: Any,
    base_val: Any,
    lm_train: np.ndarray,
    lm_calib: np.ndarray,
    lm_val: np.ndarray,
) -> tuple[Any, Any, Any]:
    if feature_set == "lm_only":
        return lm_train, lm_calib, lm_val
    if feature_set == "base_lm_proba":
        return _append(base_train, lm_train), _append(base_calib, lm_calib), _append(base_val, lm_val)
    if feature_set == "base":
        return base_train, base_calib, base_val
    if feature_set == "lm_scalar":
        scalar_width = 18
        return lm_train[:, :scalar_width], lm_calib[:, :scalar_width], lm_val[:, :scalar_width]
    if feature_set == "base_lm_scalar":
        scalar_width = 18
        return (
            _append(base_train, lm_train[:, :scalar_width]),
            _append(base_calib, lm_calib[:, :scalar_width]),
            _append(base_val, lm_val[:, :scalar_width]),
        )
    raise ValueError(f"Unsupported feature set: {feature_set}")


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    logits = np.log(np.maximum(np.asarray(proba, dtype=float), 1e-12)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    calibrated = exp / exp.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(calibrated)) or not np.allclose(calibrated.sum(axis=1), 1.0):
        raise ValueError("Invalid calibrated probabilities")
    return calibrated


def threshold_predictions(proba: np.ndarray, *, buy_threshold: float, sell_threshold: float) -> np.ndarray:
    pred = np.full(proba.shape[0], 1, dtype=int)
    buy_ok = (proba[:, 2] >= buy_threshold) & (proba[:, 2] >= proba[:, 0])
    sell_ok = (proba[:, 0] >= sell_threshold) & ~buy_ok
    pred[buy_ok] = 2
    pred[sell_ok] = 0
    return pred


def build_regime_labels(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    target_indices: np.ndarray,
    lm_train: np.ndarray,
    lm_target: np.ndarray,
) -> dict[str, np.ndarray]:
    vol, trend, hours = _past_regime_arrays(df)
    return {
        "volatility": _three_bucket(vol[train_indices], vol[target_indices], ("low_vol", "mid_vol", "high_vol")),
        "trend": _three_bucket(trend[train_indices], trend[target_indices], ("downtrend", "flat", "uptrend")),
        "session": _session_labels(hours[target_indices]),
        "lm_uncertainty": _three_bucket(lm_train[:, 3], lm_target[:, 3], ("low_entropy", "mid_entropy", "high_entropy")),
    }


def regime_rows_for_predictions(
    regime_labels: dict[str, np.ndarray],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    decision: ThresholdDecision,
) -> dict[str, list[dict[str, Any]]]:
    action_confidence = proba.max(axis=1)
    lm_confidence = np.zeros(len(y_true), dtype=float)
    result = {}
    for name, labels in regime_labels.items():
        rows = _regime_rows(labels, y_true, y_pred, action_confidence, lm_confidence)
        for row in rows:
            row["threshold_mode"] = decision.mode
            if decision.regime_thresholds and row["regime"] in decision.regime_thresholds:
                row["selected_thresholds_used"] = decision.regime_thresholds[row["regime"]]
            else:
                row["selected_thresholds_used"] = {
                    "buy_threshold": decision.buy_threshold,
                    "sell_threshold": decision.sell_threshold,
                    "temperature": decision.temperature,
                }
            row["action_rate"] = 1.0 - row["prediction_distribution"].get("HOLD", {}).get("share", 0.0)
        result[name] = rows
    return result


def _three_bucket(train_values: np.ndarray, target_values: np.ndarray, names: tuple[str, str, str]) -> np.ndarray:
    low, high = _quantile_thresholds(train_values)
    labels = np.full(len(target_values), names[1], dtype=object)
    labels[np.asarray(target_values, dtype=float) <= low] = names[0]
    labels[np.asarray(target_values, dtype=float) >= high] = names[2]
    return labels


def _session_labels(hours: np.ndarray) -> np.ndarray:
    labels = np.full(len(hours), "middle", dtype=object)
    labels[hours < 12] = "early"
    labels[hours >= 17] = "late"
    return labels


def _score_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = action_metrics(y_true, y_pred)
    return {
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "buy_f1": metrics["buy_f1"],
        "sell_f1": metrics["sell_f1"],
        "hold_f1": metrics["hold_f1"],
        "action_rate": float(np.mean(np.isin(y_pred, [0, 2]))),
    }


def _require_proba(proba: np.ndarray | None) -> np.ndarray:
    if proba is None:
        raise ValueError("Nested threshold CLI currently requires classifiers with predict_proba")
    if not np.all(np.isfinite(proba)) or not np.allclose(proba.sum(axis=1), 1.0):
        raise ValueError("Invalid action probabilities")
    return proba


def _validate_nested_samples(samples: dict[str, Any], ranges: NestedRange, window_size: int, horizon: int, lm_context_size: int) -> None:
    split_bounds = {
        "inner_train": (ranges.inner_train_start, ranges.inner_train_end),
        "calibration": (ranges.calibration_start, ranges.calibration_end),
        "outer_val": (ranges.outer_fold.val_start, ranges.outer_fold.val_end),
    }
    for split_name, sample in samples.items():
        start, end = split_bounds[split_name]
        expected = (end - start) - window_size - horizon + 1
        if sample.size != expected:
            raise ValueError(f"{split_name} sample count mismatch: got {sample.size}, expected {expected}")
        if np.any(sample.target_indices - window_size + 1 < start):
            raise ValueError(f"{split_name} action context crosses split boundary")
        if np.any(sample.target_indices - lm_context_size + 1 < start):
            raise ValueError(f"{split_name} LM context crosses split boundary")
        if np.any(sample.target_indices + horizon >= end):
            raise ValueError(f"{split_name} action label crosses split boundary")


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["vocabulary"],
            row["feature_set"],
            row["classifier"],
            row["class_weight"],
            row["action_horizon"],
            row["threshold_mode"],
            row["calibration_method"],
            row["is_oracle"],
        )
        grouped.setdefault(key, []).append(row)
    result = []
    for key, items in grouped.items():
        macro = np.asarray([item["metrics"]["macro_f1"] for item in items], dtype=float)
        buy = np.asarray([item["metrics"]["buy_f1"] for item in items], dtype=float)
        sell = np.asarray([item["metrics"]["sell_f1"] for item in items], dtype=float)
        action_rate = np.asarray([item["metrics"]["action_rate"] for item in items], dtype=float)
        result.append(
            {
                "vocabulary": key[0],
                "feature_set": key[1],
                "classifier": key[2],
                "class_weight": key[3],
                "action_horizon": int(key[4]),
                "threshold_mode": key[5],
                "calibration_method": key[6],
                "is_oracle": bool(key[7]),
                "folds": int(len(items)),
                "outer_val_macro_f1_mean": float(macro.mean()),
                "outer_val_macro_f1_std": float(macro.std(ddof=0)),
                "outer_val_macro_f1_worst": float(macro.min()),
                "buy_f1_mean": float(buy.mean()),
                "sell_f1_mean": float(sell.mean()),
                "mean_action_rate": float(action_rate.mean()),
            }
        )
    return sorted(result, key=lambda row: (row["is_oracle"], row["outer_val_macro_f1_mean"]), reverse=True)


def select_best_honest(aggregates: list[dict[str, Any]]) -> dict[str, Any] | None:
    honest = [row for row in aggregates if not row["is_oracle"]]
    if not honest:
        return None
    return max(
        enumerate(honest),
        key=lambda item: (
            item[1]["outer_val_macro_f1_mean"],
            item[1]["outer_val_macro_f1_worst"],
            item[1]["buy_f1_mean"] + item[1]["sell_f1_mean"],
            -item[0],
        ),
    )[1]


def compact_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        metrics = row["metrics"]
        compact.append(
            {
                "outer_fold_id": row["outer_fold_id"],
                "vocabulary": row["vocabulary"],
                "feature_set": row["feature_set"],
                "classifier": row["classifier"],
                "class_weight": row["class_weight"],
                "action_horizon": row["action_horizon"],
                "calibration_method": row["calibration_method"],
                "temperature": row["temperature"],
                "threshold_mode": row["threshold_mode"],
                "selected_buy_threshold": row["selected_buy_threshold"],
                "selected_sell_threshold": row["selected_sell_threshold"],
                "is_oracle": row["is_oracle"],
                "selection_metric_on_calibration": row["selection_metric_on_calibration"],
                "outer_val_macro_f1": metrics["macro_f1"],
                "outer_val_accuracy": metrics["accuracy"],
                "outer_val_balanced_accuracy": metrics["balanced_accuracy"],
                "buy_f1": metrics["buy_f1"],
                "sell_f1": metrics["sell_f1"],
                "hold_f1": metrics["hold_f1"],
                "action_rate": metrics["action_rate"],
            }
        )
    return compact


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def print_summary(aggregates: list[dict[str, Any]]) -> None:
    print("vocabulary | horizon | class_weight | mode | oracle | macro-F1 | worst | BUY F1 | SELL F1 | action_rate")
    print("--- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---:")
    for row in aggregates[:16]:
        print(
            f"{row['vocabulary']} | {row['action_horizon']} | {row['class_weight']} | "
            f"{row['threshold_mode']} | {row['is_oracle']} | {row['outer_val_macro_f1_mean']:.4f} | "
            f"{row['outer_val_macro_f1_worst']:.4f} | {row['buy_f1_mean']:.4f} | "
            f"{row['sell_f1_mean']:.4f} | {row['mean_action_rate']:.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--timeframe", default="1H")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data", default="")
    parser.add_argument("--vocab-configs", default="shape:gmm:20,shape:gmm:16")
    parser.add_argument("--feature-sets", default="lm_only")
    parser.add_argument("--classifiers", default="logreg")
    parser.add_argument("--class-weights", default="none,balanced")
    parser.add_argument("--action-horizons", default="1,3")
    parser.add_argument("--context-size", type=int, default=16)
    parser.add_argument("--action-window-size", type=int, default=32)
    parser.add_argument("--forecast-horizon", type=int, default=3)
    parser.add_argument("--lm-order", type=int, default=2)
    parser.add_argument("--lm-alpha", type=float, default=0.1)
    parser.add_argument("--fold-mode", choices=["expanding", "rolling"], default="rolling")
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument("--initial-train-size", type=int, default=12000)
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=3000)
    parser.add_argument("--step-size", type=int, default=3000)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--calibration-size", type=int, default=2500)
    parser.add_argument("--threshold-grid", default="0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--temperature-grid", default="0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--threshold-modes", default="argmax,global,regime_volatility,regime_trend")
    parser.add_argument("--selection-metric", choices=["macro_f1", "balanced_accuracy", "buy_f1", "sell_f1"], default="macro_f1")
    parser.add_argument("--min-regime-calibration-samples", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-json", default="data/reports/sber_h1_action_nested_thresholds_rolling_20260515.json")
    parser.add_argument("--output-csv", default="data/reports/sber_h1_action_nested_thresholds_rolling_20260515.csv")
    args = parser.parse_args()

    if args.quick:
        args.vocab_configs = "shape:gmm:20"
        args.action_horizons = "1"
        args.threshold_modes = "argmax,global"
        args.max_folds = min(args.max_folds, 2)

    started = time.perf_counter()
    df, data_path = load_sber_frame(args)
    folds = build_folds(args, len(df))
    vocab_configs = parse_vocab_configs(args.vocab_configs)
    feature_sets = parse_list(args.feature_sets)
    classifiers = parse_list(args.classifiers)
    class_weights = parse_class_weights(args.class_weights)
    action_horizons = parse_int_list(args.action_horizons)
    threshold_grid = parse_float_list(args.threshold_grid)
    temperature_grid = parse_float_list(args.temperature_grid)
    threshold_modes = parse_list(args.threshold_modes)
    shape_cache = {config.shape_variant: candle_shape_matrix(df, variant=config.shape_variant)[0] for config in vocab_configs}

    rows: list[dict[str, Any]] = []
    print(f"Загружено свечей: {len(df)}; файл: {data_path}")
    print(f"Outer folds: {len(folds)}; режим: {args.fold_mode}; calibration_size={args.calibration_size}")
    for fold in folds:
        ranges = nested_range(fold, args.calibration_size)
        print(
            f"Fold {fold.fold_id}: inner=[{ranges.inner_train_start}:{ranges.inner_train_end}) "
            f"calib=[{ranges.calibration_start}:{ranges.calibration_end}) val=[{fold.val_start}:{fold.val_end})"
        )
        for action_horizon in action_horizons:
            labels, future_returns, threshold = make_action_labels(df, horizon=action_horizon, commission=0.0005)
            print(f"  Action horizon {action_horizon}; label threshold={threshold:.6f}")
            for vocab_config in vocab_configs:
                print(f"    Vocabulary {vocab_config.label}")
                rows.extend(
                    run_nested_fold_vocab_horizon(
                        df,
                        shape_cache[vocab_config.shape_variant],
                        labels,
                        future_returns,
                        ranges,
                        vocab_config,
                        feature_sets=feature_sets,
                        classifiers=classifiers,
                        class_weights=class_weights,
                        context_size=args.context_size,
                        action_window_size=args.action_window_size,
                        action_horizon=action_horizon,
                        lm_order=args.lm_order,
                        lm_alpha=args.lm_alpha,
                        lm_forecast_horizon=args.forecast_horizon,
                        threshold_grid=threshold_grid,
                        temperature_grid=temperature_grid,
                        threshold_modes=threshold_modes,
                        selection_metric=args.selection_metric,
                        min_regime_calibration_samples=args.min_regime_calibration_samples,
                        random_state=args.random_state,
                    )
                )

    aggregates = aggregate_rows(rows)
    best_honest = select_best_honest(aggregates)
    payload = {
        "data_path": str(data_path),
        "rows": int(len(df)),
        "fold_mode": args.fold_mode,
        "folds": [asdict(fold) for fold in folds],
        "nested_ranges": [asdict(nested_range(fold, args.calibration_size)) for fold in folds],
        "selection": "thresholds and temperature selected on calibration split; outer validation is evaluation only; oracle rows are diagnostic and excluded from best_honest",
        "vocab_configs": [asdict(config) for config in vocab_configs],
        "feature_sets": feature_sets,
        "classifiers": classifiers,
        "class_weights": ["none" if item is None else item for item in class_weights],
        "action_horizons": action_horizons,
        "threshold_grid": threshold_grid,
        "temperature_grid": temperature_grid,
        "threshold_modes": threshold_modes,
        "fold_results": rows,
        "aggregates": aggregates,
        "best_honest": best_honest,
        "duration_sec": float(time.perf_counter() - started),
    }
    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    write_json(payload, output_json)
    write_csv(compact_csv_rows(rows), output_csv)
    print_summary(aggregates)
    if best_honest:
        print(
            "Лучший честный config: "
            f"{best_honest['vocabulary']} | horizon={best_honest['action_horizon']} | "
            f"{best_honest['class_weight']} | {best_honest['threshold_mode']}"
        )
    print(f"Записан JSON: {output_json}")
    print(f"Записан CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
