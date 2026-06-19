"""Experiment runner for candle-language models."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .candles import (
    ACTION_LABELS,
    SentenceSamples,
    candle_shape_matrix,
    label_distribution,
    make_action_labels,
    make_sentence_samples,
    split_ranges,
)
from .classifiers import ClassifierSpec, build_classifier, classifier_requires_dense, maybe_dense
from .clustering import CandleClusterer, ClusterSpec
from .vectorizers import VectorizerSpec, build_vectorizer


@dataclass(frozen=True)
class ExperimentConfig:
    """Full candle-language experiment configuration."""

    shape_variant: str
    horizon: int
    window_size: int
    commission: float
    cluster: ClusterSpec
    vectorizer: VectorizerSpec
    classifier: ClassifierSpec
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    min_return: float = 0.0

    @property
    def label(self) -> str:
        return (
            f"shape={self.shape_variant}|h={self.horizon}|w={self.window_size}|"
            f"{self.cluster.label}|{self.vectorizer.label}|{self.classifier.label}"
        )


def run_experiment(
    df: pd.DataFrame,
    config: ExperimentConfig,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run one full train/validation/test experiment."""

    started = time.perf_counter()
    df = df.sort_values("begin").reset_index(drop=True) if "begin" in df.columns else df.reset_index(drop=True)

    shape_matrix, shape_columns = candle_shape_matrix(df, variant=config.shape_variant)
    ranges = split_ranges(len(df), train_ratio=config.train_ratio, val_ratio=config.val_ratio)
    labels, future_returns, threshold = make_action_labels(
        df,
        horizon=config.horizon,
        commission=config.commission,
        min_return=config.min_return,
    )

    train_start, train_end = ranges["train"]
    clusterer = CandleClusterer(config.cluster, random_state=random_state)
    clusterer.fit(shape_matrix[train_start:train_end])

    word_ids = np.empty(len(df), dtype=int)
    word_ids[train_start:train_end] = clusterer.train_labels_
    for split_name in ("val", "test"):
        split_start, split_end = ranges[split_name]
        word_ids[split_start:split_end] = clusterer.predict(shape_matrix[split_start:split_end])
    word_tokens = clusterer.labels_to_words(word_ids)

    samples = {
        split_name: make_sentence_samples(
            word_tokens,
            labels,
            future_returns,
            split_start,
            split_end,
            config.window_size,
            config.horizon,
        )
        for split_name, (split_start, split_end) in ranges.items()
    }
    _validate_samples(samples)

    vectorizer = build_vectorizer(config.vectorizer, random_state=random_state)
    X_train = vectorizer.fit_transform(samples["train"].sentences, samples["train"].token_lists)
    X_val = vectorizer.transform(samples["val"].sentences, samples["val"].token_lists)
    X_test = vectorizer.transform(samples["test"].sentences, samples["test"].token_lists)

    classifier = build_classifier(config.classifier, random_state=random_state)
    if classifier_requires_dense(config.classifier):
        X_train = maybe_dense(X_train)
        X_val = maybe_dense(X_val)
        X_test = maybe_dense(X_test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier.fit(X_train, samples["train"].y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = {
            "train": classifier.predict(X_train),
            "val": classifier.predict(X_val),
            "test": classifier.predict(X_test),
        }

    metrics = {
        split_name: classification_metrics(samples[split_name].y, predictions[split_name])
        for split_name in ("train", "val", "test")
    }
    trading = {
        split_name: trading_metrics(
            predictions[split_name],
            samples[split_name].future_returns,
            commission=config.commission,
        )
        for split_name in ("val", "test")
    }

    finished = time.perf_counter()
    return {
        "label": config.label,
        "config": config_to_dict(config),
        "shape_columns": shape_columns,
        "target_threshold": float(threshold),
        "split_ranges": {name: [int(start), int(end)] for name, (start, end) in ranges.items()},
        "sample_counts": {name: sample.size for name, sample in samples.items()},
        "target_distribution": {name: label_distribution(sample.y) for name, sample in samples.items()},
        "cluster": clusterer.quality_,
        "metrics": metrics,
        "trading": trading,
        "duration_sec": float(finished - started),
    }


def _validate_samples(samples: dict[str, SentenceSamples]) -> None:
    for split_name, sample in samples.items():
        if sample.size == 0:
            raise ValueError(f"No samples for {split_name}")
    train_classes = np.unique(samples["train"].y)
    if len(train_classes) < 2:
        raise ValueError(f"Need at least two train classes, got {train_classes.tolist()}")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute fixed-label action classification metrics."""

    labels = [0, 1, 2]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def trading_metrics(
    y_pred: np.ndarray,
    future_returns: np.ndarray,
    commission: float = 0.0005,
) -> dict[str, float]:
    """Evaluate a simple long/short/flat policy from action predictions."""

    signals = np.zeros(len(y_pred), dtype=float)
    signals[y_pred == 2] = 1.0
    signals[y_pred == 0] = -1.0

    valid = ~np.isnan(future_returns)
    if not np.any(valid):
        return {
            "trade_rate": 0.0,
            "long_rate": 0.0,
            "short_rate": 0.0,
            "total_return": 0.0,
            "mean_return": 0.0,
            "win_rate": 0.0,
            "period_sharpe": 0.0,
        }

    signals = signals[valid]
    returns = future_returns[valid]
    gross = signals * returns
    costs = np.abs(signals) * 2.0 * commission
    net = gross - costs
    trades = signals != 0.0
    std = float(net.std())
    return {
        "trade_rate": float(trades.mean()),
        "long_rate": float((signals == 1.0).mean()),
        "short_rate": float((signals == -1.0).mean()),
        "total_return": float(net.sum()),
        "mean_return": float(net.mean()),
        "avg_trade_return": float(net[trades].mean()) if np.any(trades) else 0.0,
        "win_rate": float((net[trades] > 0.0).mean()) if np.any(trades) else 0.0,
        "period_sharpe": float(net.mean() / std) if std > 0.0 else 0.0,
    }


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert dataclass config to a compact JSON-friendly dict."""

    return {
        "shape_variant": config.shape_variant,
        "horizon": int(config.horizon),
        "window_size": int(config.window_size),
        "commission": float(config.commission),
        "train_ratio": float(config.train_ratio),
        "val_ratio": float(config.val_ratio),
        "min_return": float(config.min_return),
        "cluster": {"name": config.cluster.name, "params": dict(config.cluster.params)},
        "vectorizer": {"name": config.vectorizer.name, "params": dict(config.vectorizer.params)},
        "classifier": {"name": config.classifier.name, "params": dict(config.classifier.params)},
        "action_labels": ACTION_LABELS,
    }
