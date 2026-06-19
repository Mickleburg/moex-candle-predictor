"""Walk-forward action classification with candle-word LM-derived features."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.data import clean_candles
from src.data.split import WalkForwardRange, rolling_walk_forward_ranges, walk_forward_ranges
from src.nlp import (
    ClassifierSpec,
    ClusterSpec,
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
from src.nlp.word_lm import NGramBackoffLanguageModel, word_distribution_metrics
from src.utils.io import ensure_dir

from sber_next_word_research import find_latest_raw


ACTION_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}


@dataclass(frozen=True)
class VocabularyConfig:
    shape_variant: str
    clusterer: str
    vocab_size: int
    covariance_type: str = "diag"

    @property
    def label(self) -> str:
        if self.clusterer == "gmm":
            return f"{self.shape_variant}/gmm_{self.covariance_type}/{self.vocab_size}"
        return f"{self.shape_variant}/{self.clusterer}/{self.vocab_size}"


def parse_vocab_configs(value: str) -> list[VocabularyConfig]:
    configs = []
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        parts = item.split(":")
        if len(parts) not in {3, 4}:
            raise ValueError(f"Invalid vocab config '{item}', expected shape:clusterer:size[:covariance]")
        configs.append(
            VocabularyConfig(
                shape_variant=parts[0],
                clusterer=parts[1],
                vocab_size=int(parts[2]),
                covariance_type=parts[3] if len(parts) == 4 else "diag",
            )
        )
    return configs


def parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_cluster_spec(config: VocabularyConfig) -> ClusterSpec:
    if config.clusterer == "kmeans":
        return ClusterSpec("kmeans", {"n_clusters": config.vocab_size, "n_init": 10})
    if config.clusterer == "gmm":
        return ClusterSpec(
            "gmm",
            {"n_components": config.vocab_size, "covariance_type": config.covariance_type, "reg_covar": 1e-6},
        )
    raise ValueError(f"Unsupported clusterer: {config.clusterer}")


def build_folds(args: argparse.Namespace, n_rows: int) -> list[WalkForwardRange]:
    if args.fold_mode == "rolling":
        return rolling_walk_forward_ranges(
            n_rows,
            train_size=args.train_size,
            val_size=args.val_size,
            step_size=args.step_size,
            max_folds=args.max_folds,
            gap=args.gap,
        )
    return walk_forward_ranges(
        n_rows,
        n_splits=args.max_folds,
        initial_train_size=args.initial_train_size,
        val_size=args.val_size,
        gap=args.gap,
        min_train_size=min(1000, args.initial_train_size),
    )


def load_sber_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    data_path = Path(args.data).resolve() if args.data else find_latest_raw(REPO_ROOT / args.raw_dir, args.ticker, args.timeframe)
    df = pd.read_parquet(data_path)
    if "ticker" in df.columns:
        df = df[df["ticker"] == args.ticker]
    if "timeframe" in df.columns:
        df = df[df["timeframe"] == args.timeframe]
    df = clean_candles(df)
    df = df.sort_values("begin").reset_index(drop=True) if "begin" in df.columns else df.reset_index(drop=True)
    return df, data_path


def fit_fold_words(
    shape_matrix: np.ndarray,
    fold: WalkForwardRange,
    config: VocabularyConfig,
    *,
    random_state: int,
) -> tuple[np.ndarray, CandleClusterer]:
    clusterer = CandleClusterer(build_cluster_spec(config), random_state=random_state)
    clusterer.fit(shape_matrix[fold.train_start : fold.train_end])
    word_ids = np.full(shape_matrix.shape[0], -1, dtype=int)
    word_ids[fold.train_start : fold.train_end] = clusterer.train_labels_
    word_ids[fold.val_start : fold.val_end] = clusterer.predict(shape_matrix[fold.val_start : fold.val_end])
    return word_ids, clusterer


def run_fold_vocab(
    df: pd.DataFrame,
    shape_matrix: np.ndarray,
    labels: np.ndarray,
    future_returns: np.ndarray,
    fold: WalkForwardRange,
    vocab_config: VocabularyConfig,
    *,
    feature_sets: list[str],
    classifier_names: list[str],
    context_size: int,
    action_window_size: int,
    action_horizon: int,
    lm_order: int,
    lm_alpha: float,
    lm_forecast_horizon: int,
    random_state: int,
    include_calibration: bool = False,
    include_regime_analysis: bool = False,
    include_threshold_sweep: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    word_ids, clusterer = fit_fold_words(shape_matrix, fold, vocab_config, random_state=random_state)
    word_tokens = clusterer.labels_to_words(word_ids)
    samples = {
        "train": make_sentence_samples(
            word_tokens,
            labels,
            future_returns,
            fold.train_start,
            fold.train_end,
            action_window_size,
            action_horizon,
        ),
        "val": make_sentence_samples(
            word_tokens,
            labels,
            future_returns,
            fold.val_start,
            fold.val_end,
            action_window_size,
            action_horizon,
        ),
    }
    _validate_action_samples(samples, fold, action_window_size, action_horizon, context_size)

    vectorizer = build_vectorizer(
        VectorizerSpec(
            "cooccurrence_svd",
            {"embedding_dim": 24, "context_window": 2, "pool": ("mean", "std", "last"), "include_histogram": True},
        ),
        random_state=random_state,
    )
    X_base_train = vectorizer.fit_transform(samples["train"].sentences, samples["train"].token_lists)
    X_base_val = vectorizer.transform(samples["val"].sentences, samples["val"].token_lists)

    lm = NGramBackoffLanguageModel(order=lm_order, alpha=lm_alpha).fit(
        word_ids,
        train_start=fold.train_start,
        train_end=fold.train_end,
        n_words=clusterer.n_words_,
    )
    distance_matrix = clusterer_distance_matrix(clusterer)
    lm_scalar_train = make_lm_action_features(
        word_ids=word_ids,
        target_indices=samples["train"].target_indices,
        context_size=context_size,
        model=lm,
        distance_matrix=distance_matrix,
        include_probabilities=False,
        beam_horizon=lm_forecast_horizon,
    )
    lm_scalar_val = make_lm_action_features(
        word_ids=word_ids,
        target_indices=samples["val"].target_indices,
        context_size=context_size,
        model=lm,
        distance_matrix=distance_matrix,
        include_probabilities=False,
        beam_horizon=lm_forecast_horizon,
    )
    lm_proba_train = make_lm_action_features(
        word_ids=word_ids,
        target_indices=samples["train"].target_indices,
        context_size=context_size,
        model=lm,
        distance_matrix=distance_matrix,
        include_probabilities=True,
        beam_horizon=lm_forecast_horizon,
    )
    lm_proba_val = make_lm_action_features(
        word_ids=word_ids,
        target_indices=samples["val"].target_indices,
        context_size=context_size,
        model=lm,
        distance_matrix=distance_matrix,
        include_probabilities=True,
        beam_horizon=lm_forecast_horizon,
    )
    train_word_distribution = _word_distribution_row(
        word_ids,
        fold,
        vocab_config,
        random_state=random_state,
        n_words=clusterer.n_words_,
        split_name="train",
    )
    val_word_distribution = _word_distribution_row(
        word_ids,
        fold,
        vocab_config,
        random_state=random_state,
        n_words=clusterer.n_words_,
        split_name="val",
    )

    rows = []
    for feature_set in feature_sets:
        X_train, X_val = _build_feature_matrix(
            feature_set,
            X_base_train,
            X_base_val,
            lm_scalar_train.X,
            lm_scalar_val.X,
            lm_proba_train.X,
            lm_proba_val.X,
        )
        lm_val_scalar = _lm_scalar_dict(lm_scalar_val.X)
        for classifier_name in classifier_names:
            classifier_spec = _classifier_spec(classifier_name)
            classifier = build_classifier(classifier_spec, random_state=random_state)
            fit_train = X_train
            fit_val = X_val
            if classifier_requires_dense(classifier_spec):
                fit_train = maybe_dense(fit_train)
                fit_val = maybe_dense(fit_val)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                classifier.fit(fit_train, samples["train"].y)
                pred = classifier.predict(fit_val)
            proba = _action_probabilities(classifier, fit_val)
            calibrated_proba = _proper_action_probabilities(classifier, fit_val)
            metrics = action_metrics(samples["val"].y, pred)
            row = {
                "fold_id": int(fold.fold_id),
                "random_state": int(random_state),
                "train_start": int(fold.train_start),
                "train_end": int(fold.train_end),
                "val_start": int(fold.val_start),
                "val_end": int(fold.val_end),
                "train_rows": int(fold.train_len),
                "val_rows": int(fold.val_len),
                "vocabulary": vocab_config.label,
                "feature_set": feature_set,
                "classifier": classifier_name,
                "n_train_samples": int(samples["train"].size),
                "n_val_samples": int(samples["val"].size),
                "metrics": metrics,
                "prediction_distribution": label_distribution(pred),
                "true_distribution": label_distribution(samples["val"].y),
                "confidence": confidence_curves(samples["val"].y, pred, proba, lm_val_scalar),
            }
            if include_calibration:
                row["calibration"] = calibration_diagnostics(samples["val"].y, pred, calibrated_proba)
            if include_regime_analysis:
                row["regime_error_analysis"] = regime_error_analysis(
                    df,
                    samples["train"],
                    samples["val"],
                    pred,
                    proba,
                    lm_scalar_train.X,
                    lm_scalar_val.X,
                )
            if include_threshold_sweep:
                row["threshold_sweep"] = threshold_sweep(samples["val"].y, calibrated_proba)
            rows.append(row)
    diagnostics = {
        "fold_id": int(fold.fold_id),
        "random_state": int(random_state),
        "vocabulary": vocab_config.label,
        "train": train_word_distribution,
        "val": val_word_distribution,
    }
    return rows, diagnostics


def action_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _balanced_accuracy_fixed_labels(y_true, y_pred),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
    }
    for idx, label in enumerate([0, 1, 2]):
        name = ACTION_NAMES[label]
        result[f"{name.lower()}_precision"] = float(precision[idx])
        result[f"{name.lower()}_recall"] = float(recall[idx])
        result[f"{name.lower()}_f1"] = float(f1[idx])
        result[f"{name.lower()}_support"] = int(support[idx])
    return result


def _balanced_accuracy_fixed_labels(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for label in [0, 1, 2]:
        mask = y_true == label
        recalls.append(float(np.mean(y_pred[mask] == label)) if np.any(mask) else 0.0)
    return float(np.mean(recalls))


def calibration_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None) -> dict[str, Any]:
    """Validation-only reliability diagnostics for true probability models."""

    if proba is None:
        return {"available": False, "reason": "classifier_has_no_predict_proba"}
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] != 3:
        raise ValueError("Action probability matrix must have shape (n_samples, 3)")
    if not np.all(np.isfinite(proba)) or not np.allclose(proba.sum(axis=1), 1.0):
        raise ValueError("Action probabilities must be finite and sum to 1")

    confidence = proba.max(axis=1)
    one_hot = np.eye(3, dtype=float)[np.asarray(y_true, dtype=int)]
    brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1))) if len(y_true) else 0.0
    bins = [0.0, 0.33, 0.40, 0.50, 0.60, 0.70, 1.0 + 1e-12]
    rows = []
    ece = 0.0
    for left, right in zip(bins, bins[1:]):
        mask = (confidence >= left) & (confidence < right)
        count = int(np.count_nonzero(mask))
        if count == 0:
            rows.append(
                {
                    "bin": f"[{left:.2f},{min(right, 1.0):.2f})",
                    "count": 0,
                    "coverage": 0.0,
                    "mean_confidence": None,
                    "accuracy": None,
                    "macro_f1": None,
                    "true_distribution": {},
                    "prediction_distribution": {},
                }
            )
            continue
        bucket_accuracy = float(accuracy_score(y_true[mask], y_pred[mask]))
        bucket_confidence = float(confidence[mask].mean())
        ece += float(count / len(y_true)) * abs(bucket_accuracy - bucket_confidence)
        rows.append(
            {
                "bin": f"[{left:.2f},{min(right, 1.0):.2f})",
                "count": count,
                "coverage": float(count / len(y_true)),
                "mean_confidence": bucket_confidence,
                "accuracy": bucket_accuracy,
                "macro_f1": float(f1_score(y_true[mask], y_pred[mask], labels=[0, 1, 2], average="macro", zero_division=0)),
                "true_distribution": label_distribution(y_true[mask]),
                "prediction_distribution": label_distribution(y_pred[mask]),
            }
        )
    return {
        "available": True,
        "expected_calibration_error": float(ece),
        "brier_score": brier,
        "mean_confidence": float(confidence.mean()) if len(confidence) else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "reliability_table": rows,
    }


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray | None) -> dict[str, Any]:
    """Validation-only BUY/SELL threshold diagnostic for probabilistic classifiers."""

    if proba is None:
        return {"available": False, "reason": "classifier_has_no_predict_proba"}
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    rows = []
    for buy_threshold in thresholds:
        for sell_threshold in thresholds:
            pred = _threshold_predictions(proba, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
            metrics = action_metrics(y_true, pred)
            rows.append(
                {
                    "buy_threshold": float(buy_threshold),
                    "sell_threshold": float(sell_threshold),
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "buy_precision": metrics["buy_precision"],
                    "buy_recall": metrics["buy_recall"],
                    "buy_f1": metrics["buy_f1"],
                    "sell_precision": metrics["sell_precision"],
                    "sell_recall": metrics["sell_recall"],
                    "sell_f1": metrics["sell_f1"],
                    "action_rate": float(np.mean(np.isin(pred, [0, 2]))),
                    "hold_rate": float(np.mean(pred == 1)),
                    "prediction_distribution": label_distribution(pred),
                }
            )
    return {
        "available": True,
        "selection_note": "validation-only diagnostic; not a production threshold",
        "best_by_macro_f1": max(rows, key=lambda row: (row["macro_f1"], row["buy_f1"] + row["sell_f1"])),
        "rows": sorted(rows, key=lambda row: (row["macro_f1"], row["buy_f1"] + row["sell_f1"]), reverse=True),
    }


def regime_error_analysis(
    df: pd.DataFrame,
    train_samples: Any,
    val_samples: Any,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
    lm_scalar_train: np.ndarray,
    lm_scalar_val: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    """Validation error diagnostics by past/current regimes.

    Quantile thresholds are fitted on train sample indices only. Validation
    labels are used only inside the final regime metrics.
    """

    vol, trend, hours = _past_regime_arrays(df)
    train_idx = np.asarray(train_samples.target_indices, dtype=int)
    val_idx = np.asarray(val_samples.target_indices, dtype=int)
    regimes = {
        "volatility": _assign_three_bucket_regime(vol[val_idx], _quantile_thresholds(vol[train_idx]), ("low_vol", "mid_vol", "high_vol")),
        "trend": _assign_three_bucket_regime(trend[val_idx], _quantile_thresholds(trend[train_idx]), ("downtrend", "flat", "uptrend")),
        "session": _session_regime(hours[val_idx]),
        "lm_uncertainty": _assign_three_bucket_regime(
            lm_scalar_val[:, 3],
            _quantile_thresholds(lm_scalar_train[:, 3]),
            ("low_entropy", "mid_entropy", "high_entropy"),
        ),
    }
    action_confidence = proba.max(axis=1) if proba is not None else np.full(len(y_pred), np.nan)
    lm_confidence = lm_scalar_val[:, 0]
    return {
        name: _regime_rows(labels, val_samples.y, y_pred, action_confidence, lm_confidence)
        for name, labels in regimes.items()
    }


def confidence_curves(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
    lm_scalar: dict[str, np.ndarray],
) -> dict[str, Any]:
    if proba is None:
        action_conf = np.ones(len(y_pred), dtype=float)
        action_margin = np.zeros(len(y_pred), dtype=float)
        action_entropy = np.zeros(len(y_pred), dtype=float)
    else:
        sorted_proba = np.sort(proba, axis=1)[:, ::-1]
        action_conf = sorted_proba[:, 0]
        action_margin = sorted_proba[:, 0] - sorted_proba[:, 1]
        action_entropy = -np.sum(np.where(proba > 0, proba * np.log(np.maximum(proba, 1e-300)), 0.0), axis=1)

    curves = {
        "summary": {
            "mean_action_confidence": float(action_conf.mean()) if len(action_conf) else 0.0,
            "mean_action_margin": float(action_margin.mean()) if len(action_margin) else 0.0,
            "mean_action_entropy": float(action_entropy.mean()) if len(action_entropy) else 0.0,
            "mean_lm_top1_prob": float(lm_scalar["top1"].mean()) if len(y_pred) else 0.0,
            "mean_lm_top3_mass": float(lm_scalar["top3"].mean()) if len(y_pred) else 0.0,
        },
        "thresholds": [],
    }
    masks = [
        ("action_confidence>=0.40", action_conf >= 0.40),
        ("action_confidence>=0.50", action_conf >= 0.50),
        ("action_confidence>=0.60", action_conf >= 0.60),
        ("lm_top1_prob>=0.50", lm_scalar["top1"] >= 0.50),
        ("lm_top3_mass>=0.80", lm_scalar["top3"] >= 0.80),
        ("action_confidence>=0.50_and_lm_top1_prob>=0.50", (action_conf >= 0.50) & (lm_scalar["top1"] >= 0.50)),
    ]
    for label, mask in masks:
        curves["thresholds"].append(_covered_metrics(label, mask, y_true, y_pred))
    return curves


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["vocabulary"], row["feature_set"], row["classifier"]), []).append(row)

    aggregates = []
    for (vocabulary, feature_set, classifier), items in grouped.items():
        macro = np.asarray([item["metrics"]["macro_f1"] for item in items], dtype=float)
        acc = np.asarray([item["metrics"]["accuracy"] for item in items], dtype=float)
        buy = np.asarray([item["metrics"]["buy_f1"] for item in items], dtype=float)
        sell = np.asarray([item["metrics"]["sell_f1"] for item in items], dtype=float)
        seed_means = _metric_means_by_key(items, "random_state", "macro_f1")
        fold_means = _metric_means_by_key(items, "fold_id", "macro_f1")
        pred_buy = np.asarray([_class_share(item["prediction_distribution"], "BUY") for item in items], dtype=float)
        pred_sell = np.asarray([_class_share(item["prediction_distribution"], "SELL") for item in items], dtype=float)
        pred_hold = np.asarray([_class_share(item["prediction_distribution"], "HOLD") for item in items], dtype=float)
        aggregates.append(
            {
                "vocabulary": vocabulary,
                "feature_set": feature_set,
                "classifier": classifier,
                "folds": int(len(items)),
                "random_states": int(len({int(item.get("random_state", 0)) for item in items})),
                "val_macro_f1_mean": float(macro.mean()),
                "val_macro_f1_std": float(macro.std(ddof=0)),
                "val_macro_f1_worst": float(macro.min()),
                "val_macro_f1_std_across_random_state": float(np.std(list(seed_means.values()), ddof=0)) if seed_means else 0.0,
                "val_macro_f1_worst_random_state": float(min(seed_means.values())) if seed_means else float(macro.min()),
                "val_macro_f1_worst_fold_mean": float(min(fold_means.values())) if fold_means else float(macro.min()),
                "val_accuracy_mean": float(acc.mean()),
                "buy_f1_mean": float(buy.mean()),
                "buy_f1_std": float(buy.std(ddof=0)),
                "buy_f1_worst": float(buy.min()),
                "sell_f1_mean": float(sell.mean()),
                "sell_f1_std": float(sell.std(ddof=0)),
                "sell_f1_worst": float(sell.min()),
                "prediction_buy_share_mean": float(pred_buy.mean()),
                "prediction_buy_share_std": float(pred_buy.std(ddof=0)),
                "prediction_sell_share_mean": float(pred_sell.mean()),
                "prediction_sell_share_std": float(pred_sell.std(ddof=0)),
                "prediction_hold_share_mean": float(pred_hold.mean()),
                "prediction_hold_share_std": float(pred_hold.std(ddof=0)),
                "coverage_action_conf_050_mean": _mean_threshold(items, "action_confidence>=0.50", "coverage"),
                "macro_f1_action_conf_050_mean": _mean_threshold(items, "action_confidence>=0.50", "macro_f1"),
                "coverage_lm_top1_050_mean": _mean_threshold(items, "lm_top1_prob>=0.50", "coverage"),
                "macro_f1_lm_top1_050_mean": _mean_threshold(items, "lm_top1_prob>=0.50", "macro_f1"),
            }
        )
    return sorted(aggregates, key=lambda row: (row["val_macro_f1_mean"], row["val_macro_f1_worst"], row["val_accuracy_mean"]), reverse=True)


def select_best_by_validation(aggregates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not aggregates:
        return None
    return max(
        enumerate(aggregates),
        key=lambda item: (
            item[1]["val_macro_f1_mean"],
            item[1]["val_macro_f1_worst"],
            item[1]["buy_f1_mean"] + item[1]["sell_f1_mean"],
            -item[0],
        ),
    )[1]


def _classifier_spec(name: str) -> ClassifierSpec:
    if name == "ridge":
        return ClassifierSpec("ridge", {"alpha": 1.0})
    if name == "logreg":
        return ClassifierSpec("logreg", {"max_iter": 1000})
    if name == "linear_svc":
        return ClassifierSpec("linear_svc", {"C": 0.5})
    raise ValueError(f"Unsupported classifier: {name}")


def _build_feature_matrix(
    feature_set: str,
    base_train: Any,
    base_val: Any,
    scalar_train: np.ndarray,
    scalar_val: np.ndarray,
    proba_train: np.ndarray,
    proba_val: np.ndarray,
):
    if feature_set == "base":
        return base_train, base_val
    if feature_set == "lm_scalar":
        return scalar_train, scalar_val
    if feature_set == "base_lm_scalar":
        return _append(base_train, scalar_train), _append(base_val, scalar_val)
    if feature_set == "base_lm_proba":
        return _append(base_train, proba_train), _append(base_val, proba_val)
    if feature_set == "lm_only":
        return proba_train, proba_val
    raise ValueError(f"Unsupported feature set: {feature_set}")


def _append(base: Any, extra: np.ndarray):
    if sparse.issparse(base):
        return sparse.hstack([base, sparse.csr_matrix(extra)], format="csr")
    return np.hstack([np.asarray(base), extra])


def _proper_action_probabilities(classifier: Any, X: Any) -> np.ndarray | None:
    if not hasattr(classifier, "predict_proba"):
        return None
    raw = classifier.predict_proba(X)
    classes = classifier.classes_
    return _expand_class_probabilities(raw, classes)


def _action_probabilities(classifier: Any, X: Any) -> np.ndarray | None:
    if hasattr(classifier, "predict_proba"):
        raw = classifier.predict_proba(X)
        classes = classifier.classes_
    elif hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        raw = _softmax(scores)
        classes = classifier.classes_
    else:
        return None
    return _expand_class_probabilities(raw, classes)


def _expand_class_probabilities(raw: np.ndarray, classes: np.ndarray) -> np.ndarray:
    full = np.zeros((raw.shape[0], 3), dtype=float)
    for idx, cls in enumerate(classes):
        if 0 <= int(cls) <= 2:
            full[:, int(cls)] = raw[:, idx]
    row_sums = full.sum(axis=1, keepdims=True)
    return np.divide(full, row_sums, out=np.ones_like(full) / 3.0, where=row_sums > 0)


def _threshold_predictions(proba: np.ndarray, *, buy_threshold: float, sell_threshold: float) -> np.ndarray:
    pred = np.full(proba.shape[0], 1, dtype=int)
    sell_ok = proba[:, 0] >= sell_threshold
    buy_ok = proba[:, 2] >= buy_threshold
    buy_only = buy_ok & ~sell_ok
    sell_only = sell_ok & ~buy_ok
    both = buy_ok & sell_ok
    pred[buy_only] = 2
    pred[sell_only] = 0
    pred[both] = np.where(proba[both, 2] >= proba[both, 0], 2, 0)
    return pred


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def _lm_scalar_dict(X: np.ndarray) -> dict[str, np.ndarray]:
    return {"top1": X[:, 0], "top3": X[:, 2], "entropy": X[:, 3], "margin": X[:, 4]}


def _word_distribution_row(
    word_ids: np.ndarray,
    fold: WalkForwardRange,
    config: VocabularyConfig,
    *,
    random_state: int,
    n_words: int,
    split_name: str,
) -> dict[str, Any]:
    start, end = (fold.train_start, fold.train_end) if split_name == "train" else (fold.val_start, fold.val_end)
    words = np.asarray(word_ids[start:end], dtype=int)
    counts = np.bincount(words[(words >= 0) & (words < n_words)], minlength=n_words)
    metrics = word_distribution_metrics(words, n_words)
    return {
        "fold_id": int(fold.fold_id),
        "random_state": int(random_state),
        "vocabulary": config.label,
        "split": split_name,
        "n_words": int(n_words),
        "counts": counts.astype(int).tolist(),
        "shares": (counts / max(1, counts.sum())).astype(float).tolist(),
        **metrics,
    }


def _past_regime_arrays(df: pd.DataFrame, window: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    open_ = df["open"].astype(float).replace(0.0, np.nan)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    intrabar_range = ((high - low) / open_).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    volatility = intrabar_range.rolling(window, min_periods=1).mean().to_numpy(dtype=float)
    trend = close.pct_change(window).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    if "begin" in df.columns:
        hours = pd.to_datetime(df["begin"]).dt.hour.to_numpy(dtype=int)
    else:
        hours = np.zeros(len(df), dtype=int)
    return volatility, trend, hours


def _quantile_thresholds(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0, 0.0
    low, high = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
    return float(low), float(high)


def _assign_three_bucket_regime(values: np.ndarray, thresholds: tuple[float, float], names: tuple[str, str, str]) -> np.ndarray:
    low, high = thresholds
    labels = np.full(len(values), names[1], dtype=object)
    labels[np.asarray(values, dtype=float) <= low] = names[0]
    labels[np.asarray(values, dtype=float) >= high] = names[2]
    return labels


def _session_regime(hours: np.ndarray) -> np.ndarray:
    labels = np.full(len(hours), "middle", dtype=object)
    labels[hours < 12] = "early"
    labels[hours >= 17] = "late"
    return labels


def _regime_rows(
    regime_labels: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    action_confidence: np.ndarray,
    lm_confidence: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for label in sorted({str(item) for item in regime_labels}):
        mask = regime_labels == label
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        metrics = action_metrics(y_true[mask], y_pred[mask])
        rows.append(
            {
                "regime": label,
                "n_samples": count,
                "coverage": float(count / len(regime_labels)),
                "true_distribution": label_distribution(y_true[mask]),
                "prediction_distribution": label_distribution(y_pred[mask]),
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "buy_precision": metrics["buy_precision"],
                "buy_recall": metrics["buy_recall"],
                "buy_f1": metrics["buy_f1"],
                "sell_precision": metrics["sell_precision"],
                "sell_recall": metrics["sell_recall"],
                "sell_f1": metrics["sell_f1"],
                "hold_precision": metrics["hold_precision"],
                "hold_recall": metrics["hold_recall"],
                "hold_f1": metrics["hold_f1"],
                "mean_action_confidence": _nanmean(action_confidence[mask]),
                "mean_lm_confidence": _nanmean(lm_confidence[mask]),
            }
        )
    return rows


def _nanmean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else None


def _covered_metrics(label: str, mask: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {
            "threshold": label,
            "coverage": 0.0,
            "accuracy": None,
            "macro_f1": None,
            "buy_precision": None,
            "sell_precision": None,
            "trade_like_action_rate": None,
        }
    metrics = action_metrics(y_true[mask], y_pred[mask])
    return {
        "threshold": label,
        "coverage": float(count / len(mask)),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "buy_precision": metrics["buy_precision"],
        "sell_precision": metrics["sell_precision"],
        "buy_recall": metrics["buy_recall"],
        "sell_recall": metrics["sell_recall"],
        "hold_recall": metrics["hold_recall"],
        "trade_like_action_rate": float(np.mean(np.isin(y_pred[mask], [0, 2]))),
        "prediction_distribution": label_distribution(y_pred[mask]),
    }


def _mean_threshold(items: list[dict[str, Any]], threshold: str, key: str) -> float | None:
    values = []
    for item in items:
        for row in item["confidence"]["thresholds"]:
            if row["threshold"] == threshold and row[key] is not None:
                values.append(row[key])
    return float(np.mean(values)) if values else None


def _metric_means_by_key(items: list[dict[str, Any]], key: str, metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for item in items:
        grouped.setdefault(int(item[key]), []).append(float(item["metrics"][metric]))
    return {group: float(np.mean(values)) for group, values in grouped.items()}


def _class_share(distribution: dict[str, Any], label: str) -> float:
    item = distribution.get(label)
    return float(item.get("share", 0.0)) if isinstance(item, dict) else 0.0


def random_state_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["vocabulary"], row["feature_set"], row["classifier"], int(row["random_state"])), []).append(row)
    result = []
    for (vocabulary, feature_set, classifier, random_state), items in grouped.items():
        macro = np.asarray([item["metrics"]["macro_f1"] for item in items], dtype=float)
        buy = np.asarray([item["metrics"]["buy_f1"] for item in items], dtype=float)
        sell = np.asarray([item["metrics"]["sell_f1"] for item in items], dtype=float)
        result.append(
            {
                "vocabulary": vocabulary,
                "feature_set": feature_set,
                "classifier": classifier,
                "random_state": int(random_state),
                "folds": int(len(items)),
                "macro_f1_mean": float(macro.mean()),
                "macro_f1_std": float(macro.std(ddof=0)),
                "macro_f1_worst": float(macro.min()),
                "buy_f1_mean": float(buy.mean()),
                "sell_f1_mean": float(sell.mean()),
            }
        )
    return sorted(result, key=lambda row: (row["macro_f1_mean"], row["macro_f1_worst"]), reverse=True)


def word_distribution_stability(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for item in diagnostics:
        for split in ["train", "val"]:
            row = item[split]
            grouped.setdefault((row["vocabulary"], split), []).append(np.asarray(row["shares"], dtype=float))
    result = []
    for (vocabulary, split), shares in grouped.items():
        width = max(len(item) for item in shares)
        padded = np.vstack([np.pad(item, (0, width - len(item))) for item in shares])
        center = padded.mean(axis=0)
        l1 = np.abs(padded - center).sum(axis=1)
        result.append(
            {
                "vocabulary": vocabulary,
                "split": split,
                "items": int(len(shares)),
                "mean_l1_to_mean_distribution": float(l1.mean()),
                "max_l1_to_mean_distribution": float(l1.max()),
            }
        )
    return sorted(result, key=lambda row: (row["vocabulary"], row["split"]))


def compact_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        pred_dist = row["prediction_distribution"]
        compact.append(
            {
                "config": f"{row['vocabulary']}|{row['feature_set']}|{row['classifier']}",
                "random_state": row["random_state"],
                "fold_mode": row.get("fold_mode", ""),
                "fold_id": row["fold_id"],
                "vocabulary": row["vocabulary"],
                "feature_set": row["feature_set"],
                "classifier": row["classifier"],
                "macro_f1": row["metrics"]["macro_f1"],
                "buy_f1": row["metrics"]["buy_f1"],
                "sell_f1": row["metrics"]["sell_f1"],
                "accuracy": row["metrics"]["accuracy"],
                "balanced_accuracy": row["metrics"]["balanced_accuracy"],
                "prediction_buy_share": _class_share(pred_dist, "BUY"),
                "prediction_sell_share": _class_share(pred_dist, "SELL"),
                "prediction_hold_share": _class_share(pred_dist, "HOLD"),
            }
        )
    return compact


def _collect_optional(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    collected = []
    for row in rows:
        if key not in row:
            continue
        collected.append(
            {
                "fold_id": row["fold_id"],
                "random_state": row["random_state"],
                "vocabulary": row["vocabulary"],
                "feature_set": row["feature_set"],
                "classifier": row["classifier"],
                key: row[key],
            }
        )
    return collected


def _validate_action_samples(samples: dict[str, Any], fold: WalkForwardRange, window_size: int, horizon: int, lm_context_size: int) -> None:
    for split_name, sample in samples.items():
        split_len = fold.train_len if split_name == "train" else fold.val_len
        expected = split_len - window_size - horizon + 1
        if sample.size != expected:
            raise ValueError(f"{split_name} sample count mismatch: got {sample.size}, expected {expected}")
        start = fold.train_start if split_name == "train" else fold.val_start
        end = fold.train_end if split_name == "train" else fold.val_end
        if np.any(sample.target_indices - window_size + 1 < start):
            raise ValueError(f"{split_name} action context crosses fold boundary")
        if np.any(sample.target_indices - lm_context_size + 1 < start):
            raise ValueError(f"{split_name} LM context crosses fold boundary")
        if np.any(sample.target_indices + horizon >= end):
            raise ValueError(f"{split_name} action label crosses fold boundary")


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def print_summary(aggregates: list[dict[str, Any]]) -> None:
    print("vocabulary | features | classifier | macro-F1 | worst | accuracy | BUY F1 | SELL F1")
    print("--- | --- | --- | ---: | ---: | ---: | ---: | ---:")
    for row in aggregates[:12]:
        print(
            f"{row['vocabulary']} | {row['feature_set']} | {row['classifier']} | "
            f"{row['val_macro_f1_mean']:.4f} | {row['val_macro_f1_worst']:.4f} | "
            f"{row['val_accuracy_mean']:.4f} | {row['buy_f1_mean']:.4f} | {row['sell_f1_mean']:.4f}"
        )


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--timeframe", default="1H")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data", default="")
    parser.add_argument("--vocab-configs", default="shape:kmeans:20,shape:kmeans:16,shape:gmm:16")
    parser.add_argument("--context-size", type=int, default=16)
    parser.add_argument("--action-window-size", type=int, default=32)
    parser.add_argument("--action-horizon", type=int, default=1)
    parser.add_argument("--forecast-horizon", type=int, default=3)
    parser.add_argument("--lm-order", type=int, default=2)
    parser.add_argument("--lm-alpha", type=float, default=0.1)
    parser.add_argument("--fold-mode", choices=["expanding", "rolling"], default="expanding")
    parser.add_argument("--max-folds", type=int, default=3)
    parser.add_argument("--initial-train-size", type=int, default=12000)
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=3000)
    parser.add_argument("--step-size", type=int, default=3000)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--feature-sets", default="base,lm_scalar,base_lm_scalar,base_lm_proba,lm_only")
    parser.add_argument("--classifiers", default="ridge,logreg")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--random-states", default="")
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--include-regime-analysis", action="store_true")
    parser.add_argument("--include-threshold-sweep", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-json", default="data/reports/sber_h1_action_lm_features_expanding_20260515.json")
    parser.add_argument("--output-csv", default="data/reports/sber_h1_action_lm_features_expanding_20260515.csv")
    args = parser.parse_args()

    if args.quick:
        args.vocab_configs = "shape:kmeans:20"
        args.feature_sets = "base,base_lm_scalar,lm_only"
        args.classifiers = "ridge"
        args.max_folds = min(args.max_folds, 2)
        args.random_states = args.random_states or str(args.random_state)

    started = time.perf_counter()
    df, data_path = load_sber_frame(args)
    folds = build_folds(args, len(df))
    vocab_configs = parse_vocab_configs(args.vocab_configs)
    feature_sets = parse_list(args.feature_sets)
    classifier_names = parse_list(args.classifiers)
    random_states = parse_int_list(args.random_states) if args.random_states else [int(args.random_state)]

    labels, future_returns, threshold = make_action_labels(df, horizon=args.action_horizon, commission=0.0005)
    shape_cache = {
        config.shape_variant: candle_shape_matrix(df, variant=config.shape_variant)[0]
        for config in vocab_configs
    }

    rows: list[dict[str, Any]] = []
    word_diagnostics: list[dict[str, Any]] = []
    print(f"Загружено свечей: {len(df)}; файл: {data_path}")
    print(f"Folds: {len(folds)}; режим: {args.fold_mode}; random_states: {random_states}")
    for random_state in random_states:
        print(f"Random state {random_state}")
        for fold in folds:
            print(f"  Fold {fold.fold_id}: train=[{fold.train_start}:{fold.train_end}) val=[{fold.val_start}:{fold.val_end})")
            for vocab_config in vocab_configs:
                print(f"    Vocabulary {vocab_config.label}")
                fold_rows, diagnostics = run_fold_vocab(
                    df,
                    shape_cache[vocab_config.shape_variant],
                    labels,
                    future_returns,
                    fold,
                    vocab_config,
                    feature_sets=feature_sets,
                    classifier_names=classifier_names,
                    context_size=args.context_size,
                    action_window_size=args.action_window_size,
                    action_horizon=args.action_horizon,
                    lm_order=args.lm_order,
                    lm_alpha=args.lm_alpha,
                    lm_forecast_horizon=args.forecast_horizon,
                    random_state=random_state,
                    include_calibration=args.include_calibration,
                    include_regime_analysis=args.include_regime_analysis,
                    include_threshold_sweep=args.include_threshold_sweep,
                )
                for row in fold_rows:
                    row["fold_mode"] = args.fold_mode
                rows.extend(fold_rows)
                word_diagnostics.append(diagnostics)

    aggregates = aggregate_rows(rows)
    best = select_best_by_validation(aggregates)
    payload = {
        "data_path": str(data_path),
        "rows": int(len(df)),
        "fold_mode": args.fold_mode,
        "folds": [fold.__dict__ for fold in folds],
        "target_threshold": float(threshold),
        "selection": "validation mean macro-F1, then worst-fold macro-F1; test split is not used",
        "vocab_configs": [config.__dict__ for config in vocab_configs],
        "feature_sets": feature_sets,
        "classifiers": classifier_names,
        "random_states": random_states,
        "leakage_note": "Clusterer, vectorizer, LM and action classifier are fit on train fold only. LM features use words up to target_idx and never receive future target words.",
        "fold_results": rows,
        "aggregates": aggregates,
        "random_state_results": random_state_results(rows),
        "calibration_tables": _collect_optional(rows, "calibration"),
        "regime_error_analysis": _collect_optional(rows, "regime_error_analysis"),
        "threshold_sweep": _collect_optional(rows, "threshold_sweep"),
        "prediction_distribution_by_fold": [
            {
                "fold_id": row["fold_id"],
                "random_state": row["random_state"],
                "vocabulary": row["vocabulary"],
                "feature_set": row["feature_set"],
                "classifier": row["classifier"],
                "prediction_distribution": row["prediction_distribution"],
                "true_distribution": row["true_distribution"],
            }
            for row in rows
        ],
        "word_distribution_by_seed": word_diagnostics,
        "word_distribution_stability": word_distribution_stability(word_diagnostics),
        "best": best,
        "duration_sec": float(time.perf_counter() - started),
    }
    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    write_json(payload, output_json)
    write_csv(compact_csv_rows(rows), output_csv)
    print_summary(aggregates)
    if best:
        print(f"Лучший config по validation: {best['vocabulary']} | {best['feature_set']} | {best['classifier']}")
    print(f"Записан JSON: {output_json}")
    print(f"Записан CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
