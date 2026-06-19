"""Walk-forward validation for SBER H1 candle-language action classifiers."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.data import clean_candles
from src.data.split import WalkForwardRange, walk_forward_ranges
from src.nlp import (
    ClassifierSpec,
    ClusterSpec,
    ExperimentConfig,
    VectorizerSpec,
    candle_shape_matrix,
    label_distribution,
    make_action_labels,
    make_sentence_samples,
)
from src.nlp.classifiers import build_classifier, classifier_requires_dense, maybe_dense
from src.nlp.clustering import CandleClusterer
from src.nlp.pipeline import classification_metrics, trading_metrics
from src.nlp.vectorizers import build_vectorizer
from src.nlp.word_forecast import clusterer_distance_matrix, fit_markov_prior_features, make_markov_prior_feature_matrix
from src.utils.io import ensure_dir

from sber_next_word_research import find_latest_raw


def experiment_configs() -> dict[str, tuple[ExperimentConfig, bool]]:
    best = ExperimentConfig(
        shape_variant="shape",
        horizon=1,
        window_size=32,
        commission=0.0005,
        cluster=ClusterSpec("gmm", {"n_components": 20, "covariance_type": "diag", "reg_covar": 1e-6}),
        vectorizer=VectorizerSpec(
            "cooccurrence_svd",
            {"embedding_dim": 24, "context_window": 2, "pool": ("mean", "std", "last"), "include_histogram": True},
        ),
        classifier=ClassifierSpec("ridge", {"alpha": 1.0}),
    )
    kmeans_tfidf = ExperimentConfig(
        shape_variant="shape",
        horizon=1,
        window_size=32,
        commission=0.0005,
        cluster=ClusterSpec("kmeans", {"n_clusters": 20, "n_init": 10}),
        vectorizer=VectorizerSpec("tfidf", {"ngram_range": (1, 2), "min_df": 2, "max_features": 3000}),
        classifier=ClassifierSpec("ridge", {"alpha": 1.0}),
    )
    gmm_tfidf_svc = ExperimentConfig(
        shape_variant="shape",
        horizon=1,
        window_size=32,
        commission=0.0005,
        cluster=ClusterSpec("gmm", {"n_components": 20, "covariance_type": "diag", "reg_covar": 1e-6}),
        vectorizer=VectorizerSpec("tfidf", {"ngram_range": (1, 2), "min_df": 2, "max_features": 3000}),
        classifier=ClassifierSpec("linear_svc", {"C": 0.5}),
    )
    return {
        "best_holdout": (best, False),
        "kmeans_tfidf_ridge": (kmeans_tfidf, False),
        "gmm_tfidf_linear_svc": (gmm_tfidf_svc, False),
        "best_holdout_markov_features": (replace(best), True),
    }


def run_fold_action_experiment(
    df: pd.DataFrame,
    shape_matrix: np.ndarray,
    labels: np.ndarray,
    future_returns: np.ndarray,
    fold: WalkForwardRange,
    *,
    config_name: str,
    config: ExperimentConfig,
    use_markov_features: bool,
    random_state: int,
) -> dict[str, Any]:
    clusterer = CandleClusterer(config.cluster, random_state=random_state)
    clusterer.fit(shape_matrix[fold.train_start : fold.train_end])
    word_ids = np.full(len(df), -1, dtype=int)
    word_ids[fold.train_start : fold.train_end] = clusterer.train_labels_
    word_ids[fold.val_start : fold.val_end] = clusterer.predict(shape_matrix[fold.val_start : fold.val_end])
    word_tokens = clusterer.labels_to_words(word_ids)

    samples = {
        "train": make_sentence_samples(
            word_tokens,
            labels,
            future_returns,
            fold.train_start,
            fold.train_end,
            config.window_size,
            config.horizon,
        ),
        "val": make_sentence_samples(
            word_tokens,
            labels,
            future_returns,
            fold.val_start,
            fold.val_end,
            config.window_size,
            config.horizon,
        ),
    }
    _validate_sample_counts(samples, fold, config)

    vectorizer = build_vectorizer(config.vectorizer, random_state=random_state)
    X_train = vectorizer.fit_transform(samples["train"].sentences, samples["train"].token_lists)
    X_val = vectorizer.transform(samples["val"].sentences, samples["val"].token_lists)

    if use_markov_features:
        prior = fit_markov_prior_features(
            word_ids,
            train_start=fold.train_start,
            train_end=fold.train_end,
            n_words=clusterer.n_words_,
        )
        distance_matrix = clusterer_distance_matrix(clusterer)
        train_prior = make_markov_prior_feature_matrix(
            word_ids,
            samples["train"].target_indices,
            prior,
            distance_matrix=distance_matrix,
        )
        val_prior = make_markov_prior_feature_matrix(
            word_ids,
            samples["val"].target_indices,
            prior,
            distance_matrix=distance_matrix,
        )
        X_train = _append_features(X_train, train_prior)
        X_val = _append_features(X_val, val_prior)

    classifier = build_classifier(config.classifier, random_state=random_state)
    if classifier_requires_dense(config.classifier):
        X_train = maybe_dense(X_train)
        X_val = maybe_dense(X_val)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier.fit(X_train, samples["train"].y)
        val_pred = classifier.predict(X_val)

    metrics = classification_metrics(samples["val"].y, val_pred)
    trade = trading_metrics(val_pred, samples["val"].future_returns, commission=config.commission)
    return {
        "fold_id": int(fold.fold_id),
        "train_start": int(fold.train_start),
        "train_end": int(fold.train_end),
        "val_start": int(fold.val_start),
        "val_end": int(fold.val_end),
        "train_rows": int(fold.train_len),
        "val_rows": int(fold.val_len),
        "config_name": config_name,
        "config_label": config.label,
        "markov_next_word_features": bool(use_markov_features),
        "train_samples": int(samples["train"].size),
        "val_samples": int(samples["val"].size),
        "val_accuracy": float(metrics["accuracy"]),
        "val_macro_f1": float(metrics["macro_f1"]),
        "val_weighted_f1": float(metrics["weighted_f1"]),
        "val_class_distribution": label_distribution(samples["val"].y),
        "val_prediction_distribution": label_distribution(val_pred),
        "val_trading": trade,
        "cluster": clusterer.quality_,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["config_name"], []).append(row)

    aggregates = []
    for config_name, items in grouped.items():
        macro_f1 = np.asarray([item["val_macro_f1"] for item in items], dtype=float)
        accuracy = np.asarray([item["val_accuracy"] for item in items], dtype=float)
        aggregates.append(
            {
                "config_name": config_name,
                "markov_next_word_features": bool(items[0]["markov_next_word_features"]),
                "folds": int(len(items)),
                "val_macro_f1_mean": float(macro_f1.mean()),
                "val_macro_f1_std": float(macro_f1.std(ddof=0)),
                "val_macro_f1_min": float(macro_f1.min()),
                "val_macro_f1_max": float(macro_f1.max()),
                "val_accuracy_mean": float(accuracy.mean()),
                "val_accuracy_std": float(accuracy.std(ddof=0)),
            }
        )
    return sorted(aggregates, key=lambda row: (row["val_macro_f1_mean"], row["val_accuracy_mean"], row["config_name"]), reverse=True)


def select_best_by_walk_forward(aggregates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select by validation folds only; no test metrics are available here."""

    if not aggregates:
        return None
    return max(
        enumerate(aggregates),
        key=lambda item: (
            item[1]["val_macro_f1_mean"],
            item[1]["val_accuracy_mean"],
            -item[0],
        ),
    )[1]


def load_sber_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    data_path = Path(args.data).resolve() if args.data else find_latest_raw(REPO_ROOT / args.raw_dir, args.ticker, args.timeframe)
    df = pd.read_parquet(data_path)
    if "ticker" in df.columns:
        df = df[df["ticker"] == args.ticker]
    if "timeframe" in df.columns:
        df = df[df["timeframe"] == args.timeframe]
    return clean_candles(df), data_path


def parse_config_names(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat.pop("cluster", None)
        flat.pop("val_trading", None)
        flat.pop("val_class_distribution", None)
        flat.pop("val_prediction_distribution", None)
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)


def print_aggregates(aggregates: list[dict[str, Any]]) -> None:
    print("config | markov features | folds | val macro-F1 | std | worst fold | val accuracy")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---:")
    for row in aggregates:
        print(
            f"{row['config_name']} | {row['markov_next_word_features']} | {row['folds']} | "
            f"{row['val_macro_f1_mean']:.4f} | {row['val_macro_f1_std']:.4f} | "
            f"{row['val_macro_f1_min']:.4f} | {row['val_accuracy_mean']:.4f}"
        )


def _validate_sample_counts(samples: dict[str, Any], fold: WalkForwardRange, config: ExperimentConfig) -> None:
    expected_train = fold.train_len - config.window_size - config.horizon + 1
    expected_val = fold.val_len - config.window_size - config.horizon + 1
    if samples["train"].size != expected_train:
        raise ValueError(f"train sample count mismatch: got {samples['train'].size}, expected {expected_train}")
    if samples["val"].size != expected_val:
        raise ValueError(f"val sample count mismatch: got {samples['val'].size}, expected {expected_val}")
    if samples["train"].size == 0 or samples["val"].size == 0:
        raise ValueError("Empty train or validation samples")
    if np.any(samples["train"].target_indices + config.horizon >= fold.train_end):
        raise ValueError("Train labels cross fold boundary")
    if np.any(samples["val"].target_indices + config.horizon >= fold.val_end):
        raise ValueError("Validation labels cross fold boundary")
    if np.any(samples["val"].target_indices - config.window_size + 1 < fold.val_start):
        raise ValueError("Validation context crosses fold boundary")


def _append_features(base: Any, extra: np.ndarray):
    if sparse.issparse(base):
        return sparse.hstack([base, sparse.csr_matrix(extra)], format="csr")
    return np.hstack([np.asarray(base), extra])


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
    parser.add_argument("--configs", default="best_holdout,kmeans_tfidf_ridge,best_holdout_markov_features")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--initial-train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=3000)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-json", default="data/reports/sber_h1_nlp_walk_forward_20260515.json")
    parser.add_argument("--output-csv", default="data/reports/sber_h1_nlp_walk_forward_20260515.csv")
    args = parser.parse_args()

    started = time.perf_counter()
    df, data_path = load_sber_frame(args)
    df = df.sort_values("begin").reset_index(drop=True) if "begin" in df.columns else df.reset_index(drop=True)
    all_configs = experiment_configs()
    selected_names = parse_config_names(args.configs)
    unknown = sorted(set(selected_names) - set(all_configs))
    if unknown:
        raise ValueError(f"Unknown configs: {unknown}; available={sorted(all_configs)}")

    folds = walk_forward_ranges(
        len(df),
        n_splits=args.n_splits,
        initial_train_size=args.initial_train_size,
        val_size=args.val_size,
        gap=args.gap,
        min_train_size=min(1000, args.initial_train_size),
    )
    print(f"Loaded {len(df)} candles from {data_path}")
    print(f"Generated {len(folds)} expanding walk-forward folds")

    rows: list[dict[str, Any]] = []
    shape_cache: dict[str, np.ndarray] = {}
    label_cache: dict[tuple[int, float, float], tuple[np.ndarray, np.ndarray, float]] = {}
    for config_name in selected_names:
        config, use_markov_features = all_configs[config_name]
        if config.shape_variant not in shape_cache:
            shape_cache[config.shape_variant], _ = candle_shape_matrix(df, variant=config.shape_variant)
        label_key = (config.horizon, config.commission, config.min_return)
        if label_key not in label_cache:
            label_cache[label_key] = make_action_labels(
                df,
                horizon=config.horizon,
                commission=config.commission,
                min_return=config.min_return,
            )
        labels, future_returns, threshold = label_cache[label_key]
        print(f"Config {config_name}: {config.label}; target_threshold={threshold:.6f}")
        for fold in folds:
            print(f"  Fold {fold.fold_id}: train=[{fold.train_start}:{fold.train_end}) val=[{fold.val_start}:{fold.val_end})")
            row = run_fold_action_experiment(
                df,
                shape_cache[config.shape_variant],
                labels,
                future_returns,
                fold,
                config_name=config_name,
                config=config,
                use_markov_features=use_markov_features,
                random_state=args.random_state,
            )
            rows.append(row)

    aggregates = aggregate_rows(rows)
    best = select_best_by_walk_forward(aggregates)
    payload = {
        "data_path": str(data_path),
        "rows": int(len(df)),
        "folds": [fold.__dict__ for fold in folds],
        "configs": {
            name: {
                "label": all_configs[name][0].label,
                "markov_next_word_features": all_configs[name][1],
            }
            for name in selected_names
        },
        "selection": "validation-fold mean macro-F1, then validation-fold accuracy, then deterministic config order; no test split is used",
        "fold_results": rows,
        "aggregates": aggregates,
        "best": best,
        "duration_sec": float(time.perf_counter() - started),
    }

    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    write_json(payload, output_json)
    write_csv(rows, output_csv)
    print_aggregates(aggregates)
    if best:
        print(f"Best by walk-forward validation: {best['config_name']}")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
