"""Walk-forward validation for SBER H1 next candle-word forecasting."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.data import clean_candles
from src.data.split import WalkForwardRange, walk_forward_ranges
from src.nlp import candle_shape_matrix
from src.nlp.clustering import CandleClusterer
from src.nlp.word_forecast import (
    build_word_forecaster,
    clusterer_distance_matrix,
    evaluate_word_forecast,
    expected_next_word_sample_count,
    make_next_word_samples,
)
from src.utils.io import ensure_dir

from sber_next_word_research import build_cluster_spec, distance_matrix_summary, find_latest_raw, parse_int_list


def assign_fold_words(
    shape_matrix: np.ndarray,
    fold: WalkForwardRange,
    *,
    cluster_name: str,
    n_clusters: int,
    random_state: int,
) -> tuple[np.ndarray, CandleClusterer]:
    clusterer = CandleClusterer(build_cluster_spec(cluster_name, n_clusters), random_state=random_state)
    clusterer.fit(shape_matrix[fold.train_start : fold.train_end])
    word_ids = np.full(shape_matrix.shape[0], -1, dtype=int)
    word_ids[fold.train_start : fold.train_end] = clusterer.train_labels_
    word_ids[fold.val_start : fold.val_end] = clusterer.predict(shape_matrix[fold.val_start : fold.val_end])
    return word_ids, clusterer


def evaluate_fold_config(
    word_ids: np.ndarray,
    fold: WalkForwardRange,
    clusterer: CandleClusterer,
    *,
    context_size: int,
    forecast_horizon: int,
    model_name: str,
) -> dict[str, Any]:
    samples = {
        "train": make_next_word_samples(word_ids, fold.train_start, fold.train_end, context_size, forecast_horizon),
        "val": make_next_word_samples(word_ids, fold.val_start, fold.val_end, context_size, forecast_horizon),
    }
    for split_name, sample in samples.items():
        split_len = fold.train_len if split_name == "train" else fold.val_len
        expected = expected_next_word_sample_count(split_len, context_size, forecast_horizon)
        if sample.size != expected:
            raise ValueError(f"{split_name} sample count mismatch: got {sample.size}, expected {expected}")

    model = build_word_forecaster(model_name)
    model.fit(samples["train"].X_contexts, samples["train"].Y_future_words, n_words=clusterer.n_words_)
    distance_matrix = clusterer_distance_matrix(clusterer)

    metrics = {}
    for split_name in ("train", "val"):
        X = samples[split_name].X_contexts
        Y = samples[split_name].Y_future_words
        pred = model.predict(X)
        try:
            probabilities = model.predict_proba(X)
        except AttributeError:
            probabilities = None
        metrics[split_name] = evaluate_word_forecast(
            Y,
            pred,
            n_words=clusterer.n_words_,
            distance_matrix=distance_matrix,
            probabilities=probabilities,
            nearest_n=3,
        )

    val_h1 = metrics["val"]["per_horizon"][0]
    return {
        "fold_id": int(fold.fold_id),
        "train_start": int(fold.train_start),
        "train_end": int(fold.train_end),
        "val_start": int(fold.val_start),
        "val_end": int(fold.val_end),
        "train_rows": int(fold.train_len),
        "val_rows": int(fold.val_len),
        "model": model_name,
        "context_size": int(context_size),
        "forecast_horizon": int(forecast_horizon),
        "train_samples": int(samples["train"].size),
        "val_samples": int(samples["val"].size),
        "val_exact_acc_h1": float(val_h1["accuracy"]),
        "val_macro_f1_h1": float(val_h1["macro_f1"]),
        "val_top3_h1": _none_float(val_h1.get("top3_accuracy")),
        "val_mean_exact": float(metrics["val"]["mean_accuracy"]),
        "val_mean_macro_f1": float(metrics["val"]["mean_macro_f1"]),
        "val_sequence_exact": float(metrics["val"]["sequence_exact_match"]),
        "val_soft_similarity": _none_float(metrics["val"].get("mean_soft_similarity")),
        "val_centroid_distance": _none_float(metrics["val"].get("mean_centroid_distance")),
        "cluster_n_words": int(clusterer.n_words_),
        "centroid_distance_summary": distance_matrix_summary(distance_matrix),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["model"], row["context_size"], row["forecast_horizon"])
        grouped.setdefault(key, []).append(row)

    aggregates = []
    for (model, context_size, forecast_horizon), items in grouped.items():
        exact = np.asarray([item["val_mean_exact"] for item in items], dtype=float)
        soft = np.asarray([item["val_soft_similarity"] for item in items], dtype=float)
        seq = np.asarray([item["val_sequence_exact"] for item in items], dtype=float)
        aggregates.append(
            {
                "model": model,
                "context_size": int(context_size),
                "forecast_horizon": int(forecast_horizon),
                "folds": int(len(items)),
                "val_mean_exact_mean": float(exact.mean()),
                "val_mean_exact_std": float(exact.std(ddof=0)),
                "val_mean_exact_min": float(exact.min()),
                "val_mean_exact_max": float(exact.max()),
                "val_soft_similarity_mean": float(soft.mean()),
                "val_soft_similarity_std": float(soft.std(ddof=0)),
                "val_sequence_exact_mean": float(seq.mean()),
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (row["val_mean_exact_mean"], row["val_soft_similarity_mean"], row["model"]),
        reverse=True,
    )


def select_best_by_walk_forward(aggregates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select by validation folds only; no holdout/test metrics are read here."""

    if not aggregates:
        return None
    return max(
        enumerate(aggregates),
        key=lambda item: (
            item[1]["val_mean_exact_mean"],
            item[1]["val_soft_similarity_mean"],
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


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat.pop("centroid_distance_summary", None)
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)


def print_aggregates(aggregates: list[dict[str, Any]]) -> None:
    print("model | context | K | folds | val mean exact | std | val soft")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---:")
    for row in aggregates:
        print(
            f"{row['model']} | {row['context_size']} | {row['forecast_horizon']} | {row['folds']} | "
            f"{row['val_mean_exact_mean']:.4f} | {row['val_mean_exact_std']:.4f} | "
            f"{row['val_soft_similarity_mean']:.4f}"
        )


def _none_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


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
    parser.add_argument("--shape-variant", default="shape", choices=["ohlc", "shape", "ohlc_shape"])
    parser.add_argument("--cluster", default="gmm", choices=["kmeans", "gmm", "minibatch_kmeans"])
    parser.add_argument("--n-clusters", type=int, default=20)
    parser.add_argument("--context-sizes", default="16,32")
    parser.add_argument("--forecast-horizons", default="1,3")
    parser.add_argument("--models", default="persistence,unigram,markov1,tfidf_logreg")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--initial-train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=3000)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-json", default="data/reports/sber_h1_next_word_walk_forward_20260515.json")
    parser.add_argument("--output-csv", default="data/reports/sber_h1_next_word_walk_forward_20260515.csv")
    args = parser.parse_args()

    started = time.perf_counter()
    df, data_path = load_sber_frame(args)
    df = df.sort_values("begin").reset_index(drop=True) if "begin" in df.columns else df.reset_index(drop=True)
    shape_matrix, shape_columns = candle_shape_matrix(df, variant=args.shape_variant)
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
    for fold in folds:
        print(f"Fold {fold.fold_id}: train=[{fold.train_start}:{fold.train_end}) val=[{fold.val_start}:{fold.val_end})")
        word_ids, clusterer = assign_fold_words(
            shape_matrix,
            fold,
            cluster_name=args.cluster,
            n_clusters=args.n_clusters,
            random_state=args.random_state,
        )
        for context_size in parse_int_list(args.context_sizes):
            for forecast_horizon in parse_int_list(args.forecast_horizons):
                for model_name in [part.strip() for part in args.models.split(",") if part.strip()]:
                    print(f"  Evaluating {model_name}, context={context_size}, K={forecast_horizon}")
                    row = evaluate_fold_config(
                        word_ids,
                        fold,
                        clusterer,
                        context_size=context_size,
                        forecast_horizon=forecast_horizon,
                        model_name=model_name,
                    )
                    rows.append(row)

    aggregates = aggregate_rows(rows)
    best = select_best_by_walk_forward(aggregates)
    payload = {
        "data_path": str(data_path),
        "rows": int(len(df)),
        "shape_variant": args.shape_variant,
        "shape_columns": shape_columns,
        "cluster": {"name": args.cluster, "n_clusters": int(args.n_clusters)},
        "folds": [fold.__dict__ for fold in folds],
        "selection": "validation-fold mean exact accuracy, then validation-fold soft similarity, then deterministic config order; no test split is used",
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
        print(f"Best by walk-forward validation: {best['model']} context={best['context_size']} K={best['forecast_horizon']}")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
