"""Run next candle-word forecasting baselines on SBER H1 candles."""

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
from src.nlp import candle_shape_matrix, split_ranges
from src.nlp.clustering import CandleClusterer, ClusterSpec
from src.nlp.word_forecast import (
    build_word_forecaster,
    clusterer_distance_matrix,
    evaluate_word_forecast,
    expected_next_word_sample_count,
    make_next_word_samples,
)
from src.utils.io import ensure_dir, write_json


def find_latest_raw(raw_dir: Path, ticker: str, timeframe: str) -> Path:
    files = sorted(raw_dir.glob(f"{ticker}_{timeframe}_*.parquet"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No raw parquet files found for {ticker} {timeframe} in {raw_dir}")
    return files[0]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_cluster_spec(name: str, n_clusters: int) -> ClusterSpec:
    if name == "gmm":
        return ClusterSpec("gmm", {"n_components": n_clusters, "covariance_type": "diag", "reg_covar": 1e-6})
    return ClusterSpec(name, {"n_clusters": n_clusters, "n_init": 10})


def assign_words(
    df: pd.DataFrame,
    *,
    shape_variant: str,
    cluster: ClusterSpec,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, dict[str, tuple[int, int]], CandleClusterer]:
    df = df.sort_values("begin").reset_index(drop=True) if "begin" in df.columns else df.reset_index(drop=True)
    shape_matrix, _ = candle_shape_matrix(df, variant=shape_variant)
    ranges = split_ranges(len(df), train_ratio=train_ratio, val_ratio=val_ratio)
    train_start, train_end = ranges["train"]

    clusterer = CandleClusterer(cluster, random_state=random_state)
    clusterer.fit(shape_matrix[train_start:train_end])

    word_ids = np.empty(len(df), dtype=int)
    word_ids[train_start:train_end] = clusterer.train_labels_
    for split_name in ("val", "test"):
        split_start, split_end = ranges[split_name]
        word_ids[split_start:split_end] = clusterer.predict(shape_matrix[split_start:split_end])
    return word_ids, ranges, clusterer


def evaluate_config(
    word_ids: np.ndarray,
    ranges: dict[str, tuple[int, int]],
    clusterer: CandleClusterer,
    *,
    context_size: int,
    forecast_horizon: int,
    model_name: str,
) -> dict[str, Any]:
    samples = {
        split_name: make_next_word_samples(word_ids, start, end, context_size, forecast_horizon)
        for split_name, (start, end) in ranges.items()
    }
    for split_name, sample in samples.items():
        expected = expected_next_word_sample_count(ranges[split_name][1] - ranges[split_name][0], context_size, forecast_horizon)
        if sample.size != expected:
            raise ValueError(f"{split_name} sample count mismatch: got {sample.size}, expected {expected}")

    model = build_word_forecaster(model_name)
    model.fit(samples["train"].X_contexts, samples["train"].Y_future_words, n_words=clusterer.n_words_)
    distance_matrix = clusterer_distance_matrix(clusterer)

    metrics = {}
    for split_name in ("train", "val", "test"):
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

    return {
        "model": model_name,
        "context_size": int(context_size),
        "forecast_horizon": int(forecast_horizon),
        "sample_counts": {name: sample.size for name, sample in samples.items()},
        "metrics": metrics,
    }


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    val_h1 = result["metrics"]["val"]["per_horizon"][0]
    test_h1 = result["metrics"]["test"]["per_horizon"][0]
    return {
        "model": result["model"],
        "context_size": result["context_size"],
        "forecast_horizon": result["forecast_horizon"],
        "val_exact_acc_h1": val_h1["accuracy"],
        "val_macro_f1_h1": val_h1["macro_f1"],
        "val_soft_similarity": result["metrics"]["val"].get("mean_soft_similarity"),
        "val_sequence_exact": result["metrics"]["val"]["sequence_exact_match"],
        "test_exact_acc_h1": test_h1["accuracy"],
        "test_macro_f1_h1": test_h1["macro_f1"],
        "test_soft_similarity": result["metrics"]["test"].get("mean_soft_similarity"),
        "test_sequence_exact": result["metrics"]["test"]["sequence_exact_match"],
    }


def distance_matrix_summary(distance_matrix: np.ndarray) -> dict[str, float]:
    nonzero = distance_matrix[distance_matrix > 0]
    if len(nonzero) == 0:
        return {"min_nonzero": 0.0, "median_nonzero": 0.0, "mean_nonzero": 0.0, "max": 0.0}
    return {
        "min_nonzero": float(nonzero.min()),
        "median_nonzero": float(np.median(nonzero)),
        "mean_nonzero": float(nonzero.mean()),
        "max": float(nonzero.max()),
    }


def random_similarity_baseline(
    word_ids: np.ndarray,
    ranges: dict[str, tuple[int, int]],
    distance_matrix: np.ndarray,
    *,
    context_size: int,
    forecast_horizon: int,
    random_state: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_state)
    n_words = int(distance_matrix.shape[0])
    baselines = {}
    for split_name in ("val", "test"):
        start, end = ranges[split_name]
        sample = make_next_word_samples(word_ids, start, end, context_size, forecast_horizon)
        random_pred = rng.integers(0, n_words, size=sample.Y_future_words.shape)
        baselines[split_name] = evaluate_word_forecast(
            sample.Y_future_words,
            random_pred,
            n_words=n_words,
            distance_matrix=distance_matrix,
            probabilities=None,
            nearest_n=3,
        )
    return baselines


def select_best_by_validation(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select using validation-only next-word quality."""

    if not results:
        return None
    return max(
        enumerate(results),
        key=lambda item: (
            item[1]["metrics"]["val"]["mean_accuracy"],
            item[1]["metrics"]["val"].get("mean_soft_similarity", -1.0),
            -item[0],
        ),
    )[1]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def print_table(rows: list[dict[str, Any]]) -> None:
    print("model | context | K | val acc@1 | val soft | test acc@1 | test soft")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---:")
    for row in rows:
        print(
            f"{row['model']} | {row['context_size']} | {row['forecast_horizon']} | "
            f"{row['val_exact_acc_h1']:.4f} | {row['val_soft_similarity']:.4f} | "
            f"{row['test_exact_acc_h1']:.4f} | {row['test_soft_similarity']:.4f}"
        )


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
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-json", default="data/reports/sber_h1_next_word_research_20260515.json")
    parser.add_argument("--output-csv", default="data/reports/sber_h1_next_word_research_20260515.csv")
    args = parser.parse_args()

    data_path = Path(args.data).resolve() if args.data else find_latest_raw(REPO_ROOT / args.raw_dir, args.ticker, args.timeframe)
    df = pd.read_parquet(data_path)
    if "ticker" in df.columns:
        df = df[df["ticker"] == args.ticker]
    if "timeframe" in df.columns:
        df = df[df["timeframe"] == args.timeframe]
    df = clean_candles(df)
    print(f"Loaded {len(df)} candles from {data_path}")

    cluster = build_cluster_spec(args.cluster, args.n_clusters)
    word_ids, ranges, clusterer = assign_words(
        df,
        shape_variant=args.shape_variant,
        cluster=cluster,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
    )
    print(f"Assigned {clusterer.n_words_} candle words; ranges={ranges}")
    distance_matrix = clusterer_distance_matrix(clusterer)

    results = []
    for context_size in parse_int_list(args.context_sizes):
        for forecast_horizon in parse_int_list(args.forecast_horizons):
            for model_name in [part.strip() for part in args.models.split(",") if part.strip()]:
                started = time.perf_counter()
                print(f"Evaluating {model_name}, context={context_size}, K={forecast_horizon}")
                result = evaluate_config(
                    word_ids,
                    ranges,
                    clusterer,
                    context_size=context_size,
                    forecast_horizon=forecast_horizon,
                    model_name=model_name,
                )
                result["duration_sec"] = float(time.perf_counter() - started)
                results.append(result)

    rows = [result_row(result) for result in results]
    rows = sorted(rows, key=lambda row: (row["val_exact_acc_h1"], row["val_soft_similarity"], row["model"]), reverse=True)
    best = select_best_by_validation(results)
    payload = {
        "data_path": str(data_path),
        "rows": int(len(df)),
        "shape_variant": args.shape_variant,
        "cluster": {
            "name": cluster.name,
            "params": dict(cluster.params),
            "quality": clusterer.quality_,
            "centroid_distance_summary": distance_matrix_summary(distance_matrix),
        },
        "split_ranges": {name: [int(start), int(end)] for name, (start, end) in ranges.items()},
        "selection": "validation mean_accuracy, then validation mean_soft_similarity, then deterministic config order; test is report-only",
        "soft_similarity_sanity": {
            "random_uniform_baseline": random_similarity_baseline(
                word_ids,
                ranges,
                distance_matrix,
                context_size=parse_int_list(args.context_sizes)[0],
                forecast_horizon=parse_int_list(args.forecast_horizons)[0],
                random_state=args.random_state,
            ),
            "note": "Soft similarity uses exp(-centroid_distance / median_nonzero_train_centroid_distance).",
        },
        "best": best,
        "results": results,
    }

    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    write_json(payload, output_json)
    write_csv(rows, output_csv)
    print_table(rows)
    if best:
        print(f"Best by validation: {best['model']} context={best['context_size']} K={best['forecast_horizon']}")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
