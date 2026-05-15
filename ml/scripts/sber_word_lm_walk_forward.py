"""Walk-forward n-gram language-model research for SBER H1 candle words."""

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
from src.nlp.word_forecast import clusterer_distance_matrix, expected_next_word_sample_count, make_next_word_samples
from src.nlp.word_lm import (
    NGramBackoffLanguageModel,
    evaluate_language_model,
    transition_entropy,
    word_distribution_metrics,
)
from src.utils.io import ensure_dir

from sber_next_word_research import build_cluster_spec, find_latest_raw, parse_int_list


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


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


def fit_fold_vocabulary(
    shape_matrix: np.ndarray,
    fold: WalkForwardRange,
    *,
    cluster_name: str,
    vocab_size: int,
    random_state: int,
) -> tuple[np.ndarray, CandleClusterer]:
    clusterer = CandleClusterer(build_cluster_spec(cluster_name, vocab_size), random_state=random_state)
    clusterer.fit(shape_matrix[fold.train_start : fold.train_end])
    word_ids = np.full(shape_matrix.shape[0], -1, dtype=int)
    word_ids[fold.train_start : fold.train_end] = clusterer.train_labels_
    word_ids[fold.val_start : fold.val_end] = clusterer.predict(shape_matrix[fold.val_start : fold.val_end])
    return word_ids, clusterer


def evaluate_lm_config(
    word_ids: np.ndarray,
    fold: WalkForwardRange,
    clusterer: CandleClusterer,
    *,
    shape_variant: str,
    cluster_name: str,
    vocab_size: int,
    context_size: int,
    forecast_horizon: int,
    order: int,
    alpha: float,
    beam_width: int,
) -> dict[str, Any]:
    train_samples = make_next_word_samples(word_ids, fold.train_start, fold.train_end, context_size, forecast_horizon)
    val_samples = make_next_word_samples(word_ids, fold.val_start, fold.val_end, context_size, forecast_horizon)
    _validate_sample_count(train_samples.size, fold.train_len, context_size, forecast_horizon, "train")
    _validate_sample_count(val_samples.size, fold.val_len, context_size, forecast_horizon, "val")

    model = NGramBackoffLanguageModel(order=order, alpha=alpha).fit(
        word_ids,
        train_start=fold.train_start,
        train_end=fold.train_end,
        n_words=clusterer.n_words_,
    )
    distance_matrix = clusterer_distance_matrix(clusterer)
    metrics = evaluate_language_model(
        model,
        val_samples.X_contexts,
        val_samples.Y_future_words,
        distance_matrix=distance_matrix,
        beam_width=beam_width,
    )
    h1 = metrics["per_horizon"][0]
    return {
        "fold_id": int(fold.fold_id),
        "train_start": int(fold.train_start),
        "train_end": int(fold.train_end),
        "val_start": int(fold.val_start),
        "val_end": int(fold.val_end),
        "train_rows": int(fold.train_len),
        "val_rows": int(fold.val_len),
        "shape_variant": shape_variant,
        "clusterer": cluster_name,
        "vocabulary_size": int(clusterer.n_words_),
        "requested_vocabulary_size": int(vocab_size),
        "context_size": int(context_size),
        "forecast_horizon": int(forecast_horizon),
        "markov_order": int(order),
        "smoothing_alpha": float(alpha),
        "beam_width": int(beam_width),
        "train_samples": int(train_samples.size),
        "val_samples": int(val_samples.size),
        "token_accuracy_at_1": float(metrics["mean_accuracy_at_1"]),
        "token_top3_accuracy": float(metrics["mean_top3_accuracy"]),
        "token_top5_accuracy": float(metrics["mean_top5_accuracy"]),
        "token_mrr": float(metrics["mean_mrr"]),
        "token_nll": float(metrics["mean_token_nll"]),
        "perplexity": float(metrics["perplexity"]),
        "sequence_exact": float(metrics["sequence_exact_match"]),
        "beam_contains_true_sequence": float(metrics["beam_contains_true_sequence"]),
        "sequence_nll": float(metrics["sequence_nll"]),
        "centroid_trajectory_distance": _optional_float(metrics.get("average_centroid_trajectory_distance")),
        "soft_similarity": _optional_float(metrics.get("mean_soft_similarity")),
        "h1_accuracy_at_1": float(h1["accuracy_at_1"]),
        "h1_top3_accuracy": float(h1["top3_accuracy"]),
        "h1_top5_accuracy": float(h1["top5_accuracy"]),
        "h1_mrr": float(h1["mean_reciprocal_rank"]),
    }


def vocabulary_audit_row(
    word_ids: np.ndarray,
    fold: WalkForwardRange,
    clusterer: CandleClusterer,
    *,
    shape_variant: str,
    cluster_name: str,
    vocab_size: int,
    context_size: int,
    order: int,
    alpha: float,
    beam_width: int,
) -> dict[str, Any]:
    train_words = word_ids[fold.train_start : fold.train_end]
    metrics = word_distribution_metrics(train_words, clusterer.n_words_)
    metrics["transition_entropy"] = transition_entropy(train_words, clusterer.n_words_)

    val_samples = make_next_word_samples(word_ids, fold.val_start, fold.val_end, context_size, 1)
    model = NGramBackoffLanguageModel(order=order, alpha=alpha).fit(
        word_ids,
        train_start=fold.train_start,
        train_end=fold.train_end,
        n_words=clusterer.n_words_,
    )
    lm_metrics = evaluate_language_model(
        model,
        val_samples.X_contexts,
        val_samples.Y_future_words,
        distance_matrix=clusterer_distance_matrix(clusterer),
        beam_width=beam_width,
    )
    return {
        "fold_id": int(fold.fold_id),
        "shape_variant": shape_variant,
        "clusterer": cluster_name,
        "requested_vocabulary_size": int(vocab_size),
        "vocabulary_size": int(clusterer.n_words_),
        "cluster_quality": clusterer.quality_,
        "word_distribution": metrics,
        "validation_next_token": {
            "order": int(order),
            "context_size": int(context_size),
            "token_nll": float(lm_metrics["mean_token_nll"]),
            "perplexity": float(lm_metrics["perplexity"]),
            "top3_accuracy": float(lm_metrics["mean_top3_accuracy"]),
            "top5_accuracy": float(lm_metrics["mean_top5_accuracy"]),
        },
    }


def aggregate_lm_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["shape_variant"],
            row["clusterer"],
            row["requested_vocabulary_size"],
            row["context_size"],
            row["forecast_horizon"],
            row["markov_order"],
            row["smoothing_alpha"],
        )
        grouped.setdefault(key, []).append(row)

    aggregates = []
    for key, items in grouped.items():
        token_nll = np.asarray([item["token_nll"] for item in items], dtype=float)
        ppl = np.asarray([item["perplexity"] for item in items], dtype=float)
        top3 = np.asarray([item["token_top3_accuracy"] for item in items], dtype=float)
        seq_exact = np.asarray([item["sequence_exact"] for item in items], dtype=float)
        soft = np.asarray([item["soft_similarity"] for item in items if item["soft_similarity"] is not None], dtype=float)
        aggregates.append(
            {
                "shape_variant": key[0],
                "clusterer": key[1],
                "requested_vocabulary_size": int(key[2]),
                "context_size": int(key[3]),
                "forecast_horizon": int(key[4]),
                "markov_order": int(key[5]),
                "smoothing_alpha": float(key[6]),
                "folds": int(len(items)),
                "token_nll_mean": float(token_nll.mean()),
                "token_nll_std": float(token_nll.std(ddof=0)),
                "token_nll_min": float(token_nll.min()),
                "token_nll_max": float(token_nll.max()),
                "perplexity_mean": float(ppl.mean()),
                "perplexity_std": float(ppl.std(ddof=0)),
                "token_top3_mean": float(top3.mean()),
                "sequence_exact_mean": float(seq_exact.mean()),
                "soft_similarity_mean": float(soft.mean()) if len(soft) else None,
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (row["token_nll_mean"], row["perplexity_mean"], -row["token_top3_mean"]),
    )


def aggregate_vocabulary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["shape_variant"], row["clusterer"], row["requested_vocabulary_size"])
        grouped.setdefault(key, []).append(row)

    aggregates = []
    for (shape_variant, clusterer, vocab_size), items in grouped.items():
        entropy = np.asarray([item["word_distribution"]["normalized_entropy"] for item in items], dtype=float)
        dominant = np.asarray([item["word_distribution"]["dominant_share"] for item in items], dtype=float)
        trans_entropy = np.asarray([item["word_distribution"]["transition_entropy"] for item in items], dtype=float)
        perplexity = np.asarray([item["validation_next_token"]["perplexity"] for item in items], dtype=float)
        top3 = np.asarray([item["validation_next_token"]["top3_accuracy"] for item in items], dtype=float)
        aggregates.append(
            {
                "shape_variant": shape_variant,
                "clusterer": clusterer,
                "requested_vocabulary_size": int(vocab_size),
                "folds": int(len(items)),
                "normalized_entropy_mean": float(entropy.mean()),
                "normalized_entropy_std": float(entropy.std(ddof=0)),
                "dominant_share_mean": float(dominant.mean()),
                "dominant_share_std": float(dominant.std(ddof=0)),
                "transition_entropy_mean": float(trans_entropy.mean()),
                "validation_next_token_perplexity_mean": float(perplexity.mean()),
                "validation_next_token_top3_mean": float(top3.mean()),
            }
        )
    return sorted(aggregates, key=lambda row: (row["validation_next_token_perplexity_mean"], -row["validation_next_token_top3_mean"]))


def select_best_by_validation(aggregates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select only by validation aggregate NLL/perplexity."""

    if not aggregates:
        return None
    return min(
        enumerate(aggregates),
        key=lambda item: (
            item[1]["token_nll_mean"],
            item[1]["perplexity_mean"],
            -item[1]["token_top3_mean"],
            item[0],
        ),
    )[1]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)


def print_summary(aggregates: list[dict[str, Any]]) -> None:
    print("model | context | K | order | alpha | val token NLL | perplexity | top3 | seq exact")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for row in aggregates[:12]:
        print(
            f"{row['shape_variant']}/{row['clusterer']}/{row['requested_vocabulary_size']} | "
            f"{row['context_size']} | {row['forecast_horizon']} | {row['markov_order']} | "
            f"{row['smoothing_alpha']:.3f} | {row['token_nll_mean']:.4f} | "
            f"{row['perplexity_mean']:.4f} | {row['token_top3_mean']:.4f} | "
            f"{row['sequence_exact_mean']:.4f}"
        )


def _validate_sample_count(sample_count: int, split_len: int, context_size: int, forecast_horizon: int, split_name: str) -> None:
    expected = expected_next_word_sample_count(split_len, context_size, forecast_horizon)
    if sample_count != expected:
        raise ValueError(f"{split_name} sample count mismatch: got {sample_count}, expected {expected}")
    if sample_count <= 0:
        raise ValueError(f"{split_name} has no samples")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


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
    parser.add_argument("--shape-variants", default="shape")
    parser.add_argument("--clusterers", default="gmm")
    parser.add_argument("--vocab-sizes", default="20")
    parser.add_argument("--context-sizes", default="16,32")
    parser.add_argument("--forecast-horizons", default="3,5")
    parser.add_argument("--orders", default="1,2,3")
    parser.add_argument("--smoothing-alphas", default="0.1")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--initial-train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=3000)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--include-vocabulary-audit", action="store_true", default=True)
    parser.add_argument("--skip-vocabulary-audit", action="store_false", dest="include_vocabulary_audit")
    parser.add_argument("--vocab-audit-shape-variants", default="ohlc,shape")
    parser.add_argument("--vocab-audit-clusterers", default="kmeans,gmm")
    parser.add_argument("--vocab-audit-sizes", default="8,16,20,32,48,64")
    parser.add_argument("--vocab-audit-order", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-json", default="data/reports/sber_h1_word_lm_walk_forward_20260515.json")
    parser.add_argument("--output-csv", default="data/reports/sber_h1_word_lm_walk_forward_20260515.csv")
    args = parser.parse_args()

    if args.quick:
        args.context_sizes = "16"
        args.forecast_horizons = "3"
        args.orders = "1,2"
        args.vocab_audit_sizes = "8,20"
        args.vocab_audit_clusterers = "kmeans,gmm"

    started = time.perf_counter()
    df, data_path = load_sber_frame(args)
    folds = walk_forward_ranges(
        len(df),
        n_splits=args.n_splits,
        initial_train_size=args.initial_train_size,
        val_size=args.val_size,
        gap=args.gap,
        min_train_size=min(1000, args.initial_train_size),
    )
    print(f"Загружено {len(df)} свечей из {data_path}")
    print(f"Сформировано walk-forward folds: {len(folds)}")

    main_shape_variants = parse_str_list(args.shape_variants)
    main_clusterers = parse_str_list(args.clusterers)
    main_vocab_sizes = parse_int_list(args.vocab_sizes)
    context_sizes = parse_int_list(args.context_sizes)
    forecast_horizons = parse_int_list(args.forecast_horizons)
    orders = parse_int_list(args.orders)
    alphas = parse_float_list(args.smoothing_alphas)
    shape_cache = {variant: candle_shape_matrix(df, variant=variant)[0] for variant in set(main_shape_variants + parse_str_list(args.vocab_audit_shape_variants))}

    rows: list[dict[str, Any]] = []
    vocabulary_rows: list[dict[str, Any]] = []
    for fold in folds:
        print(f"Fold {fold.fold_id}: train=[{fold.train_start}:{fold.train_end}) val=[{fold.val_start}:{fold.val_end})")
        for shape_variant in main_shape_variants:
            for cluster_name in main_clusterers:
                for vocab_size in main_vocab_sizes:
                    word_ids, clusterer = fit_fold_vocabulary(
                        shape_cache[shape_variant],
                        fold,
                        cluster_name=cluster_name,
                        vocab_size=vocab_size,
                        random_state=args.random_state,
                    )
                    for context_size in context_sizes:
                        for forecast_horizon in forecast_horizons:
                            for order in orders:
                                for alpha in alphas:
                                    print(
                                        f"  LM {shape_variant}/{cluster_name}/{vocab_size}: "
                                        f"context={context_size}, K={forecast_horizon}, order={order}, alpha={alpha}"
                                    )
                                    rows.append(
                                        evaluate_lm_config(
                                            word_ids,
                                            fold,
                                            clusterer,
                                            shape_variant=shape_variant,
                                            cluster_name=cluster_name,
                                            vocab_size=vocab_size,
                                            context_size=context_size,
                                            forecast_horizon=forecast_horizon,
                                            order=order,
                                            alpha=alpha,
                                            beam_width=args.beam_width,
                                        )
                                    )

        if args.include_vocabulary_audit:
            for shape_variant in parse_str_list(args.vocab_audit_shape_variants):
                for cluster_name in parse_str_list(args.vocab_audit_clusterers):
                    for vocab_size in parse_int_list(args.vocab_audit_sizes):
                        word_ids, clusterer = fit_fold_vocabulary(
                            shape_cache[shape_variant],
                            fold,
                            cluster_name=cluster_name,
                            vocab_size=vocab_size,
                            random_state=args.random_state,
                        )
                        vocabulary_rows.append(
                            vocabulary_audit_row(
                                word_ids,
                                fold,
                                clusterer,
                                shape_variant=shape_variant,
                                cluster_name=cluster_name,
                                vocab_size=vocab_size,
                                context_size=min(context_sizes),
                                order=args.vocab_audit_order,
                                alpha=alphas[0],
                                beam_width=args.beam_width,
                            )
                        )

    aggregates = aggregate_lm_rows(rows)
    vocabulary_aggregates = aggregate_vocabulary_rows(vocabulary_rows) if vocabulary_rows else []
    best = select_best_by_validation(aggregates)
    payload = {
        "data_path": str(data_path),
        "rows": int(len(df)),
        "folds": [fold.__dict__ for fold in folds],
        "selection": "validation mean token NLL, then validation perplexity, then validation top-k; test split is not used",
        "teacher_forcing_note": "Token NLL/top-k are evaluated with true previous target tokens as evaluation context; free-running sequence metrics use greedy/beam decoding.",
        "fold_results": rows,
        "aggregates": aggregates,
        "best": best,
        "vocabulary_audit": {
            "fold_results": vocabulary_rows,
            "aggregates": vocabulary_aggregates,
            "note": "Vocabulary statistics are train-only; validation next-token metrics are evaluation-only.",
        },
        "duration_sec": float(time.perf_counter() - started),
    }

    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    write_json(payload, output_json)
    write_csv(rows, output_csv)
    print_summary(aggregates)
    if best:
        print(
            "Лучший config по validation: "
            f"{best['shape_variant']}/{best['clusterer']}/{best['requested_vocabulary_size']} "
            f"context={best['context_size']} K={best['forecast_horizon']} order={best['markov_order']}"
        )
    print(f"Записан JSON: {output_json}")
    print(f"Записан CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
