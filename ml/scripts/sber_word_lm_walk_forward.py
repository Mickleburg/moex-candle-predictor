"""Walk-forward n-gram LM and vocabulary selection for SBER H1 candle words."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.data import clean_candles
from src.data.split import WalkForwardRange, rolling_walk_forward_ranges, walk_forward_ranges
from src.nlp import candle_shape_matrix
from src.nlp.clustering import CandleClusterer, ClusterSpec
from src.nlp.word_forecast import clusterer_distance_matrix, expected_next_word_sample_count, make_next_word_samples
from src.nlp.word_lm import (
    NGramBackoffLanguageModel,
    confidence_analysis,
    error_analysis,
    evaluate_language_model,
    transition_quality_metrics,
    word_distribution_metrics,
)
from src.utils.io import ensure_dir

from sber_next_word_research import find_latest_raw, parse_int_list


@dataclass(frozen=True)
class VocabularySpec:
    """Vocabulary configuration for one fold-local clusterer fit."""

    shape_variant: str
    clusterer: str
    requested_vocabulary_size: int
    covariance_type: str | None = None

    @property
    def label(self) -> str:
        if self.clusterer == "gmm":
            return f"{self.shape_variant}/gmm_{self.covariance_type}/{self.requested_vocabulary_size}"
        return f"{self.shape_variant}/{self.clusterer}/{self.requested_vocabulary_size}"

    @property
    def clusterer_label(self) -> str:
        if self.clusterer == "gmm":
            return f"gmm_{self.covariance_type}"
        return self.clusterer


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


def iter_vocab_specs(args: argparse.Namespace) -> list[VocabularySpec]:
    specs: list[VocabularySpec] = []
    for shape_variant in parse_str_list(args.shape_variants):
        for clusterer in parse_str_list(args.clusterers):
            for vocab_size in parse_int_list(args.vocab_sizes):
                if clusterer == "gmm":
                    for covariance_type in parse_str_list(args.gmm_covariance_types):
                        specs.append(VocabularySpec(shape_variant, clusterer, vocab_size, covariance_type))
                else:
                    specs.append(VocabularySpec(shape_variant, clusterer, vocab_size))
    return specs


def build_cluster_spec(spec: VocabularySpec) -> ClusterSpec:
    if spec.clusterer == "kmeans":
        return ClusterSpec("kmeans", {"n_clusters": spec.requested_vocabulary_size, "n_init": 10})
    if spec.clusterer == "gmm":
        return ClusterSpec(
            "gmm",
            {
                "n_components": spec.requested_vocabulary_size,
                "covariance_type": spec.covariance_type or "diag",
                "reg_covar": 1e-6,
            },
        )
    raise ValueError(f"Unsupported clusterer for vocabulary selection: {spec.clusterer}")


def fit_fold_vocabulary(
    shape_matrix: np.ndarray,
    fold: WalkForwardRange,
    *,
    spec: VocabularySpec,
    random_state: int,
) -> tuple[np.ndarray, CandleClusterer]:
    clusterer = CandleClusterer(build_cluster_spec(spec), random_state=random_state)
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
    spec: VocabularySpec,
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
    metrics = evaluate_language_model(
        model,
        val_samples.X_contexts,
        val_samples.Y_future_words,
        distance_matrix=clusterer_distance_matrix(clusterer),
        beam_width=beam_width,
    )
    h1 = metrics["per_horizon"][0]
    return {
        "fold_id": int(fold.fold_id),
        "train_start": int(fold.train_start),
        "train_end": int(fold.train_end),
        "val_start": int(fold.val_start),
        "val_end": int(fold.val_end),
        "fold_mode_train_len": int(fold.train_len),
        "fold_mode_val_len": int(fold.val_len),
        "shape_variant": spec.shape_variant,
        "clusterer": spec.clusterer_label,
        "clusterer_family": spec.clusterer,
        "gmm_covariance_type": spec.covariance_type,
        "vocab_size_requested": int(spec.requested_vocabulary_size),
        "vocab_size_observed": int(clusterer.n_words_),
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
    spec: VocabularySpec,
    context_size: int,
    order: int,
    alpha: float,
    beam_width: int,
) -> dict[str, Any]:
    train_words = word_ids[fold.train_start : fold.train_end]
    distribution = word_distribution_metrics(train_words, clusterer.n_words_)
    transition = transition_quality_metrics(train_words, clusterer.n_words_)
    counts = np.bincount(train_words[(train_words >= 0) & (train_words < clusterer.n_words_)], minlength=clusterer.n_words_)
    total = counts.sum() if counts.sum() else 1
    probabilities = counts / total

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
        "shape_variant": spec.shape_variant,
        "clusterer": spec.clusterer_label,
        "clusterer_family": spec.clusterer,
        "gmm_covariance_type": spec.covariance_type,
        "vocab_size_requested": int(spec.requested_vocabulary_size),
        "vocab_size_observed": int(clusterer.n_words_),
        "empty_clusters": int(max(0, spec.requested_vocabulary_size - distribution["nonempty_words"])),
        "observed_vocab_ratio": float(distribution["nonempty_words"] / max(1, spec.requested_vocabulary_size)),
        "word_counts": counts.astype(int).tolist(),
        "word_probabilities": probabilities.astype(float).tolist(),
        "word_distribution": distribution,
        "transition": transition,
        "cluster_quality": clusterer.quality_,
        "validation_next_token": {
            "order": int(order),
            "context_size": int(context_size),
            "token_nll": float(lm_metrics["mean_token_nll"]),
            "perplexity": float(lm_metrics["perplexity"]),
            "top3_accuracy": float(lm_metrics["mean_top3_accuracy"]),
            "top5_accuracy": float(lm_metrics["mean_top5_accuracy"]),
            "mrr": float(lm_metrics["mean_mrr"]),
            "sequence_exact": float(lm_metrics["sequence_exact_match"]),
            "sequence_nll": float(lm_metrics["sequence_nll"]),
        },
    }


def aggregate_lm_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["shape_variant"],
            row["clusterer"],
            row["vocab_size_requested"],
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
        top5 = np.asarray([item["token_top5_accuracy"] for item in items], dtype=float)
        mrr = np.asarray([item["token_mrr"] for item in items], dtype=float)
        seq_exact = np.asarray([item["sequence_exact"] for item in items], dtype=float)
        aggregates.append(
            {
                "shape_variant": key[0],
                "clusterer": key[1],
                "vocab_size_requested": int(key[2]),
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
                "top3_mean": float(top3.mean()),
                "top5_mean": float(top5.mean()),
                "mrr_mean": float(mrr.mean()),
                "sequence_exact_mean": float(seq_exact.mean()),
            }
        )
    return sorted(aggregates, key=lambda row: (row["token_nll_mean"], row["perplexity_mean"], -row["top3_mean"]))


def aggregate_vocabulary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["shape_variant"], row["clusterer"], row["vocab_size_requested"])
        grouped.setdefault(key, []).append(row)

    aggregates = []
    for (shape_variant, clusterer, vocab_size), items in grouped.items():
        wd = [item["word_distribution"] for item in items]
        tr = [item["transition"] for item in items]
        val = [item["validation_next_token"] for item in items]
        stability = _distribution_stability([item["word_probabilities"] for item in items])
        aggregates.append(
            {
                "shape_variant": shape_variant,
                "clusterer": clusterer,
                "vocab_size_requested": int(vocab_size),
                "vocab_size_observed_mean": float(np.mean([item["vocab_size_observed"] for item in items])),
                "empty_clusters_mean": float(np.mean([item["empty_clusters"] for item in items])),
                "observed_vocab_ratio_mean": float(np.mean([item["observed_vocab_ratio"] for item in items])),
                "normalized_entropy_mean": _mean(wd, "normalized_entropy"),
                "normalized_entropy_std": _std(wd, "normalized_entropy"),
                "effective_vocab_size_mean": _mean(wd, "effective_vocab_size"),
                "dominant_share_mean": _mean(wd, "dominant_share"),
                "top3_share_mean": _mean(wd, "top3_share"),
                "top5_share_mean": _mean(wd, "top5_share"),
                "rare_word_share_mean": _mean(wd, "rare_word_share"),
                "transition_entropy_mean": _mean(tr, "transition_entropy"),
                "self_transition_rate_mean": _mean(tr, "self_transition_rate"),
                "average_outgoing_transition_entropy_mean": _mean(tr, "average_outgoing_transition_entropy"),
                "fold_distribution_l1_mean": stability["mean_pairwise_l1"],
                "fold_distribution_js_mean": stability["mean_pairwise_js"],
                "validation_token_nll_mean": _mean(val, "token_nll"),
                "validation_perplexity_mean": _mean(val, "perplexity"),
                "validation_top3_mean": _mean(val, "top3_accuracy"),
                "validation_top5_mean": _mean(val, "top5_accuracy"),
                "validation_mrr_mean": _mean(val, "mrr"),
                "validation_sequence_exact_mean": _mean(val, "sequence_exact"),
            }
        )
    return sorted(aggregates, key=lambda row: (row["validation_perplexity_mean"], -row["validation_top3_mean"]))


def apply_vocabulary_constraints(
    vocabulary_aggregates: list[dict[str, Any]],
    *,
    min_norm_entropy: float,
    max_dominant_share: float,
    max_top3_share: float,
    min_observed_vocab_ratio: float,
) -> list[dict[str, Any]]:
    constrained = []
    for row in vocabulary_aggregates:
        reasons = []
        if row["normalized_entropy_mean"] < min_norm_entropy:
            reasons.append(f"normalized_entropy<{min_norm_entropy}")
        if row["dominant_share_mean"] > max_dominant_share:
            reasons.append(f"dominant_share>{max_dominant_share}")
        if row["top3_share_mean"] > max_top3_share:
            reasons.append(f"top3_share>{max_top3_share}")
        if row["observed_vocab_ratio_mean"] < min_observed_vocab_ratio:
            reasons.append(f"observed_vocab_ratio<{min_observed_vocab_ratio}")
        item = dict(row)
        item["accepted_by_constraints"] = not reasons
        item["rejection_reason"] = "; ".join(reasons) if reasons else ""
        constrained.append(item)
    return constrained


def build_candidate_table(vocabulary_rows: list[dict[str, Any]], lm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_lm_by_vocab: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in lm_rows:
        key = (row["shape_variant"], row["clusterer"], row["vocab_size_requested"])
        current = best_lm_by_vocab.get(key)
        if current is None or (row["token_nll_mean"], row["perplexity_mean"], -row["top3_mean"]) < (
            current["token_nll_mean"],
            current["perplexity_mean"],
            -current["top3_mean"],
        ):
            best_lm_by_vocab[key] = row

    candidates = []
    for vocab in vocabulary_rows:
        key = (vocab["shape_variant"], vocab["clusterer"], vocab["vocab_size_requested"])
        lm = best_lm_by_vocab.get(key, {})
        candidates.append(
            {
                "shape_variant": vocab["shape_variant"],
                "clusterer": vocab["clusterer"],
                "vocab_size": vocab["vocab_size_requested"],
                "vocab_size_observed": vocab["vocab_size_observed_mean"],
                "norm_entropy": vocab["normalized_entropy_mean"],
                "dominant_share": vocab["dominant_share_mean"],
                "top3_share": vocab["top3_share_mean"],
                "top5_share": vocab["top5_share_mean"],
                "rare_word_share": vocab["rare_word_share_mean"],
                "effective_vocab_size": vocab["effective_vocab_size_mean"],
                "transition_entropy": vocab["transition_entropy_mean"],
                "self_transition_rate": vocab["self_transition_rate_mean"],
                "fold_distribution_l1": vocab["fold_distribution_l1_mean"],
                "token_nll_mean": lm.get("token_nll_mean"),
                "token_nll_std": lm.get("token_nll_std"),
                "perplexity_mean": lm.get("perplexity_mean"),
                "top3_mean": lm.get("top3_mean"),
                "mrr_mean": lm.get("mrr_mean"),
                "sequence_exact_mean": lm.get("sequence_exact_mean"),
                "best_context_size": lm.get("context_size"),
                "best_forecast_horizon": lm.get("forecast_horizon"),
                "best_markov_order": lm.get("markov_order"),
                "accepted_by_constraints": vocab["accepted_by_constraints"],
                "rejection_reason": vocab["rejection_reason"],
            }
        )
    return sorted(candidates, key=lambda row: (not row["accepted_by_constraints"], row["token_nll_mean"] if row["token_nll_mean"] is not None else float("inf")))


def select_best_by_validation(aggregates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not aggregates:
        return None
    return min(enumerate(aggregates), key=lambda item: (item[1]["token_nll_mean"], item[1]["perplexity_mean"], -item[1]["top3_mean"], item[0]))[1]


def select_best_constrained(aggregates: list[dict[str, Any]], constrained_vocab: list[dict[str, Any]]) -> dict[str, Any] | None:
    accepted = {
        (row["shape_variant"], row["clusterer"], row["vocab_size_requested"])
        for row in constrained_vocab
        if row["accepted_by_constraints"]
    }
    candidates = [
        row
        for row in aggregates
        if (row["shape_variant"], row["clusterer"], row["vocab_size_requested"]) in accepted
    ]
    return select_best_by_validation(candidates)


def run_best_diagnostics(
    best: dict[str, Any] | None,
    *,
    folds: list[WalkForwardRange],
    shape_cache: dict[str, np.ndarray],
    random_state: int,
    beam_width: int,
) -> dict[str, Any]:
    if best is None:
        return {}
    spec = _best_to_spec(best)
    confidence_rows = []
    error_rows = []
    for fold in folds:
        word_ids, clusterer = fit_fold_vocabulary(shape_cache[spec.shape_variant], fold, spec=spec, random_state=random_state)
        samples = make_next_word_samples(word_ids, fold.val_start, fold.val_end, int(best["context_size"]), int(best["forecast_horizon"]))
        model = NGramBackoffLanguageModel(order=int(best["markov_order"]), alpha=float(best["smoothing_alpha"])).fit(
            word_ids,
            train_start=fold.train_start,
            train_end=fold.train_end,
            n_words=clusterer.n_words_,
        )
        confidence_rows.append({"fold_id": int(fold.fold_id), **confidence_analysis(model, samples.X_contexts, samples.Y_future_words)})
        error_rows.append(
            {
                "fold_id": int(fold.fold_id),
                **error_analysis(
                    model,
                    samples.X_contexts,
                    samples.Y_future_words,
                    distance_matrix=clusterer_distance_matrix(clusterer),
                ),
            }
        )
    return {
        "config": best,
        "confidence_by_fold": confidence_rows,
        "error_by_fold": error_rows,
        "confidence_summary": _summarize_confidence(confidence_rows),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)


def print_summary(candidates: list[dict[str, Any]], best_constrained: dict[str, Any] | None) -> None:
    print("Словарь | accepted | NLL | PPL | top3 | entropy | dominant | top3_share | reason")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---")
    for row in candidates[:12]:
        print(
            f"{row['shape_variant']}/{row['clusterer']}/{row['vocab_size']} | "
            f"{row['accepted_by_constraints']} | {_fmt(row['token_nll_mean'])} | {_fmt(row['perplexity_mean'])} | "
            f"{_fmt(row['top3_mean'])} | {_fmt(row['norm_entropy'])} | {_fmt(row['dominant_share'])} | "
            f"{_fmt(row['top3_share'])} | {row['rejection_reason']}"
        )
    if best_constrained:
        print(
            "Лучший допустимый словарь: "
            f"{best_constrained['shape_variant']}/{best_constrained['clusterer']}/{best_constrained['vocab_size_requested']} "
            f"context={best_constrained['context_size']} K={best_constrained['forecast_horizon']} "
            f"order={best_constrained['markov_order']}"
        )


def _validate_sample_count(sample_count: int, split_len: int, context_size: int, forecast_horizon: int, split_name: str) -> None:
    expected = expected_next_word_sample_count(split_len, context_size, forecast_horizon)
    if sample_count != expected:
        raise ValueError(f"{split_name} sample count mismatch: got {sample_count}, expected {expected}")
    if sample_count <= 0:
        raise ValueError(f"{split_name} has no samples")


def _distribution_stability(distributions: list[list[float]]) -> dict[str, float]:
    if len(distributions) < 2:
        return {"mean_pairwise_l1": 0.0, "mean_pairwise_js": 0.0}
    max_len = max(len(row) for row in distributions)
    padded = []
    for row in distributions:
        arr = np.zeros(max_len, dtype=float)
        arr[: len(row)] = row
        if arr.sum() > 0:
            arr = arr / arr.sum()
        padded.append(arr)
    l1_values = []
    js_values = []
    for i in range(len(padded)):
        for j in range(i + 1, len(padded)):
            p = padded[i]
            q = padded[j]
            l1_values.append(float(np.abs(p - q).sum()))
            m = 0.5 * (p + q)
            js_values.append(0.5 * _kl(p, m) + 0.5 * _kl(q, m))
    return {"mean_pairwise_l1": float(np.mean(l1_values)), "mean_pairwise_js": float(np.mean(js_values))}


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-300))))


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def _std(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.std([row[key] for row in rows], ddof=0))


def _best_to_spec(best: dict[str, Any]) -> VocabularySpec:
    clusterer = "gmm" if str(best["clusterer"]).startswith("gmm") else str(best["clusterer"])
    covariance = None
    if str(best["clusterer"]).startswith("gmm_"):
        covariance = str(best["clusterer"]).split("_", 1)[1]
    return VocabularySpec(str(best["shape_variant"]), clusterer, int(best["vocab_size_requested"]), covariance)


def _summarize_confidence(confidence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not confidence_rows:
        return {}
    summaries = [row["summary"] for row in confidence_rows]
    abstention = []
    labels = confidence_rows[0]["abstention_curves"]["top1_probability"]
    for idx, item in enumerate(labels):
        values = [row["abstention_curves"]["top1_probability"][idx] for row in confidence_rows]
        valid_acc = [value["accuracy_at_1"] for value in values if value["accuracy_at_1"] is not None]
        abstention.append(
            {
                "label": item["label"],
                "coverage_mean": float(np.mean([value["coverage"] for value in values])),
                "accuracy_at_1_mean": float(np.mean(valid_acc)) if valid_acc else None,
            }
        )
    return {
        "mean_top1_probability": _mean(summaries, "mean_top1_probability"),
        "mean_top3_probability_mass": _mean(summaries, "mean_top3_probability_mass"),
        "mean_distribution_entropy": _mean(summaries, "mean_distribution_entropy"),
        "accuracy_at_1": _mean(summaries, "accuracy_at_1"),
        "top3_accuracy": _mean(summaries, "top3_accuracy"),
        "abstention_top1_probability": abstention,
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


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
    parser.add_argument("--gmm-covariance-types", default="diag")
    parser.add_argument("--context-sizes", default="16,32")
    parser.add_argument("--forecast-horizons", default="3,5")
    parser.add_argument("--orders", default="1,2,3")
    parser.add_argument("--smoothing-alphas", default="0.1")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--fold-mode", choices=["expanding", "rolling"], default="expanding")
    parser.add_argument("--max-folds", type=int, default=3)
    parser.add_argument("--initial-train-size", type=int, default=12000)
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=3000)
    parser.add_argument("--step-size", type=int, default=3000)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--min-norm-entropy", type=float, default=0.50)
    parser.add_argument("--max-dominant-share", type=float, default=0.55)
    parser.add_argument("--max-top3-share", type=float, default=0.80)
    parser.add_argument("--min-observed-vocab-ratio", type=float, default=0.80)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-json", default="data/reports/sber_h1_word_lm_walk_forward_20260515.json")
    parser.add_argument("--output-csv", default="data/reports/sber_h1_word_lm_walk_forward_20260515.csv")
    args = parser.parse_args()

    if args.quick:
        args.shape_variants = "shape"
        args.clusterers = "kmeans,gmm"
        args.vocab_sizes = "16,20"
        args.context_sizes = "16"
        args.forecast_horizons = "3"
        args.orders = "1,2"
        args.max_folds = min(args.max_folds, 2)

    started = time.perf_counter()
    df, data_path = load_sber_frame(args)
    folds = build_folds(args, len(df))
    print(f"Загружено свечей: {len(df)}; файл: {data_path}")
    print(f"Folds: {len(folds)}; режим: {args.fold_mode}")

    vocab_specs = iter_vocab_specs(args)
    context_sizes = parse_int_list(args.context_sizes)
    forecast_horizons = parse_int_list(args.forecast_horizons)
    orders = parse_int_list(args.orders)
    alphas = parse_float_list(args.smoothing_alphas)
    shape_cache = {variant: candle_shape_matrix(df, variant=variant)[0] for variant in sorted({spec.shape_variant for spec in vocab_specs})}

    lm_rows: list[dict[str, Any]] = []
    vocabulary_rows: list[dict[str, Any]] = []
    for fold in folds:
        print(f"Fold {fold.fold_id}: train=[{fold.train_start}:{fold.train_end}) val=[{fold.val_start}:{fold.val_end})")
        for spec in vocab_specs:
            word_ids, clusterer = fit_fold_vocabulary(
                shape_cache[spec.shape_variant],
                fold,
                spec=spec,
                random_state=args.random_state,
            )
            vocabulary_rows.append(
                vocabulary_audit_row(
                    word_ids,
                    fold,
                    clusterer,
                    spec=spec,
                    context_size=min(context_sizes),
                    order=min(orders),
                    alpha=alphas[0],
                    beam_width=args.beam_width,
                )
            )
            for context_size in context_sizes:
                for forecast_horizon in forecast_horizons:
                    for order in orders:
                        for alpha in alphas:
                            lm_rows.append(
                                evaluate_lm_config(
                                    word_ids,
                                    fold,
                                    clusterer,
                                    spec=spec,
                                    context_size=context_size,
                                    forecast_horizon=forecast_horizon,
                                    order=order,
                                    alpha=alpha,
                                    beam_width=args.beam_width,
                                )
                            )

    lm_aggregates = aggregate_lm_rows(lm_rows)
    vocabulary_aggregates = aggregate_vocabulary_rows(vocabulary_rows)
    constrained_vocab = apply_vocabulary_constraints(
        vocabulary_aggregates,
        min_norm_entropy=args.min_norm_entropy,
        max_dominant_share=args.max_dominant_share,
        max_top3_share=args.max_top3_share,
        min_observed_vocab_ratio=args.min_observed_vocab_ratio,
    )
    candidate_table = build_candidate_table(constrained_vocab, lm_aggregates)
    best_unconstrained = select_best_by_validation(lm_aggregates)
    best_constrained = select_best_constrained(lm_aggregates, constrained_vocab)
    diagnostics = run_best_diagnostics(
        best_constrained,
        folds=folds,
        shape_cache=shape_cache,
        random_state=args.random_state,
        beam_width=args.beam_width,
    )
    rejected = [row for row in constrained_vocab if not row["accepted_by_constraints"]]
    rejected = sorted(rejected, key=lambda row: (row["validation_perplexity_mean"], row["dominant_share_mean"]))[:12]

    payload = {
        "data_path": str(data_path),
        "rows": int(len(df)),
        "fold_mode": args.fold_mode,
        "folds": [fold.__dict__ for fold in folds],
        "constraints": {
            "min_norm_entropy": float(args.min_norm_entropy),
            "max_dominant_share": float(args.max_dominant_share),
            "max_top3_share": float(args.max_top3_share),
            "min_observed_vocab_ratio": float(args.min_observed_vocab_ratio),
        },
        "selection": "Filter vocabularies by train-only balance constraints, then select by validation token NLL/perplexity/top-k; no test split is used.",
        "fold_results": lm_rows,
        "lm_aggregates": lm_aggregates,
        "vocabulary_fold_results": vocabulary_rows,
        "vocabulary_aggregates": constrained_vocab,
        "candidate_table": candidate_table,
        "best_unconstrained": best_unconstrained,
        "best_constrained": best_constrained,
        "top_rejected_vocabularies": rejected,
        "best_constrained_diagnostics": diagnostics,
        "duration_sec": float(time.perf_counter() - started),
    }

    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    write_json(payload, output_json)
    write_csv(candidate_table, output_csv)
    print_summary(candidate_table, best_constrained)
    print(f"Записан JSON: {output_json}")
    print(f"Записан CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
