"""One-shot final evaluation for a frozen LM-derived action candidate.

This script is intentionally narrow: it evaluates a preselected research
candidate on the untouched chronological test split and does not tune anything
on test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    split_ranges,
)
from src.nlp.classifiers import build_classifier, classifier_requires_dense, maybe_dense
from src.nlp.vectorizers import build_vectorizer
from src.nlp.word_forecast import clusterer_distance_matrix
from src.nlp.word_lm import NGramBackoffLanguageModel
from src.utils.io import ensure_dir

from sber_action_nested_thresholds import (
    NestedRange,
    ThresholdDecision,
    _jsonable,
    _require_proba,
    action_metrics,
    apply_temperature,
    build_feature_matrices,
    build_regime_feature_matrix,
    build_regime_labels,
    fit_words_for_nested,
    negative_log_likelihood,
    parse_float_list,
    parse_vocab_configs,
    regime_rows_for_predictions,
    resolve_class_weight,
    result_row,
    select_global_thresholds,
)
from sber_action_lm_features_walk_forward import (
    _proper_action_probabilities,
    load_sber_frame,
)


def parse_random_state_policy(value: str) -> int:
    if value.startswith("fixed:"):
        return int(value.split(":", 1)[1])
    return int(value)


def final_nested_range(n_rows: int, calibration_size: int) -> tuple[NestedRange, dict[str, tuple[int, int]]]:
    ranges = split_ranges(n_rows, train_ratio=0.7, val_ratio=0.15)
    dev_start = ranges["train"][0]
    dev_end = ranges["test"][0]
    test_start, test_end = ranges["test"]
    outer = WalkForwardRange(
        fold_id=0,
        train_start=dev_start,
        train_end=dev_end,
        val_start=test_start,
        val_end=test_end,
    )
    calibration_start = dev_end - calibration_size
    if calibration_start <= dev_start:
        raise ValueError("calibration_size leaves no development training rows")
    nested = NestedRange(
        outer_fold=outer,
        inner_train_start=dev_start,
        inner_train_end=calibration_start,
        calibration_start=calibration_start,
        calibration_end=dev_end,
    )
    return nested, ranges


def evaluate_frozen_candidate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    df, data_path = load_sber_frame(args)
    nested, holdout_ranges = final_nested_range(len(df), args.calibration_size)
    vocab_config = parse_vocab_configs(args.vocab_config)[0]
    random_state = parse_random_state_policy(args.random_state_policy)

    labels, future_returns, label_threshold = make_action_labels(df, horizon=args.action_horizon, commission=0.0005)
    shape_matrix = candle_shape_matrix(df, variant=vocab_config.shape_variant)[0]
    word_ids, clusterer = fit_words_for_nested(shape_matrix, nested, vocab_config, random_state=random_state)
    word_tokens = clusterer.labels_to_words(word_ids)
    samples = {
        "inner_train": make_sentence_samples(
            word_tokens,
            labels,
            future_returns,
            nested.inner_train_start,
            nested.inner_train_end,
            args.action_window_size,
            args.action_horizon,
        ),
        "calibration": make_sentence_samples(
            word_tokens,
            labels,
            future_returns,
            nested.calibration_start,
            nested.calibration_end,
            args.action_window_size,
            args.action_horizon,
        ),
        "outer_val": make_sentence_samples(
            word_tokens,
            labels,
            future_returns,
            nested.outer_fold.val_start,
            nested.outer_fold.val_end,
            args.action_window_size,
            args.action_horizon,
        ),
    }

    vectorizer = build_vectorizer(
        VectorizerSpec(
            "cooccurrence_svd",
            {"embedding_dim": 24, "context_window": 2, "pool": ("mean", "std", "last"), "include_histogram": True},
        ),
        random_state=random_state,
    )
    X_base_train = vectorizer.fit_transform(samples["inner_train"].sentences, samples["inner_train"].token_lists)
    X_base_calib = vectorizer.transform(samples["calibration"].sentences, samples["calibration"].token_lists)
    X_base_test = vectorizer.transform(samples["outer_val"].sentences, samples["outer_val"].token_lists)

    lm = NGramBackoffLanguageModel(order=args.lm_order, alpha=args.lm_alpha).fit(
        word_ids,
        train_start=nested.inner_train_start,
        train_end=nested.inner_train_end,
        n_words=clusterer.n_words_,
    )
    distance_matrix = clusterer_distance_matrix(clusterer)
    lm_train = make_lm_action_features(
        word_ids=word_ids,
        target_indices=samples["inner_train"].target_indices,
        context_size=args.context_size,
        model=lm,
        distance_matrix=distance_matrix,
        include_probabilities=True,
        beam_horizon=args.forecast_horizon,
    )
    lm_calib = make_lm_action_features(
        word_ids=word_ids,
        target_indices=samples["calibration"].target_indices,
        context_size=args.context_size,
        model=lm,
        distance_matrix=distance_matrix,
        include_probabilities=True,
        beam_horizon=args.forecast_horizon,
    )
    lm_test = make_lm_action_features(
        word_ids=word_ids,
        target_indices=samples["outer_val"].target_indices,
        context_size=args.context_size,
        model=lm,
        distance_matrix=distance_matrix,
        include_probabilities=True,
        beam_horizon=args.forecast_horizon,
    )

    regime_train = build_regime_feature_matrix(
        df, samples["inner_train"].target_indices, samples["inner_train"].target_indices, lm_train.X, lm_train.X
    )
    regime_calib = build_regime_feature_matrix(
        df, samples["inner_train"].target_indices, samples["calibration"].target_indices, lm_train.X, lm_calib.X
    )
    regime_test = build_regime_feature_matrix(
        df, samples["inner_train"].target_indices, samples["outer_val"].target_indices, lm_train.X, lm_test.X
    )
    regime_labels = build_regime_labels(
        df, samples["inner_train"].target_indices, samples["outer_val"].target_indices, lm_train.X, lm_test.X
    )

    X_train, X_calib, X_test = build_feature_matrices(
        args.feature_set,
        X_base_train,
        X_base_calib,
        X_base_test,
        lm_train.X,
        lm_calib.X,
        lm_test.X,
        regime_train,
        regime_calib,
        regime_test,
    )

    classifier = build_classifier(
        ClassifierSpec(args.classifier, {"max_iter": 1000, "class_weight": resolve_class_weight(args.class_weight)}),
        random_state=random_state,
    )
    fit_train = X_train
    fit_calib = X_calib
    fit_test = X_test
    if classifier_requires_dense(ClassifierSpec(args.classifier)):
        fit_train = maybe_dense(fit_train)
        fit_calib = maybe_dense(fit_calib)
        fit_test = maybe_dense(fit_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier.fit(fit_train, samples["inner_train"].y)
    calib_proba = _require_proba(_proper_action_probabilities(classifier, fit_calib))
    test_proba = _require_proba(_proper_action_probabilities(classifier, fit_test))

    if args.decision == "argmax":
        decision = ThresholdDecision(
            mode="argmax",
            calibration_method="none",
            selection_objective="frozen_argmax",
            temperature_selection="none",
            temperature=1.0,
            buy_threshold=None,
            sell_threshold=None,
            calibration_score=float(action_metrics(samples["calibration"].y, np.argmax(calib_proba, axis=1))["macro_f1"]),
            calibration_metrics=action_metrics(samples["calibration"].y, np.argmax(calib_proba, axis=1)),
        )
    elif args.decision == "global":
        decision = select_global_thresholds(
            samples["calibration"].y,
            calib_proba,
            parse_float_list(args.threshold_grid),
            parse_float_list(args.temperature_grid),
            selection_objective=args.selection_objective,
            temperature_selection=args.temperature_selection,
            target_action_rate=args.target_action_rate,
            action_rate_penalty=args.action_rate_penalty,
            mode="global",
        )
    else:
        raise ValueError(f"Unsupported decision: {args.decision}")

    row = result_row(
        nested,
        samples["outer_val"].y,
        test_proba,
        decision,
        vocab_config=vocab_config,
        feature_set=args.feature_set,
        classifier_name=args.classifier,
        class_weight=args.class_weight,
        action_horizon=args.action_horizon,
        random_state=random_state,
        regime_labels=regime_labels,
    )
    test_calibrated = apply_temperature(test_proba, decision.temperature)
    test_nll = negative_log_likelihood(samples["outer_val"].y, test_calibrated)
    payload = {
        "purpose": "one-shot untouched test evaluation for a frozen research candidate; test is report-only",
        "data_path": str(data_path),
        "rows": int(len(df)),
        "holdout_ranges": {key: [int(start), int(end)] for key, (start, end) in holdout_ranges.items()},
        "nested_range": asdict(nested),
        "frozen_config": {
            "vocabulary": vocab_config.label,
            "feature_set": args.feature_set,
            "classifier": args.classifier,
            "class_weight": args.class_weight,
            "action_horizon": args.action_horizon,
            "decision": args.decision,
            "context_size": args.context_size,
            "forecast_horizon": args.forecast_horizon,
            "lm_order": args.lm_order,
            "lm_alpha": args.lm_alpha,
            "random_state_policy": args.random_state_policy,
        },
        "label_threshold": float(label_threshold),
        "n_train_samples": int(samples["inner_train"].size),
        "n_calibration_samples": int(samples["calibration"].size),
        "n_test_samples": int(samples["outer_val"].size),
        "test_result": row,
        "test_nll": float(test_nll),
        "test_true_distribution": label_distribution(samples["outer_val"].y),
        "duration_sec": float(time.perf_counter() - started),
    }
    return payload


def compact_csv_row(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload["test_result"]
    metrics = result["metrics"]
    config = payload["frozen_config"]
    return {
        **config,
        "n_train_samples": payload["n_train_samples"],
        "n_calibration_samples": payload["n_calibration_samples"],
        "n_test_samples": payload["n_test_samples"],
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "buy_f1": metrics["buy_f1"],
        "sell_f1": metrics["sell_f1"],
        "hold_f1": metrics["hold_f1"],
        "buy_sell_hmean_f1": metrics["buy_sell_hmean_f1"],
        "action_rate": metrics["action_rate"],
        "test_nll": payload["test_nll"],
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)


def write_csv(row: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    pd.DataFrame([row]).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--timeframe", default="1H")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data", default="")
    parser.add_argument("--vocab-config", default="shape:gmm:20")
    parser.add_argument("--feature-set", default="lm_regime")
    parser.add_argument("--classifier", default="logreg")
    parser.add_argument("--class-weight", default="action_boost_1.2")
    parser.add_argument("--action-horizon", type=int, default=1)
    parser.add_argument("--decision", choices=["argmax", "global"], default="argmax")
    parser.add_argument("--context-size", type=int, default=16)
    parser.add_argument("--action-window-size", type=int, default=32)
    parser.add_argument("--forecast-horizon", type=int, default=3)
    parser.add_argument("--lm-order", type=int, default=2)
    parser.add_argument("--lm-alpha", type=float, default=0.1)
    parser.add_argument("--calibration-size", type=int, default=2500)
    parser.add_argument("--random-state-policy", default="fixed:42")
    parser.add_argument("--threshold-grid", default="0.20,0.225,0.25,0.275,0.30,0.325,0.35,0.375,0.40")
    parser.add_argument("--temperature-grid", default="0.60,0.70,0.80,0.90,1.00,1.10,1.20,1.30,1.40,1.50")
    parser.add_argument("--selection-objective", default="macro_f1")
    parser.add_argument("--temperature-selection", choices=["nll", "macro_f1", "objective"], default="objective")
    parser.add_argument("--target-action-rate", type=float, default=0.50)
    parser.add_argument("--action-rate-penalty", type=float, default=0.10)
    parser.add_argument("--output-json", default="data/reports/sber_h1_action_final_eval_20260515.json")
    parser.add_argument("--output-csv", default="data/reports/sber_h1_action_final_eval_20260515.csv")
    args = parser.parse_args()

    payload = evaluate_frozen_candidate(args)
    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    write_json(payload, output_json)
    write_csv(compact_csv_row(payload), output_csv)
    metrics = payload["test_result"]["metrics"]
    print("Frozen candidate final test evaluation")
    print(f"macro-F1={metrics['macro_f1']:.4f}; BUY F1={metrics['buy_f1']:.4f}; SELL F1={metrics['sell_f1']:.4f}; HOLD F1={metrics['hold_f1']:.4f}")
    print(f"action_rate={metrics['action_rate']:.4f}; n_test_samples={payload['n_test_samples']}")
    print(f"JSON: {output_json}")
    print(f"CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
