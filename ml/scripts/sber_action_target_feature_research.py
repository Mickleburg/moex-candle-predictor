"""Research CLI for alternative action targets and past-only feature baselines."""

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
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.nlp import (
    ActionTargetSpec,
    ClassifierSpec,
    VectorizerSpec,
    candle_shape_matrix,
    make_continuous_past_features,
    make_lm_action_features,
    make_research_action_targets,
    make_sentence_samples,
    standardize_by_train,
    target_analysis,
)
from src.nlp.classifiers import build_classifier, classifier_requires_dense, maybe_dense
from src.nlp.vectorizers import build_vectorizer
from src.nlp.word_forecast import clusterer_distance_matrix
from src.nlp.word_lm import NGramBackoffLanguageModel
from src.utils.io import ensure_dir

from sber_action_lm_features_walk_forward import (
    _proper_action_probabilities,
    action_metrics,
    build_folds,
    label_distribution,
    load_sber_frame,
    parse_list,
    parse_vocab_configs,
)
from sber_action_nested_thresholds import (
    _jsonable,
    build_feature_matrices,
    build_regime_feature_matrix,
    fit_words_for_nested,
    nested_range,
    parse_class_weights,
    parse_float_list,
    parse_int_list,
    resolve_class_weight,
)


def build_target_specs(args: argparse.Namespace) -> list[ActionTargetSpec]:
    specs: list[ActionTargetSpec] = []
    action_horizons = parse_int_list(args.action_horizons)
    vol_windows = parse_int_list(args.vol_windows)
    vol_ks = parse_float_list(args.vol_ks)
    barrier_horizons = parse_int_list(args.barrier_horizons)
    barrier_vol_windows = parse_int_list(args.barrier_vol_windows)
    barrier_up_ks = parse_float_list(args.barrier_up_ks)
    barrier_down_ks = parse_float_list(args.barrier_down_ks)
    neutral_buy = parse_float_list(args.buy_threshold_mults)
    neutral_sell = parse_float_list(args.sell_threshold_mults)
    for mode in parse_list(args.target_modes):
        if mode == "return_threshold":
            for horizon in action_horizons:
                specs.append(ActionTargetSpec(mode=mode, horizon=horizon))
        elif mode == "volatility_adjusted_return":
            for horizon in action_horizons:
                for window in vol_windows:
                    for vol_k in vol_ks:
                        specs.append(ActionTargetSpec(mode=mode, horizon=horizon, vol_window=window, vol_k=vol_k))
        elif mode == "triple_barrier":
            for horizon in barrier_horizons:
                for window in barrier_vol_windows:
                    for up_k in barrier_up_ks:
                        for down_k in barrier_down_ks:
                            specs.append(
                                ActionTargetSpec(
                                    mode=mode,
                                    barrier_horizon=horizon,
                                    barrier_vol_window=window,
                                    barrier_up_k=up_k,
                                    barrier_down_k=down_k,
                                )
                            )
        elif mode == "neutral_zone_return":
            for horizon in action_horizons:
                for buy_mult in neutral_buy:
                    for sell_mult in neutral_sell:
                        specs.append(
                            ActionTargetSpec(
                                mode=mode,
                                horizon=horizon,
                                buy_threshold_mult=buy_mult,
                                sell_threshold_mult=sell_mult,
                            )
                        )
        else:
            raise ValueError(f"Unsupported target mode: {mode}")
    return specs


def uses_lm_features(feature_sets: list[str]) -> bool:
    return any(name in {"lm_regime", "lm_regime_continuous"} for name in feature_sets)


def build_feature_set_matrices(
    feature_set: str,
    *,
    lm_train: np.ndarray | None,
    lm_calib: np.ndarray | None,
    lm_val: np.ndarray | None,
    regime_train: np.ndarray | None,
    regime_calib: np.ndarray | None,
    regime_val: np.ndarray | None,
    cont_train: np.ndarray,
    cont_calib: np.ndarray,
    cont_val: np.ndarray,
) -> tuple[Any, Any, Any]:
    if feature_set == "continuous_regime":
        return cont_train, cont_calib, cont_val
    if feature_set == "lm_regime":
        if lm_train is None or regime_train is None:
            raise ValueError("lm_regime requires LM features")
        scalar_width = 18
        return (
            np.hstack([lm_train[:, :scalar_width], regime_train]),
            np.hstack([lm_calib[:, :scalar_width], regime_calib]),
            np.hstack([lm_val[:, :scalar_width], regime_val]),
        )
    if feature_set == "lm_regime_continuous":
        if lm_train is None or regime_train is None:
            raise ValueError("lm_regime_continuous requires LM features")
        scalar_width = 18
        return (
            np.hstack([lm_train[:, :scalar_width], regime_train, cont_train]),
            np.hstack([lm_calib[:, :scalar_width], regime_calib, cont_calib]),
            np.hstack([lm_val[:, :scalar_width], regime_val, cont_val]),
        )
    raise ValueError(f"Unsupported feature set: {feature_set}")


def run_fold_target(
    df: pd.DataFrame,
    shape_matrix: np.ndarray,
    continuous_matrix: np.ndarray,
    target_spec: ActionTargetSpec,
    ranges: Any,
    vocab_config: Any,
    *,
    feature_sets: list[str],
    models: list[str],
    class_weights: list[str | None],
    context_size: int,
    action_window_size: int,
    lm_order: int,
    lm_alpha: float,
    lm_forecast_horizon: int,
    random_state: int,
) -> list[dict[str, Any]]:
    target = make_research_action_targets(df, target_spec)
    dummy_tokens = ["w000"] * len(df)
    word_ids = None
    clusterer = None
    word_tokens = dummy_tokens
    if uses_lm_features(feature_sets):
        word_ids, clusterer = fit_words_for_nested(shape_matrix, ranges, vocab_config, random_state=random_state)
        word_tokens = clusterer.labels_to_words(word_ids)

    samples = {
        "inner_train": make_sentence_samples(
            word_tokens,
            target.labels,
            target.future_returns,
            ranges.inner_train_start,
            ranges.inner_train_end,
            action_window_size,
            target.effective_horizon,
        ),
        "calibration": make_sentence_samples(
            word_tokens,
            target.labels,
            target.future_returns,
            ranges.calibration_start,
            ranges.calibration_end,
            action_window_size,
            target.effective_horizon,
        ),
        "outer_val": make_sentence_samples(
            word_tokens,
            target.labels,
            target.future_returns,
            ranges.outer_fold.val_start,
            ranges.outer_fold.val_end,
            action_window_size,
            target.effective_horizon,
        ),
    }
    _validate_samples(samples, ranges, action_window_size, target.effective_horizon, context_size)

    cont_train = standardize_by_train(
        continuous_matrix, samples["inner_train"].target_indices, samples["inner_train"].target_indices
    )
    cont_calib = standardize_by_train(
        continuous_matrix, samples["inner_train"].target_indices, samples["calibration"].target_indices
    )
    cont_val = standardize_by_train(
        continuous_matrix, samples["inner_train"].target_indices, samples["outer_val"].target_indices
    )

    lm_train = lm_calib = lm_val = None
    regime_train = regime_calib = regime_val = None
    if uses_lm_features(feature_sets):
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
            include_probabilities=False,
            beam_horizon=lm_forecast_horizon,
        ).X
        lm_calib = make_lm_action_features(
            word_ids=word_ids,
            target_indices=samples["calibration"].target_indices,
            context_size=context_size,
            model=lm,
            distance_matrix=distance_matrix,
            include_probabilities=False,
            beam_horizon=lm_forecast_horizon,
        ).X
        lm_val = make_lm_action_features(
            word_ids=word_ids,
            target_indices=samples["outer_val"].target_indices,
            context_size=context_size,
            model=lm,
            distance_matrix=distance_matrix,
            include_probabilities=False,
            beam_horizon=lm_forecast_horizon,
        ).X
        regime_train = build_regime_feature_matrix(
            df, samples["inner_train"].target_indices, samples["inner_train"].target_indices, lm_train, lm_train
        )
        regime_calib = build_regime_feature_matrix(
            df, samples["inner_train"].target_indices, samples["calibration"].target_indices, lm_train, lm_calib
        )
        regime_val = build_regime_feature_matrix(
            df, samples["inner_train"].target_indices, samples["outer_val"].target_indices, lm_train, lm_val
        )

    rows = []
    for feature_set in feature_sets:
        X_train, _, X_val = build_feature_set_matrices(
            feature_set,
            lm_train=lm_train,
            lm_calib=lm_calib,
            lm_val=lm_val,
            regime_train=regime_train,
            regime_calib=regime_calib,
            regime_val=regime_val,
            cont_train=cont_train,
            cont_calib=cont_calib,
            cont_val=cont_val,
        )
        for model_name in models:
            for class_weight in class_weights:
                pred, proba = fit_predict_model(
                    X_train,
                    samples["inner_train"].y,
                    X_val,
                    model_name=model_name,
                    class_weight=class_weight,
                    random_state=random_state,
                )
                metrics = action_metrics(samples["outer_val"].y, pred)
                metrics["action_rate"] = float(np.mean(np.isin(pred, [0, 2])))
                metrics["hold_rate"] = float(np.mean(pred == 1))
                metrics["buy_sell_hmean_f1"] = float(
                    2.0 * metrics["buy_f1"] * metrics["sell_f1"] / (metrics["buy_f1"] + metrics["sell_f1"] + 1e-12)
                )
                rows.append(
                    {
                        "target_mode": target_spec.mode,
                        "target_label": target_spec.label,
                        "target_params": asdict(target_spec),
                        "feature_set": feature_set,
                        "model": model_name,
                        "class_weight": "none" if class_weight is None else str(class_weight),
                        "fold_id": int(ranges.outer_fold.fold_id),
                        "random_state": int(random_state),
                        "n_train": int(samples["inner_train"].size),
                        "n_calibration": int(samples["calibration"].size),
                        "n_validation": int(samples["outer_val"].size),
                        "target_distribution_train": label_distribution(samples["inner_train"].y),
                        "target_distribution_val": label_distribution(samples["outer_val"].y),
                        "prediction_distribution": label_distribution(pred),
                        "metrics": metrics,
                        "has_predict_proba": proba is not None,
                        "target_analysis": target_analysis(target.labels, target.future_returns, target.metadata),
                    }
                )
    return rows


def fit_predict_model(
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    *,
    model_name: str,
    class_weight: str | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    params: dict[str, Any] = {}
    fit_kwargs: dict[str, Any] = {}
    resolved_weight = resolve_class_weight(class_weight)
    model_lower = model_name.lower()
    if model_lower in {"logreg", "ridge", "extra_trees", "lightgbm"}:
        params["class_weight"] = resolved_weight
    if model_lower == "extra_trees":
        params.update({"n_estimators": 200, "min_samples_leaf": 2, "n_jobs": -1})
    if model_lower == "hist_gb":
        fit_kwargs["sample_weight"] = sample_weights(y_train, resolved_weight)

    classifier = build_classifier(ClassifierSpec(model_name, params), random_state=random_state)
    fit_train = X_train
    fit_val = X_val
    if classifier_requires_dense(ClassifierSpec(model_name)):
        fit_train = maybe_dense(fit_train)
        fit_val = maybe_dense(fit_val)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fit_kwargs.get("sample_weight") is None:
            classifier.fit(fit_train, y_train)
        else:
            classifier.fit(fit_train, y_train, sample_weight=fit_kwargs["sample_weight"])
    pred = classifier.predict(fit_val)
    proba = _proper_action_probabilities(classifier, fit_val)
    return np.asarray(pred, dtype=int), proba


def sample_weights(y: np.ndarray, class_weight: str | dict[int, float] | None) -> np.ndarray | None:
    if class_weight is None:
        return None
    if class_weight == "balanced":
        values, counts = np.unique(y, return_counts=True)
        total = float(len(y))
        weights = {int(value): total / (len(values) * count) for value, count in zip(values, counts)}
    elif isinstance(class_weight, dict):
        weights = class_weight
    else:
        raise ValueError(f"Unsupported sample weight mode: {class_weight}")
    return np.asarray([weights.get(int(label), 1.0) for label in y], dtype=float)


def _validate_samples(samples: dict[str, Any], ranges: Any, window_size: int, horizon: int, context_size: int) -> None:
    bounds = {
        "inner_train": (ranges.inner_train_start, ranges.inner_train_end),
        "calibration": (ranges.calibration_start, ranges.calibration_end),
        "outer_val": (ranges.outer_fold.val_start, ranges.outer_fold.val_end),
    }
    for name, sample in samples.items():
        start, end = bounds[name]
        if sample.size != (end - start) - window_size - horizon + 1:
            raise ValueError(f"{name} sample count mismatch")
        if np.any(sample.target_indices - window_size + 1 < start):
            raise ValueError(f"{name} feature window crosses split boundary")
        if np.any(sample.target_indices - context_size + 1 < start):
            raise ValueError(f"{name} LM context crosses split boundary")
        if np.any(sample.target_indices + horizon >= end):
            raise ValueError(f"{name} target horizon crosses split boundary")


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["target_label"], row["feature_set"], row["model"], row["class_weight"])
        grouped.setdefault(key, []).append(row)
    result = []
    for key, items in grouped.items():
        macro = np.asarray([item["metrics"]["macro_f1"] for item in items], dtype=float)
        buy = np.asarray([item["metrics"]["buy_f1"] for item in items], dtype=float)
        sell = np.asarray([item["metrics"]["sell_f1"] for item in items], dtype=float)
        hold = np.asarray([item["metrics"]["hold_f1"] for item in items], dtype=float)
        action = np.asarray([item["metrics"]["action_rate"] for item in items], dtype=float)
        hmean = np.asarray([item["metrics"]["buy_sell_hmean_f1"] for item in items], dtype=float)
        result.append(
            {
                "target_label": key[0],
                "feature_set": key[1],
                "model": key[2],
                "class_weight": key[3],
                "folds": int(len(items)),
                "macro_f1_mean": float(macro.mean()),
                "macro_f1_std": float(macro.std(ddof=0)),
                "macro_f1_worst": float(macro.min()),
                "buy_f1_mean": float(buy.mean()),
                "sell_f1_mean": float(sell.mean()),
                "hold_f1_mean": float(hold.mean()),
                "buy_sell_hmean_f1": float(hmean.mean()),
                "action_rate_mean": float(action.mean()),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            row["macro_f1_mean"],
            row["macro_f1_worst"],
            row["buy_sell_hmean_f1"],
            -abs(row["action_rate_mean"] - 0.5),
        ),
        reverse=True,
    )


def compact_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        metrics = row["metrics"]
        compact.append(
            {
                "target_label": row["target_label"],
                "target_mode": row["target_mode"],
                "feature_set": row["feature_set"],
                "model": row["model"],
                "class_weight": row["class_weight"],
                "fold_id": row["fold_id"],
                "n_train": row["n_train"],
                "n_calibration": row["n_calibration"],
                "n_validation": row["n_validation"],
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "buy_f1": metrics["buy_f1"],
                "sell_f1": metrics["sell_f1"],
                "hold_f1": metrics["hold_f1"],
                "buy_sell_hmean_f1": metrics["buy_sell_hmean_f1"],
                "action_rate": metrics["action_rate"],
                "hold_rate": metrics["hold_rate"],
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


def print_summary(aggregates: list[dict[str, Any]]) -> None:
    print("target | features | model | weight | macro-F1 | worst | BUY F1 | SELL F1 | HOLD F1 | action_rate")
    print("--- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---:")
    for row in aggregates[:16]:
        print(
            f"{row['target_label']} | {row['feature_set']} | {row['model']} | {row['class_weight']} | "
            f"{row['macro_f1_mean']:.4f} | {row['macro_f1_worst']:.4f} | {row['buy_f1_mean']:.4f} | "
            f"{row['sell_f1_mean']:.4f} | {row['hold_f1_mean']:.4f} | {row['action_rate_mean']:.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--timeframe", default="1H")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data", default="")
    parser.add_argument("--target-modes", default="return_threshold,volatility_adjusted_return")
    parser.add_argument("--action-horizons", default="1")
    parser.add_argument("--vol-windows", default="16")
    parser.add_argument("--vol-ks", default="1.0")
    parser.add_argument("--barrier-horizons", default="3")
    parser.add_argument("--barrier-vol-windows", default="16")
    parser.add_argument("--barrier-up-ks", default="1.0")
    parser.add_argument("--barrier-down-ks", default="1.0")
    parser.add_argument("--buy-threshold-mults", default="1.5")
    parser.add_argument("--sell-threshold-mults", default="1.5")
    parser.add_argument("--feature-sets", default="lm_regime,continuous_regime,lm_regime_continuous")
    parser.add_argument("--models", default="logreg,hist_gb")
    parser.add_argument("--vocab-config", default="shape:gmm:20")
    parser.add_argument("--class-weights", default="balanced,action_boost_1.2")
    parser.add_argument("--context-size", type=int, default=16)
    parser.add_argument("--action-window-size", type=int, default=32)
    parser.add_argument("--forecast-horizon", type=int, default=3)
    parser.add_argument("--lm-order", type=int, default=2)
    parser.add_argument("--lm-alpha", type=float, default=0.1)
    parser.add_argument("--fold-mode", choices=["expanding", "rolling"], default="rolling")
    parser.add_argument("--initial-train-size", type=int, default=12000)
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=3000)
    parser.add_argument("--step-size", type=int, default=3000)
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--calibration-size", type=int, default=2500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-test", action="store_true", help="Kept for explicit research-only CLI calls; test is never used.")
    parser.add_argument("--output-json", default="data/reports/sber_h1_target_feature_research_20260515.json")
    parser.add_argument("--output-csv", default="data/reports/sber_h1_target_feature_research_20260515.csv")
    args = parser.parse_args()

    if args.quick:
        args.max_folds = min(args.max_folds, 2)
        args.vol_windows = "16"
        args.vol_ks = "1.0"
        args.models = ",".join(parse_list(args.models)[:2])

    started = time.perf_counter()
    df, data_path = load_sber_frame(args)
    folds = build_folds(args, len(df))
    vocab_config = parse_vocab_configs(args.vocab_config)[0]
    target_specs = build_target_specs(args)
    feature_sets = parse_list(args.feature_sets)
    models = parse_list(args.models)
    class_weights = parse_class_weights(args.class_weights)
    shape_matrix = candle_shape_matrix(df, variant=vocab_config.shape_variant)[0]
    continuous_matrix, continuous_names = make_continuous_past_features(df)

    rows: list[dict[str, Any]] = []
    print(f"Загружено свечей: {len(df)}; файл: {data_path}")
    print(f"Folds: {len(folds)}; test не используется; target specs: {len(target_specs)}")
    for fold in folds:
        ranges = nested_range(fold, args.calibration_size)
        print(
            f"Fold {fold.fold_id}: inner=[{ranges.inner_train_start}:{ranges.inner_train_end}) "
            f"calib=[{ranges.calibration_start}:{ranges.calibration_end}) val=[{fold.val_start}:{fold.val_end})"
        )
        for target_spec in target_specs:
            print(f"  Target {target_spec.label}")
            rows.extend(
                run_fold_target(
                    df,
                    shape_matrix,
                    continuous_matrix,
                    target_spec,
                    ranges,
                    vocab_config,
                    feature_sets=feature_sets,
                    models=models,
                    class_weights=class_weights,
                    context_size=args.context_size,
                    action_window_size=args.action_window_size,
                    lm_order=args.lm_order,
                    lm_alpha=args.lm_alpha,
                    lm_forecast_horizon=args.forecast_horizon,
                    random_state=args.random_state,
                )
            )

    aggregates = aggregate_rows(rows)
    best = aggregates[0] if aggregates else None
    payload = {
        "purpose": "validation-only target and past-feature research; test split is not used",
        "data_path": str(data_path),
        "rows": int(len(df)),
        "fold_mode": args.fold_mode,
        "folds": [asdict(fold) for fold in folds],
        "vocab_config": asdict(vocab_config),
        "target_specs": [asdict(spec) for spec in target_specs],
        "feature_sets": feature_sets,
        "models": models,
        "class_weights": ["none" if item is None else item for item in class_weights],
        "continuous_feature_names": continuous_names,
        "fold_results": rows,
        "aggregates": aggregates,
        "best_validation_only": best,
        "baseline_note": "current validation baseline is shape/gmm_diag/20 + lm_regime + logreg + action_boost_1.2 + argmax, macro-F1 about 0.4238-0.4265; test 0.4055 is not used here",
        "duration_sec": float(time.perf_counter() - started),
    }
    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    write_json(payload, output_json)
    write_csv(compact_csv_rows(rows), output_csv)
    print_summary(aggregates)
    if best:
        print(
            "Лучший validation-only config: "
            f"{best['target_label']} | {best['feature_set']} | {best['model']} | {best['class_weight']} | "
            f"macro-F1={best['macro_f1_mean']:.4f}"
        )
    print(f"JSON: {output_json}")
    print(f"CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
