"""Research CLI for alternative action targets and past-only feature baselines."""

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
    triple_barrier_details,
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


@dataclass(frozen=True)
class ModelConfig:
    """Resolved model config for compact CLI grids."""

    name: str
    label: str
    params: dict[str, Any]


def build_target_specs(args: argparse.Namespace) -> list[ActionTargetSpec]:
    specs: list[ActionTargetSpec] = []
    action_horizons = parse_int_list(args.action_horizons)
    return_threshold_mults = parse_float_list(args.return_threshold_mults)
    vol_windows = parse_int_list(args.vol_windows)
    vol_ks = parse_float_list(args.vol_ks)
    barrier_horizons = parse_int_list(args.barrier_horizons)
    barrier_vol_windows = parse_int_list(args.barrier_vol_windows)
    barrier_up_ks = parse_float_list(args.barrier_up_k_values or args.barrier_up_ks)
    barrier_down_ks = parse_float_list(args.barrier_down_k_values or args.barrier_down_ks)
    neutral_buy = parse_float_list(args.buy_threshold_mults)
    neutral_sell = parse_float_list(args.sell_threshold_mults)
    for mode in parse_list(args.target_modes):
        if mode == "return_threshold":
            for horizon in action_horizons:
                for threshold_mult in return_threshold_mults:
                    specs.append(ActionTargetSpec(mode=mode, horizon=horizon, return_threshold_mult=threshold_mult))
        elif mode == "volatility_adjusted_return":
            for horizon in action_horizons:
                for window in vol_windows:
                    for vol_k in vol_ks:
                        specs.append(ActionTargetSpec(mode=mode, horizon=horizon, vol_window=window, vol_k=vol_k))
        elif mode == "triple_barrier":
            if args.barrier_k_values:
                barrier_up_ks = parse_float_list(args.barrier_k_values)
                barrier_down_ks = parse_float_list(args.barrier_k_values)
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
    return any(name.startswith("lm_regime") and not name.endswith("_no_lm") for name in feature_sets)


def build_model_configs(args: argparse.Namespace) -> list[ModelConfig]:
    configs: list[ModelConfig] = []
    for model in parse_list(args.models):
        model_lower = model.lower()
        if model_lower == "logreg":
            penalties = parse_list(args.logreg_penalties)
            solvers = parse_list(args.logreg_solvers)
            for c_value in parse_float_list(args.logreg_c_values):
                for penalty in penalties:
                    for solver in solvers:
                        if not _valid_logreg_combo(penalty, solver):
                            continue
                        label = f"logreg:C={c_value:g}:penalty={penalty}:solver={solver}"
                        configs.append(
                            ModelConfig(
                                name="logreg",
                                label=label,
                                params={"C": float(c_value), "penalty": penalty, "solver": solver, "max_iter": 2000},
                            )
                        )
        elif model_lower == "hist_gb":
            for learning_rate in parse_float_list(args.hist_gb_learning_rates):
                for max_leaf_nodes in parse_int_list(args.hist_gb_max_leaf_nodes):
                    for l2 in parse_float_list(args.hist_gb_l2):
                        for max_iter in parse_int_list(args.hist_gb_max_iter):
                            configs.append(
                                ModelConfig(
                                    name="hist_gb",
                                    label=(
                                        f"hist_gb:iter={max_iter}:lr={learning_rate:g}:"
                                        f"leaf={max_leaf_nodes}:l2={l2:g}"
                                    ),
                                    params={
                                        "max_iter": int(max_iter),
                                        "learning_rate": float(learning_rate),
                                        "max_leaf_nodes": int(max_leaf_nodes),
                                        "l2_regularization": float(l2),
                                    },
                                )
                            )
        elif model_lower == "extra_trees":
            for max_depth in _parse_optional_int_list(args.extra_trees_max_depths):
                for min_leaf in parse_int_list(args.extra_trees_min_samples_leaf):
                    for max_features in parse_list(args.extra_trees_max_features):
                        label_depth = "none" if max_depth is None else str(max_depth)
                        configs.append(
                            ModelConfig(
                                name="extra_trees",
                                label=f"extra_trees:depth={label_depth}:leaf={min_leaf}:maxfeat={max_features}",
                                params={
                                    "n_estimators": int(args.extra_trees_n_estimators),
                                    "max_depth": max_depth,
                                    "min_samples_leaf": int(min_leaf),
                                    "max_features": _parse_max_features(max_features),
                                    "n_jobs": -1,
                                },
                            )
                        )
        else:
            configs.append(ModelConfig(name=model_lower, label=model_lower, params={}))
    if not configs:
        raise ValueError("No valid model configs were generated")
    return configs


def parse_random_states(args: argparse.Namespace) -> list[int]:
    return parse_int_list(args.random_states) if args.random_states else [int(args.random_state)]


def _valid_logreg_combo(penalty: str, solver: str) -> bool:
    if penalty == "l1":
        return solver in {"liblinear", "saga"}
    if penalty == "l2":
        return solver in {"lbfgs", "liblinear", "saga"}
    return False


def _parse_optional_int_list(value: str) -> list[int | None]:
    result: list[int | None] = []
    for item in parse_list(value):
        result.append(None if item.lower() in {"none", "null"} else int(item))
    return result


def _parse_max_features(value: str) -> str | float:
    return value if value in {"sqrt", "log2"} else float(value)


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
    continuous_names: list[str],
) -> tuple[Any, Any, Any]:
    base_feature_set = _base_feature_set(feature_set)
    cont_train_f, cont_calib_f, cont_val_f = _filter_continuous_features(
        feature_set, cont_train, cont_calib, cont_val, continuous_names
    )
    if base_feature_set in {"continuous_regime", "lm_regime_continuous_no_lm"}:
        return cont_train_f, cont_calib_f, cont_val_f
    if base_feature_set == "lm_regime":
        if lm_train is None or regime_train is None:
            raise ValueError("lm_regime requires LM features")
        reg_train, reg_calib, reg_val = _filter_regime_features(feature_set, regime_train, regime_calib, regime_val)
        scalar_width = 18
        return (
            np.hstack([lm_train[:, :scalar_width], reg_train]),
            np.hstack([lm_calib[:, :scalar_width], reg_calib]),
            np.hstack([lm_val[:, :scalar_width], reg_val]),
        )
    if base_feature_set == "lm_regime_continuous":
        if lm_train is None or regime_train is None:
            raise ValueError("lm_regime_continuous requires LM features")
        reg_train, reg_calib, reg_val = _filter_regime_features(feature_set, regime_train, regime_calib, regime_val)
        scalar_width = 18
        return (
            np.hstack([lm_train[:, :scalar_width], reg_train, cont_train_f]),
            np.hstack([lm_calib[:, :scalar_width], reg_calib, cont_calib_f]),
            np.hstack([lm_val[:, :scalar_width], reg_val, cont_val_f]),
        )
    raise ValueError(f"Unsupported feature set: {feature_set}")


def _base_feature_set(feature_set: str) -> str:
    if feature_set.startswith("lm_regime_continuous_no_"):
        return "lm_regime_continuous_no_lm" if feature_set.endswith("_no_lm") else "lm_regime_continuous"
    if feature_set.startswith("continuous_no_"):
        return "continuous_regime"
    return feature_set


def _filter_continuous_features(
    feature_set: str,
    train: np.ndarray,
    calib: np.ndarray,
    val: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = continuous_feature_mask(feature_set, names)
    return train[:, keep], calib[:, keep], val[:, keep]


def continuous_feature_mask(feature_set: str, names: list[str]) -> np.ndarray:
    """Return selected continuous columns for an ablation feature set."""

    remove_group = _removed_group(feature_set)
    keep = np.ones(len(names), dtype=bool)
    if remove_group is None:
        return keep
    for idx, name in enumerate(names):
        if _continuous_group(name) == remove_group:
            keep[idx] = False
    if not np.any(keep):
        raise ValueError(f"Ablation removed all continuous features: {feature_set}")
    return keep


def _removed_group(feature_set: str) -> str | None:
    for suffix, group in {
        "_no_session": "session",
        "_no_volume": "volume",
        "_no_returns": "returns",
        "_no_volatility": "volatility",
        "_no_candle_shape": "candle_shape",
    }.items():
        if feature_set.endswith(suffix):
            return group
    return None


def _continuous_group(name: str) -> str:
    if name.startswith("ret_") or name.startswith("ema_distance_"):
        return "returns"
    if name.startswith("vol_") or name.startswith("range_mean_"):
        return "volatility"
    if "volume" in name:
        return "volume"
    if name.startswith(("hour_", "dow_", "large_time_gap")):
        return "session"
    if name in {"body_signed", "body_abs", "range_to_open", "upper_shadow", "lower_shadow", "close_position_in_candle"}:
        return "candle_shape"
    return "other"


def _filter_regime_features(
    feature_set: str,
    train: np.ndarray,
    calib: np.ndarray,
    val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    remove_group = _removed_group(feature_set)
    if remove_group is None:
        return train, calib, val
    keep = np.ones(train.shape[1], dtype=bool)
    if remove_group == "volatility":
        keep[0:3] = False
    elif remove_group == "returns":
        keep[3:6] = False
    elif remove_group == "session":
        keep[6:9] = False
    if not np.any(keep):
        raise ValueError(f"Ablation removed all regime features: {feature_set}")
    return train[:, keep], calib[:, keep], val[:, keep]


def run_fold_target(
    df: pd.DataFrame,
    shape_matrix: np.ndarray,
    continuous_matrix: np.ndarray,
    continuous_names: list[str],
    target_spec: ActionTargetSpec,
    ranges: Any,
    vocab_config: Any,
    *,
    feature_sets: list[str],
    models: list[ModelConfig],
    class_weights: list[str | None],
    context_size: int,
    action_window_size: int,
    lm_order: int,
    lm_alpha: float,
    lm_forecast_horizon: int,
    random_state: int,
    include_target_audit: bool = False,
    include_economic_sanity: bool = False,
    dump_target_audit_samples: int = 0,
) -> list[dict[str, Any]]:
    target = make_research_action_targets(df, target_spec)
    target_detail = triple_barrier_details(df, target_spec) if target_spec.mode == "triple_barrier" else None
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
            continuous_names=continuous_names,
        )
        for model_config in models:
            for class_weight in class_weights:
                pred, proba, model_diagnostics = fit_predict_model(
                    X_train,
                    samples["inner_train"].y,
                    X_val,
                    model_config=model_config,
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
                        "model": model_config.label,
                        "model_name": model_config.name,
                        "model_params": model_config.params,
                        "class_weight": "none" if class_weight is None else str(class_weight),
                        "fold_id": int(ranges.outer_fold.fold_id),
                        "random_state": int(random_state),
                        "n_train": int(samples["inner_train"].size),
                        "n_calibration": int(samples["calibration"].size),
                        "n_validation": int(samples["outer_val"].size),
                        "target_distribution_train": label_distribution(samples["inner_train"].y),
                        "target_distribution_calibration": label_distribution(samples["calibration"].y),
                        "target_distribution_val": label_distribution(samples["outer_val"].y),
                        "prediction_distribution": label_distribution(pred),
                        "metrics": metrics,
                        "has_predict_proba": proba is not None,
                        "model_diagnostics": model_diagnostics,
                        "target_analysis": target_analysis(target.labels, target.future_returns, target.metadata),
                        "target_audit": (
                            target_audit_for_indices(target_detail, samples, df, max_samples=dump_target_audit_samples)
                            if include_target_audit and target_detail is not None
                            else {}
                        ),
                        "economic_sanity": (
                            economic_sanity(samples["outer_val"].y, pred, samples["outer_val"].target_indices, target.future_returns, target_detail)
                            if include_economic_sanity
                            else {}
                        ),
                    }
                )
    return rows


def target_audit_for_indices(
    detail: dict[str, Any],
    samples: dict[str, Any],
    df: pd.DataFrame,
    *,
    max_samples: int = 0,
) -> dict[str, Any]:
    result = {
        "train": target_audit_summary(detail, samples["inner_train"].target_indices, samples["inner_train"].y),
        "calibration": target_audit_summary(detail, samples["calibration"].target_indices, samples["calibration"].y),
        "validation": target_audit_summary(detail, samples["outer_val"].target_indices, samples["outer_val"].y),
    }
    if max_samples > 0:
        result["sample_dump"] = target_audit_sample_dump(detail, df, samples["outer_val"].target_indices[:max_samples])
    return result


def target_audit_summary(detail: dict[str, Any], indices: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    idx = np.asarray(indices, dtype=int)
    labels = np.asarray(labels, dtype=int)
    outcomes = np.asarray(detail["outcome"], dtype=object)[idx]
    time_to = np.asarray(detail["time_to_barrier"], dtype=float)[idx]
    future_return = np.asarray(detail["future_return"], dtype=float)[idx]
    mfe = np.asarray(detail["mfe"], dtype=float)[idx]
    mae = np.asarray(detail["mae"], dtype=float)[idx]
    valid_time = np.isfinite(time_to)
    result: dict[str, Any] = {
        "target_distribution": label_distribution(labels),
        "share_upper_first": float(np.mean(outcomes == "upper_first")) if len(outcomes) else 0.0,
        "share_lower_first": float(np.mean(outcomes == "lower_first")) if len(outcomes) else 0.0,
        "share_vertical_timeout": float(np.mean(outcomes == "vertical_timeout")) if len(outcomes) else 0.0,
        "share_ambiguous": float(np.mean(outcomes == "ambiguous")) if len(outcomes) else 0.0,
        "mean_time_to_barrier": float(np.nanmean(time_to[valid_time])) if np.any(valid_time) else 0.0,
        "median_time_to_barrier": float(np.nanmedian(time_to[valid_time])) if np.any(valid_time) else 0.0,
    }
    per_label: dict[str, Any] = {}
    for label_id, label_name in {0: "SELL", 1: "HOLD", 2: "BUY"}.items():
        mask = labels == label_id
        per_label[label_name] = {
            "count": int(np.count_nonzero(mask)),
            "mean_time_to_barrier": _safe_mean(time_to[mask]),
            "median_time_to_barrier": _safe_median(time_to[mask]),
            "mean_future_return": _safe_mean(future_return[mask]),
            "median_future_return": _safe_median(future_return[mask]),
            "mean_mfe": _safe_mean(mfe[mask]),
            "mean_mae": _safe_mean(mae[mask]),
        }
    result["by_label"] = per_label
    return result


def target_audit_sample_dump(detail: dict[str, Any], df: pd.DataFrame, indices: np.ndarray) -> list[dict[str, Any]]:
    begin = pd.to_datetime(df["begin"]) if "begin" in df.columns else pd.Series(np.arange(len(df)))
    rows = []
    horizon = 0
    for idx in np.asarray(indices, dtype=int):
        time_to = detail["time_to_barrier"][idx]
        horizon = max(horizon, int(time_to) if np.isfinite(time_to) else 0)
    horizon = max(horizon, 1)
    for idx in np.asarray(indices, dtype=int):
        future_slice = slice(idx + 1, min(len(df), idx + horizon + 1))
        rows.append(
            {
                "sample_idx": int(idx),
                "decision_time": str(begin.iloc[idx]),
                "close_t": float(detail["close"][idx]),
                "past_vol_t": float(detail["past_volatility"][idx]),
                "upper_barrier": float(detail["upper_barrier"][idx]),
                "lower_barrier": float(detail["lower_barrier"][idx]),
                "future_highs": [float(x) for x in df["high"].astype(float).to_numpy()[future_slice]],
                "future_lows": [float(x) for x in df["low"].astype(float).to_numpy()[future_slice]],
                "future_timestamps": [str(x) for x in begin.iloc[future_slice].tolist()],
                "label": int(detail["labels"][idx]),
                "outcome": str(detail["outcome"][idx]),
                "time_to_barrier": float(detail["time_to_barrier"][idx]) if np.isfinite(detail["time_to_barrier"][idx]) else None,
                "max_feature_time": str(begin.iloc[idx]),
            }
        )
    return rows


def economic_sanity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_indices: np.ndarray,
    future_returns: np.ndarray,
    target_detail: dict[str, Any] | None,
) -> dict[str, Any]:
    idx = np.asarray(target_indices, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    returns = np.asarray(future_returns, dtype=float)[idx]
    result: dict[str, Any] = {
        "mean_realized_return_by_prediction": {},
        "median_realized_return_by_prediction": {},
    }
    for label_id, label_name in {0: "SELL", 1: "HOLD", 2: "BUY"}.items():
        mask = pred == label_id
        result["mean_realized_return_by_prediction"][label_name] = _safe_mean(returns[mask])
        result["median_realized_return_by_prediction"][label_name] = _safe_median(returns[mask])
        result[f"{label_name.lower()}_prediction"] = {
            "count": int(np.count_nonzero(mask)),
            "mean_realized_return": _safe_mean(returns[mask]),
            "median_realized_return": _safe_median(returns[mask]),
        }
    buy_mask = pred == 2
    sell_mask = pred == 0
    result["directional_hit_rate_for_BUY"] = float(np.mean(returns[buy_mask] > 0.0)) if np.any(buy_mask) else 0.0
    result["directional_hit_rate_for_SELL"] = float(np.mean(returns[sell_mask] < 0.0)) if np.any(sell_mask) else 0.0
    result["hold_mean_abs_future_return"] = _safe_mean(np.abs(returns[pred == 1]))
    if target_detail is not None:
        outcomes = np.asarray(target_detail["outcome"], dtype=object)[idx]
        result["predicted_BUY_upper_hit_rate"] = float(np.mean(outcomes[buy_mask] == "upper_first")) if np.any(buy_mask) else 0.0
        result["predicted_SELL_lower_hit_rate"] = float(np.mean(outcomes[sell_mask] == "lower_first")) if np.any(sell_mask) else 0.0
        action_mask = np.isin(pred, [0, 2])
        result["predicted_action_barrier_hit_rate"] = float(
            np.mean(np.isin(outcomes[action_mask], ["upper_first", "lower_first"]))
        ) if np.any(action_mask) else 0.0
    return result


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else 0.0


def _safe_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else 0.0


def fit_predict_model(
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    *,
    model_config: ModelConfig,
    class_weight: str | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    params: dict[str, Any] = dict(model_config.params)
    fit_kwargs: dict[str, Any] = {}
    resolved_weight = resolve_class_weight(class_weight)
    model_lower = model_config.name.lower()
    if model_lower in {"logreg", "ridge", "extra_trees", "lightgbm"}:
        params["class_weight"] = resolved_weight
    if model_lower == "hist_gb":
        fit_kwargs["sample_weight"] = sample_weights(y_train, resolved_weight)

    classifier = build_classifier(ClassifierSpec(model_config.name, params), random_state=random_state)
    fit_train = X_train
    fit_val = X_val
    if classifier_requires_dense(ClassifierSpec(model_config.name)):
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
    diagnostics: dict[str, Any] = {}
    coef = getattr(classifier, "coef_", None)
    if coef is not None:
        coef_arr = np.asarray(coef, dtype=float)
        diagnostics["coefficient_sparsity"] = float(np.mean(np.abs(coef_arr) < 1e-12))
        diagnostics["n_coefficients"] = int(coef_arr.size)
    return np.asarray(pred, dtype=int), proba, diagnostics


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
        accuracy = np.asarray([item["metrics"]["accuracy"] for item in items], dtype=float)
        balanced = np.asarray([item["metrics"]["balanced_accuracy"] for item in items], dtype=float)
        buy = np.asarray([item["metrics"]["buy_f1"] for item in items], dtype=float)
        sell = np.asarray([item["metrics"]["sell_f1"] for item in items], dtype=float)
        hold = np.asarray([item["metrics"]["hold_f1"] for item in items], dtype=float)
        action = np.asarray([item["metrics"]["action_rate"] for item in items], dtype=float)
        hold_rate = np.asarray([item["metrics"]["hold_rate"] for item in items], dtype=float)
        hmean = np.asarray([item["metrics"]["buy_sell_hmean_f1"] for item in items], dtype=float)
        pred_buy = np.asarray([_label_share(item["prediction_distribution"], "BUY") for item in items], dtype=float)
        pred_sell = np.asarray([_label_share(item["prediction_distribution"], "SELL") for item in items], dtype=float)
        pred_hold = np.asarray([_label_share(item["prediction_distribution"], "HOLD") for item in items], dtype=float)
        seed_means = _group_metric_mean(items, "random_state", "macro_f1")
        fold_means = _group_metric_mean(items, "fold_id", "macro_f1")
        result.append(
            {
                "target_label": key[0],
                "target_mode": items[0]["target_mode"],
                "feature_set": key[1],
                "model": key[2],
                "class_weight": key[3],
                "n_folds": int(len({item["fold_id"] for item in items})),
                "n_rows": int(len(items)),
                "random_states": sorted({int(item["random_state"]) for item in items}),
                "mean_macro_f1": float(macro.mean()),
                "std_macro_f1": float(macro.std(ddof=0)),
                "worst_macro_f1": float(macro.min()),
                "std_across_folds": float(np.asarray(list(fold_means.values()), dtype=float).std(ddof=0)) if fold_means else 0.0,
                "std_across_seeds": float(np.asarray(list(seed_means.values()), dtype=float).std(ddof=0)) if seed_means else 0.0,
                "mean_accuracy": float(accuracy.mean()),
                "mean_balanced_accuracy": float(balanced.mean()),
                "mean_buy_f1": float(buy.mean()),
                "mean_sell_f1": float(sell.mean()),
                "mean_hold_f1": float(hold.mean()),
                "buy_sell_hmean": float(hmean.mean()),
                "mean_action_rate": float(action.mean()),
                "std_action_rate": float(action.std(ddof=0)),
                "mean_hold_rate": float(hold_rate.mean()),
                "mean_prediction_buy_share": float(pred_buy.mean()),
                "mean_prediction_sell_share": float(pred_sell.mean()),
                "mean_prediction_hold_share": float(pred_hold.mean()),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            row["mean_macro_f1"],
            row["worst_macro_f1"],
            row["buy_sell_hmean"],
            -abs(row["mean_action_rate"] - 0.5),
        ),
        reverse=True,
    )


def _group_metric_mean(items: list[dict[str, Any]], group_key: str, metric_name: str) -> dict[Any, float]:
    grouped: dict[Any, list[float]] = {}
    for item in items:
        grouped.setdefault(item[group_key], []).append(float(item["metrics"][metric_name]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _label_share(distribution: dict[str, Any], label: str) -> float:
    return float(distribution.get(label, {}).get("share", 0.0))


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
                "random_state": row["random_state"],
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
    print("AGGREGATE RESULTS, not fold-level rows")
    print("target | features | model | weight | macro-F1 | worst | BUY F1 | SELL F1 | HOLD F1 | action_rate")
    print("--- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---:")
    for row in aggregates[:16]:
        print(
            f"{row['target_label']} | {row['feature_set']} | {row['model']} | {row['class_weight']} | "
            f"{row['mean_macro_f1']:.4f} | {row['worst_macro_f1']:.4f} | {row['mean_buy_f1']:.4f} | "
            f"{row['mean_sell_f1']:.4f} | {row['mean_hold_f1']:.4f} | {row['mean_action_rate']:.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--timeframe", default="1H")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data", default="")
    parser.add_argument("--target-modes", default="return_threshold,volatility_adjusted_return")
    parser.add_argument("--action-horizons", default="1")
    parser.add_argument("--return-threshold-mults", default="1.0")
    parser.add_argument("--vol-windows", default="16")
    parser.add_argument("--vol-ks", default="1.0")
    parser.add_argument("--barrier-horizons", default="3")
    parser.add_argument("--barrier-vol-windows", default="16")
    parser.add_argument("--barrier-up-ks", default="1.0")
    parser.add_argument("--barrier-down-ks", default="1.0")
    parser.add_argument("--barrier-up-k-values", default="")
    parser.add_argument("--barrier-down-k-values", default="")
    parser.add_argument("--barrier-k-values", default="")
    parser.add_argument("--buy-threshold-mults", default="1.5")
    parser.add_argument("--sell-threshold-mults", default="1.5")
    parser.add_argument("--feature-sets", default="lm_regime,continuous_regime,lm_regime_continuous")
    parser.add_argument("--models", default="logreg,hist_gb")
    parser.add_argument("--logreg-c-values", default="1.0")
    parser.add_argument("--logreg-penalties", default="l2")
    parser.add_argument("--logreg-solvers", default="lbfgs")
    parser.add_argument("--hist-gb-max-iter", default="200")
    parser.add_argument("--hist-gb-learning-rates", default="0.05")
    parser.add_argument("--hist-gb-max-leaf-nodes", default="31")
    parser.add_argument("--hist-gb-l2", default="0.0")
    parser.add_argument("--extra-trees-n-estimators", type=int, default=300)
    parser.add_argument("--extra-trees-max-depths", default="none")
    parser.add_argument("--extra-trees-min-samples-leaf", default="5")
    parser.add_argument("--extra-trees-max-features", default="sqrt")
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
    parser.add_argument("--random-states", default="")
    parser.add_argument("--include-target-audit", action="store_true")
    parser.add_argument("--include-economic-sanity", action="store_true")
    parser.add_argument("--dump-target-audit-samples", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-test", action="store_true", help="Kept for explicit research-only CLI calls; test is never used.")
    parser.add_argument("--output-json", default="data/reports/sber_h1_target_feature_research_20260515.json")
    parser.add_argument("--output-csv", default="data/reports/sber_h1_target_feature_research_20260515.csv")
    parser.add_argument("--output-aggregate-csv", default="")
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
    models = build_model_configs(args)
    class_weights = parse_class_weights(args.class_weights)
    random_states = parse_random_states(args)
    shape_matrix = candle_shape_matrix(df, variant=vocab_config.shape_variant)[0]
    continuous_matrix, continuous_names = make_continuous_past_features(df)

    rows: list[dict[str, Any]] = []
    print(f"Загружено свечей: {len(df)}; файл: {data_path}")
    print(f"Folds: {len(folds)}; test не используется; target specs: {len(target_specs)}; random_states: {random_states}")
    for random_state in random_states:
        print(f"Random state {random_state}")
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
                        continuous_names,
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
                        random_state=random_state,
                        include_target_audit=args.include_target_audit,
                        include_economic_sanity=args.include_economic_sanity,
                        dump_target_audit_samples=args.dump_target_audit_samples,
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
        "models": [asdict(model) for model in models],
        "class_weights": ["none" if item is None else item for item in class_weights],
        "random_states": random_states,
        "continuous_feature_names": continuous_names,
        "fold_results": rows,
        "aggregates": aggregates,
        "best_validation_only": best,
        "baseline_note": "current validation baseline is shape/gmm_diag/20 + lm_regime + logreg + action_boost_1.2 + argmax, macro-F1 about 0.4238-0.4265; test 0.4055 is not used here",
        "duration_sec": float(time.perf_counter() - started),
    }
    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    output_aggregate_csv = (
        REPO_ROOT / args.output_aggregate_csv
        if args.output_aggregate_csv
        else output_csv.with_name(f"{output_csv.stem}.aggregate{output_csv.suffix}")
    )
    write_json(payload, output_json)
    write_csv(compact_csv_rows(rows), output_csv)
    write_csv(aggregates, output_aggregate_csv)
    print_summary(aggregates)
    if best:
        print(
            "Лучший validation-only aggregate config: "
            f"{best['target_label']} | {best['feature_set']} | {best['model']} | {best['class_weight']} | "
            f"mean macro-F1={best['mean_macro_f1']:.4f}; worst={best['worst_macro_f1']:.4f}"
        )
    print(f"JSON: {output_json}")
    print(f"Fold-level CSV: {output_csv}")
    print(f"Aggregate CSV: {output_aggregate_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
