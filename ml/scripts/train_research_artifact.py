"""Train a local research artifact for the frozen SBER H1 candidate.

The artifact is intended for contract/integration testing of
``candle_batch JSON -> ml_prediction JSON``. It is not a production trading
artifact and it does not run final test evaluation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.data import clean_candles  # noqa: E402
from src.nlp import ActionTargetSpec, ClassifierSpec, make_continuous_past_features, make_research_action_targets, make_sentence_samples  # noqa: E402
from src.nlp.classifiers import build_classifier  # noqa: E402
from src.utils.io import ensure_dir  # noqa: E402


VALIDATION_METRICS = {
    "validation_macro_f1_mean": 0.4685,
    "validation_macro_f1_worst_fold": 0.4522,
    "validation_buy_f1": 0.4044,
    "validation_sell_f1": 0.4377,
    "validation_hold_f1": 0.5634,
    "validation_action_rate": 0.6708,
    "seed_robustness": {
        "seeds": [7, 13, 21, 42, 100],
        "std_across_seeds": 0.0008,
        "std_across_folds": 0.0087,
        "worst_seed_macro_f1": 0.4676,
        "best_seed_macro_f1": 0.4695,
    },
}


def find_latest_raw(raw_dir: Path, ticker: str, timeframe: str) -> Path:
    pattern = f"{ticker}_{timeframe}_*.parquet"
    matches = sorted(raw_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No raw parquet files found for pattern {raw_dir / pattern}")
    return matches[-1]


def load_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    data_path = Path(args.data).resolve() if args.data else find_latest_raw(REPO_ROOT / args.raw_dir, args.ticker, args.timeframe)
    df = pd.read_parquet(data_path)
    if "ticker" in df.columns:
        df = df[df["ticker"] == args.ticker]
    if "timeframe" in df.columns:
        df = df[df["timeframe"] == args.timeframe]
    df = clean_candles(df)
    df = df.sort_values("begin").reset_index(drop=True) if "begin" in df.columns else df.reset_index(drop=True)
    return df, data_path


def build_target_spec(args: argparse.Namespace) -> ActionTargetSpec:
    if args.target_mode != "triple_barrier":
        raise ValueError("Only triple_barrier is supported by this frozen artifact script")
    return ActionTargetSpec(
        mode="triple_barrier",
        barrier_horizon=int(args.barrier_horizon),
        barrier_vol_window=int(args.barrier_vol_window),
        barrier_up_k=float(args.barrier_up_k),
        barrier_down_k=float(args.barrier_down_k),
    )


def parse_optional_int(value: str) -> int | None:
    return None if value.lower() in {"none", "null"} else int(value)


def parse_max_features(value: str) -> str | float:
    return value if value in {"sqrt", "log2"} else float(value)


def label_distribution(y: np.ndarray) -> dict[str, Any]:
    names = {0: "SELL", 1: "HOLD", 2: "BUY"}
    values, counts = np.unique(np.asarray(y, dtype=int), return_counts=True)
    total = float(len(y))
    return {
        names.get(int(value), str(int(value))): {"count": int(count), "share": float(count / total)}
        for value, count in zip(values, counts)
    }


def train_artifact(args: argparse.Namespace) -> dict[str, Any]:
    df, data_path = load_frame(args)
    if args.training_protocol != "development_only":
        raise ValueError("This script currently supports only --training-protocol development_only")

    development_end = int(len(df) * float(args.development_ratio))
    if development_end <= args.action_window_size + args.barrier_horizon:
        raise ValueError("Development range is too small for action window and target horizon")

    target_spec = build_target_spec(args)
    target = make_research_action_targets(df, target_spec)
    dummy_tokens = ["w000"] * len(df)
    samples = make_sentence_samples(
        dummy_tokens,
        target.labels,
        target.future_returns,
        0,
        development_end,
        int(args.action_window_size),
        target.effective_horizon,
    )
    if samples.size == 0:
        raise ValueError("No development samples were generated")
    if np.any(samples.target_indices + target.effective_horizon >= development_end):
        raise ValueError("Artifact target horizon crosses development boundary")

    feature_matrix, feature_names = make_continuous_past_features(df)
    if args.feature_set != "continuous_regime":
        raise ValueError("This frozen artifact script currently supports only continuous_regime")

    train_rows = feature_matrix[samples.target_indices]
    feature_mean = np.nanmean(train_rows, axis=0)
    feature_std = np.nanstd(train_rows, axis=0)
    feature_mean = np.nan_to_num(feature_mean, nan=0.0, posinf=0.0, neginf=0.0)
    feature_std = np.nan_to_num(feature_std, nan=1.0, posinf=1.0, neginf=1.0)
    feature_std = np.where(feature_std < 1e-12, 1.0, feature_std)
    X_train = (train_rows - feature_mean) / feature_std
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    class_weight = None if args.class_weight == "none" else args.class_weight
    if args.model != "extra_trees":
        raise ValueError("This frozen artifact script currently supports only extra_trees")
    model = build_classifier(
        ClassifierSpec(
            "extra_trees",
            {
                "n_estimators": int(args.n_estimators),
                "max_depth": parse_optional_int(args.max_depth),
                "min_samples_leaf": int(args.min_samples_leaf),
                "max_features": parse_max_features(args.max_features),
                "class_weight": class_weight,
                "n_jobs": -1,
            },
        ),
        random_state=int(args.random_state),
    )
    model.fit(X_train, samples.y)

    output_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    ensure_dir(output_dir)

    metadata = {
        "artifact_id": "research_triple_barrier_sber_h1_20260515",
        "model_version": "research_triple_barrier_sber_h1_20260515",
        "artifact_type": "research",
        "is_production": False,
        "ticker": args.ticker,
        "timeframe": args.timeframe,
        "model_family": "triple_barrier_extra_trees",
        "model_class": "ExtraTreesClassifier",
        "target": target_spec.label,
        "feature_set": args.feature_set,
        "class_weight": args.class_weight,
        "n_estimators": int(args.n_estimators),
        "min_samples_leaf": int(args.min_samples_leaf),
        "max_depth": args.max_depth,
        "max_features": args.max_features,
        "random_state": int(args.random_state),
        "probabilities_calibrated": False,
        "min_candles_for_prediction": 1,
        "recommended_min_candles": int(args.action_window_size),
        "training_protocol": "fit frozen research candidate on first 85% development data only; no test tuning",
        "created_at": "2026-05-15",
        "notes": "Research artifact for integration testing. Not a production trading artifact.",
        **VALIDATION_METRICS,
    }
    target_config = {
        "target_mode": "triple_barrier",
        "horizon": int(args.barrier_horizon),
        "vol_window": int(args.barrier_vol_window),
        "up_k": float(args.barrier_up_k),
        "down_k": float(args.barrier_down_k),
        "target_label": target_spec.label,
        "label_order": ["SELL", "HOLD", "BUY"],
    }
    feature_config = {
        "feature_set": args.feature_set,
        "past_only": True,
        "standardization": "fit_on_development_training_samples",
        "feature_groups": ["returns", "volatility", "candle_shape", "volume", "session", "trend"],
        "feature_columns": list(feature_names),
        "standardization_mean": [float(value) for value in feature_mean],
        "standardization_std": [float(value) for value in feature_std],
    }
    label_mapping = {
        "internal_to_contract": {"SELL": "sell", "HOLD": "hold", "BUY": "buy"},
        "contract_to_internal": {"sell": "SELL", "hold": "HOLD", "buy": "BUY"},
    }
    schema_version = {
        "artifact_schema_version": 1,
        "ml_prediction_schema": "contracts/ml_prediction.schema.json",
        "candle_batch_schema": "contracts/candle_batch.schema.json",
    }
    training_summary = {
        "data_path": str(data_path),
        "raw_rows_after_cleaning": int(len(df)),
        "development_ratio": float(args.development_ratio),
        "development_end": int(development_end),
        "untouched_tail_start": int(development_end),
        "untouched_tail_rows": int(len(df) - development_end),
        "action_window_size": int(args.action_window_size),
        "effective_target_horizon": int(target.effective_horizon),
        "n_training_samples": int(samples.size),
        "first_training_target_idx": int(samples.target_indices[0]),
        "last_training_target_idx": int(samples.target_indices[-1]),
        "training_class_distribution": label_distribution(samples.y),
        "model_classes": [int(value) for value in model.classes_],
    }

    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "target_config.json", target_config)
    write_json(output_dir / "feature_config.json", feature_config)
    write_json(output_dir / "label_mapping.json", label_mapping)
    write_json(output_dir / "schema_version.json", schema_version)
    write_json(output_dir / "feature_columns.json", list(feature_names))
    write_json(output_dir / "training_summary.json", training_summary)

    return {
        "output_dir": str(output_dir),
        "metadata": metadata,
        "training_summary": training_summary,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--timeframe", default="1H")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data", default="")
    parser.add_argument("--target-mode", default="triple_barrier")
    parser.add_argument("--barrier-horizon", type=int, default=3)
    parser.add_argument("--barrier-vol-window", type=int, default=12)
    parser.add_argument("--barrier-up-k", type=float, default=1.25)
    parser.add_argument("--barrier-down-k", type=float, default=1.25)
    parser.add_argument("--feature-set", default="continuous_regime")
    parser.add_argument("--model", default="extra_trees")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--max-depth", default="none")
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--class-weight", default="none")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--training-protocol", default="development_only")
    parser.add_argument("--development-ratio", type=float, default=0.85)
    parser.add_argument("--action-window-size", type=int, default=32)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    result = train_artifact(args)
    print("Research artifact written:")
    print(f"  output_dir: {result['output_dir']}")
    print(f"  artifact_id: {result['metadata']['artifact_id']}")
    print(f"  target: {result['metadata']['target']}")
    print(f"  feature_set: {result['metadata']['feature_set']}")
    print(f"  n_training_samples: {result['training_summary']['n_training_samples']}")
    print("  is_production: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
