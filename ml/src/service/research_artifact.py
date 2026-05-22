"""Research artifact loading and contract inference for the ML block.

This module intentionally handles research artifacts only. It does not mark
models as production-ready and does not change the legacy FastAPI predictor.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.nlp import make_continuous_past_features

from .contracts import CandleBatch, ModelContractMetadata, build_ml_prediction_response, dataframe_as_of


REQUIRED_ARTIFACT_FILES = (
    "model.pkl",
    "feature_config.json",
    "target_config.json",
    "metadata.json",
    "label_mapping.json",
    "schema_version.json",
)


@dataclass(frozen=True)
class ResearchArtifact:
    """Loaded research artifact bundle."""

    artifact_dir: Path
    model: Any
    metadata: dict[str, Any]
    feature_config: dict[str, Any]
    target_config: dict[str, Any]
    label_mapping: dict[str, Any]
    schema_version: dict[str, Any]
    feature_columns: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    training_summary: dict[str, Any]


@dataclass(frozen=True)
class ArtifactPrediction:
    """Prediction values ready for ml_prediction response building."""

    probabilities: dict[str, float]
    confidence: float
    diagnostics: dict[str, Any]


def artifact_bundle_available(path: str | Path) -> bool:
    """Return whether a path looks like a complete research artifact bundle."""

    artifact_path = Path(path)
    return artifact_path.exists() and all((artifact_path / name).exists() for name in REQUIRED_ARTIFACT_FILES)


def load_research_artifact(path: str | Path) -> ResearchArtifact:
    """Load a research artifact bundle from disk."""

    artifact_dir = Path(path)
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (artifact_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Research artifact is incomplete, missing files: {missing}")

    with (artifact_dir / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)

    metadata = _read_json(artifact_dir / "metadata.json")
    feature_config = _read_json(artifact_dir / "feature_config.json")
    target_config = _read_json(artifact_dir / "target_config.json")
    label_mapping = _read_json(artifact_dir / "label_mapping.json")
    schema_version = _read_json(artifact_dir / "schema_version.json")
    training_summary_path = artifact_dir / "training_summary.json"
    training_summary = _read_json(training_summary_path) if training_summary_path.exists() else {}

    feature_columns = list(feature_config.get("feature_columns") or _read_json(artifact_dir / "feature_columns.json"))
    feature_mean = np.asarray(feature_config.get("standardization_mean"), dtype=float)
    feature_std = np.asarray(feature_config.get("standardization_std"), dtype=float)
    if not feature_columns:
        raise ValueError("Research artifact has no feature columns")
    if len(feature_columns) != len(feature_mean) or len(feature_columns) != len(feature_std):
        raise ValueError("Feature columns and standardization vectors have different lengths")
    if not np.all(np.isfinite(feature_mean)) or not np.all(np.isfinite(feature_std)):
        raise ValueError("Research artifact standardization vectors contain non-finite values")
    feature_std = np.where(feature_std < 1e-12, 1.0, feature_std)

    return ResearchArtifact(
        artifact_dir=artifact_dir,
        model=model,
        metadata=metadata,
        feature_config=feature_config,
        target_config=target_config,
        label_mapping=label_mapping,
        schema_version=schema_version,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
        training_summary=training_summary,
    )


def predict_with_artifact(artifact: ResearchArtifact, candle_batch_df: pd.DataFrame) -> ArtifactPrediction:
    """Predict contract probabilities for the latest candle in a batch."""

    min_candles = int(artifact.metadata.get("min_candles_for_prediction", 1))
    if len(candle_batch_df) < min_candles:
        return _safe_hold_prediction(
            artifact,
            error="insufficient_history",
            message=f"Need at least {min_candles} candles, got {len(candle_batch_df)}.",
            n_candles=len(candle_batch_df),
        )

    feature_matrix, feature_names = make_continuous_past_features(candle_batch_df)
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    missing_columns = [name for name in artifact.feature_columns if name not in name_to_idx]
    if missing_columns:
        raise ValueError(f"Input feature builder did not produce artifact columns: {missing_columns}")

    indices = [name_to_idx[name] for name in artifact.feature_columns]
    row = feature_matrix[-1:, indices]
    row = (row - artifact.feature_mean) / artifact.feature_std
    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(row)):
        raise ValueError("Artifact inference features contain non-finite values")

    if not hasattr(artifact.model, "predict_proba"):
        raise ValueError("Research artifact model does not support predict_proba")
    raw_proba = np.asarray(artifact.model.predict_proba(row)[0], dtype=float)
    probabilities = _map_model_probabilities(artifact, raw_proba)
    confidence = float(max(probabilities.values()))
    diagnostics = _base_diagnostics(artifact, n_candles=len(candle_batch_df))
    diagnostics.update(
        {
            "artifact_missing": False,
            "prediction_mode": "research_artifact",
            "probabilities_calibrated": bool(artifact.metadata.get("probabilities_calibrated", False)),
            "feature_columns_count": int(len(artifact.feature_columns)),
        }
    )
    return ArtifactPrediction(probabilities=probabilities, confidence=confidence, diagnostics=diagnostics)


def build_artifact_prediction_response(
    *,
    batch: CandleBatch,
    df: pd.DataFrame,
    artifact: ResearchArtifact,
) -> dict[str, Any]:
    """Build an ml_prediction response using a loaded research artifact."""

    prediction = predict_with_artifact(artifact, df)
    metadata = model_contract_metadata(artifact)
    return build_ml_prediction_response(
        ticker=batch.ticker,
        timeframe=batch.timeframe,
        as_of=dataframe_as_of(df),
        probabilities=prediction.probabilities,
        confidence=prediction.confidence,
        expected_return=None,
        metadata=metadata,
        diagnostics=prediction.diagnostics,
    )


def model_contract_metadata(artifact: ResearchArtifact) -> ModelContractMetadata:
    """Convert artifact metadata to the public ml_prediction metadata fields."""

    metadata = artifact.metadata
    return ModelContractMetadata(
        model_version=str(metadata.get("model_version", metadata.get("artifact_id", "research-artifact"))),
        model_family=str(metadata.get("model_family", "triple_barrier_extra_trees")),
        target=str(metadata.get("target", artifact.target_config.get("target_label", ""))),
        feature_set=str(metadata.get("feature_set", artifact.feature_config.get("feature_set", ""))),
        validation_macro_f1=float(metadata.get("validation_macro_f1_mean", metadata.get("validation_macro_f1", 0.0))),
        is_production=bool(metadata.get("is_production", False)),
        class_weight=metadata.get("class_weight"),
    )


def _map_model_probabilities(artifact: ResearchArtifact, raw_proba: np.ndarray) -> dict[str, float]:
    internal_to_contract = artifact.label_mapping.get("internal_to_contract", {})
    class_values = getattr(artifact.model, "classes_", np.asarray([0, 1, 2], dtype=int))
    probabilities = {"buy": 0.0, "hold": 0.0, "sell": 0.0}
    for class_value, probability in zip(class_values, raw_proba):
        label = _internal_label_name(class_value)
        contract_key = internal_to_contract.get(label)
        if contract_key in probabilities:
            probabilities[contract_key] += float(probability)
    total = float(sum(probabilities.values()))
    if not np.isfinite(total) or total <= 0.0:
        return {"buy": 0.0, "hold": 1.0, "sell": 0.0}
    return {key: float(max(0.0, value) / total) for key, value in probabilities.items()}


def _internal_label_name(value: Any) -> str:
    try:
        label_id = int(value)
    except (TypeError, ValueError):
        return str(value).upper()
    return {0: "SELL", 1: "HOLD", 2: "BUY"}.get(label_id, str(label_id))


def _safe_hold_prediction(
    artifact: ResearchArtifact,
    *,
    error: str,
    message: str,
    n_candles: int,
) -> ArtifactPrediction:
    diagnostics = _base_diagnostics(artifact, n_candles=n_candles)
    diagnostics.update(
        {
            "artifact_missing": False,
            "error": error,
            "message": message,
            "prediction_mode": "safe_hold",
        }
    )
    return ArtifactPrediction(
        probabilities={"buy": 0.0, "hold": 1.0, "sell": 0.0},
        confidence=0.0,
        diagnostics=diagnostics,
    )


def _base_diagnostics(artifact: ResearchArtifact, *, n_candles: int) -> dict[str, Any]:
    metadata = artifact.metadata
    return {
        "artifact_id": metadata.get("artifact_id"),
        "artifact_type": metadata.get("artifact_type", "research"),
        "artifact_path": str(artifact.artifact_dir),
        "model_family": metadata.get("model_family"),
        "target": metadata.get("target"),
        "feature_set": metadata.get("feature_set"),
        "is_production": bool(metadata.get("is_production", False)),
        "n_candles": int(n_candles),
        "recommended_min_candles": int(metadata.get("recommended_min_candles", 32)),
    }


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
