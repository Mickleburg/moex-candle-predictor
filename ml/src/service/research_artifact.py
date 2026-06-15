"""Research artifact loading and contract inference for the ML block.

Supports two artifact types detected via metadata.json model_family:
  - "triple_barrier_extra_trees" (default): loads model.pkl (sklearn), uses
    make_continuous_past_features for a single-row feature vector.
  - "triple_barrier_lstm": loads model.pt (PyTorch state_dict) + model_config.json,
    uses build_per_step_features for a SEQ_LEN-step sequence window.
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

# LSTM dependencies loaded lazily to avoid hard torch dependency for ET-only deployments
_TORCH_AVAILABLE: bool | None = None


def _check_torch() -> bool:
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


_BASE_BARRIER_THRESHOLD = 0.001   # 2 * 0.0005 commission, matches triple_barrier_details


def _parse_triple_barrier_params(target_str: str | None) -> dict[str, float] | None:
    """Parse 'triple_barrier:h3:w12:up1.25:down1.25' -> barrier params, else None."""
    if not target_str or not str(target_str).startswith("triple_barrier"):
        return None
    params = {"horizon": 3, "vol_window": 12, "up_k": 1.25, "down_k": 1.25}
    for tok in str(target_str).split(":")[1:]:
        try:
            if tok.startswith("h"):
                params["horizon"] = int(tok[1:])
            elif tok.startswith("w"):
                params["vol_window"] = int(tok[1:])
            elif tok.startswith("up"):
                params["up_k"] = float(tok[2:])
            elif tok.startswith("down"):
                params["down_k"] = float(tok[4:])
        except (ValueError, IndexError):
            continue
    return params


def _compute_signal_context(
    artifact: "ResearchArtifact",
    candle_batch_df: pd.DataFrame,
    probabilities: dict[str, float],
) -> tuple[float | None, dict[str, Any] | None]:
    """Compute prediction-intrinsic forecast context (expected_return + signal_context).

    Informational only — downstream owns thresholds/stops/sizing. Returns (None, None)
    for non triple-barrier targets or when history is too short for volatility.
    """
    target_str = str(artifact.metadata.get("target", "")) or str(
        artifact.target_config.get("target_label", "")
    )
    params = _parse_triple_barrier_params(target_str)
    if params is None or candle_batch_df.empty:
        return None, None

    from src.nlp.targets import past_return_volatility

    try:
        vol_arr = past_return_volatility(candle_batch_df, int(params["vol_window"]))
    except Exception:
        return None, None
    vol = float(vol_arr[-1]) if len(vol_arr) else 0.0

    up_ret = max(_BASE_BARRIER_THRESHOLD, float(params["up_k"]) * vol)
    dn_ret = max(_BASE_BARRIER_THRESHOLD, float(params["down_k"]) * vol)
    close = float(candle_batch_df["close"].iloc[-1])

    p_buy = float(probabilities.get("buy", 0.0))
    p_sell = float(probabilities.get("sell", 0.0))
    expected_return = p_buy * up_ret - p_sell * dn_ret

    timeframe = ""
    if "timeframe" in candle_batch_df.columns and len(candle_batch_df):
        timeframe = str(candle_batch_df["timeframe"].iloc[-1])

    context = {
        "horizon_bars": int(params["horizon"]),
        "horizon_timeframe": timeframe,
        "upper_return": up_ret,
        "lower_return": dn_ret,
        "upper_barrier": close * (1.0 + up_ret),
        "lower_barrier": close * (1.0 - dn_ret),
        "volatility": vol,
        "reference_close": close,
        "calibrated": bool(artifact.metadata.get("probabilities_calibrated", False)),
    }
    return float(expected_return), context


REQUIRED_ARTIFACT_FILES = (
    "feature_config.json",
    "target_config.json",
    "metadata.json",
    "label_mapping.json",
    "schema_version.json",
)

# ET artifacts require model.pkl; LSTM artifacts require model.pt + model_config.json
_ET_EXTRA_FILES = ("model.pkl",)
_LSTM_EXTRA_FILES = ("model.pt", "model_config.json")


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
    expected_return: float | None = None
    signal_context: dict[str, Any] | None = None


def artifact_bundle_available(path: str | Path) -> bool:
    """Return whether a path looks like a complete research artifact bundle."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        return False
    if not all((artifact_path / name).exists() for name in REQUIRED_ARTIFACT_FILES):
        return False
    # Accept either ET (model.pkl) or LSTM (model.pt + model_config.json)
    has_et = (artifact_path / "model.pkl").exists()
    has_lstm = (artifact_path / "model.pt").exists() and (artifact_path / "model_config.json").exists()
    return has_et or has_lstm


def load_research_artifact(path: str | Path) -> ResearchArtifact:
    """Load a research artifact bundle from disk (ET or LSTM)."""

    artifact_dir = Path(path)
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (artifact_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Research artifact is incomplete, missing files: {missing}")

    # Detect artifact type and load model accordingly
    if (artifact_dir / "model.pkl").exists():
        with (artifact_dir / "model.pkl").open("rb") as handle:
            model = pickle.load(handle)
    elif (artifact_dir / "model.pt").exists():
        if not _check_torch():
            raise ImportError("PyTorch is required to load LSTM artifacts. Install with: pip install torch")
        import torch
        from src.models.lstm_model import CandleLSTM
        model_config = _read_json(artifact_dir / "model_config.json")
        lstm = CandleLSTM.from_config(model_config)
        state_dict = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
        lstm.load_state_dict(state_dict)
        lstm.eval()
        model = lstm
    else:
        raise FileNotFoundError(f"No model file found in {artifact_dir} (expected model.pkl or model.pt)")

    metadata = _read_json(artifact_dir / "metadata.json")
    feature_config = _read_json(artifact_dir / "feature_config.json")
    target_config = _read_json(artifact_dir / "target_config.json")
    label_mapping = _read_json(artifact_dir / "label_mapping.json")
    schema_version = _read_json(artifact_dir / "schema_version.json")
    training_summary_path = artifact_dir / "training_summary.json"
    training_summary = _read_json(training_summary_path) if training_summary_path.exists() else {}

    # ET artifacts use feature_columns + standardization_mean/std
    # LSTM artifacts use feature_names + normalization_mean/std
    is_lstm = feature_config.get("model_type") == "lstm"
    if is_lstm:
        feature_columns = list(feature_config.get("feature_names", []))
        feature_mean = np.asarray(feature_config.get("normalization_mean"), dtype=float)
        feature_std  = np.asarray(feature_config.get("normalization_std"),  dtype=float)
    else:
        feature_columns_raw = feature_config.get("feature_columns")
        if not feature_columns_raw and (artifact_dir / "feature_columns.json").exists():
            feature_columns_raw = _read_json(artifact_dir / "feature_columns.json")
        feature_columns = list(feature_columns_raw or [])
        feature_mean = np.asarray(feature_config.get("standardization_mean"), dtype=float)
        feature_std  = np.asarray(feature_config.get("standardization_std"),  dtype=float)

    if not feature_columns:
        raise ValueError("Research artifact has no feature columns")
    if len(feature_columns) != len(feature_mean) or len(feature_columns) != len(feature_std):
        raise ValueError("Feature columns and normalisation vectors have different lengths")
    if not np.all(np.isfinite(feature_mean)) or not np.all(np.isfinite(feature_std)):
        raise ValueError("Research artifact normalisation vectors contain non-finite values")
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

    model_family = str(artifact.metadata.get("model_family", "triple_barrier_extra_trees"))
    if model_family == "triple_barrier_lstm":
        return _predict_with_lstm(artifact, candle_batch_df)
    return _predict_with_et(artifact, candle_batch_df)


def _predict_with_et(artifact: ResearchArtifact, candle_batch_df: pd.DataFrame) -> ArtifactPrediction:
    """ExtraTrees inference: single-row flat feature vector."""
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
    diagnostics.update({
        "artifact_missing": False,
        "prediction_mode": "research_artifact",
        "probabilities_calibrated": bool(artifact.metadata.get("probabilities_calibrated", False)),
        "feature_columns_count": int(len(artifact.feature_columns)),
    })
    expected_return, signal_context = _compute_signal_context(artifact, candle_batch_df, probabilities)
    return ArtifactPrediction(
        probabilities=probabilities, confidence=confidence, diagnostics=diagnostics,
        expected_return=expected_return, signal_context=signal_context,
    )


def _lstm_feature_matrix(artifact: ResearchArtifact, candle_batch_df: pd.DataFrame) -> np.ndarray:
    """Build the LSTM per-step feature matrix, with orthogonal features if the artifact uses them.

    Orthogonal artifacts declare `feature_config.orthogonal_groups`; the orthogonal series are
    self-fetched by the ML block via the market-context provider (NOT supplied through the
    contract). Plain artifacts use the 14-dim OHLCV/time features.
    """
    from src.models.lstm_model import build_per_step_features

    groups = artifact.feature_config.get("orthogonal_groups")
    if groups:
        from src.features.orthogonal import build_combined_features
        from .market_context import get_market_context

        ticker = str(artifact.metadata.get("ticker", "")).upper()
        ortho = get_market_context().get_ortho_series()
        mat, _ = build_combined_features(candle_batch_df, ortho, ticker, groups=tuple(groups))
        return mat
    return build_per_step_features(candle_batch_df)


def _predict_with_lstm(artifact: ResearchArtifact, candle_batch_df: pd.DataFrame) -> ArtifactPrediction:
    """LSTM inference: SEQ_LEN-step sequence window."""
    import torch

    seq_len = int(artifact.feature_config.get("seq_len", 32))
    norm_mean = artifact.feature_mean   # shape (input_dim,)
    norm_std  = artifact.feature_std    # shape (input_dim,)

    feat_mat = _lstm_feature_matrix(artifact, candle_batch_df)   # (N, input_dim)
    if len(feat_mat) < seq_len:
        return _safe_hold_prediction(
            artifact,
            error="insufficient_history",
            message=f"LSTM needs at least {seq_len} candles, got {len(feat_mat)}.",
            n_candles=len(candle_batch_df),
        )

    # Guard: built feature dim must match the artifact's normalisation/input dim.
    if feat_mat.shape[1] != len(norm_mean):
        raise ValueError(
            f"Feature dimension mismatch: builder produced {feat_mat.shape[1]} features but the "
            f"artifact expects {len(norm_mean)} (feature_set='{artifact.feature_config.get('feature_set', '')}', "
            f"orthogonal_groups={artifact.feature_config.get('orthogonal_groups')})."
        )

    # Normalise and take the last seq_len steps
    feat_norm = np.nan_to_num((feat_mat - norm_mean) / norm_std, nan=0.0, posinf=0.0, neginf=0.0)
    window = feat_norm[-seq_len:].astype(np.float32)       # (seq_len, input_dim)
    x = torch.from_numpy(window).unsqueeze(0)              # (1, seq_len, input_dim)

    artifact.model.eval()
    with torch.no_grad():
        logits = artifact.model(x)
        raw_proba = torch.softmax(logits, dim=1).numpy()[0]   # (3,)

    # Map class indices to contract keys (SELL=0, HOLD=1, BUY=2)
    label_map = artifact.label_mapping.get("internal_to_contract", {})
    probabilities = {
        label_map.get("SELL", "sell"): float(raw_proba[0]),
        label_map.get("HOLD", "hold"): float(raw_proba[1]),
        label_map.get("BUY",  "buy"):  float(raw_proba[2]),
    }
    # Ensure keys match contract spec
    probabilities = {k.lower(): v for k, v in probabilities.items()}
    confidence = float(max(probabilities.values()))

    diagnostics = _base_diagnostics(artifact, n_candles=len(candle_batch_df))
    diagnostics.update({
        "artifact_missing": False,
        "prediction_mode": "research_artifact_lstm",
        "probabilities_calibrated": bool(artifact.metadata.get("probabilities_calibrated", False)),
        "seq_len": seq_len,
        "note": "Entry threshold/hold/stop are risk_manager concerns; see signal_context for barriers.",
    })
    expected_return, signal_context = _compute_signal_context(artifact, candle_batch_df, probabilities)
    return ArtifactPrediction(
        probabilities=probabilities, confidence=confidence, diagnostics=diagnostics,
        expected_return=expected_return, signal_context=signal_context,
    )


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
        expected_return=prediction.expected_return,
        metadata=metadata,
        diagnostics=prediction.diagnostics,
        signal_context=prediction.signal_context,
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
