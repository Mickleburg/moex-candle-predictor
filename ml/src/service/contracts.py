"""JSON contract helpers for ML block integration.

The functions in this module implement the repository-level contracts from
``contracts/candle_batch.schema.json`` and ``contracts/ml_prediction.schema.json``
without changing the legacy FastAPI ``/predict`` path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_CANDLE_FIELDS = ("begin", "open", "high", "low", "close", "volume")
PROBABILITY_KEYS = ("buy", "hold", "sell")


@dataclass(frozen=True)
class CandleBatch:
    """Parsed candle batch input contract."""

    ticker: str
    timeframe: str
    candles: list[dict[str, Any]]


@dataclass(frozen=True)
class ModelContractMetadata:
    """Research/default model metadata for ML prediction contract output."""

    model_version: str
    model_family: str
    target: str
    feature_set: str
    validation_macro_f1: float
    is_production: bool = False
    class_weight: str | None = None


CURRENT_RESEARCH_DEFAULT = ModelContractMetadata(
    model_version="research-triple-barrier-2026-05-15",
    model_family="triple_barrier_extra_trees",
    target="triple_barrier:h3:w12:up1.25:down1.25",
    feature_set="continuous_regime",
    validation_macro_f1=0.4685,
    is_production=False,
    class_weight="none",
)


def load_candle_batch_json(path_or_dict: str | Path | dict[str, Any]) -> CandleBatch:
    """Load and validate a candle batch JSON payload."""

    if isinstance(path_or_dict, (str, Path)):
        with Path(path_or_dict).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif isinstance(path_or_dict, dict):
        payload = path_or_dict
    else:
        raise TypeError("path_or_dict must be a path or dictionary")

    if not isinstance(payload, dict):
        raise ValueError("Candle batch payload must be a JSON object")
    for field in ("ticker", "timeframe", "candles"):
        if field not in payload:
            raise ValueError(f"Missing candle batch field: {field}")
    ticker = str(payload["ticker"]).strip()
    timeframe = str(payload["timeframe"]).strip()
    candles = payload["candles"]
    if not ticker:
        raise ValueError("ticker must be non-empty")
    if not timeframe:
        raise ValueError("timeframe must be non-empty")
    if not isinstance(candles, list) or not candles:
        raise ValueError("candles must be a non-empty array")
    for idx, candle in enumerate(candles):
        if not isinstance(candle, dict):
            raise ValueError(f"candle[{idx}] must be an object")
        missing = [field for field in REQUIRED_CANDLE_FIELDS if field not in candle]
        if missing:
            raise ValueError(f"candle[{idx}] missing required fields: {missing}")
    return CandleBatch(ticker=ticker, timeframe=timeframe, candles=list(candles))


def candle_batch_to_dataframe(batch: CandleBatch) -> pd.DataFrame:
    """Convert a contract candle batch to a normalized, sorted DataFrame."""

    rows: list[dict[str, Any]] = []
    for candle in batch.candles:
        row = dict(candle)
        row.setdefault("ticker", batch.ticker)
        row.setdefault("timeframe", batch.timeframe)
        rows.append(row)
    df = pd.DataFrame(rows)

    df["begin"] = pd.to_datetime(df["begin"], errors="coerce")
    if df["begin"].isna().any():
        raise ValueError("Candle batch contains invalid begin timestamps")
    if df["begin"].duplicated().any():
        duplicates = df.loc[df["begin"].duplicated(), "begin"].astype(str).tolist()
        raise ValueError(f"Duplicate candle begin timestamps are not supported: {duplicates}")

    for column, expected in (("ticker", batch.ticker), ("timeframe", batch.timeframe)):
        values = df[column].dropna().astype(str).str.strip()
        if values.nunique() > 1:
            raise ValueError(f"Mixed {column} values are not supported in one candle batch")
        if len(values) and values.iloc[0] != expected:
            raise ValueError(f"Candle-level {column} does not match batch {column}")

    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any() or not np.isfinite(df[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Candle column {column} contains non-finite values")

    invalid_ohlc = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (
        df["low"] > df[["open", "close", "high"]].min(axis=1)
    )
    if invalid_ohlc.any():
        raise ValueError("Candle batch contains invalid OHLC rows")
    if (df["volume"] < 0).any():
        raise ValueError("Candle volume must be non-negative")

    df = df.sort_values("begin").reset_index(drop=True)
    if "value" not in df.columns:
        df["value"] = df["close"] * df["volume"]
    return df


def dataframe_as_of(df: pd.DataFrame) -> str:
    """Return contract timestamp for the latest candle in a DataFrame."""

    if df.empty:
        raise ValueError("Cannot compute as_of for an empty DataFrame")
    value = pd.Timestamp(df["begin"].iloc[-1])
    return value.isoformat()


def build_ml_prediction_response(
    *,
    ticker: str,
    timeframe: str,
    as_of: str,
    probabilities: dict[str, float] | None = None,
    confidence: float | None = None,
    expected_return: float | None = None,
    metadata: ModelContractMetadata = CURRENT_RESEARCH_DEFAULT,
    diagnostics: dict[str, Any] | None = None,
    signal_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON object compatible with ``ml_prediction.schema.json``.

    ``signal_context`` is optional prediction-intrinsic forecast context (horizon,
    volatility-scaled barrier levels, calibration flag). It is INFORMATION for the
    downstream risk_manager — never a trading command. Omitted from the
    response when not provided.
    """

    probabilities = dict(probabilities or {"buy": 0.0, "hold": 1.0, "sell": 0.0})
    _validate_probability_dict(probabilities)
    if confidence is None:
        confidence = float(max(probabilities.values()))
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be in [0, 1]")

    base_diagnostics = {
        "feature_set": metadata.feature_set,
        "validation_macro_f1": float(metadata.validation_macro_f1),
        "is_production": bool(metadata.is_production),
    }
    if metadata.class_weight is not None:
        base_diagnostics["class_weight"] = metadata.class_weight
    if diagnostics:
        base_diagnostics.update(diagnostics)

    response: dict[str, Any] = {
        "ticker": ticker,
        "timeframe": timeframe,
        "as_of": as_of,
        "model_version": metadata.model_version,
        "model_family": metadata.model_family,
        "target": metadata.target,
        "probabilities": {key: float(probabilities[key]) for key in PROBABILITY_KEYS},
        "confidence": float(confidence),
        "expected_return": expected_return,
        "diagnostics": base_diagnostics,
    }
    if signal_context is not None:
        response["signal_context"] = dict(signal_context)
    return response


def build_artifact_missing_response(
    *,
    batch: CandleBatch,
    df: pd.DataFrame,
    artifact_dir: str | Path,
    metadata: ModelContractMetadata = CURRENT_RESEARCH_DEFAULT,
    message: str | None = None,
) -> dict[str, Any]:
    """Return an explicit contract-compatible response when no artifact exists."""

    artifact_path = Path(artifact_dir)
    diagnostics = {
        "artifact_missing": True,
        "artifact_path": str(artifact_path),
        "n_candles": int(len(df)),
        "message": message
        or "Research candidate is documented but no fitted production artifact is available yet.",
    }
    return build_ml_prediction_response(
        ticker=batch.ticker,
        timeframe=batch.timeframe,
        as_of=dataframe_as_of(df),
        probabilities={"buy": 0.0, "hold": 1.0, "sell": 0.0},
        confidence=0.0,
        expected_return=None,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def metadata_from_dict(payload: dict[str, Any]) -> ModelContractMetadata:
    """Parse metadata dictionary for future artifact bundles."""

    return ModelContractMetadata(
        model_version=str(payload.get("model_version", CURRENT_RESEARCH_DEFAULT.model_version)),
        model_family=str(payload.get("model_family", CURRENT_RESEARCH_DEFAULT.model_family)),
        target=str(payload.get("target", CURRENT_RESEARCH_DEFAULT.target)),
        feature_set=str(payload.get("feature_set", CURRENT_RESEARCH_DEFAULT.feature_set)),
        validation_macro_f1=float(payload.get("validation_macro_f1", CURRENT_RESEARCH_DEFAULT.validation_macro_f1)),
        is_production=bool(payload.get("is_production", False)),
        class_weight=payload.get("class_weight"),
    )


def metadata_to_dict(metadata: ModelContractMetadata) -> dict[str, Any]:
    """Serialize metadata for CLI JSON output or future artifact manifests."""

    return asdict(metadata)


def _validate_probability_dict(probabilities: dict[str, float]) -> None:
    if set(probabilities) != set(PROBABILITY_KEYS):
        raise ValueError(f"probabilities must have keys: {PROBABILITY_KEYS}")
    values = np.asarray([float(probabilities[key]) for key in PROBABILITY_KEYS], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities contain non-finite values")
    if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
        raise ValueError("probabilities must be in [0, 1]")
    if not np.isclose(values.sum(), 1.0, atol=1e-6):
        raise ValueError(f"probabilities must sum to 1.0, got {values.sum()}")
