"""Per-ticker model routing for the ML block.

The ML block serves `ml_prediction` PER TICKER, and each ticker needs its OWN model:
price dynamics differ across SBER/GAZP/LKOH and joint training dilutes the signal
(see research). This module is the request-handling layer: an incoming candle_batch
carries a `ticker`, and the router resolves the ticker-specific research artifact,
loads it (cached), and returns the ml_prediction contract response. If no artifact
exists for that ticker, it returns a graceful `artifact_missing` response.

Resolution convention (first existing wins), per ticker T at 1H:
    ml/artifacts/research_lstm_v2_<t>_h1/         (preferred — LSTM v2)
    ml/artifacts/research_triple_barrier_<t>_h1/  (fallback — ExtraTrees)

Today only SBER has packaged artifacts, so GAZP/LKOH route to artifact_missing until
their models are packaged. The router does not decide trades — it only forecasts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import (
    CandleBatch,
    build_artifact_missing_response,
    candle_batch_to_dataframe,
    load_candle_batch_json,
)
from .research_artifact import (
    ResearchArtifact,
    artifact_bundle_available,
    build_artifact_prediction_response,
    load_research_artifact,
)

ML_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = ML_DIR / "artifacts"

SUPPORTED_TICKERS = ("SBER", "GAZP", "LKOH")

# Preference order; "{t}" is the lowercase ticker. First existing bundle wins.
_ARTIFACT_TEMPLATES = (
    "research_lstm_v2_{t}_h1",
    "research_triple_barrier_{t}_h1",
)


def resolve_artifact_dir(ticker: str, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT) -> Path | None:
    """Return the artifact directory for a ticker, or None if none is available."""
    t = str(ticker).strip().lower()
    if not t:
        return None
    for template in _ARTIFACT_TEMPLATES:
        candidate = Path(artifact_root) / template.format(t=t)
        if artifact_bundle_available(candidate):
            return candidate
    return None


class TickerModelRouter:
    """Routes a candle_batch to the ticker-specific model and runs inference.

    Loaded artifacts are cached in-memory so repeated requests for the same ticker
    do not reload from disk.
    """

    def __init__(self, artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT) -> None:
        self.artifact_root = Path(artifact_root)
        self._cache: dict[Path, ResearchArtifact] = {}

    def available_tickers(self) -> list[str]:
        """Tickers that currently have a loadable artifact."""
        return [t for t in SUPPORTED_TICKERS if resolve_artifact_dir(t, self.artifact_root) is not None]

    def _load(self, artifact_dir: Path) -> ResearchArtifact:
        artifact = self._cache.get(artifact_dir)
        if artifact is None:
            artifact = load_research_artifact(artifact_dir)
            self._cache[artifact_dir] = artifact
        return artifact

    def predict(self, batch: CandleBatch, df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Resolve the ticker's model and return an ml_prediction response."""
        if df is None:
            df = candle_batch_to_dataframe(batch)

        artifact_dir = resolve_artifact_dir(batch.ticker, self.artifact_root)
        if artifact_dir is None:
            expected = self.artifact_root / _ARTIFACT_TEMPLATES[0].format(t=batch.ticker.strip().lower())
            return build_artifact_missing_response(
                batch=batch,
                df=df,
                artifact_dir=expected,
                message=(
                    f"No research artifact for ticker '{batch.ticker}'. "
                    f"Tickers with models: {self.available_tickers() or 'none'}."
                ),
            )

        artifact = self._load(artifact_dir)
        # Safety: a ticker-routed artifact must have been trained for this ticker.
        artifact_ticker = str(artifact.metadata.get("ticker", "")).strip().upper()
        if artifact_ticker and artifact_ticker != batch.ticker.strip().upper():
            raise ValueError(
                f"Artifact at {artifact_dir} is for ticker {artifact_ticker!r}, "
                f"but the request is for {batch.ticker!r}."
            )
        return build_artifact_prediction_response(batch=batch, df=df, artifact=artifact)


def predict_candle_batch(
    payload: str | Path | dict[str, Any] | CandleBatch,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """End-to-end convenience: candle_batch (path/dict/CandleBatch) -> routed ml_prediction."""
    batch = payload if isinstance(payload, CandleBatch) else load_candle_batch_json(payload)
    df = candle_batch_to_dataframe(batch)
    return TickerModelRouter(artifact_root).predict(batch, df)
