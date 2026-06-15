"""Market-context provider — the ML block self-fetches orthogonal data at inference.

Architectural decision: the candle_batch input contract carries only the TARGET ticker's
candles (keeps the contract simple as more tickers are added). Models that use orthogonal
drivers (e.g. LKOH needs Brent + IMOEX/RTSI) obtain that data HERE, inside the ML block,
not via the contract.

Current source: the local parquet store in data/raw (downloaded by scripts/download_candles.py
and scripts/download_futures_continuous.py). For a live deployment, `refresh_recent()` is the
extension point to pull the latest bars from MOEX ISS before serving.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.orthogonal import ORTHO_TICKERS, load_ortho_series

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "raw"


class MarketContextProvider:
    """Provides aligned orthogonal series for the ML block's own use at inference."""

    def __init__(self, data_dir: str | Path = _DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self._series: dict[str, pd.DataFrame] | None = None

    def get_ortho_series(self, instruments=ORTHO_TICKERS) -> dict[str, pd.DataFrame]:
        """Return {instrument: [begin, close]} (MSK tz-aware), loaded from the local store.

        Cached after first load. Missing instruments are skipped (tolerant loader).
        """
        if self._series is None:
            self._series = load_ortho_series(str(self.data_dir), instruments)
        return self._series

    def refresh_recent(self) -> None:
        """Extension point for live deployments: pull latest bars from MOEX ISS.

        Not needed for research/backtest (the local store already spans the period). A live
        server would fetch the newest candles for each orthogonal instrument here and merge
        them into the local store / cache before serving.
        """
        self._series = None  # force reload from the (externally refreshed) store


_DEFAULT_PROVIDER: MarketContextProvider | None = None


def get_market_context() -> MarketContextProvider:
    """Process-wide singleton provider (artifacts are cached; so is this)."""
    global _DEFAULT_PROVIDER
    if _DEFAULT_PROVIDER is None:
        _DEFAULT_PROVIDER = MarketContextProvider()
    return _DEFAULT_PROVIDER
