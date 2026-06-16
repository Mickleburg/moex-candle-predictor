"""V2 Layer-4 serving: the ML block's RISK-ANALYTICS output for risk_manager.

After the directional alpha (price and news) was found not to generalize, the ML block's
honest, deployable role is risk analytics: forward-volatility forecasting (H4) and a
market-regime novelty gate (H5), turned into per-ticker sizing + a portfolio exposure scalar.
This module assembles the frozen `risk_analytics` contract. It is INFORMATION for risk_manager
(vol-targeting + regime gating), never a trade decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..features.regime import regime_features, rolling_mahalanobis

REPO_ROOT = Path(__file__).resolve().parents[3]
EWMA_LAMBDA = 0.94


def ewma_vol_forecast(panel: pd.DataFrame, as_of: pd.Timestamp,
                      lam: float = EWMA_LAMBDA) -> pd.Series:
    """Past-only EWMA per-bar volatility forecast per ticker at as_of (H4)."""
    r = panel.loc[:as_of].pct_change()
    var = r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var.iloc[-1])


def regime_state(panel: pd.DataFrame, market: pd.Series, as_of: pd.Timestamp,
                 min_train: int = 250) -> dict[str, float | bool]:
    """Past-only regime novelty + suggested exposure scalar at as_of (H5)."""
    feat = regime_features(panel.loc[:as_of], market.loc[:as_of])
    dist = rolling_mahalanobis(feat, min_train=min_train).dropna()
    if len(dist) == 0:
        return {"distance": 0.0, "percentile": 0.5, "novel": False, "exposure_scalar": 1.0}
    cur = float(dist.iloc[-1])
    pct = float((dist <= cur).mean())
    novel = pct >= 2.0 / 3.0
    # smooth gross-exposure multiplier: 1 below the 2/3 quantile, ramp to 0 at the top tail
    exposure = float(np.clip(1.0 - max(0.0, pct - 2.0 / 3.0) / (1.0 / 3.0), 0.0, 1.0))
    return {"distance": round(cur, 4), "percentile": round(pct, 4),
            "novel": bool(novel), "exposure_scalar": round(exposure, 4)}


def build_risk_analytics(
    panel: pd.DataFrame,
    market: pd.Series,
    *,
    as_of: pd.Timestamp | None = None,
    horizon_bars: int = 10,
    timeframe: str = "1D",
    model_version: str = "risk_analytics_v0",
    is_production: bool = False,
) -> dict[str, Any]:
    """Assemble a dict compatible with `risk_analytics.schema.json`."""
    if as_of is None:
        as_of = panel.index[-1]
    vol = ewma_vol_forecast(panel, as_of)
    valid = vol[np.isfinite(vol) & (vol > 0)]
    inv = 1.0 / valid
    inv = inv / inv.sum() if inv.sum() > 0 else inv

    per_ticker = []
    for t in panel.columns:
        v = float(vol[t]) if np.isfinite(vol[t]) else 0.0
        ok = np.isfinite(vol[t]) and vol[t] > 0
        per_ticker.append({
            "ticker": t,
            "vol_forecast": round(v, 6),
            "inv_vol_weight": round(float(inv[t]), 6) if ok and t in inv else 0.0,
            "valid": bool(ok),
        })

    return {
        "as_of": as_of.isoformat(),
        "timeframe": timeframe,
        "horizon": {"bars": int(horizon_bars), "timeframe": timeframe},
        "universe": list(panel.columns),
        "per_ticker": per_ticker,
        "regime": regime_state(panel, market, as_of),
        "model_version": model_version,
        "is_production": bool(is_production),
    }


def validate_against_schema(payload: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError:
        return
    schema = json.loads(
        (REPO_ROOT / "contracts" / "risk_analytics.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)
