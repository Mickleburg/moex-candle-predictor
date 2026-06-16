"""Regime detector (H5) — flag when the current market state is UNLIKE the training
distribution, so risk_manager can abstain / downsize there.

Premise: a model is least reliable in novel regimes (shocks, structural breaks). If we can
flag those past-only, we gate trades and improve robustness — independent of the alpha source
(price OR news). This module provides the regime vector and a rolling Mahalanobis novelty
distance; the research script validates that predictability degrades as distance rises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_features(panel: pd.DataFrame, market: pd.Series, vol_window: int = 20) -> pd.DataFrame:
    """Past-only daily regime descriptor for the universe + market.

    Features:
      mkt_vol     EWMA volatility of the market index (RiskMetrics-ish)
      mkt_trend   market momentum over vol_window
      xsec_disp   cross-sectional dispersion of daily returns (rolling mean of std-across-names)
      mkt_absret  recent average |market return| (stress proxy)
    """
    rets = panel.pct_change()
    rm = market.pct_change()
    mkt_vol = np.sqrt(rm.pow(2).ewm(alpha=1 - 0.94, adjust=False).mean())
    mkt_trend = market / market.shift(vol_window) - 1.0
    xsec_disp = rets.std(axis=1).rolling(vol_window).mean()
    mkt_absret = rm.abs().rolling(vol_window).mean()
    feat = pd.DataFrame({
        "mkt_vol": mkt_vol, "mkt_trend": mkt_trend,
        "xsec_disp": xsec_disp, "mkt_absret": mkt_absret,
    }).reindex(panel.index)
    return feat


def rolling_mahalanobis(feat: pd.DataFrame, min_train: int = 250) -> pd.Series:
    """Novelty distance of each date's regime vs the EXPANDING past distribution (past-only).

    At date t: fit mean/cov on feat[:t] (strictly before t), return Mahalanobis distance of
    feat[t]. NaN until min_train history. Higher = more novel/unlike training.
    """
    X = feat.to_numpy(float)
    n, d = X.shape
    out = np.full(n, np.nan)
    for t in range(min_train, n):
        past = X[:t]
        past = past[np.all(np.isfinite(past), axis=1)]
        row = X[t]
        if len(past) < min_train or not np.all(np.isfinite(row)):
            continue
        mu = past.mean(axis=0)
        cov = np.cov(past, rowvar=False) + 1e-9 * np.eye(d)
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue
        diff = row - mu
        out[t] = float(np.sqrt(diff @ inv @ diff))
    return pd.Series(out, index=feat.index, name="regime_distance")
