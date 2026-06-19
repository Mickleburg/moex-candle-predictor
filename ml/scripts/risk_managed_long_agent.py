"""Layer-4 trading agent: risk-managed LONG basket vs buy-and-hold.

The honest realization of the ML block's role. We have NO directional alpha, so the agent
does not pick winners; it takes the equity-premium assumption (the basket drifts up over time)
and makes that exposure smarter with our two validated risk signals:
  * inverse-vol weights (H4) — calm names bigger, jumpy names smaller (low-vol / risk-parity tilt);
  * regime gate (H5) — full exposure in normal regimes, cut to cash when novelty spikes (e.g. 2022).

Compared against equal-weight buy-and-hold. Daily long-only backtest, weekly rebalance, fees.
Judge by RISK-ADJUSTED return (Sharpe, max drawdown) and shock behavior (2022), not raw return.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from src.features.cross_sectional import load_panels  # noqa: E402
from src.features.regime import regime_features, rolling_mahalanobis  # noqa: E402

FORWARD_START = pd.Timestamp("2025-01-01", tz="Europe/Moscow")
FEE_ONEWAY = 0.0005
REBALANCE = 5            # trading days
BARS_PER_YEAR = 247


def ewma_vol_panel(panel, lam=0.94):
    r = panel.pct_change()
    return np.sqrt(r.pow(2).ewm(alpha=1 - lam, adjust=False).mean())


def exposure_series(dist: pd.Series) -> pd.Series:
    """Past-only regime exposure scalar in [0,1] from the novelty distance (expanding percentile)."""
    d = dist.to_numpy()
    out = np.full(len(d), 1.0)
    for t in range(len(d)):
        if not np.isfinite(d[t]):
            out[t] = 0.0          # no regime info yet -> stay flat (conservative)
            continue
        past = d[: t + 1][np.isfinite(d[: t + 1])]
        pct = float((past <= d[t]).mean())
        out[t] = float(np.clip(1.0 - max(0.0, pct - 2 / 3) / (1 / 3), 0.0, 1.0))
    return pd.Series(out, index=dist.index)


def target_weights(invvol_row, expo, use_vol, use_regime, n):
    if use_vol and np.isfinite(invvol_row).any():
        w = np.where(np.isfinite(invvol_row) & (invvol_row > 0), invvol_row, 0.0)
        w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    else:
        w = np.full(n, 1.0 / n)
    if use_regime:
        w = w * expo                      # scale gross exposure (rest -> cash)
    return w


def backtest(panel, vol_panel, expo, *, use_vol, use_regime):
    rets = panel.pct_change().fillna(0.0).to_numpy()
    invvol = (1.0 / vol_panel.replace(0, np.nan)).to_numpy()
    expo_v = expo.to_numpy()
    n = panel.shape[1]
    idx = panel.index
    w = np.zeros(n)
    eq = []
    start = 60                            # warm-up for vol/regime
    for i in range(len(idx)):
        if i > start:
            port_ret = float(w @ rets[i])             # cash earns 0
            gross = 1.0 + port_ret
            w = w * (1.0 + rets[i]) / gross if gross > 0 else w   # drift
        else:
            port_ret = 0.0
        eq.append(port_ret)
        if i >= start and (i - start) % REBALANCE == 0:
            tgt = target_weights(invvol[i], expo_v[i], use_vol, use_regime, n)
            turnover = float(np.abs(tgt - w).sum())
            eq[-1] -= turnover * FEE_ONEWAY            # rebalance cost
            w = tgt
    return pd.Series(eq, index=idx)


def stats(daily: pd.Series, lo=None, hi=None):
    s = daily.copy()
    if lo is not None:
        s = s[s.index >= lo]
    if hi is not None:
        s = s[s.index < hi]
    if len(s) == 0:
        return {}
    eq = (1 + s).cumprod()
    ann_ret = (1 + s.mean()) ** BARS_PER_YEAR - 1
    ann_vol = s.std() * np.sqrt(BARS_PER_YEAR)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    return {"cum": float(eq.iloc[-1] - 1), "ann": float(ann_ret),
            "vol": float(ann_vol), "sharpe": float(sharpe), "maxdd": dd}


def row(name, daily):
    full = stats(daily)
    isd = stats(daily, hi=FORWARD_START)
    fwd = stats(daily, lo=FORWARD_START)
    y22 = stats(daily, lo=pd.Timestamp("2022-01-01", tz="Europe/Moscow"),
                hi=pd.Timestamp("2023-01-01", tz="Europe/Moscow"))
    print(f"{name:16} | full cum {full['cum']:>+6.2f} Sh {full['sharpe']:>+5.2f} "
          f"DD {full['maxdd']:>+6.2f} | FWD cum {fwd['cum']:>+5.2f} Sh {fwd['sharpe']:>+5.2f} "
          f"DD {fwd['maxdd']:>+6.2f} | 2022 cum {y22['cum']:>+5.2f} DD {y22['maxdd']:>+6.2f}")


def main() -> int:
    panel, _, market = load_panels(timeframe="1D")
    vol_panel = ewma_vol_panel(panel)
    dist = rolling_mahalanobis(regime_features(panel, market), min_train=250)
    expo = exposure_series(dist)
    print(f"Risk-managed LONG agent vs buy-and-hold ({panel.shape[1]} names, "
          f"{panel.index.min().date()}..{panel.index.max().date()}, weekly rebal, fee 5bps)\n")
    print(f"{'strategy':16} | {'-------- full --------':^24} | {'------- forward ------':^22} | "
          f"{'--- 2022 shock ---':^18}")
    row("buy&hold (eq)", backtest(panel, vol_panel, expo, use_vol=False, use_regime=False))
    row("inv-vol", backtest(panel, vol_panel, expo, use_vol=True, use_regime=False))
    row("eq + regime", backtest(panel, vol_panel, expo, use_vol=False, use_regime=True))
    row("inv-vol + regime", backtest(panel, vol_panel, expo, use_vol=True, use_regime=True))
    print("\nThe agent (inv-vol + regime) should beat buy&hold on Sharpe and especially max")
    print("drawdown / 2022, at some cost to raw return — risk-managed, not alpha. is_production=false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
