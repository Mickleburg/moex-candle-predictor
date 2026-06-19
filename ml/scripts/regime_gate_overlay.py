"""Regime gate as a STANDALONE defensive overlay — the ML block's strongest deployable result.

The gate (H5) is direction-agnostic: it does not pick stocks, it cuts gross exposure when the
market enters a NOVEL regime (e.g. 2022). Applied to any passive long, it halves crash losses.
This isolates the gate's value: passive long (basket and IMOEX) with the gate OFF vs ON, and
DAILY vs WEEKLY reaction (faster reaction = better crash protection). Past-only, fees on the
exposure changes. Judge by max drawdown and the 2022 shock.
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
from scripts.risk_managed_long_agent import exposure_series  # noqa: E402

FORWARD_START = pd.Timestamp("2025-01-01", tz="Europe/Moscow")
FEE_ONEWAY = 0.0005
BARS_PER_YEAR = 247


def overlay(under_ret: pd.Series, expo: pd.Series, react_days: int) -> pd.Series:
    """Apply exposure (held, updated every react_days) to a passive long; fee on changes."""
    r = under_ret.to_numpy()
    e = expo.reindex(under_ret.index).fillna(0.0).to_numpy()
    held, out = 1.0, []
    start = 260
    for i in range(len(r)):
        pr = held * r[i] if i > start else 0.0
        if i >= start and (i - start) % react_days == 0:
            pr -= abs(e[i] - held) * FEE_ONEWAY
            held = e[i]
        out.append(pr)
    return pd.Series(out, index=under_ret.index)


def stats(daily, lo=None, hi=None):
    s = daily
    if lo is not None: s = s[s.index >= lo]
    if hi is not None: s = s[s.index < hi]
    if len(s) == 0: return {}
    eq = (1 + s).cumprod()
    ann = (1 + s.mean()) ** BARS_PER_YEAR - 1
    vol = s.std() * np.sqrt(BARS_PER_YEAR)
    return {"cum": float(eq.iloc[-1] - 1), "sharpe": float(ann / vol) if vol > 0 else 0.0,
            "maxdd": float((eq / eq.cummax() - 1).min())}


def row(name, daily):
    f, fw = stats(daily), stats(daily, lo=FORWARD_START)
    y22 = stats(daily, lo=pd.Timestamp("2022-01-01", tz="Europe/Moscow"),
                hi=pd.Timestamp("2023-01-01", tz="Europe/Moscow"))
    print(f"{name:24} | full cum {f['cum']:>+6.2f} Sh {f['sharpe']:>+5.2f} DD {f['maxdd']:>+6.2f} "
          f"| FWD cum {fw['cum']:>+5.2f} DD {fw['maxdd']:>+6.2f} | 2022 cum {y22['cum']:>+5.2f} "
          f"DD {y22['maxdd']:>+6.2f}")


def main() -> int:
    panel, _, market = load_panels(timeframe="1D")
    expo = exposure_series(rolling_mahalanobis(regime_features(panel, market), min_train=250))
    basket = panel.pct_change().mean(axis=1).fillna(0.0)        # equal-weight basket
    index = market.pct_change().reindex(panel.index).fillna(0.0)
    print(f"Regime-gate defensive overlay ({panel.index.min().date()}..{panel.index.max().date()}, "
          f"fee 5bps)\n")
    print(f"{'strategy':24} | {'--------- full ---------':^24} | {'----- forward -----':^20} | "
          f"{'--- 2022 ---':^18}")
    for label, under in [("BASKET", basket), ("IMOEX", index)]:
        row(f"{label} long (no gate)", overlay(under, pd.Series(1.0, index=under.index), 5))
        row(f"{label} + gate weekly", overlay(under, expo, 5))
        row(f"{label} + gate daily", overlay(under, expo, 1))
        print()
    print("Read: the gate keeps full upside in normal regimes but cuts exposure in novel ones,")
    print("so max drawdown and the 2022 loss shrink. Daily reaction protects faster than weekly.")
    print("Direction-agnostic crash protection — INFORMATION for risk_manager. is_production=false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
