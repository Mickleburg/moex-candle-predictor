"""H5 — Regime detector: does cross-sectional predictability degrade in novel regimes?

If a past-only novelty distance (rolling Mahalanobis of the regime vector) flags the periods
where any model is least reliable, risk_manager can gate/downsize there. Validation:
  1. Face validity — distance should spike at known shocks (2022-02, sanctions, etc.).
  2. Premise — bucket decision dates by distance; show cross-sectional momentum IC is LOWER
     (and outcome dispersion HIGHER) in high-distance buckets, in IS and forward.
If so, a regime gate is a justified robustness overlay for the eventual news signal.
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


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    ra = pd.Series(a[m]).rank().to_numpy(float); rb = pd.Series(b[m]).rank().to_numpy(float)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else np.nan


def main() -> int:
    panel, _, market = load_panels(timeframe="1D")
    feat = regime_features(panel, market)
    dist = rolling_mahalanobis(feat, min_train=250)
    print(f"Loaded {panel.shape[1]} tickers x {len(panel)} days; "
          f"regime distance defined on {dist.notna().sum()} days\n")

    # 1) Face validity — biggest novelty spikes
    print("=== Top-10 regime-novelty spikes (should match known shocks) ===")
    for d, v in dist.dropna().sort_values(ascending=False).head(10).items():
        print(f"  {d.date()}  distance={v:.2f}")

    # 2) Premise — IC & dispersion by distance bucket
    H, L = 20, 20
    mom = panel / panel.shift(L) - 1.0
    fwd = panel.shift(-H) / panel - 1.0
    fwd_rel = fwd.sub(fwd.mean(axis=1), axis=0)
    rows = []
    for t in panel.index:
        if not np.isfinite(dist.get(t, np.nan)):
            continue
        ic = spearman(mom.loc[t].to_numpy(), fwd_rel.loc[t].to_numpy())
        disp = float(np.nanstd(fwd.loc[t].to_numpy()))   # outcome dispersion
        rows.append((t, dist[t], ic, disp))
    df = pd.DataFrame(rows, columns=["t", "dist", "ic", "disp"]).dropna().set_index("t")

    def report(name, sub):
        if len(sub) < 30:
            print(f"  {name}: too few ({len(sub)})"); return
        q = sub["dist"].quantile([1/3, 2/3]).to_numpy()
        lo = sub[sub["dist"] <= q[0]]; hi = sub[sub["dist"] >= q[1]]
        print(f"  {name:8} | low-novelty  IC={lo['ic'].mean():+.4f} disp={lo['disp'].mean():.4f}"
              f"  | high-novelty IC={hi['ic'].mean():+.4f} disp={hi['disp'].mean():.4f}"
              f"  | corr(dist,|ic|)={sub['dist'].corr(sub['ic'].abs()):+.3f}")

    print("\n=== Predictability vs regime novelty (momentum IC, L20/H20) ===")
    print("  premise holds if high-novelty IC is LOWER and dispersion HIGHER than low-novelty")
    report("ALL", df)
    report("IS", df[df.index < FORWARD_START])
    report("FORWARD", df[df.index >= FORWARD_START])

    print("\nUse: risk_manager gates/downsizes when rolling_mahalanobis(t) is in its high")
    print("quantile (novel regime). Past-only, model-agnostic — protects the news signal too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
