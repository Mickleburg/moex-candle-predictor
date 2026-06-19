"""Risk overlay on the deployment-sim gate: vol-targeting (H4) + regime gating (H5).

Capstone that wires the validated risk pieces ON TOP of the cross-sectional ranker, in the
honest deployment simulation. Four variants compared on the SAME rolling-retrain Ridge ranker
(beta-residual target), fresh forward, fees:

    raw        equal-weight long top-k / short bottom-k
    +vol       inverse-EWMA-vol position sizing (H4): downsize high-vol names
    +regime    skip the trade when regime novelty (H5) is in its high past-only quantile
    +both      vol sizing AND regime gate

On the current NULL price signal the expected win is risk reduction (drawdown / loss tails),
not return. The point is to prove the overlay machinery and have it ready for the news signal,
which runs through the exact same path (swap the score source in build_feature_panels()).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from src.features.cross_sectional import load_panels, relative_target  # noqa: E402
from src.features.regime import regime_features, rolling_mahalanobis  # noqa: E402
from scripts.xsec_deployment_sim import build_feature_panels  # noqa: E402

try:
    from sklearn.linear_model import Ridge
except ImportError:
    Ridge = None

FORWARD_START = pd.Timestamp("2025-01-01", tz="Europe/Moscow")
FEE_ONEWAY = 0.0005


def ewma_vol_panel(panel: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    r = panel.pct_change()
    return np.sqrt(r.pow(2).ewm(alpha=1 - lam, adjust=False).mean())


def leg_weights(idxs, inv_vol, use_vol):
    if not use_vol:
        return np.full(len(idxs), 1.0 / len(idxs))
    w = inv_vol[idxs]
    w = np.where(np.isfinite(w) & (w > 0), w, np.nanmedian(w))
    return w / w.sum()


def run(panel, target, feat_panels, vol_panel, dist, *, horizon, k,
        use_vol, use_regime, retrain_every=20, train_min=300):
    names = list(feat_panels.keys())
    idx = panel.index
    n = len(idx)
    fee = 4.0 * FEE_ONEWAY
    tgt = target.to_numpy()
    fwd = (panel.shift(-horizon) / panel - 1.0).to_numpy()
    invvol = (1.0 / vol_panel.replace(0, np.nan)).to_numpy()

    # past-only high-novelty threshold (expanding 2/3 quantile of distance)
    dist_v = dist.to_numpy()

    Xpool, ypool, model = [], [], None
    rets, steps, next_trade = [], 0, 0
    for i in range(n - horizon):
        d = i - horizon
        if d >= 0:
            Xr, yr = _row(feat_panels, names, d), tgt[d]
            m = np.all(np.isfinite(Xr), axis=1) & np.isfinite(yr)
            if m.any():
                Xpool.append(Xr[m]); ypool.append(yr[m])
        if i < next_trade:
            continue
        next_trade = i + horizon
        steps += 1
        Xi = _row(feat_panels, names, i)
        valid = np.all(np.isfinite(Xi), axis=1)
        if valid.sum() < 2 * k or Ridge is None:
            continue
        total = sum(len(x) for x in Xpool)
        if total >= train_min and (model is None or steps % retrain_every == 0):
            model = Ridge(alpha=10.0).fit(np.vstack(Xpool), np.concatenate(ypool))
        if model is None:
            continue
        # regime gate: skip if current novelty >= past-only 2/3 quantile
        if use_regime and np.isfinite(dist_v[i]):
            past = dist_v[:i][np.isfinite(dist_v[:i])]
            if len(past) >= 100 and dist_v[i] >= np.quantile(past, 2 / 3):
                rets.append((idx[i], 0.0))           # abstain (flat)
                continue
        scores = np.full(Xi.shape[0], np.nan)
        scores[valid] = model.predict(Xi[valid])
        order = np.argsort(np.where(valid, scores, np.nan))
        order = order[~np.isnan(scores[order])]
        if len(order) < 2 * k:
            continue
        longs, shorts = order[-k:], order[:k]
        wl, ws = leg_weights(longs, invvol[i], use_vol), leg_weights(shorts, invvol[i], use_vol)
        ret = float(np.nansum(wl * fwd[i][longs]) - np.nansum(ws * fwd[i][shorts])) - fee
        rets.append((idx[i], ret))

    s = pd.DataFrame(rets, columns=["t", "r"]).set_index("t")["r"].dropna()
    return s


def _row(feat_panels, names, i):
    return np.column_stack([feat_panels[f].iloc[i].to_numpy() for f in names])


def stats(s):
    fw = s[s.index >= FORWARD_START]
    if len(fw) == 0:
        return {}
    eq = (1 + fw).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    traded = fw[fw != 0]
    return {"fwd_cum": round(float(eq.iloc[-1] - 1), 4), "fwd_dd": round(dd, 4),
            "fwd_vol": round(float(fw.std()), 4), "win": round(float((traded > 0).mean()), 3)
            if len(traded) else 0.0, "n": len(fw), "n_traded": int(len(traded))}


def main() -> int:
    if Ridge is None:
        print("sklearn missing"); return 1
    panel, sector_panel, market = load_panels(timeframe="1D")
    feat_panels = build_feature_panels(panel)
    vol_panel = ewma_vol_panel(panel)
    dist = rolling_mahalanobis(regime_features(panel, market), min_train=250)
    H, k = 20, 3
    target = relative_target(panel, H, "beta_residual", sector_panel, market)
    print(f"Risk overlay on deployment gate (H={H}, k={k}, target=beta_residual)\n")
    print(f"{'variant':10} | {'FWD cum':>8} {'FWD maxDD':>9} {'FWD vol':>8} {'win':>5} "
          f"{'n_traded':>8}")
    for label, uv, ur in [("raw", False, False), ("+vol", True, False),
                          ("+regime", False, True), ("+both", True, True)]:
        s = run(panel, target, feat_panels, vol_panel, dist,
                horizon=H, k=k, use_vol=uv, use_regime=ur)
        m = stats(s)
        print(f"{label:10} | {m.get('fwd_cum',0):>8.4f} {m.get('fwd_dd',0):>9.4f} "
              f"{m.get('fwd_vol',0):>8.4f} {m.get('win',0):>5.2f} {m.get('n_traded',0):>8}")
    print("\nRead: vol-sizing should cut FWD vol/drawdown; regime gate should cut drawdown by")
    print("abstaining in novel regimes (fewer trades). On the null price signal, judge by RISK")
    print("(maxDD, vol), not return. Same path will size/gate the future news signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
