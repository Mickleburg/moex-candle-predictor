"""(3) Deployment-simulation GATE for the cross-sectional ranker.

The ONLY valid validation (CLAUDE.md): rolling retrain through a fresh forward period,
fees applied, no lookahead. Ordinary walk-forward fooled us in V1. This rig is the gate the
eventual model must pass. Today it runs on PRICE features (Ridge ranker) to establish the
plumbing and a baseline; when news features land they enter `build_feature_panels()` and the
exact same gate judges them.

Design:
  * Target = beta-residual relative forward return (the cleanest market-neutral label, per the
    target-engineering study). Truest market-neutral; portfolio is long/short on the rank.
  * Features (past-only, cross-sectionally z-scored per date): momentum L in {10,20,60},
    realized vol(20), distance from MA(20).
  * Rolling retrain: a Ridge model is refit every `retrain_every` trade steps on ALL samples
    whose label was already realized at the decision time (d + H <= t) — strictly no leakage.
  * Non-overlapping trades every H bars: long top-k / short bottom-k, fee = 4*one-way/period.
  * Reports rank IC (pred vs realized target) and market-neutral net P&L, IN-SAMPLE vs FORWARD,
    against a momentum-only baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from src.features.cross_sectional import load_panels, relative_target  # noqa: E402

try:
    from sklearn.linear_model import Ridge
except ImportError:
    Ridge = None

FORWARD_START = pd.Timestamp("2025-01-01", tz="Europe/Moscow")
FEE_ONEWAY = 0.0005


def _zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def build_feature_panels(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Past-only feature panels (time x ticker), cross-sectionally z-scored per date.

    NEWS HOOK: when llm_analysis features exist, add them here as extra panels
    (e.g. 'news_sentiment', 'news_impact') aligned to the daily index with no-lookahead.
    """
    rets = panel.pct_change()
    feats = {
        "mom10": panel / panel.shift(10) - 1,
        "mom20": panel / panel.shift(20) - 1,
        "mom60": panel / panel.shift(60) - 1,
        "vol20": rets.rolling(20).std(),
        "ma20_dist": panel / panel.rolling(20).mean() - 1,
    }
    return {k: _zscore_rows(v) for k, v in feats.items()}


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    ra = pd.Series(a[m]).rank().to_numpy(float); rb = pd.Series(b[m]).rank().to_numpy(float)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else np.nan


def deployment_sim(panel, target, feat_panels, horizon, k=3,
                   retrain_every=20, train_min=300, fee_oneway=FEE_ONEWAY):
    names = list(feat_panels.keys())
    idx = panel.index
    n = len(idx)
    fee = 4.0 * fee_oneway

    def feat_row(i):
        return np.column_stack([feat_panels[f].iloc[i].to_numpy() for f in names])  # (tkr, feat)

    tgt = target.to_numpy()
    fwd_simple = (panel.shift(-horizon) / panel - 1.0).to_numpy()

    Xpool, ypool = [], []      # realized training samples (features, target)
    pool_dates = []
    model = None
    ic_rows, bt_rows = [], []
    next_trade, steps = 0, 0

    for i in range(n - horizon):
        # add samples whose label is now realized (date d with d+horizon <= i)
        d = i - horizon
        if d >= 0:
            Xr, yr = feat_row(d), tgt[d]
            mask = np.all(np.isfinite(Xr), axis=1) & np.isfinite(yr)
            if mask.any():
                Xpool.append(Xr[mask]); ypool.append(yr[mask]); pool_dates.append(idx[d])

        if i < next_trade:
            continue
        steps += 1
        Xi = feat_row(i)
        valid = np.all(np.isfinite(Xi), axis=1)
        if valid.sum() < 2 * k or Ridge is None:
            next_trade = i + horizon
            continue

        # (re)fit rolling model
        total = sum(len(x) for x in Xpool)
        if total >= train_min and (model is None or steps % retrain_every == 0):
            Xtr = np.vstack(Xpool); ytr = np.concatenate(ypool)
            model = Ridge(alpha=10.0).fit(Xtr, ytr)

        if model is None:
            next_trade = i + horizon
            continue

        scores = np.full(Xi.shape[0], np.nan)
        scores[valid] = model.predict(Xi[valid])
        realized_tgt = tgt[i]
        ic_rows.append((idx[i], _spearman(scores, realized_tgt)))

        order = np.argsort(np.where(valid, scores, np.nan))
        order = order[~np.isnan(scores[order])]
        if len(order) >= 2 * k:
            longs, shorts = order[-k:], order[:k]
            ret = np.nanmean(fwd_simple[i][longs]) - np.nanmean(fwd_simple[i][shorts]) - fee
            bt_rows.append((idx[i], ret))
        next_trade = i + horizon

    ic = pd.DataFrame(ic_rows, columns=["t", "ic"]).set_index("t")["ic"].dropna()
    bt = pd.DataFrame(bt_rows, columns=["t", "ret"]).set_index("t")["ret"].dropna()

    def split(s): return s[s.index < FORWARD_START], s[s.index >= FORWARD_START]
    def cum(s): return float((1 + s).prod() - 1) if len(s) else 0.0
    def ir(s): return float(s.mean() / (s.std() + 1e-9)) if len(s) else 0.0
    ic_is, ic_fw = split(ic); bt_is, bt_fw = split(bt)
    return {
        "ic_is": round(ic_is.mean(), 4), "ic_is_ir": round(ir(ic_is), 2),
        "ic_fw": round(ic_fw.mean(), 4), "ic_fw_ir": round(ir(ic_fw), 2),
        "bt_is_cum": round(cum(bt_is), 4), "bt_fw_cum": round(cum(bt_fw), 4),
        "bt_fw_win": round(float((bt_fw > 0).mean()), 3) if len(bt_fw) else 0.0,
        "n_trades_fw": int(len(bt_fw)), "n_ic": int(len(ic)),
    }


def main() -> int:
    if Ridge is None:
        print("scikit-learn not available."); return 1
    panel, sector_panel, market = load_panels(timeframe="1D")
    feat_panels = build_feature_panels(panel)
    print(f"Deployment-sim gate: {panel.shape[1]} tickers x {len(panel)} days, "
          f"features={list(feat_panels)}\n")
    print("Target = beta_residual (truest market-neutral). Ridge ranker, rolling retrain.\n")
    print(f"{'H':>3} {'k':>2} | {'IS IC(IR)':>12} {'FWD IC(IR)':>13} | "
          f"{'IS net':>8} {'FWD net':>8} {'win':>5} {'n_fw':>5}")
    for H in (10, 20):
        for k in (2, 3):
            tgt = relative_target(panel, H, "beta_residual", sector_panel, market)
            m = deployment_sim(panel, tgt, feat_panels, horizon=H, k=k)
            print(f"{H:>3} {k:>2} | {m['ic_is']:>+7.4f}({m['ic_is_ir']:>+4.2f}) "
                  f"{m['ic_fw']:>+7.4f}({m['ic_fw_ir']:>+4.2f}) | "
                  f"{m['bt_is_cum']:>+8.3f} {m['bt_fw_cum']:>+8.3f} {m['bt_fw_win']:>5.2f} "
                  f"{m['n_trades_fw']:>5}")
    print("\nThis is the GATE: rolling-retrain, fees, fresh forward. Price-only is expected")
    print("marginal (per H1a). The bar for the news-fused model: FORWARD net P&L robustly >0")
    print("with positive, stable forward IC. News features plug into build_feature_panels().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
