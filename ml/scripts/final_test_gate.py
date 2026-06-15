"""
FINAL LOCKED TEST-SET GATE — SBER + LKOH. One-shot, irreversible (invariant #1).

Evaluates the PACKAGED production artifacts (frozen weights, frozen normalisation) on the
untouched test split (last 15%) with the exact production rule. NOTHING is tuned here.

Rigor (must not miss any of these):
  * Test split = rows [int(len*0.85), len). Artifacts trained on the first 85% only.
  * FROZEN artifact: model weights + normalisation mean/std come from the artifact (training),
    never recomputed on test (no test-statistic leakage).
  * No lookahead: window [t-32, t), entry at close[t], barriers from past_vol[t]; future used
    only for the trade P&L.
  * Same production rule as validation: BUY conf>0.50, hold 3h + lower-barrier stop, no
    take-profit, long-only, skip weekend sessions, fee 0.05% one-way.
  * LKOH self-fetches Brent/IMOEX/RTSI (frozen orthogonal normalisation).
  * Faithfulness check vs the live router for one test candle.
Pre-flight asserts abort BEFORE any backtest if the setup is wrong.

Result: ml/docs/research/FINAL_test_gate_<ts>.json  (+ console report)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
sys.path.insert(0, str(ML_DIR))

import numpy as np
import pandas as pd
import torch

from src.data.load import load_candles
from src.service.research_artifact import load_research_artifact, _lstm_feature_matrix
from src.service.model_registry import TickerModelRouter, resolve_artifact_dir
from src.service.contracts import load_candle_batch_json
from scripts.sber_edge_analysis import long_stop_return, summarize, sharpe
from scripts.sber_backtest_research import HOURS_PER_YEAR
from src.nlp.targets import ActionTargetSpec, make_research_action_targets, triple_barrier_details

RESULTS_DIR = ML_DIR / "docs" / "research"
DEV_RATIO = 0.85
SEQ_LEN = 32
HORIZON = 3
FEE = 0.0005
BUY = 2

TARGET_SPEC = ActionTargetSpec(mode="triple_barrier", barrier_horizon=HORIZON, barrier_vol_window=12,
                               barrier_up_k=1.25, barrier_down_k=1.25)

VAL = {
    "SBER": {"return": 0.1896, "sharpe": 14.95, "win": 0.732, "trades": 41},
    "LKOH": {"return": 0.1173, "sharpe": 5.42, "win": 0.568, "trades": 81},
}


def buy_and_hold(close, lo, hi):
    seg = close[lo:hi]
    rets = np.diff(seg) / seg[:-1]
    eq = np.concatenate([[1.0], np.cumprod(1 + rets)])
    peak = np.maximum.accumulate(eq)
    dd = float(np.min(eq / np.where(peak < 1e-12, 1.0, peak)) - 1)
    shp = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(HOURS_PER_YEAR)) if rets.std() > 0 else 0.0
    return {"return": float(eq[-1] - 1), "sharpe": shp, "max_drawdown": dd}


def evaluate(ticker: str) -> dict:
    print("\n" + "=" * 72)
    print(f"  {ticker}")
    print("=" * 72)

    df = load_candles("data/raw", ticker=ticker, timeframe="1H", tz_aware=True).sort_values("begin").reset_index(drop=True)
    n = len(df)
    dev_end = int(n * DEV_RATIO)
    begin = pd.to_datetime(df["begin"])
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()

    # ── PRE-FLIGHT ─────────────────────────────────────────────────────────────
    art_dir = resolve_artifact_dir(ticker)
    assert art_dir is not None, f"No artifact for {ticker}"
    art = load_research_artifact(art_dir)
    feat = _lstm_feature_matrix(art, df)                      # uses the SAME builder as deployment
    norm_mean, norm_std = art.feature_mean, art.feature_std
    assert feat.shape[1] == len(norm_mean), f"feat dim {feat.shape[1]} != artifact {len(norm_mean)}"
    det = triple_barrier_details(df, TARGET_SPEC)
    labels = make_research_action_targets(df, TARGET_SPEC).labels

    test_lo = dev_end
    test_hi = n - HORIZON
    test_targets = [t for t in range(test_lo, test_hi) if labels[t] != -1]
    assert min(test_targets) >= dev_end, "LEAKAGE: a test target falls inside the training span!"
    print(f"  rows={n}  dev_end={dev_end} (first {DEV_RATIO:.0%})  test=[{test_lo}:{n}]")
    print(f"  test period: {begin.iloc[test_lo]}  ..  {begin.iloc[n-1]}")
    print(f"  test candles with valid labels: {len(test_targets)}")
    print(f"  artifact: {art_dir.name} | input_dim={feat.shape[1]} | "
          f"ortho_groups={art.feature_config.get('orthogonal_groups')} | frozen normalisation: YES")

    # FROZEN normalisation (never recomputed on test)
    feat_norm = np.nan_to_num((feat - norm_mean) / norm_std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Faithfulness vs the live router (one test candle)
    probe_t = test_targets[len(test_targets) // 2]
    win = feat_norm[probe_t - SEQ_LEN:probe_t][None, :, :]
    art.model.eval()
    with torch.no_grad():
        my_proba = torch.softmax(art.model(torch.from_numpy(win)), 1).numpy()[0]
    # Long batch (250) so the router's rolling/ewm features converge to full-series values.
    sub = df.iloc[probe_t - 249:probe_t + 1]                  # 250-candle batch ending at probe_t (inclusive)
    candles = [{"begin": b.isoformat(), "open": float(o), "high": float(h), "low": float(l),
                "close": float(c), "volume": float(v)}
               for b, o, h, l, c, v in zip(sub["begin"], sub["open"], sub["high"], sub["low"], sub["close"], sub["volume"])]
    rr = TickerModelRouter().predict(load_candle_batch_json({"ticker": ticker, "timeframe": "1H", "candles": candles}))
    router_buy = rr["probabilities"]["buy"]
    # router predicts label for the candle AFTER the batch's last (window = last 32 candles); align to probe_t+1
    win2 = feat_norm[probe_t + 1 - SEQ_LEN:probe_t + 1][None, :, :]
    with torch.no_grad():
        my_proba2 = torch.softmax(art.model(torch.from_numpy(win2)), 1).numpy()[0]
    faithful = abs(float(my_proba2[2]) - router_buy) < 1e-3
    print(f"  faithfulness vs router (buy proba): replicated={my_proba2[2]:.5f} router={router_buy:.5f} "
          f"match={faithful}")
    assert faithful, "Replicated inference does not match the deployed router!"

    # ── GATE: predict all test targets with the FROZEN artifact ────────────────
    windows = np.stack([feat_norm[t - SEQ_LEN:t] for t in test_targets]).astype(np.float32)
    with torch.no_grad():
        proba = torch.softmax(art.model(torch.from_numpy(windows)), 1).numpy()
    argmax, conf = proba.argmax(1), proba.max(1)

    up, dn, fut = det["upper_return"], det["lower_return"], det["future_return"]
    rets, holds, free_at = [], [], -1
    n_buy_signals = int(((argmax == BUY) & (conf > 0.50)).sum())
    for i, t in enumerate(test_targets):
        if t < free_at or argmax[i] != BUY or conf[i] <= 0.50:
            continue
        if begin.iloc[t].dayofweek >= 5 or t + HORIZON >= len(close):
            continue
        r, h, _ = long_stop_return(int(t), close, high, low, float(up[t]), float(dn[t]), float(fut[t]))
        rets.append(r); holds.append(h); free_at = t + int(np.ceil(h))
    bt = summarize(rets, holds)
    # max drawdown on the trade-equity curve (summarize doesn't provide it)
    if rets:
        eq = np.cumprod(1 + np.array(rets)); peak = np.maximum.accumulate(eq)
        bt["max_drawdown"] = float(np.min(eq / np.where(peak < 1e-12, 1.0, peak)) - 1)
    else:
        bt["max_drawdown"] = 0.0
    bh = buy_and_hold(close, test_lo, n)
    val = VAL[ticker]

    print(f"\n  --- TEST RESULT ({ticker}) ---")
    print(f"  conf>0.50 BUY signals on test: {n_buy_signals} | trades after weekend filter+cooldown: {bt['n']}")
    print(f"  TEST:  ret={bt['total_return']:+.2%}  Sharpe={bt['sharpe']:.2f}  win={bt['win_rate']:.1%}  "
          f"DD={bt['max_drawdown']:.2%}  mean/trade={bt['mean_ret']:+.3%}")
    print(f"  VAL:   ret={val['return']:+.2%}  Sharpe={val['sharpe']:.2f}  win={val['win']:.1%}  trades={val['trades']}")
    print(f"  Buy&Hold (test): ret={bh['return']:+.2%}  Sharpe={bh['sharpe']:.2f}  DD={bh['max_drawdown']:.2%}")

    return {
        "ticker": ticker, "artifact": art_dir.name,
        "test_period": [str(begin.iloc[test_lo]), str(begin.iloc[n - 1])],
        "test_candles_valid": len(test_targets),
        "conf_gt_050_buy_signals": n_buy_signals,
        "test_backtest": bt, "validation": val, "buy_and_hold_test": bh,
        "faithful_vs_router": bool(faithful),
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("#" * 72)
    print("#  FINAL LOCKED TEST-SET GATE — one-shot, irreversible")
    print("#  Frozen artifacts on the untouched last 15%. No tuning. Production rule.")
    print("#" * 72)
    results = {tk: evaluate(tk) for tk in ["SBER", "LKOH"]}

    out = {"experiment": "final_test_gate", "timestamp": ts, "git_branch": "ml-expirement",
           "rule": "BUY conf>0.50; hold 3h + lower-barrier stop; no take-profit; long-only; skip weekends; fee 0.05%",
           "dev_ratio": DEV_RATIO, "results": results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"FINAL_test_gate_{ts}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
