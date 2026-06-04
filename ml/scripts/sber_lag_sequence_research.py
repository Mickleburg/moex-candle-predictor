"""
Lag-sequence feature experiment for SBER H1 triple-barrier ExtraTrees.

Hypothesis:
    The current continuous_regime feature set captures a 27-dim snapshot at time t.
    Cumulative returns (ret_3, ret_6...) lose trajectory shape: a 3h return of +0.5%
    could be three equal steps up OR one big spike then reversal.
    Individual hourly lag returns expose the SHAPE of price movement, letting ExtraTrees
    find patterns like '3 bearish hours followed by volume spike' = reversal signal.

Conditions (walk-forward, 4 folds, seeds=[7,42,100]):
    1. baseline    — 27 continuous_regime features
    2. lag_only    — 26 lag-sequence features (trajectory shape, multi-day, streaks)
    3. combined    — 27 + 26 = 53 features

Baseline walk-forward F1: 0.4738 (from time-ablation experiment).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
sys.path.insert(0, str(ML_DIR))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

from src.data.load import load_candles
from src.data.split import time_split, walk_forward_ranges
from src.nlp.action_features import make_continuous_past_features, make_lag_sequence_features
from src.nlp.targets import ActionTargetSpec, make_research_action_targets

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_MD = ML_DIR / "docs" / "research" / "sber_h1_lag_sequence_2026-06-03.md"

TARGET_SPEC = ActionTargetSpec(
    mode="triple_barrier", barrier_horizon=3, barrier_vol_window=12,
    barrier_up_k=1.25, barrier_down_k=1.25,
)
MODEL_PARAMS = dict(n_estimators=300, max_depth=None, min_samples_leaf=20,
                    max_features="sqrt", n_jobs=-1)
SEEDS = [7, 42, 100]
WF_INITIAL_TRAIN = 12_000
WF_VAL_SIZE = 2_000
WF_N_SPLITS = 4


def load_data():
    df = load_candles(str(DATA_DIR), ticker="SBER", timeframe="1H")
    df["begin"] = pd.to_datetime(df["begin"], utc=True)
    df = df.sort_values("begin").reset_index(drop=True)
    labels = make_research_action_targets(df, TARGET_SPEC).labels
    return df, labels


def standardize(X_tr, X_ot):
    m = X_tr.mean(axis=0); s = X_tr.std(axis=0)
    s = np.where(s < 1e-12, 1.0, s)
    return np.nan_to_num((X_tr - m) / s), np.nan_to_num((X_ot - m) / s)


def run_experiment(feat_matrix, feat_names, labels, n_trainval, name):
    folds = walk_forward_ranges(
        n_trainval, n_splits=WF_N_SPLITS,
        initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE
    )
    results = []
    for fold in folds:
        tr = np.arange(fold.train_start, fold.train_end)
        va = np.arange(fold.val_start, fold.val_end)
        tr = tr[labels[tr] != -1]; va = va[labels[va] != -1]
        if len(tr) < 500 or len(va) < 50:
            continue
        X_tr_s, X_va_s = standardize(feat_matrix[tr], feat_matrix[va])
        y_tr = labels[tr]; y_va = labels[va]
        fold_f1 = []
        fold_imps = []
        for seed in SEEDS:
            m = ExtraTreesClassifier(random_state=seed, **MODEL_PARAMS)
            m.fit(X_tr_s, y_tr)
            preds = m.predict(X_va_s)
            macro = f1_score(y_va, preds, average="macro")
            pc = f1_score(y_va, preds, average=None, labels=[0, 1, 2])
            results.append({"fold": fold.fold_id, "seed": seed, "macro": macro,
                             "sell": pc[0], "hold": pc[1], "buy": pc[2],
                             "imp": m.feature_importances_})
            fold_f1.append(macro)
            fold_imps.append(m.feature_importances_)
        print(f"  [{name}] fold {fold.fold_id}: {[f'{f:.4f}' for f in fold_f1]}")

    if not results:
        return {}
    macros = np.array([r["macro"] for r in results])
    fold_means = {}
    for r in results:
        fold_means.setdefault(r["fold"], []).append(r["macro"])
    mean_imp = np.mean([r["imp"] for r in results], axis=0)
    top15_idx = np.argsort(mean_imp)[::-1][:15]
    return {
        "name": name,
        "n_features": len(feat_names),
        "mean_f1": float(macros.mean()),
        "std_f1": float(macros.std()),
        "worst_fold": float(min(np.mean(v) for v in fold_means.values())),
        "sell_f1": float(np.mean([r["sell"] for r in results])),
        "hold_f1": float(np.mean([r["hold"] for r in results])),
        "buy_f1":  float(np.mean([r["buy"]  for r in results])),
        "top15": [(feat_names[i], float(mean_imp[i])) for i in top15_idx],
    }


def write_report(baseline, lag_only, combined):
    d1 = lag_only["mean_f1"] - baseline["mean_f1"]
    d2 = combined["mean_f1"] - baseline["mean_f1"]

    lines = [
        "# SBER H1 -- Lag Sequence Features -- 2026-06-03", "",
        "## Hypothesis",
        "Individual 1h lag returns expose price trajectory shape lost by cumulative returns.",
        "Patterns like '3 bearish hours + volume spike' become learnable by ExtraTrees.", "",
        "## Method",
        "- Model: ExtraTreesClassifier (n=300, depth=None, leaf=20, sqrt)",
        "- Target: triple_barrier:h3:w12:up1.25:down1.25",
        f"- Walk-forward: {WF_N_SPLITS} folds, initial_train={WF_INITIAL_TRAIN}, val={WF_VAL_SIZE}",
        f"- Seeds: {SEEDS}", "",
        "## New lag features (26 total)",
        "- lag_ret_2..lag_ret_10: individual 1h returns 2..10 steps back (9 features)",
        "- lag_body_1..5: signed candle body k steps back (5 features)",
        "- lag_vol_ratio_1..5: volume ratio k steps back (5 features)",
        "- ret_day_1..3: ~1/2/3 trading-day cumulative returns (3 features)",
        "- day_range, close_in_day_range: position within last-session high-low (2 features)",
        "- up_streak, down_streak: consecutive up/down hourly returns (2 features)", "",
        "## Results", "",
        "| Condition | Features | F1 mean+-std | Worst fold | SELL | HOLD | BUY | Delta |",
        "|-----------|---------|-------------|------------|------|------|-----|-------|",
        f"| baseline | {baseline['n_features']} | {baseline['mean_f1']:.4f}+-{baseline['std_f1']:.4f} | {baseline['worst_fold']:.4f} | {baseline['sell_f1']:.4f} | {baseline['hold_f1']:.4f} | {baseline['buy_f1']:.4f} | -- |",
        f"| lag_only | {lag_only['n_features']} | {lag_only['mean_f1']:.4f}+-{lag_only['std_f1']:.4f} | {lag_only['worst_fold']:.4f} | {lag_only['sell_f1']:.4f} | {lag_only['hold_f1']:.4f} | {lag_only['buy_f1']:.4f} | {d1:+.4f} |",
        f"| combined | {combined['n_features']} | {combined['mean_f1']:.4f}+-{combined['std_f1']:.4f} | {combined['worst_fold']:.4f} | {combined['sell_f1']:.4f} | {combined['hold_f1']:.4f} | {combined['buy_f1']:.4f} | {d2:+.4f} |",
        "", "## Top-15 Feature Importances (combined)", "",
        "| Rank | Feature | Importance |",
        "|------|---------|-----------|",
    ]
    for rank, (n, imp) in enumerate(combined["top15"], 1):
        tag = " [LAG]" if n.startswith("lag_") or n.startswith("ret_day") or n in ("day_range", "close_in_day_range", "up_streak", "down_streak") else ""
        lines.append(f"| {rank} | {n}{tag} | {imp:.4f} |")

    lines += ["", "## Conclusion", ""]
    if d2 > 0.01:
        verdict = (
            f"Lag features IMPROVED combined F1 by {d2:+.4f}. "
            f"Trajectory shape is meaningful signal for triple-barrier prediction.\n"
            f"Recommendation: replace continuous_regime with combined feature set as new frozen candidate.\n"
            f"Next: retrain artifact, run smoke tests, calibrate if F1 > 0.55."
        )
    elif d2 > 0:
        verdict = (
            f"Marginal improvement {d2:+.4f}. Trajectory features add some value.\n"
            f"Consider expanding lags further (n_lags=15, session lags) or moving to LSTM."
        )
    else:
        verdict = (
            f"Lag features did not improve F1 (delta={d2:+.4f}).\n"
            f"ExtraTrees cannot leverage temporal patterns even when lags are explicit.\n"
            f"This confirms LSTM is necessary: must move to Step 4 (sequence model)."
        )
    lines.append(verdict)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to {OUTPUT_MD}")


def main():
    print("Loading SBER 1H data...")
    df, labels = load_data()
    n_trainval = int(len(df) * 0.85)
    print(f"  {len(df)} candles, first {n_trainval} rows used (test excluded)")

    print("Building feature matrices...")
    X_cont, n_cont = make_continuous_past_features(df)
    X_lag, n_lag = make_lag_sequence_features(df, n_lags=10)
    X_comb = np.hstack([X_cont, X_lag])
    n_comb = n_cont + n_lag
    print(f"  continuous: {X_cont.shape}  lag: {X_lag.shape}  combined: {X_comb.shape}")

    print(f"\n{'='*60}\nBASELINE (continuous_regime, {len(n_cont)} features)\n{'='*60}")
    baseline = run_experiment(X_cont, n_cont, labels, n_trainval, "baseline")
    print(f"  => macro-F1: {baseline['mean_f1']:.4f} +- {baseline['std_f1']:.4f}  worst={baseline['worst_fold']:.4f}")

    print(f"\n{'='*60}\nLAG ONLY ({len(n_lag)} features)\n{'='*60}")
    lag_only = run_experiment(X_lag, n_lag, labels, n_trainval, "lag_only")
    print(f"  => macro-F1: {lag_only['mean_f1']:.4f} +- {lag_only['std_f1']:.4f}  worst={lag_only['worst_fold']:.4f}")

    print(f"\n{'='*60}\nCOMBINED (continuous + lag, {len(n_comb)} features)\n{'='*60}")
    combined = run_experiment(X_comb, n_comb, labels, n_trainval, "combined")
    print(f"  => macro-F1: {combined['mean_f1']:.4f} +- {combined['std_f1']:.4f}  worst={combined['worst_fold']:.4f}")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"  baseline: {baseline['mean_f1']:.4f}  lag_only: {lag_only['mean_f1']:.4f}  combined: {combined['mean_f1']:.4f}")
    print(f"  combined delta vs baseline: {combined['mean_f1'] - baseline['mean_f1']:+.4f}")

    write_report(baseline, lag_only, combined)


if __name__ == "__main__":
    main()
