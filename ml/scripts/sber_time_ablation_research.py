"""
Time feature ablation for SBER H1 triple-barrier ExtraTrees.

Hypothesis:
    hour_cos/hour_sin/dow_cos/dow_sin dominate 58% of feature importance.
    Removing them forces the model to rely on price-structure features and
    may improve generalisation (or reveal how much of the baseline is pure
    intraday seasonality).

Method:
    Two feature sets compared side-by-side:
      - baseline  : all 27 continuous_regime features (frozen candidate)
      - no_time   : 23 features — baseline minus {hour_sin, hour_cos,
                    dow_sin, dow_cos}

    Walk-forward expanding-window validation:
      - n_rows used: train+val split (test never touched)
      - initial_train_size: 12000 candles
      - val_size: 2000 candles per fold
      - n_splits: 4 folds

    Seeds: [7, 42, 100] — aggregate across seeds.

    Model: ExtraTreesClassifier (n=300, depth=None, leaf=20, sqrt) — frozen candidate spec.
    Target: triple_barrier:h3:w12:up1.25:down1.25

Result saved to: ml/docs/research/sber_h1_time_ablation_2026-06-02.md
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
from sklearn.metrics import f1_score, classification_report

from src.data.load import load_candles
from src.data.split import time_split, walk_forward_ranges
from src.nlp.action_features import make_continuous_past_features
from src.nlp.targets import ActionTargetSpec, make_research_action_targets

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_MD = ML_DIR / "docs" / "research" / "sber_h1_time_ablation_2026-06-02.md"

TARGET_SPEC = ActionTargetSpec(
    mode="triple_barrier",
    barrier_horizon=3,
    barrier_vol_window=12,
    barrier_up_k=1.25,
    barrier_down_k=1.25,
)

MODEL_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=20,
    max_features="sqrt",
    n_jobs=-1,
)

SEEDS = [7, 42, 100]
TIME_FEATURES = {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}
LABEL_NAMES = ["SELL", "HOLD", "BUY"]

# Walk-forward config (train+val only — test excluded)
WF_INITIAL_TRAIN = 12_000
WF_VAL_SIZE = 2_000
WF_N_SPLITS = 4


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    df = load_candles(str(DATA_DIR), ticker="SBER", timeframe="1H")
    df["begin"] = pd.to_datetime(df["begin"], utc=True)
    df = df.sort_values("begin").reset_index(drop=True)

    # Compute targets for the full series
    result = make_research_action_targets(df, TARGET_SPEC)
    labels = result.labels  # shape (N,), -1 for rows without enough future data

    return df, labels


# ── Single fold evaluation ────────────────────────────────────────────────────

def eval_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> dict:
    model = ExtraTreesClassifier(random_state=seed, **MODEL_PARAMS)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    macro_f1 = f1_score(y_val, preds, average="macro")
    per_class = f1_score(y_val, preds, average=None, labels=[0, 1, 2])
    importances = model.feature_importances_

    return {
        "macro_f1": macro_f1,
        "sell_f1": per_class[0],
        "hold_f1": per_class[1],
        "buy_f1": per_class[2],
        "importances": importances,
    }


# ── Walk-forward across seeds ─────────────────────────────────────────────────

def run_experiment(
    feat_matrix: np.ndarray,
    feat_names: list[str],
    labels: np.ndarray,
    n_rows_trainval: int,
    experiment_name: str,
) -> dict:
    """Run walk-forward CV for a given feature matrix, return aggregate stats."""
    folds = walk_forward_ranges(
        n_rows_trainval,
        n_splits=WF_N_SPLITS,
        initial_train_size=WF_INITIAL_TRAIN,
        val_size=WF_VAL_SIZE,
    )

    all_results = []  # (fold_id, seed, metrics_dict)

    for fold in folds:
        train_idx = np.arange(fold.train_start, fold.train_end)
        val_idx = np.arange(fold.val_start, fold.val_end)

        # Filter out -1 labels
        train_valid = train_idx[labels[train_idx] != -1]
        val_valid = val_idx[labels[val_idx] != -1]

        if len(train_valid) < 500 or len(val_valid) < 50:
            print(f"  [WARN] fold {fold.fold_id}: too few valid samples, skipping")
            continue

        X_train = feat_matrix[train_valid]
        y_train = labels[train_valid]
        X_val = feat_matrix[val_valid]
        y_val = labels[val_valid]

        # Standardize using train statistics only (no leakage)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        X_train_s = (X_train - mean) / std
        X_val_s = (X_val - mean) / std
        X_train_s = np.nan_to_num(X_train_s)
        X_val_s = np.nan_to_num(X_val_s)

        for seed in SEEDS:
            metrics = eval_fold(X_train_s, y_train, X_val_s, y_val, seed)
            metrics["fold_id"] = fold.fold_id
            metrics["seed"] = seed
            metrics["train_size"] = len(train_valid)
            metrics["val_size"] = len(val_valid)
            all_results.append(metrics)

        fold_f1s = [r["macro_f1"] for r in all_results if r["fold_id"] == fold.fold_id]
        print(
            f"  fold {fold.fold_id}: train={len(train_valid):>5d}  val={len(val_valid):>5d}  "
            f"macro-F1 (seeds): {[f'{f:.4f}' for f in fold_f1s]}"
        )

    if not all_results:
        raise RuntimeError("No results collected — check data/fold config")

    macro_f1s = np.array([r["macro_f1"] for r in all_results])
    sell_f1s = np.array([r["sell_f1"] for r in all_results])
    hold_f1s = np.array([r["hold_f1"] for r in all_results])
    buy_f1s = np.array([r["buy_f1"] for r in all_results])

    # Worst fold: mean across seeds for each fold, then min
    fold_means = {}
    for r in all_results:
        fold_means.setdefault(r["fold_id"], []).append(r["macro_f1"])
    worst_fold_f1 = min(np.mean(v) for v in fold_means.values())

    # Top feature importances (averaged across all runs)
    mean_imp = np.mean([r["importances"] for r in all_results], axis=0)
    top_idx = np.argsort(mean_imp)[::-1][:15]

    time_imp = sum(mean_imp[i] for i, n in enumerate(feat_names) if n in TIME_FEATURES)

    return {
        "name": experiment_name,
        "n_features": len(feat_names),
        "mean_macro_f1": float(macro_f1s.mean()),
        "std_macro_f1": float(macro_f1s.std()),
        "worst_fold_f1": float(worst_fold_f1),
        "mean_sell_f1": float(sell_f1s.mean()),
        "mean_hold_f1": float(hold_f1s.mean()),
        "mean_buy_f1": float(buy_f1s.mean()),
        "time_feature_importance": float(time_imp),
        "top15_features": [(feat_names[i], float(mean_imp[i])) for i in top_idx],
        "n_folds_evaluated": len(fold_means),
        "n_seeds": len(SEEDS),
    }


# ── Report generation ─────────────────────────────────────────────────────────

def write_report(baseline: dict, no_time: dict) -> None:
    delta_mean = no_time["mean_macro_f1"] - baseline["mean_macro_f1"]
    delta_worst = no_time["worst_fold_f1"] - baseline["worst_fold_f1"]

    lines = [
        "# SBER H1 — Time Feature Ablation — 2026-06-02",
        "",
        "## Hypothesis",
        "hour_cos/hour_sin account for 58% of ExtraTrees feature importance in the frozen",
        "candidate. Removing them may force the model to learn price structure and improve",
        "generalisation, or may reveal that intraday seasonality is genuine signal.",
        "",
        "## Method",
        f"- Model: ExtraTreesClassifier (n=300, depth=None, leaf=20, max_features=sqrt)",
        f"- Target: triple_barrier:h3:w12:up1.25:down1.25",
        f"- Walk-forward: {WF_N_SPLITS} expanding folds, initial_train={WF_INITIAL_TRAIN}, val_size={WF_VAL_SIZE}",
        f"- Seeds: {SEEDS}",
        f"- Removed features (no_time): hour_sin, hour_cos, dow_sin, dow_cos",
        "",
        "## Results",
        "",
        "| Metric | Baseline (27 feat) | No time (23 feat) | Delta |",
        "|--------|-------------------|-------------------|-------|",
        f"| Val macro-F1 (mean ± std) | {baseline['mean_macro_f1']:.4f} ± {baseline['std_macro_f1']:.4f} | {no_time['mean_macro_f1']:.4f} ± {no_time['std_macro_f1']:.4f} | {delta_mean:+.4f} |",
        f"| Worst fold F1 | {baseline['worst_fold_f1']:.4f} | {no_time['worst_fold_f1']:.4f} | {delta_worst:+.4f} |",
        f"| SELL F1 | {baseline['mean_sell_f1']:.4f} | {no_time['mean_sell_f1']:.4f} | {no_time['mean_sell_f1'] - baseline['mean_sell_f1']:+.4f} |",
        f"| HOLD F1 | {baseline['mean_hold_f1']:.4f} | {no_time['mean_hold_f1']:.4f} | {no_time['mean_hold_f1'] - baseline['mean_hold_f1']:+.4f} |",
        f"| BUY F1 | {baseline['mean_buy_f1']:.4f} | {no_time['mean_buy_f1']:.4f} | {no_time['mean_buy_f1'] - baseline['mean_buy_f1']:+.4f} |",
        f"| Time feature importance | {baseline['time_feature_importance']:.4f} | {no_time['time_feature_importance']:.4f} | — |",
        "",
        "## Top-15 Feature Importances",
        "",
        "### Baseline (with time features)",
        "",
        "| Rank | Feature | Importance |",
        "|------|---------|-----------|",
    ]
    for rank, (name, imp) in enumerate(baseline["top15_features"], 1):
        marker = " ★" if name in TIME_FEATURES else ""
        lines.append(f"| {rank} | {name}{marker} | {imp:.4f} |")

    lines += [
        "",
        "### No-time (without hour/dow features)",
        "",
        "| Rank | Feature | Importance |",
        "|------|---------|-----------|",
    ]
    for rank, (name, imp) in enumerate(no_time["top15_features"], 1):
        lines.append(f"| {rank} | {name} | {imp:.4f} |")

    lines += [
        "",
        "## Conclusion",
    ]

    if abs(delta_mean) < 0.005:
        conclusion = (
            f"Removing time features had negligible impact on macro-F1 "
            f"(Δ={delta_mean:+.4f}). Time features capture genuine intraday "
            f"seasonality that is not redundant with price features."
        )
    elif delta_mean > 0.005:
        conclusion = (
            f"Removing time features IMPROVED macro-F1 by {delta_mean:+.4f}. "
            f"Time features were causing the model to overfit to intraday patterns "
            f"at the expense of generalisation. Recommend using no_time feature set going forward."
        )
    else:
        conclusion = (
            f"Removing time features HURT macro-F1 by {delta_mean:.4f}. "
            f"Intraday seasonality (hour_cos/sin) is genuine signal. "
            f"Consider keeping time features but reducing their dominance via "
            f"feature importance capping or regularisation."
        )

    lines.append(conclusion)
    lines.append("")
    lines.append(f"Next step: {'calibration (Step 2)' if delta_mean <= 0 else 'use no_time features in Step 3 Word2Vec experiment'}.")

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to {OUTPUT_MD}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading SBER 1H data...")
    df, labels = load_data()
    print(f"  Total candles: {len(df)}")
    print(f"  Valid labels:  {(labels != -1).sum()}")

    # Exclude test split from walk-forward (keep only train+val)
    _, val_df, _ = time_split(df, 0.70, 0.15)
    n_trainval = len(df) - (len(df) - (df.index.get_loc(val_df.index[-1]) + 1))
    # Simpler: trainval rows = first 70%+15% = 85%
    n_trainval = int(len(df) * 0.85)
    print(f"  Using first {n_trainval} rows (train+val, test excluded)")

    print("\nBuilding feature matrix...")
    feat_matrix, feat_names = make_continuous_past_features(df)
    print(f"  Feature shape: {feat_matrix.shape}")
    print(f"  Features: {feat_names}")

    # Baseline: all features
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: baseline (all features, n={})".format(len(feat_names)))
    print("="*60)
    baseline = run_experiment(feat_matrix, feat_names, labels, n_trainval, "baseline")

    # No-time: drop time features
    time_cols_idx = [i for i, n in enumerate(feat_names) if n in TIME_FEATURES]
    keep_idx = [i for i in range(len(feat_names)) if i not in time_cols_idx]
    feat_matrix_no_time = feat_matrix[:, keep_idx]
    feat_names_no_time = [feat_names[i] for i in keep_idx]
    print(f"\nDropping time features: {[feat_names[i] for i in time_cols_idx]}")

    print(f"\n{'='*60}")
    print("EXPERIMENT 2: no_time (n={})".format(len(feat_names_no_time)))
    print("="*60)
    no_time = run_experiment(feat_matrix_no_time, feat_names_no_time, labels, n_trainval, "no_time")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"  baseline  macro-F1: {baseline['mean_macro_f1']:.4f} ± {baseline['std_macro_f1']:.4f}  (worst: {baseline['worst_fold_f1']:.4f})")
    print(f"  no_time   macro-F1: {no_time['mean_macro_f1']:.4f} ± {no_time['std_macro_f1']:.4f}  (worst: {no_time['worst_fold_f1']:.4f})")
    delta = no_time["mean_macro_f1"] - baseline["mean_macro_f1"]
    print(f"  Delta:    {delta:+.4f}")
    print(f"\n  baseline  time importance: {baseline['time_feature_importance']:.4f}")
    print(f"  no_time   time importance: {no_time['time_feature_importance']:.4f}")

    write_report(baseline, no_time)


if __name__ == "__main__":
    main()
