"""
Probability calibration experiment for SBER H1 triple-barrier ExtraTrees.

Hypothesis:
    ExtraTrees predict_proba returns uncalibrated scores. Only 17% of predictions
    have confidence > 0.5, and max confidence is ~0.54. Isotonic calibration on
    held-out data should spread probabilities more meaningfully without losing F1.

Method:
    Walk-forward validation with calibration split inside each fold:
      - Main train:   fold.train_start .. train_end - cal_size
      - Cal set:      last cal_size rows of train (never seen by ExtraTrees during fit)
      - Val:          fold.val_start .. fold.val_end
    Two conditions compared:
      - uncalibrated: ExtraTrees.predict_proba (raw)
      - calibrated:   CalibratedClassifierCV(cv='prefit', method='isotonic')
                      fitted on cal set after ExtraTrees is trained

    Cal set size: 2000 rows (15% of initial_train_size = 12000 → 10200 main + 1800 cal).
    Walk-forward: 4 folds, initial_train=12000, val_size=2000.
    Seeds: [7, 42, 100].

    ECE = Expected Calibration Error (15 bins, weighted average).
    Confidence distribution: % with max-prob > {0.40, 0.45, 0.50}.

Result saved to: ml/docs/research/sber_h1_calibration_2026-06-02.md
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
sys.path.insert(0, str(ML_DIR))

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.data.load import load_candles
from src.data.split import time_split, walk_forward_ranges
from src.nlp.action_features import make_continuous_past_features
from src.nlp.targets import ActionTargetSpec, make_research_action_targets

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_MD = ML_DIR / "docs" / "research" / "sber_h1_calibration_2026-06-02.md"

TARGET_SPEC = ActionTargetSpec(
    mode="triple_barrier",
    barrier_horizon=3,
    barrier_vol_window=12,
    barrier_up_k=1.25,
    barrier_down_k=1.25,
)
MODEL_PARAMS = dict(n_estimators=300, max_depth=None, min_samples_leaf=20, max_features="sqrt", n_jobs=-1)
SEEDS = [7, 42, 100]
LABEL_NAMES = ["SELL", "HOLD", "BUY"]

WF_INITIAL_TRAIN = 12_000
WF_VAL_SIZE = 2_000
WF_N_SPLITS = 4
CAL_SIZE = 1_800   # held-out calibration set size within training window


# ── ECE metric ────────────────────────────────────────────────────────────────

def ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error averaged over classes (one-vs-rest)."""
    n_classes = proba.shape[1]
    errors = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        total = len(y_bin)
        err = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (proba[:, c] >= lo) & (proba[:, c] < hi)
            if mask.sum() == 0:
                continue
            frac_pos = y_bin[mask].mean()
            mean_conf = proba[:, c][mask].mean()
            err += (mask.sum() / total) * abs(frac_pos - mean_conf)
        errors.append(err)
    return float(np.mean(errors))


def confidence_pct(proba: np.ndarray, threshold: float) -> float:
    return float((proba.max(axis=1) > threshold).mean())


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    df = load_candles(str(DATA_DIR), ticker="SBER", timeframe="1H")
    df["begin"] = pd.to_datetime(df["begin"], utc=True)
    df = df.sort_values("begin").reset_index(drop=True)
    result = make_research_action_targets(df, TARGET_SPEC)
    return df, result.labels


# ── Calibration helpers ───────────────────────────────────────────────────────

def _isotonic_calibrate(proba_cal: np.ndarray, y_cal: np.ndarray, proba_test: np.ndarray) -> np.ndarray:
    """Per-class one-vs-rest isotonic calibration."""
    n_classes = proba_cal.shape[1]
    calibrated = np.zeros_like(proba_test, dtype=float)
    for c in range(n_classes):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(proba_cal[:, c], (y_cal == c).astype(float))
        calibrated[:, c] = ir.predict(proba_test[:, c])
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    return calibrated / row_sums


def _platt_calibrate(proba_cal: np.ndarray, y_cal: np.ndarray, proba_test: np.ndarray) -> np.ndarray:
    """Per-class Platt scaling (sigmoid): logistic regression on raw probability scores.

    Has only 2 parameters per class — far less overfitting risk than isotonic
    on small calibration sets (~600 samples per class).
    """
    n_classes = proba_cal.shape[1]
    calibrated = np.zeros_like(proba_test, dtype=float)
    for c in range(n_classes):
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        lr.fit(proba_cal[:, c].reshape(-1, 1), (y_cal == c).astype(int))
        calibrated[:, c] = lr.predict_proba(proba_test[:, c].reshape(-1, 1))[:, 1]
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    return calibrated / row_sums


# ── Single fold ───────────────────────────────────────────────────────────────

def run_fold(
    X_main: np.ndarray, y_main: np.ndarray,
    X_cal: np.ndarray, y_cal: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    seed: int,
) -> dict:
    model = ExtraTreesClassifier(random_state=seed, **MODEL_PARAMS)
    model.fit(X_main, y_main)

    proba_uncal = model.predict_proba(X_val)
    preds_uncal = model.predict(X_val)
    macro_uncal = f1_score(y_val, preds_uncal, average="macro")

    proba_cal_raw = model.predict_proba(X_cal)
    proba_val_raw = model.predict_proba(X_val)

    # Isotonic (non-parametric, may overfit on small cal set)
    proba_iso = _isotonic_calibrate(proba_cal_raw, y_cal, proba_val_raw)
    preds_iso = np.argmax(proba_iso, axis=1)
    macro_iso = f1_score(y_val, preds_iso, average="macro")

    # Platt / sigmoid (2 params per class, more stable on small cal set)
    proba_platt = _platt_calibrate(proba_cal_raw, y_cal, proba_val_raw)
    preds_platt = np.argmax(proba_platt, axis=1)
    macro_platt = f1_score(y_val, preds_platt, average="macro")

    def _pc(proba, t):
        return confidence_pct(proba, t)

    return {
        "macro_uncal": macro_uncal,
        "macro_iso": macro_iso,
        "macro_platt": macro_platt,
        "ece_uncal": ece(y_val, proba_uncal),
        "ece_iso": ece(y_val, proba_iso),
        "ece_platt": ece(y_val, proba_platt),
        "conf40_uncal": _pc(proba_uncal, 0.40), "conf45_uncal": _pc(proba_uncal, 0.45), "conf50_uncal": _pc(proba_uncal, 0.50),
        "conf40_iso": _pc(proba_iso, 0.40), "conf45_iso": _pc(proba_iso, 0.45), "conf50_iso": _pc(proba_iso, 0.50),
        "conf40_platt": _pc(proba_platt, 0.40), "conf45_platt": _pc(proba_platt, 0.45), "conf50_platt": _pc(proba_platt, 0.50),
        "sell_f1_iso": f1_score(y_val, preds_iso, average=None, labels=[0, 1, 2])[0],
        "hold_f1_iso": f1_score(y_val, preds_iso, average=None, labels=[0, 1, 2])[1],
        "buy_f1_iso": f1_score(y_val, preds_iso, average=None, labels=[0, 1, 2])[2],
        "sell_f1_platt": f1_score(y_val, preds_platt, average=None, labels=[0, 1, 2])[0],
        "hold_f1_platt": f1_score(y_val, preds_platt, average=None, labels=[0, 1, 2])[1],
        "buy_f1_platt": f1_score(y_val, preds_platt, average=None, labels=[0, 1, 2])[2],
        "_proba_uncal": proba_uncal,
        "_proba_iso": proba_iso,
        "_proba_platt": proba_platt,
        "_y_val": y_val,
    }


def run_walk_forward(feat_matrix: np.ndarray, feat_names: list[str], labels: np.ndarray, n_trainval: int) -> dict:
    folds = walk_forward_ranges(
        n_trainval, n_splits=WF_N_SPLITS, initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE
    )

    all_results = []
    all_proba_uncal = []  # collect last seed's val probas for overall calibration plot
    all_proba_cal = []
    all_y_val = []

    for fold in folds:
        train_idx = np.arange(fold.train_start, fold.train_end)
        val_idx = np.arange(fold.val_start, fold.val_end)
        train_valid = train_idx[labels[train_idx] != -1]
        val_valid = val_idx[labels[val_idx] != -1]

        if len(train_valid) < CAL_SIZE + 500:
            print(f"  [WARN] fold {fold.fold_id}: skipping (too few samples)")
            continue

        # Split train into main + cal
        main_idx = train_valid[:-CAL_SIZE]
        cal_idx = train_valid[-CAL_SIZE:]

        X_train_full = feat_matrix[train_valid]
        # Standardize on main train portion only
        X_main_raw = feat_matrix[main_idx]
        mean = X_main_raw.mean(axis=0)
        std = X_main_raw.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)

        def standardize(X):
            return np.nan_to_num((X - mean) / std)

        X_main_s = standardize(feat_matrix[main_idx])
        X_cal_s = standardize(feat_matrix[cal_idx])
        X_val_s = standardize(feat_matrix[val_valid])

        y_main = labels[main_idx]
        y_cal = labels[cal_idx]
        y_val = labels[val_valid]

        fold_f1_uncal = []
        fold_f1_iso = []
        fold_f1_platt = []
        last_seed_result = None

        for seed in SEEDS:
            res = run_fold(X_main_s, y_main, X_cal_s, y_cal, X_val_s, y_val, seed)
            all_results.append({**{k: v for k, v in res.items() if not k.startswith("_")},
                                  "fold_id": fold.fold_id, "seed": seed})
            fold_f1_uncal.append(res["macro_uncal"])
            fold_f1_iso.append(res["macro_iso"])
            fold_f1_platt.append(res["macro_platt"])
            last_seed_result = res

        print(
            f"  fold {fold.fold_id}: main={len(main_idx):>5d}  cal={len(cal_idx)}  val={len(val_valid):>5d}  "
            f"uncal={[f'{f:.4f}' for f in fold_f1_uncal]}  "
            f"iso={[f'{f:.4f}' for f in fold_f1_iso]}  "
            f"platt={[f'{f:.4f}' for f in fold_f1_platt]}"
        )

        if last_seed_result is not None:
            all_proba_uncal.append(last_seed_result["_proba_uncal"])
            all_proba_cal.append(last_seed_result["_proba_iso"])
            all_y_val.append(last_seed_result["_y_val"])

    if not all_results:
        raise RuntimeError("No results collected")

    def mean_std(key):
        vals = np.array([r[key] for r in all_results])
        return float(vals.mean()), float(vals.std())

    # Calibration curves aggregated across all val folds (last seed each)
    agg_proba_uncal = np.vstack(all_proba_uncal)
    agg_proba_cal = np.vstack(all_proba_cal)
    agg_y = np.concatenate(all_y_val)

    cal_curves = {}
    for c, name in enumerate(LABEL_NAMES):
        prob_true_u, prob_pred_u = calibration_curve((agg_y == c).astype(int), agg_proba_uncal[:, c], n_bins=8, strategy="quantile")
        prob_true_c, prob_pred_c = calibration_curve((agg_y == c).astype(int), agg_proba_cal[:, c], n_bins=8, strategy="quantile")
        cal_curves[name] = {
            "uncal_mean_conf": prob_pred_u.tolist(),
            "uncal_frac_pos": prob_true_u.tolist(),
            "cal_mean_conf": prob_pred_c.tolist(),
            "cal_frac_pos": prob_true_c.tolist(),
        }

    mn_u, st_u = mean_std("macro_uncal")
    mn_iso, st_iso = mean_std("macro_iso")
    mn_platt, st_platt = mean_std("macro_platt")
    worst_fold = {}
    for r in all_results:
        for k in ("macro_uncal", "macro_iso", "macro_platt"):
            worst_fold.setdefault((r["fold_id"], k), []).append(r[k])

    def worst(key):
        by_fold = {}
        for r in all_results:
            by_fold.setdefault(r["fold_id"], []).append(r[key])
        return min(np.mean(v) for v in by_fold.values())

    return {
        "macro_uncal_mean": mn_u, "macro_uncal_std": st_u,
        "macro_iso_mean": mn_iso, "macro_iso_std": st_iso,
        "macro_platt_mean": mn_platt, "macro_platt_std": st_platt,
        "worst_fold_uncal": worst("macro_uncal"),
        "worst_fold_iso": worst("macro_iso"),
        "worst_fold_platt": worst("macro_platt"),
        "ece_uncal": mean_std("ece_uncal")[0],
        "ece_iso": mean_std("ece_iso")[0],
        "ece_platt": mean_std("ece_platt")[0],
        "conf40_uncal": mean_std("conf40_uncal")[0], "conf45_uncal": mean_std("conf45_uncal")[0], "conf50_uncal": mean_std("conf50_uncal")[0],
        "conf40_iso": mean_std("conf40_iso")[0], "conf45_iso": mean_std("conf45_iso")[0], "conf50_iso": mean_std("conf50_iso")[0],
        "conf40_platt": mean_std("conf40_platt")[0], "conf45_platt": mean_std("conf45_platt")[0], "conf50_platt": mean_std("conf50_platt")[0],
        "sell_f1_iso": mean_std("sell_f1_iso")[0],
        "hold_f1_iso": mean_std("hold_f1_iso")[0],
        "buy_f1_iso": mean_std("buy_f1_iso")[0],
        "sell_f1_platt": mean_std("sell_f1_platt")[0],
        "hold_f1_platt": mean_std("hold_f1_platt")[0],
        "buy_f1_platt": mean_std("buy_f1_platt")[0],
        "calibration_curves": cal_curves,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(r: dict) -> None:
    lines = [
        "# SBER H1 — Probability Calibration — 2026-06-02",
        "",
        "## Hypothesis",
        "ExtraTrees predict_proba returns uncalibrated scores. Isotonic/Platt calibration",
        "on held-out data will spread probabilities without hurting F1.",
        "",
        "## Method",
        "- Model: ExtraTreesClassifier (n=300, depth=None, leaf=20, sqrt) frozen candidate",
        "- Target: triple_barrier:h3:w12:up1.25:down1.25",
        f"- Walk-forward: {WF_N_SPLITS} folds, initial_train={WF_INITIAL_TRAIN}, val={WF_VAL_SIZE}",
        f"- Cal set: last {CAL_SIZE} rows of each fold's train window (never seen by ExtraTrees)",
        "- Isotonic: per-class IsotonicRegression (non-parametric, ~600 cal samples per class)",
        "- Platt: per-class LogisticRegression on raw scores (2 params, less overfitting risk)",
        f"- Seeds: {SEEDS}",
        "",
        "## Results — F1 and Calibration Quality",
        "",
        "| Metric | Uncalibrated | Isotonic | Platt |",
        "|--------|-------------|----------|-------|",
        f"| Val macro-F1 (mean ± std) | {r['macro_uncal_mean']:.4f} ± {r['macro_uncal_std']:.4f} | {r['macro_iso_mean']:.4f} ± {r['macro_iso_std']:.4f} | {r['macro_platt_mean']:.4f} ± {r['macro_platt_std']:.4f} |",
        f"| Worst fold F1 | {r['worst_fold_uncal']:.4f} | {r['worst_fold_iso']:.4f} | {r['worst_fold_platt']:.4f} |",
        f"| ECE (↓ better) | {r['ece_uncal']:.4f} | {r['ece_iso']:.4f} | {r['ece_platt']:.4f} |",
        "",
        "## Results — Confidence Coverage",
        "",
        "| Threshold | Uncalibrated | Isotonic | Platt |",
        "|-----------|-------------|----------|-------|",
        f"| > 0.40 | {r['conf40_uncal']:.1%} | {r['conf40_iso']:.1%} | {r['conf40_platt']:.1%} |",
        f"| > 0.45 | {r['conf45_uncal']:.1%} | {r['conf45_iso']:.1%} | {r['conf45_platt']:.1%} |",
        f"| > 0.50 | {r['conf50_uncal']:.1%} | {r['conf50_iso']:.1%} | {r['conf50_platt']:.1%} |",
        "",
        "## Per-class F1",
        "",
        "| Class | Isotonic | Platt |",
        "|-------|----------|-------|",
        f"| SELL | {r['sell_f1_iso']:.4f} | {r['sell_f1_platt']:.4f} |",
        f"| HOLD | {r['hold_f1_iso']:.4f} | {r['hold_f1_platt']:.4f} |",
        f"| BUY | {r['buy_f1_iso']:.4f} | {r['buy_f1_platt']:.4f} |",
        "",
        "## Calibration Curves — Isotonic (aggregated val folds, quantile bins)",
        "",
        "Ideal: mean_conf == frac_pos (on diagonal). Underconfident: curve below diagonal.",
        "",
    ]

    for cls_name, curves in r["calibration_curves"].items():
        lines.append(f"### {cls_name}")
        lines.append("")
        lines.append("| Bin | Uncal conf | Uncal frac_pos | Cal (iso) conf | Cal frac_pos |")
        lines.append("|-----|-----------|----------------|---------------|--------------|")
        for i in range(len(curves["uncal_mean_conf"])):
            lines.append(
                f"| {i+1} | {curves['uncal_mean_conf'][i]:.3f} | {curves['uncal_frac_pos'][i]:.3f} "
                f"| {curves['cal_mean_conf'][i]:.3f} | {curves['cal_frac_pos'][i]:.3f} |"
            )
        lines.append("")

    lines += ["## Conclusion and Decision", ""]

    iso_delta_f1 = r["macro_iso_mean"] - r["macro_uncal_mean"]
    platt_delta_f1 = r["macro_platt_mean"] - r["macro_uncal_mean"]
    iso_delta_ece = r["ece_iso"] - r["ece_uncal"]
    platt_delta_ece = r["ece_platt"] - r["ece_uncal"]

    lines.append(f"**Isotonic**: F1 {iso_delta_f1:+.4f}, ECE {iso_delta_ece:+.4f}, conf>0.5: {r['conf50_uncal']:.1%} → {r['conf50_iso']:.1%}")
    lines.append(f"**Platt**:    F1 {platt_delta_f1:+.4f}, ECE {platt_delta_ece:+.4f}, conf>0.5: {r['conf50_uncal']:.1%} → {r['conf50_platt']:.1%}")
    lines.append("")

    if abs(iso_delta_f1) < 0.005 and iso_delta_ece < -0.003:
        decision = "Isotonic: SUCCESS — minimal F1 cost, better calibration. Add to artifact."
    elif abs(platt_delta_f1) < 0.005 and platt_delta_ece < -0.003:
        decision = "Platt: SUCCESS — minimal F1 cost, better calibration. Add to artifact."
    else:
        best_method = "Platt" if abs(platt_delta_f1) < abs(iso_delta_f1) else "Isotonic"
        decision = (
            f"Neither method improves ECE significantly while preserving F1.\n\n"
            f"**Key finding**: Uncalibrated ECE={r['ece_uncal']:.4f} is already reasonable for a 3-class problem.\n"
            f"ExtraTrees probabilities are not as miscalibrated as expected.\n\n"
            f"**Decision**: Keep uncalibrated model. Add calibration only as optional post-processing step\n"
            f"in risk_manager when threshold-based filtering is needed.\n"
            f"Set `probabilities_calibrated: false` in artifact metadata (truthful).\n\n"
            f"Better path: improve signal quality first (Word2Vec embeddings, Step 3),\n"
            f"then calibrate the stronger model."
        )

    lines.append(decision)
    lines.append("")
    lines.append("Next step: Step 3 — Word2Vec candle embeddings (primary quality improvement).")

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to {OUTPUT_MD}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading SBER 1H data...")
    df, labels = load_data()
    print(f"  Total candles: {len(df)}")

    n_trainval = int(len(df) * 0.85)
    print(f"  Using first {n_trainval} rows (train+val, test excluded)")

    print("Building feature matrix...")
    feat_matrix, feat_names = make_continuous_past_features(df)
    print(f"  Feature shape: {feat_matrix.shape}")

    print(f"\n{'='*60}")
    print("Walk-forward calibration experiment (Isotonic vs Platt)")
    print("="*60)
    results = run_walk_forward(feat_matrix, feat_names, labels, n_trainval)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"  Uncalibrated  macro-F1: {results['macro_uncal_mean']:.4f} ± {results['macro_uncal_std']:.4f}  ECE={results['ece_uncal']:.4f}  conf>0.5={results['conf50_uncal']:.1%}")
    print(f"  Isotonic      macro-F1: {results['macro_iso_mean']:.4f} ± {results['macro_iso_std']:.4f}  ECE={results['ece_iso']:.4f}  conf>0.5={results['conf50_iso']:.1%}")
    print(f"  Platt         macro-F1: {results['macro_platt_mean']:.4f} ± {results['macro_platt_std']:.4f}  ECE={results['ece_platt']:.4f}  conf>0.5={results['conf50_platt']:.1%}")

    write_report(results)


if __name__ == "__main__":
    main()
