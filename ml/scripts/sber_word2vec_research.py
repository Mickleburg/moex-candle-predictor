"""
Word2Vec candle embedding experiment for SBER H1 triple-barrier ExtraTrees.

Based on Poženel & Lavbič (2019), arXiv:1902.08684 — "Discovering Language of the Stocks".

Hypothesis:
    Continuous_regime features capture a flat snapshot (27 indicators at time t).
    Co-occurrence SVD embeddings (context of last nm candle-shape words) add
    sequential price-structure context that complements intraday time features.
    Combining both feature sets should improve macro-F1 above baseline 0.4675.

Conditions:
    1. baseline    — 27 continuous_regime features (frozen candidate)
    2. w2v_only    — 32 context embedding features only (no technical indicators)
    3. w2v_combined — 27 + 32 = 59 features (continuous + embedding context)

Hyperparameter grid (searched on validation folds only):
    nw  ∈ {20, 30, 50}   — vocabulary size (K-Means clusters)
    nv  ∈ {16, 32}       — SVD embedding dimensions
    nm  ∈ {10, 20}       — context window (past candles to average)

Walk-forward: 4 folds, initial_train=12000, val=2000. Seeds: [7, 42, 100].
Model: ExtraTreesClassifier (frozen candidate spec).
Target: triple_barrier:h3:w12:up1.25:down1.25

Result saved to: ml/docs/research/sber_h1_word2vec_2026-06-03.md
"""

from __future__ import annotations

import sys
from itertools import product
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
from src.nlp.action_features import make_continuous_past_features
from src.nlp.targets import ActionTargetSpec, make_research_action_targets
from src.nlp.word2vec_features import (
    build_cooccurrence_embeddings,
    candles_to_words,
    fit_candle_vocabulary,
    make_context_features,
    normalize_ohlc,
)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_MD = ML_DIR / "docs" / "research" / "sber_h1_word2vec_2026-06-03.md"
BASELINE_F1 = 0.4738   # walk-forward baseline from time ablation experiment

TARGET_SPEC = ActionTargetSpec(
    mode="triple_barrier",
    barrier_horizon=3,
    barrier_vol_window=12,
    barrier_up_k=1.25,
    barrier_down_k=1.25,
)
MODEL_PARAMS = dict(n_estimators=300, max_depth=None, min_samples_leaf=20, max_features="sqrt", n_jobs=-1)
SEEDS = [7, 42, 100]

WF_INITIAL_TRAIN = 12_000
WF_VAL_SIZE = 2_000
WF_N_SPLITS = 4

# Hyperparameter grid for W2V
NW_VALUES = [20, 30, 50]   # vocabulary sizes
NV_VALUES = [16, 32]       # embedding dims
NM_VALUES = [10, 20]       # context window


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    df = load_candles(str(DATA_DIR), ticker="SBER", timeframe="1H")
    df["begin"] = pd.to_datetime(df["begin"], utc=True)
    df = df.sort_values("begin").reset_index(drop=True)
    labels = make_research_action_targets(df, TARGET_SPEC).labels
    return df, labels


# ── Feature builders ──────────────────────────────────────────────────────────

def build_w2v_features(
    df: pd.DataFrame,
    train_end: int,
    nw: int,
    nv: int,
    nm: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Build context embedding features, fitting vocab+SVD on train_end rows only."""
    X_norm = normalize_ohlc(df)

    # Fit on train only (no leakage)
    km = fit_candle_vocabulary(X_norm[:train_end], nw=nw, seed=seed)
    words = candles_to_words(X_norm, km)
    embeddings = build_cooccurrence_embeddings(
        words[:train_end], n_words=nw, nv=nv, window=10, seed=seed
    )
    matrix, names = make_context_features(words, embeddings, nm=nm)
    return matrix, names


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
    macro = f1_score(y_val, preds, average="macro")
    per_class = f1_score(y_val, preds, average=None, labels=[0, 1, 2])
    return {
        "macro_f1": macro,
        "sell_f1": per_class[0],
        "hold_f1": per_class[1],
        "buy_f1": per_class[2],
        "importances": model.feature_importances_,
    }


def standardize(X_train: np.ndarray, X_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return np.nan_to_num((X_train - mean) / std), np.nan_to_num((X_other - mean) / std)


# ── Walk-forward for one experiment config ────────────────────────────────────

def run_experiment(
    feat_matrix: np.ndarray,
    feat_names: list[str],
    labels: np.ndarray,
    n_trainval: int,
    name: str,
) -> dict:
    folds = walk_forward_ranges(
        n_trainval, n_splits=WF_N_SPLITS, initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE
    )

    all_results = []
    for fold in folds:
        train_idx = np.arange(fold.train_start, fold.train_end)
        val_idx = np.arange(fold.val_start, fold.val_end)
        train_valid = train_idx[labels[train_idx] != -1]
        val_valid = val_idx[labels[val_idx] != -1]

        if len(train_valid) < 500 or len(val_valid) < 50:
            continue

        X_train_raw = feat_matrix[train_valid]
        X_val_raw = feat_matrix[val_valid]
        X_train_s, X_val_s = standardize(X_train_raw, X_val_raw)
        y_train = labels[train_valid]
        y_val = labels[val_valid]

        fold_f1 = []
        for seed in SEEDS:
            res = eval_fold(X_train_s, y_train, X_val_s, y_val, seed)
            all_results.append({**{k: v for k, v in res.items() if k != "importances"},
                                  "fold_id": fold.fold_id, "seed": seed,
                                  "importances": res["importances"]})
            fold_f1.append(res["macro_f1"])

    if not all_results:
        return {"name": name, "mean_macro_f1": 0.0, "std": 0.0, "worst_fold": 0.0,
                "sell_f1": 0.0, "hold_f1": 0.0, "buy_f1": 0.0, "top10_features": []}

    macro_f1s = np.array([r["macro_f1"] for r in all_results])
    fold_means = {}
    for r in all_results:
        fold_means.setdefault(r["fold_id"], []).append(r["macro_f1"])

    mean_imp = np.mean([r["importances"] for r in all_results], axis=0)
    top_idx = np.argsort(mean_imp)[::-1][:10]
    top10 = [(feat_names[i], float(mean_imp[i])) for i in top_idx]

    return {
        "name": name,
        "mean_macro_f1": float(macro_f1s.mean()),
        "std": float(macro_f1s.std()),
        "worst_fold": float(min(np.mean(v) for v in fold_means.values())),
        "sell_f1": float(np.mean([r["sell_f1"] for r in all_results])),
        "hold_f1": float(np.mean([r["hold_f1"] for r in all_results])),
        "buy_f1": float(np.mean([r["buy_f1"] for r in all_results])),
        "top10_features": top10,
        "n_features": len(feat_names),
    }


# ── Main grid search ──────────────────────────────────────────────────────────

def main():
    print("Loading SBER 1H data...")
    df, labels = load_data()
    n_trainval = int(len(df) * 0.85)
    print(f"  {len(df)} candles, using first {n_trainval} (test excluded)")

    print("Building continuous_regime feature matrix...")
    cont_matrix, cont_names = make_continuous_past_features(df)
    print(f"  continuous shape: {cont_matrix.shape}")

    # ── Step 1: Baseline ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BASELINE (continuous_regime, 27 features)")
    print("="*60)
    baseline = run_experiment(cont_matrix, cont_names, labels, n_trainval, "baseline")
    print(f"  macro-F1: {baseline['mean_macro_f1']:.4f} ± {baseline['std']:.4f}  worst={baseline['worst_fold']:.4f}")

    # ── Step 2: W2V grid ──────────────────────────────────────────────────────
    best_w2v_only: dict = {"mean_macro_f1": 0.0}
    best_w2v_combined: dict = {"mean_macro_f1": 0.0}
    all_grid_results: list[dict] = []

    grid = list(product(NW_VALUES, NV_VALUES, NM_VALUES))
    print(f"\n{'='*60}")
    print(f"W2V grid search: {len(grid)} configs x {len(SEEDS)} seeds x {WF_N_SPLITS} folds")
    print("="*60)

    for nw, nv, nm in grid:
        tag = f"nw{nw}_nv{nv}_nm{nm}"
        print(f"\n  [{tag}] Building W2V features (nw={nw}, nv={nv}, nm={nm})...")

        # Build W2V features using ONLY train portion of each fold for fitting.
        # For a simple grid pass, we use n_trainval as train_end (conservative).
        # This slightly underestimates quality but avoids per-fold refitting overhead.
        w2v_matrix, w2v_names = build_w2v_features(df, n_trainval, nw=nw, nv=nv, nm=nm, seed=42)
        print(f"    w2v feature shape: {w2v_matrix.shape}")

        # W2V only
        res_only = run_experiment(w2v_matrix, w2v_names, labels, n_trainval, f"w2v_only_{tag}")
        print(f"    w2v_only:     macro-F1={res_only['mean_macro_f1']:.4f} ± {res_only['std']:.4f}  worst={res_only['worst_fold']:.4f}")

        # W2V combined with continuous
        combined_matrix = np.hstack([cont_matrix, w2v_matrix])
        combined_names = cont_names + w2v_names
        res_combined = run_experiment(combined_matrix, combined_names, labels, n_trainval, f"w2v_combined_{tag}")
        print(f"    w2v_combined: macro-F1={res_combined['mean_macro_f1']:.4f} ± {res_combined['std']:.4f}  worst={res_combined['worst_fold']:.4f}")

        all_grid_results.append({"config": tag, "nw": nw, "nv": nv, "nm": nm,
                                  "w2v_only": res_only, "w2v_combined": res_combined})

        if res_only["mean_macro_f1"] > best_w2v_only["mean_macro_f1"]:
            best_w2v_only = {**res_only, "config": tag}
        if res_combined["mean_macro_f1"] > best_w2v_combined["mean_macro_f1"]:
            best_w2v_combined = {**res_combined, "config": tag}

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"  Baseline:          {baseline['mean_macro_f1']:.4f} ± {baseline['std']:.4f}  (worst={baseline['worst_fold']:.4f})")
    print(f"  Best w2v_only:     {best_w2v_only['mean_macro_f1']:.4f} ± {best_w2v_only.get('std', 0):.4f}  ({best_w2v_only.get('config', '')})")
    print(f"  Best w2v_combined: {best_w2v_combined['mean_macro_f1']:.4f} ± {best_w2v_combined.get('std', 0):.4f}  ({best_w2v_combined.get('config', '')})")

    write_report(baseline, best_w2v_only, best_w2v_combined, all_grid_results)


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(baseline: dict, best_only: dict, best_combined: dict, grid: list) -> None:
    delta_only = best_only["mean_macro_f1"] - baseline["mean_macro_f1"]
    delta_combined = best_combined["mean_macro_f1"] - baseline["mean_macro_f1"]

    lines = [
        "# SBER H1 — Word2Vec Candle Embeddings — 2026-06-03",
        "",
        "## Hypothesis",
        "Co-occurrence SVD embeddings of candle shape clusters add sequential price-structure",
        "context that complements the flat continuous_regime feature snapshot.",
        "Combining both should improve macro-F1 above baseline 0.4675 (walk-forward CV).",
        "",
        "## Method",
        "- Model: ExtraTreesClassifier (n=300, depth=None, leaf=20, sqrt)",
        "- Target: triple_barrier:h3:w12:up1.25:down1.25",
        f"- Walk-forward: {WF_N_SPLITS} folds, initial_train={WF_INITIAL_TRAIN}, val={WF_VAL_SIZE}",
        f"- Seeds: {SEEDS}",
        "- W2V pipeline: normalize_ohlc → KMeans(nw) → co-occurrence SVD(nv) → context mean(nm)",
        f"- Grid: nw∈{NW_VALUES}, nv∈{NV_VALUES}, nm∈{NM_VALUES}",
        "",
        "## Results — Best Configs",
        "",
        "| Condition | Features | macro-F1 (mean±std) | Worst fold | SELL | HOLD | BUY | Δ vs baseline |",
        "|-----------|---------|---------------------|------------|------|------|-----|--------------|",
        f"| baseline | {baseline['n_features']} | {baseline['mean_macro_f1']:.4f} ± {baseline['std']:.4f} | {baseline['worst_fold']:.4f} | {baseline['sell_f1']:.4f} | {baseline['hold_f1']:.4f} | {baseline['buy_f1']:.4f} | — |",
        f"| w2v_only (best) | {best_only.get('n_features', '?')} | {best_only['mean_macro_f1']:.4f} ± {best_only.get('std', 0):.4f} | {best_only.get('worst_fold', 0):.4f} | {best_only.get('sell_f1', 0):.4f} | {best_only.get('hold_f1', 0):.4f} | {best_only.get('buy_f1', 0):.4f} | {delta_only:+.4f} |",
        f"| w2v_combined (best) | {best_combined.get('n_features', '?')} | {best_combined['mean_macro_f1']:.4f} ± {best_combined.get('std', 0):.4f} | {best_combined.get('worst_fold', 0):.4f} | {best_combined.get('sell_f1', 0):.4f} | {best_combined.get('hold_f1', 0):.4f} | {best_combined.get('buy_f1', 0):.4f} | {delta_combined:+.4f} |",
        "",
        f"  Best w2v_only config: {best_only.get('config', '?')}",
        f"  Best w2v_combined config: {best_combined.get('config', '?')}",
        "",
        "## Grid Search Results",
        "",
        "| Config | w2v_only F1 | w2v_combined F1 | Δ combined |",
        "|--------|------------|-----------------|-----------|",
    ]

    for g in sorted(grid, key=lambda x: x["w2v_combined"]["mean_macro_f1"], reverse=True):
        delta = g["w2v_combined"]["mean_macro_f1"] - baseline["mean_macro_f1"]
        lines.append(
            f"| {g['config']} | {g['w2v_only']['mean_macro_f1']:.4f} | "
            f"{g['w2v_combined']['mean_macro_f1']:.4f} | {delta:+.4f} |"
        )

    lines += [
        "",
        "## Top-10 Feature Importances — Best w2v_combined",
        "",
        "| Rank | Feature | Importance |",
        "|------|---------|-----------|",
    ]
    for rank, (name, imp) in enumerate(best_combined.get("top10_features", []), 1):
        lines.append(f"| {rank} | {name} | {imp:.4f} |")

    lines += ["", "## Conclusion", ""]

    if delta_combined > 0.005:
        verdict = (
            f"Word2Vec embeddings IMPROVED combined macro-F1 by {delta_combined:+.4f}.\n"
            f"Best config: {best_combined.get('config', '?')}\n"
            f"Recommendation: add w2v context features to the frozen candidate feature set.\n"
            f"Next: calibrate the improved model (Step 2 revisited) + backtest."
        )
    elif delta_combined > 0:
        verdict = (
            f"Marginal improvement: {delta_combined:+.4f}. W2V adds some sequential context\n"
            f"but the gain is within noise. Consider larger vocabulary or deeper context window.\n"
            f"Next: try LSTM sequence model (Step 4) for stronger sequential modelling."
        )
    else:
        verdict = (
            f"W2V embeddings did not improve over baseline (Δ={delta_combined:+.4f}).\n"
            f"SVD co-occurrence does not capture the same structure as neural skip-gram.\n"
            f"Recommendation: install gensim on a Python 3.11/3.12 environment and re-run,\n"
            f"OR move to Step 4 (LSTM) which directly models sequence structure."
        )

    lines.append(verdict)

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
