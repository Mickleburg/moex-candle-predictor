"""
Target horizon ablation: triple_barrier h=3,4,6,8,12 on SBER H1.

Hypothesis:
    h=3 (3-hour horizon) is too noisy — price can hit a barrier and immediately
    reverse, creating label flips that corrupt training. At h=6 (full session)
    or h=12 (two sessions), the signal is cleaner: if the barrier was genuinely
    hit, it tends to persist. Both ET and LSTM should improve.

    We also vary vol_window (12 vs 24) since with longer horizons a wider
    volatility lookback makes sense.

Grid:
    horizon ∈ {3, 4, 6, 8, 12}
    vol_window = 12 (fixed — compare to baseline)
    up_k = down_k = 1.25 (fixed)
    Model: ExtraTreesClassifier (frozen candidate spec) — fast, no GPU

Walk-forward: 4 folds, initial_train=12000, val=2000, seeds=[7,42,100].
Result: ml/docs/research/sber_h1_target_horizon_2026-06-03.md
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
from src.data.split import walk_forward_ranges
from src.nlp.action_features import make_continuous_past_features
from src.nlp.targets import ActionTargetSpec, make_research_action_targets

DATA_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_MD = ML_DIR / "docs" / "research" / "sber_h1_target_horizon_2026-06-03.md"

MODEL_PARAMS = dict(n_estimators=300, max_depth=None, min_samples_leaf=20,
                    max_features="sqrt", n_jobs=-1)
SEEDS = [7, 42, 100]
WF_INITIAL_TRAIN = 12_000
WF_VAL_SIZE = 2_000
WF_N_SPLITS = 4

HORIZONS = [3, 4, 6, 8, 12]
VOL_WINDOW = 12
UP_K = DOWN_K = 1.25


def load_data():
    df = load_candles(str(DATA_DIR), ticker="SBER", timeframe="1H")
    df["begin"] = pd.to_datetime(df["begin"], utc=True)
    return df.sort_values("begin").reset_index(drop=True)


def run_horizon(df, feat_matrix, feat_names, horizon, n_trainval):
    spec = ActionTargetSpec(
        mode="triple_barrier", barrier_horizon=horizon,
        barrier_vol_window=VOL_WINDOW, barrier_up_k=UP_K, barrier_down_k=DOWN_K,
    )
    labels = make_research_action_targets(df, spec).labels
    valid_total = (labels[:n_trainval] != -1).sum()

    # Class distribution on train+val
    vals, cnts = np.unique(labels[:n_trainval][labels[:n_trainval] != -1], return_counts=True)
    dist = {int(v): float(c / cnts.sum()) for v, c in zip(vals, cnts)}

    folds = walk_forward_ranges(
        n_trainval, n_splits=WF_N_SPLITS,
        initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE,
    )
    all_results = []
    for fold in folds:
        tr = np.arange(fold.train_start, fold.train_end)
        va = np.arange(fold.val_start, fold.val_end)
        tr = tr[labels[tr] != -1]
        va = va[labels[va] != -1]
        if len(tr) < 500 or len(va) < 50:
            continue
        mean = feat_matrix[tr].mean(axis=0)
        std = feat_matrix[tr].std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        X_tr = np.nan_to_num((feat_matrix[tr] - mean) / std)
        X_va = np.nan_to_num((feat_matrix[va] - mean) / std)
        for seed in SEEDS:
            m = ExtraTreesClassifier(random_state=seed, **MODEL_PARAMS)
            m.fit(X_tr, labels[tr])
            p = m.predict(X_va)
            macro = f1_score(labels[va], p, average="macro")
            pc = f1_score(labels[va], p, average=None, labels=[0, 1, 2])
            all_results.append({"fold": fold.fold_id, "seed": seed,
                                 "macro": macro, "sell": pc[0], "hold": pc[1], "buy": pc[2]})

    if not all_results:
        return None
    macros = np.array([r["macro"] for r in all_results])
    fold_means = {}
    for r in all_results:
        fold_means.setdefault(r["fold"], []).append(r["macro"])
    return {
        "horizon": horizon,
        "mean_f1": float(macros.mean()),
        "std_f1": float(macros.std()),
        "worst_fold": float(min(np.mean(v) for v in fold_means.values())),
        "sell_f1": float(np.mean([r["sell"] for r in all_results])),
        "hold_f1": float(np.mean([r["hold"] for r in all_results])),
        "buy_f1":  float(np.mean([r["buy"]  for r in all_results])),
        "valid_labels": int(valid_total),
        "class_dist": dist,
    }


def write_report(results):
    baseline = next(r for r in results if r["horizon"] == 3)
    lines = [
        "# SBER H1 -- Target Horizon Ablation -- 2026-06-03", "",
        "## Hypothesis",
        "Triple-barrier h=3 is too noisy. Longer horizon = cleaner labels = higher F1.", "",
        f"- Model: ExtraTreesClassifier (frozen candidate spec)",
        f"- vol_window={VOL_WINDOW}, up_k=down_k={UP_K}",
        f"- Walk-forward: {WF_N_SPLITS} folds, initial_train={WF_INITIAL_TRAIN}, val={WF_VAL_SIZE}",
        f"- Seeds: {SEEDS}", "",
        "## Results", "",
        "| Horizon | F1 mean+-std | Worst fold | SELL | HOLD | BUY | Delta vs h=3 | Valid labels |",
        "|---------|-------------|------------|------|------|-----|-------------|-------------|",
    ]
    for r in results:
        d = r["mean_f1"] - baseline["mean_f1"]
        marker = " <--" if r["horizon"] != 3 and d == max(
            x["mean_f1"] - baseline["mean_f1"] for x in results if x["horizon"] != 3
        ) else ""
        lines.append(
            f"| h={r['horizon']} | {r['mean_f1']:.4f}+-{r['std_f1']:.4f} | "
            f"{r['worst_fold']:.4f} | {r['sell_f1']:.4f} | {r['hold_f1']:.4f} | "
            f"{r['buy_f1']:.4f} | {d:+.4f} | {r['valid_labels']}{marker} |"
        )
    lines += ["", "## Class Distribution by Horizon", "",
              "| Horizon | SELL% | HOLD% | BUY% |",
              "|---------|-------|-------|------|"]
    for r in results:
        d = r["class_dist"]
        lines.append(f"| h={r['horizon']} | {d.get(0,0):.1%} | {d.get(1,0):.1%} | {d.get(2,0):.1%} |")

    best = max(results, key=lambda r: r["mean_f1"])
    delta_best = best["mean_f1"] - baseline["mean_f1"]

    lines += ["", "## Conclusion", ""]
    if delta_best > 0.01:
        lines.append(
            f"Best horizon: h={best['horizon']} with macro-F1={best['mean_f1']:.4f} "
            f"(delta={delta_best:+.4f} vs h=3).\n"
            f"Longer horizon reduces label noise. "
            f"Recommendation: update frozen candidate to h={best['horizon']}.\n"
            f"Next: run LSTM with h={best['horizon']} to confirm improvement transfers."
        )
    else:
        lines.append(
            f"Horizon changes give marginal gains (best delta={delta_best:+.4f}).\n"
            f"The performance ceiling is not driven by label noise at h=3.\n"
            f"The signal itself is limited at this timeframe+ticker combination.\n"
            f"Next: try Transformer architecture or multi-ticker training."
        )
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to {OUTPUT_MD}")


def main():
    print("Loading SBER 1H...")
    df = load_data()
    n_trainval = int(len(df) * 0.85)
    print(f"  {len(df)} candles, using {n_trainval} (test excluded)")

    print("Building features...")
    feat_matrix, feat_names = make_continuous_past_features(df)

    results = []
    for h in HORIZONS:
        print(f"\n{'='*50}\nh={h}  (triple_barrier:h{h}:w{VOL_WINDOW}:up{UP_K}:down{DOWN_K})")
        r = run_horizon(df, feat_matrix, feat_names, h, n_trainval)
        if r:
            results.append(r)
            print(f"  macro-F1: {r['mean_f1']:.4f}+-{r['std_f1']:.4f}  "
                  f"worst={r['worst_fold']:.4f}  "
                  f"S={r['sell_f1']:.3f} H={r['hold_f1']:.3f} B={r['buy_f1']:.3f}  "
                  f"valid={r['valid_labels']}")

    print(f"\n{'='*50}\nSUMMARY")
    base_f1 = next(r["mean_f1"] for r in results if r["horizon"] == 3)
    for r in results:
        print(f"  h={r['horizon']:2d}: {r['mean_f1']:.4f}  delta={r['mean_f1']-base_f1:+.4f}")
    best = max(results, key=lambda r: r["mean_f1"])
    print(f"  Best: h={best['horizon']} ({best['mean_f1']:.4f})")

    write_report(results)


if __name__ == "__main__":
    main()
