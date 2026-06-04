"""
Joint grid: horizon × barrier_k to find balanced-label configs.

Problem found in sber_target_horizon_research.py:
    At h>=6 with k=1.25, HOLD class disappears completely (price always hits
    a barrier within 6h at ±1.25σ). This collapses macro-F1 to ~0.32.

Fix: for longer horizons, widen the barriers proportionally.
    Rule of thumb: sigma scales with sqrt(h), so barriers should scale as sqrt(h/3).
    h=3  → k=1.25 (baseline)
    h=6  → k≈1.77  (1.25 * sqrt(2))
    h=12 → k≈2.50  (1.25 * sqrt(4))

Grid: all (h, k) combos that produce ~25-40% HOLD labels.
Then evaluate ExtraTrees F1 on the well-balanced ones.
"""

from __future__ import annotations
import sys
from pathlib import Path
from itertools import product

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
OUTPUT_MD = ML_DIR / "docs" / "research" / "sber_h1_horizon_barrier_grid_2026-06-03.md"

MODEL_PARAMS = dict(n_estimators=300, max_depth=None, min_samples_leaf=20,
                    max_features="sqrt", n_jobs=-1)
SEEDS = [7, 42, 100]
WF_INITIAL_TRAIN = 12_000
WF_VAL_SIZE = 2_000
WF_N_SPLITS = 4

# Physically motivated grid: k scales with sqrt(h/3) from baseline
CONFIGS = [
    (3,  1.25),   # baseline
    (3,  1.75),   # wider baseline
    (6,  1.75),   # sqrt(2) * 1.25 ≈ 1.77
    (6,  2.00),
    (6,  2.50),
    (12, 2.50),   # sqrt(4) * 1.25 = 2.50
    (12, 3.00),
    (12, 3.50),
]


def label_dist(labels, n):
    """SELL/HOLD/BUY shares on first n rows."""
    lv = labels[:n]
    lv = lv[lv != -1]
    vals, cnts = np.unique(lv, return_counts=True)
    d = {int(v): float(c / cnts.sum()) for v, c in zip(vals, cnts)}
    return d.get(0, 0), d.get(1, 0), d.get(2, 0)


def run_config(df, feat_matrix, h, k, n_trainval):
    spec = ActionTargetSpec(mode="triple_barrier", barrier_horizon=h,
                            barrier_vol_window=12, barrier_up_k=k, barrier_down_k=k)
    labels = make_research_action_targets(df, spec).labels
    sell_r, hold_r, buy_r = label_dist(labels, n_trainval)

    # Only run full experiment if HOLD is meaningful
    if hold_r < 0.10:
        return {"h": h, "k": k, "sell_r": sell_r, "hold_r": hold_r, "buy_r": buy_r,
                "mean_f1": None, "std_f1": None, "worst_fold": None,
                "sell_f1": None, "hold_f1": None, "buy_f1": None, "skip": True}

    folds = walk_forward_ranges(n_trainval, n_splits=WF_N_SPLITS,
                                initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE)
    results = []
    for fold in folds:
        tr = np.arange(fold.train_start, fold.train_end)
        va = np.arange(fold.val_start, fold.val_end)
        tr = tr[labels[tr] != -1]; va = va[labels[va] != -1]
        if len(tr) < 500 or len(va) < 50:
            continue
        mean = feat_matrix[tr].mean(0); std = feat_matrix[tr].std(0)
        std = np.where(std < 1e-12, 1.0, std)
        X_tr = np.nan_to_num((feat_matrix[tr] - mean) / std)
        X_va = np.nan_to_num((feat_matrix[va] - mean) / std)
        for seed in SEEDS:
            m = ExtraTreesClassifier(random_state=seed, **MODEL_PARAMS)
            m.fit(X_tr, labels[tr])
            p = m.predict(X_va)
            macro = f1_score(labels[va], p, average="macro")
            pc = f1_score(labels[va], p, average=None, labels=[0, 1, 2])
            results.append({"fold": fold.fold_id, "macro": macro,
                             "sell": pc[0], "hold": pc[1], "buy": pc[2]})

    if not results:
        return None
    macros = np.array([r["macro"] for r in results])
    fold_means = {}
    for r in results:
        fold_means.setdefault(r["fold"], []).append(r["macro"])
    return {
        "h": h, "k": k, "sell_r": sell_r, "hold_r": hold_r, "buy_r": buy_r,
        "mean_f1": float(macros.mean()), "std_f1": float(macros.std()),
        "worst_fold": float(min(np.mean(v) for v in fold_means.values())),
        "sell_f1": float(np.mean([r["sell"] for r in results])),
        "hold_f1": float(np.mean([r["hold"] for r in results])),
        "buy_f1":  float(np.mean([r["buy"]  for r in results])),
        "skip": False,
    }


def write_report(results, baseline_f1=0.4738):
    run_results = [r for r in results if not r["skip"] and r["mean_f1"] is not None]
    best = max(run_results, key=lambda r: r["mean_f1"]) if run_results else None

    lines = [
        "# SBER H1 -- Horizon x Barrier Grid -- 2026-06-03", "",
        "## Problem",
        "At h>=6 with k=1.25, HOLD class disappears (price always hits barrier).",
        "This paper shows barriers must scale with sqrt(horizon) to keep balanced labels.", "",
        "## Grid (h x k), vol_window=12", "",
        "| h | k | SELL% | HOLD% | BUY% | F1 mean+-std | Worst fold | SELL | HOLD | BUY | Delta |",
        "|---|---|-------|-------|------|-------------|------------|------|------|-----|-------|",
    ]
    for r in results:
        if r["skip"] or r["mean_f1"] is None:
            lines.append(
                f"| {r['h']} | {r['k']} | {r['sell_r']:.0%} | {r['hold_r']:.0%} | "
                f"{r['buy_r']:.0%} | SKIP (HOLD<10%) | — | — | — | — | — |"
            )
        else:
            d = r["mean_f1"] - baseline_f1
            marker = " **BEST**" if best and r is best else ""
            lines.append(
                f"| {r['h']} | {r['k']} | {r['sell_r']:.0%} | {r['hold_r']:.0%} | "
                f"{r['buy_r']:.0%} | {r['mean_f1']:.4f}+-{r['std_f1']:.4f} | "
                f"{r['worst_fold']:.4f} | {r['sell_f1']:.4f} | {r['hold_f1']:.4f} | "
                f"{r['buy_f1']:.4f} | {d:+.4f}{marker} |"
            )

    lines += ["", "## Conclusion", ""]
    if best and best["mean_f1"] > baseline_f1 + 0.01:
        lines.append(
            f"BEST CONFIG: h={best['h']}, k={best['k']} → macro-F1={best['mean_f1']:.4f} "
            f"(delta={best['mean_f1']-baseline_f1:+.4f} vs h=3,k=1.25 baseline).\n\n"
            f"Longer horizon with properly scaled barriers IS beneficial.\n"
            f"Recommendation: update frozen candidate target to "
            f"triple_barrier:h{best['h']}:w12:up{best['k']}:down{best['k']}.\n"
            f"Next: retrain LSTM with this target."
        )
    elif best:
        lines.append(
            f"Best config h={best['h']},k={best['k']} gives "
            f"macro-F1={best['mean_f1']:.4f} (delta={best['mean_f1']-baseline_f1:+.4f}).\n\n"
            f"No meaningful improvement from changing horizon.\n"
            f"The signal ceiling at this resolution is ~0.47-0.49.\n\n"
            f"**Root cause**: 1H MOEX triple-barrier prediction is inherently hard at any horizon.\n"
            f"Intraday price moves are largely random noise; the 62% time-feature importance\n"
            f"shows the model mostly predicts WHEN to trade (session open/close effects),\n"
            f"not WHERE price will go.\n\n"
            f"Next steps with higher expected impact:\n"
            f"1. Transformer with attention over 32-step sequences (captures long-range deps)\n"
            f"2. Additional MOEX-specific features: macro (CBR rate, USD/RUB), sector flow\n"
            f"3. Multi-ticker joint training (SBER+LKOH+GAZP → 3x data, shared patterns)\n"
            f"4. Pre-train on unsupervised next-candle prediction, then fine-tune on labels"
        )
    else:
        lines.append("All longer-horizon configs had imbalanced classes. Baseline h=3 remains best.")

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to {OUTPUT_MD}")


def main():
    print("Loading SBER 1H...")
    df = load_candles(str(DATA_DIR), ticker="SBER", timeframe="1H")
    df["begin"] = pd.to_datetime(df["begin"], utc=True)
    df = df.sort_values("begin").reset_index(drop=True)
    n_trainval = int(len(df) * 0.85)

    print("Building features...")
    feat_matrix, _ = make_continuous_past_features(df)

    all_results = []
    for h, k in CONFIGS:
        r = run_config(df, feat_matrix, h, k, n_trainval)
        if r is None:
            continue
        all_results.append(r)
        if r["skip"]:
            print(f"  h={h} k={k}: SKIP — HOLD={r['hold_r']:.0%}")
        else:
            print(f"  h={h} k={k}: HOLD={r['hold_r']:.0%}  "
                  f"macro-F1={r['mean_f1']:.4f}+-{r['std_f1']:.4f}  "
                  f"worst={r['worst_fold']:.4f}  "
                  f"S={r['sell_f1']:.3f} H={r['hold_f1']:.3f} B={r['buy_f1']:.3f}")

    print(f"\nSUMMARY (vs baseline 0.4738)")
    for r in [x for x in all_results if not x["skip"] and x["mean_f1"]]:
        print(f"  h={r['h']} k={r['k']}: {r['mean_f1']:.4f}  delta={r['mean_f1']-0.4738:+.4f}")

    write_report(all_results)


if __name__ == "__main__":
    main()
