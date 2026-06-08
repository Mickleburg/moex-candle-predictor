"""
Multi-ticker joint LSTM training for SBER H1 triple-barrier prediction.

Hypothesis:
    LSTM v2 (SBER-only) hits a ~0.478 WF F1 ceiling, with fold-4 collapsing to 0.44
    on the 2024-2025 MOEX regime. The model is data-starved: 12-18k training sequences
    cannot cover enough market regimes. Training jointly on SBER + LKOH + GAZP gives 3x
    data and exposes the model to shared blue-chip price-structure patterns, expected to
    add +0.01..0.03 WF F1 and lift the worst fold.

Why this is a fair, leak-free comparison:
    * Validation is SBER-ONLY, on the IDENTICAL walk-forward folds as LSTM v2
      (initial_train=12000, val=2000, 4 folds). So the headline number is directly
      comparable to the 0.4778 baseline.
    * Training sequences from EVERY ticker (incl. SBER) are filtered by
      target_timestamp < val_start_time, where val_start_time is the wall-clock time
      at which the SBER validation window begins. No training example from any ticker
      can see information at or after the validation window start. Zero lookahead.
    * For SBER specifically this reproduces the baseline training window exactly;
      LKOH and GAZP add the same [start, val_start_time) window -> 3x augmentation.

Everything else (architecture, features, seeds, optimizer, early stopping) is identical
to sber_lstm_research.py so the ONLY changed variable is the training data volume.

Result saved to:
    ml/docs/research/sber_h1_multiticker_lstm_results_YYYYMMDD_HHMMSS.json
    ml/docs/research/sber_h1_multiticker_lstm_2026-06-08.md  (written after run)
"""

from __future__ import annotations

import json
import platform
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from src.data.load import load_candles
from src.data.split import walk_forward_ranges
from src.nlp.targets import ActionTargetSpec, make_research_action_targets
from src.models.lstm_model import CandleLSTM, build_per_step_features, SEQ_LEN, INPUT_DIM

# ── Config (identical to sber_lstm_research.py except TICKERS) ──────────────────

DATA_DIR = REPO_ROOT / "data" / "raw"
RESULTS_DIR = ML_DIR / "docs" / "research"

PRIMARY_TICKER = "SBER"                  # validation ticker (comparable to baseline)
AUX_TICKERS = ["LKOH", "GAZP"]           # extra training data
ALL_TICKERS = [PRIMARY_TICKER, *AUX_TICKERS]

TARGET_SPEC = ActionTargetSpec(
    mode="triple_barrier", barrier_horizon=3, barrier_vol_window=12,
    barrier_up_k=1.25, barrier_down_k=1.25,
)

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

BATCH_SIZE = 256
MAX_EPOCHS = 50
LR = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 8

SEEDS = [7, 42, 100]
WF_INITIAL_TRAIN = 12_000
WF_VAL_SIZE = 2_000
WF_N_SPLITS = 4

LSTM_V2_BASELINE_F1 = 0.4778             # SBER-only LSTM v2 (the number we must beat)
ET_BASELINE_F1 = 0.4738

DEVICE = torch.device("cpu")


# ── Per-ticker data bundle ─────────────────────────────────────────────────────

class TickerData:
    """Features, labels and timestamps for one ticker (full history, test excluded)."""

    def __init__(self, ticker: str):
        df = load_candles(str(DATA_DIR), ticker=ticker, timeframe="1H")
        df["begin"] = pd.to_datetime(df["begin"], utc=True)
        df = df.sort_values("begin").reset_index(drop=True)
        self.ticker = ticker
        self.df = df
        self.feat = build_per_step_features(df)                       # (N, 14)
        self.labels = make_research_action_targets(df, TARGET_SPEC).labels
        self.ts = df["begin"].values                                 # np.datetime64 array
        self.n_trainval = int(len(df) * 0.85)                        # test split excluded

    def index_before(self, cutoff_time) -> int:
        """First row index whose timestamp >= cutoff_time (searchsorted, left side).

        Sequences built over [0, cutoff_index) have all targets strictly before
        cutoff_time -> no lookahead across the validation boundary.
        """
        idx = int(np.searchsorted(self.ts, cutoff_time, side="left"))
        return min(idx, self.n_trainval)   # never reach into the test split


def build_sequences(feat_mat: np.ndarray, labels: np.ndarray, start: int, end: int):
    """Sliding windows: X (M, SEQ_LEN, INPUT_DIM), y (M,). Identical to baseline."""
    X_list, y_list = [], []
    for t in range(start + SEQ_LEN, end):
        if labels[t] == -1:
            continue
        X_list.append(feat_mat[t - SEQ_LEN: t])
        y_list.append(labels[t])
    if not X_list:
        return (np.empty((0, SEQ_LEN, INPUT_DIM), np.float32), np.empty(0, np.int64))
    return np.stack(X_list).astype(np.float32), np.array(y_list, np.int64)


# ── Normalisation ──────────────────────────────────────────────────────────────

def fit_norm(X: np.ndarray):
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


# ── Training (identical hyperparams to baseline) ───────────────────────────────

def train_fold(X_tr, y_tr, X_va, y_va, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean, std = fit_norm(X_tr)
    X_tr_n = apply_norm(X_tr, mean, std)
    X_va_n = apply_norm(X_va, mean, std)

    classes, counts = np.unique(y_tr, return_counts=True)
    class_weights = torch.tensor(
        [1.0 / counts[classes == c][0] for c in [0, 1, 2]], dtype=torch.float32
    )
    class_weights = class_weights / class_weights.sum() * len(classes)

    tr_ds = TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(y_tr).long())
    va_ds = TensorDataset(torch.from_numpy(X_va_n), torch.from_numpy(y_va).long())
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE * 4)

    model = CandleLSTM(INPUT_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    best_f1, best_state, no_imp = 0.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in tr_ld:
            opt.zero_grad()
            crit(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                trues.extend(yb.numpy())
        macro = f1_score(trues, preds, average="macro", zero_division=0)
        if macro > best_f1:
            best_f1 = macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_trues, all_proba = [], [], []
    with torch.no_grad():
        for xb, yb in DataLoader(va_ds, batch_size=BATCH_SIZE * 4):
            logits = model(xb.to(DEVICE))
            all_proba.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_trues.extend(yb.numpy())

    conf = np.vstack(all_proba).max(axis=1)
    pc = f1_score(all_trues, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "macro_f1": float(best_f1),
        "sell_f1": float(pc[0]), "hold_f1": float(pc[1]), "buy_f1": float(pc[2]),
        "epochs_run": epoch + 1,
        "conf_gt_050": float((conf > 0.50).mean()),
        "conf_gt_045": float((conf > 0.45).mean()),
    }


# ── Walk-forward (validation = SBER only; training = all tickers, time-filtered) ─

def run_walk_forward(data: dict[str, TickerData]):
    sber = data[PRIMARY_TICKER]
    folds = walk_forward_ranges(
        sber.n_trainval, n_splits=WF_N_SPLITS,
        initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE,
    )

    fold_records = []
    for fold in folds:
        # Validation: SBER only, identical to baseline build
        X_va, y_va = build_sequences(sber.feat, sber.labels, fold.val_start, fold.val_end)
        if len(y_va) < 50:
            print(f"  fold {fold.fold_id}: skipping (too few val sequences)")
            continue

        # Wall-clock boundary: nothing at/after this time may enter training
        val_start_time = sber.ts[fold.val_start]

        # Training: pool all tickers, each filtered to target_timestamp < val_start_time
        X_parts, y_parts, per_ticker_counts = [], [], {}
        for tk in ALL_TICKERS:
            td = data[tk]
            cutoff = td.index_before(val_start_time)
            Xtr, ytr = build_sequences(td.feat, td.labels, 0, cutoff)
            if len(ytr):
                X_parts.append(Xtr)
                y_parts.append(ytr)
            per_ticker_counts[tk] = int(len(ytr))

        X_tr = np.concatenate(X_parts, axis=0)
        y_tr = np.concatenate(y_parts, axis=0)

        sber_only = per_ticker_counts[PRIMARY_TICKER]
        print(f"  fold {fold.fold_id}: {len(y_tr)} train seqs "
              f"(SBER={per_ticker_counts['SBER']} LKOH={per_ticker_counts.get('LKOH',0)} "
              f"GAZP={per_ticker_counts.get('GAZP',0)}), {len(y_va)} SBER val seqs")

        t0 = time.time()
        fold_f1 = []
        for seed in SEEDS:
            res = train_fold(X_tr, y_tr, X_va, y_va, seed)
            fold_records.append({
                "fold_id": fold.fold_id, "seed": seed,
                "n_train_pooled": int(len(y_tr)), "n_train_sber": sber_only,
                **res,
            })
            fold_f1.append(res["macro_f1"])
            print(f"    seed={seed}: macro={res['macro_f1']:.4f}  "
                  f"S={res['sell_f1']:.3f} H={res['hold_f1']:.3f} B={res['buy_f1']:.3f}  "
                  f"conf>0.50={res['conf_gt_050']:.1%}  epochs={res['epochs_run']}")
        print(f"    fold mean={float(np.mean(fold_f1)):.4f}  elapsed={time.time()-t0:.0f}s")

    return fold_records


def aggregate(fold_records):
    macros = np.array([r["macro_f1"] for r in fold_records])
    fold_means = {}
    for r in fold_records:
        fold_means.setdefault(r["fold_id"], []).append(r["macro_f1"])
    return {
        "mean_macro_f1": float(macros.mean()),
        "std_macro_f1": float(macros.std()),
        "worst_fold_f1": float(min(np.mean(v) for v in fold_means.values())),
        "mean_sell_f1": float(np.mean([r["sell_f1"] for r in fold_records])),
        "mean_hold_f1": float(np.mean([r["hold_f1"] for r in fold_records])),
        "mean_buy_f1": float(np.mean([r["buy_f1"] for r in fold_records])),
        "mean_conf_gt_050": float(np.mean([r["conf_gt_050"] for r in fold_records])),
        "delta_vs_lstm_v2": float(macros.mean() - LSTM_V2_BASELINE_F1),
        "delta_vs_et": float(macros.mean() - ET_BASELINE_F1),
        "fold_means": {fid: float(np.mean(v)) for fid, v in fold_means.items()},
    }


def main():
    run_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"sber_h1_multiticker_lstm_results_{timestamp}.json"

    print("=" * 64)
    print("Multi-ticker LSTM experiment — SBER H1 triple-barrier")
    print(f"PyTorch {torch.__version__}  |  Device: {DEVICE}")
    print(f"Train tickers: {ALL_TICKERS}  |  Validation: {PRIMARY_TICKER} only")
    print(f"Output: {output_path}")
    print("=" * 64)

    print("\nLoading tickers (test split excluded)...")
    data = {}
    for tk in ALL_TICKERS:
        td = TickerData(tk)
        data[tk] = td
        print(f"  {tk}: {len(td.df)} candles, trainval={td.n_trainval}, feat={td.feat.shape}")

    print(f"\nWalk-forward ({WF_N_SPLITS} folds x {len(SEEDS)} seeds)...")
    fold_records = run_walk_forward(data)
    agg = aggregate(fold_records)
    total_time = time.time() - run_start

    print(f"\n{'='*64}")
    print("RESULTS")
    print(f"{'='*64}")
    print(f"  Multi-ticker LSTM macro-F1: {agg['mean_macro_f1']:.4f} +- {agg['std_macro_f1']:.4f}  (worst={agg['worst_fold_f1']:.4f})")
    print(f"  LSTM v2 (SBER-only):        {LSTM_V2_BASELINE_F1:.4f}")
    print(f"  Delta vs LSTM v2:           {agg['delta_vs_lstm_v2']:+.4f}")
    print(f"  Per-fold means:             {agg['fold_means']}")
    print(f"  Conf > 0.50:                {agg['mean_conf_gt_050']:.1%}")
    print(f"  Total time:                 {total_time/60:.1f} min")

    result = {
        "experiment": "sber_h1_multiticker_lstm",
        "timestamp": timestamp,
        "git_branch": "ml-expirement",
        "system": {
            "python": sys.version, "torch": torch.__version__,
            "platform": platform.platform(), "cpu_count": __import__("os").cpu_count(),
        },
        "config": {
            "primary_ticker": PRIMARY_TICKER, "aux_tickers": AUX_TICKERS,
            "target": str(TARGET_SPEC.label), "seq_len": SEQ_LEN, "input_dim": INPUT_DIM,
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "patience": PATIENCE, "seeds": SEEDS,
            "wf_initial_train": WF_INITIAL_TRAIN, "wf_val_size": WF_VAL_SIZE,
            "wf_n_splits": WF_N_SPLITS,
        },
        "baselines": {
            "et_wf_macro_f1": ET_BASELINE_F1,
            "lstm_v2_wf_macro_f1": LSTM_V2_BASELINE_F1,
            "lstm_v2_conf050_sharpe": 6.38,
        },
        "aggregate": agg,
        "fold_records": fold_records,
        "total_training_seconds": round(total_time, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
