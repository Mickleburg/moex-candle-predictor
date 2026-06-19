"""
LSTM sequence model for SBER H1 triple-barrier prediction.

Core idea:
    ExtraTrees sees a flat 27-dim snapshot at time t. No architecture can learn
    from it that "3 consecutive bearish candles after a resistance test" means
    something different from "one big drop". LSTM directly models the SEQUENCE:
    hidden state accumulates pattern memory across all 32 preceding hours.

Architecture:
    Input:  (batch, SEQ_LEN=32, INPUT_DIM=9) normalized OHLCV-derived features
    LSTM:   hidden_size=128, num_layers=2, dropout=0.3, bidirectional=False
    Head:   Linear(128, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 3)
    Loss:   CrossEntropyLoss
    Optim:  Adam(lr=0.001, weight_decay=1e-4) with CosineAnnealingLR

Input features per timestep (9 dim, all scale-invariant):
    - ret_1h:        (close[t] - close[t-1]) / close[t-1]
    - ret_3h:        (close[t] - close[t-3]) / close[t-3]   (momentum)
    - body:          (close[t] - open[t]) / open[t]          (candle direction)
    - range_:        (high[t] - low[t]) / open[t]            (volatility of candle)
    - upper_shadow:  (high[t] - max(o,c)) / open[t]
    - lower_shadow:  (min(o,c) - low[t]) / open[t]
    - vol_ratio:     volume[t] / rolling_20_mean(volume)      (unusual activity)
    - vol_std:       (vol - mean) / std                       (z-score)
    - close_pos:     (close - low) / (high - low)             (close in candle range)

Walk-forward: same 4 folds, initial_train=12000, val=2000, seeds=[7,42,100].
Result saved to: ml/docs/research/sber_h1_lstm_2026-06-03.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
sys.path.insert(0, str(ML_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.load import load_candles
from src.data.split import walk_forward_ranges
from src.nlp.targets import ActionTargetSpec, make_research_action_targets
from sklearn.metrics import f1_score

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_MD = ML_DIR / "docs" / "research" / "sber_h1_lstm_v2_2026-06-03.md"
BASELINE_F1 = 0.4738

TARGET_SPEC = ActionTargetSpec(
    mode="triple_barrier", barrier_horizon=3, barrier_vol_window=12,
    barrier_up_k=1.25, barrier_down_k=1.25,
)

SEQ_LEN = 32          # 32-hour sliding window (~4 trading days)
INPUT_DIM = 14        # features per timestep (9 OHLCV + 4 time + 1 trend)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

BATCH_SIZE = 256
MAX_EPOCHS = 50
LR = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 8          # early stopping patience on val macro-F1

SEEDS = [7, 42, 100]
WF_INITIAL_TRAIN = 12_000
WF_VAL_SIZE = 2_000
WF_N_SPLITS = 4

DEVICE = torch.device("cpu")


# ── Feature engineering ───────────────────────────────────────────────────────

def build_per_step_features(df: pd.DataFrame) -> np.ndarray:
    """Build 14-dim feature vector per timestep (past-only, no lookahead).

    9 scale-invariant OHLCV features  +  4 time-of-day/week features
    + 1 trend (EMA-8 distance). Time features contribute 62% of ExtraTrees
    importance — including them is critical for fold 4 regime.

    Returns shape (N, 14).
    """
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    v = df["volume"].astype(float).values

    safe_o = np.where(np.abs(o) < 1e-12, np.nan, o)
    safe_c_prev = np.roll(c, 1); safe_c_prev[0] = np.nan
    safe_c_prev3 = np.roll(c, 3); safe_c_prev3[:3] = np.nan
    safe_hl = np.where((h - l) < 1e-12, np.nan, h - l)

    ret_1h = (c - safe_c_prev) / np.where(np.abs(safe_c_prev) < 1e-12, np.nan, safe_c_prev)
    ret_3h = (c - safe_c_prev3) / np.where(np.abs(safe_c_prev3) < 1e-12, np.nan, safe_c_prev3)
    body = (c - o) / safe_o
    range_ = (h - l) / safe_o
    upper_shadow = (h - np.maximum(o, c)) / safe_o
    lower_shadow = (np.minimum(o, c) - l) / safe_o
    close_pos = (c - l) / safe_hl

    # Volume (past-only rolling stats)
    v_s = pd.Series(v)
    vol_mean = v_s.shift(1).rolling(20, min_periods=4).mean().values
    vol_std  = v_s.shift(1).rolling(20, min_periods=4).std().values
    vol_ratio = v / np.where(np.abs(vol_mean) < 1e-12, np.nan, vol_mean)
    vol_z = (v - vol_mean) / np.where(np.abs(vol_std) < 1e-12, 1.0, vol_std)

    # Time features — carry 62% of ExtraTrees signal; LSTM needs them too
    if "begin" in df.columns:
        begin = pd.to_datetime(df["begin"])
        hour = begin.dt.hour.astype(float).values
        dow  = begin.dt.dayofweek.astype(float).values
    else:
        hour = np.zeros(len(df))
        dow  = np.zeros(len(df))
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    dow_sin  = np.sin(2.0 * np.pi * dow  / 7.0)
    dow_cos  = np.cos(2.0 * np.pi * dow  / 7.0)

    # EMA-8 distance: tells the LSTM where price is relative to short-term trend
    c_s = pd.Series(c)
    ema8 = c_s.ewm(span=8, adjust=False).mean().values
    safe_ema8 = np.where(np.abs(ema8) < 1e-12, np.nan, ema8)
    ema_dist = (c - ema8) / safe_ema8

    mat = np.column_stack([
        ret_1h, ret_3h, body, range_, upper_shadow, lower_shadow, close_pos,
        vol_ratio, vol_z,
        hour_sin, hour_cos, dow_sin, dow_cos,
        ema_dist,
    ])
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat.astype(np.float32)


def build_sequences(
    feat_mat: np.ndarray,   # (N, INPUT_DIM)
    labels: np.ndarray,     # (N,) with -1 for invalid
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows: X shape (M, SEQ_LEN, INPUT_DIM), y shape (M,)."""
    X_list, y_list = [], []
    for t in range(start + SEQ_LEN, end):
        if labels[t] == -1:
            continue
        window = feat_mat[t - SEQ_LEN: t]   # past SEQ_LEN candles, no lookahead
        X_list.append(window)
        y_list.append(labels[t])
    if not X_list:
        return np.empty((0, SEQ_LEN, INPUT_DIM), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)


# ── LSTM model ────────────────────────────────────────────────────────────────

class CandleLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float, n_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            dropout=dropout if layers > 1 else 0.0,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])   # last layer's hidden state


# ── Training ──────────────────────────────────────────────────────────────────

def normalize_sequences(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score normalization per feature using train statistics."""
    n, s, f = X_train.shape
    X_flat = X_train.reshape(-1, f)
    mean = X_flat.mean(axis=0)
    std = X_flat.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    X_train_n = ((X_train - mean) / std).astype(np.float32)
    X_val_n = ((X_val - mean) / std).astype(np.float32)
    return X_train_n, X_val_n


def train_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_n, X_val_n = normalize_sequences(X_train, X_val)

    # Class weights for imbalanced classes
    classes, counts = np.unique(y_train, return_counts=True)
    class_weights = torch.tensor(
        [1.0 / counts[classes == c][0] for c in [0, 1, 2]], dtype=torch.float32
    )
    class_weights = class_weights / class_weights.sum() * len(classes)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_n),
        torch.from_numpy(y_train).long()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_n),
        torch.from_numpy(y_val).long()
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 4)

    model = CandleLSTM(INPUT_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_f1 = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(DEVICE))
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(yb.numpy())

        macro = f1_score(all_true, all_preds, average="macro", zero_division=0)
        if macro > best_f1:
            best_f1 = macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    # Restore best model, compute final metrics
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(DEVICE))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())

    per_class = f1_score(all_true, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "macro_f1": best_f1,
        "sell_f1": per_class[0],
        "hold_f1": per_class[1],
        "buy_f1": per_class[2],
        "epochs_run": epoch + 1,
    }


# ── Walk-forward ──────────────────────────────────────────────────────────────

def run_walk_forward(feat_mat: np.ndarray, labels: np.ndarray, n_trainval: int) -> dict:
    folds = walk_forward_ranges(
        n_trainval, n_splits=WF_N_SPLITS,
        initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE
    )

    all_results = []
    for fold in folds:
        t0 = time.time()
        X_tr, y_tr = build_sequences(feat_mat, labels, fold.train_start, fold.train_end)
        X_va, y_va = build_sequences(feat_mat, labels, fold.val_start, fold.val_end)

        if len(y_tr) < 200 or len(y_va) < 50:
            print(f"  fold {fold.fold_id}: skipping (too few sequences)")
            continue

        print(f"  fold {fold.fold_id}: {len(y_tr)} train seqs, {len(y_va)} val seqs")
        fold_f1 = []
        for seed in SEEDS:
            res = train_fold(X_tr, y_tr, X_va, y_va, seed)
            all_results.append({**res, "fold_id": fold.fold_id, "seed": seed})
            fold_f1.append(res["macro_f1"])
            print(f"    seed={seed}: macro-F1={res['macro_f1']:.4f}  "
                  f"(SELL={res['sell_f1']:.3f} HOLD={res['hold_f1']:.3f} BUY={res['buy_f1']:.3f})"
                  f"  epochs={res['epochs_run']}")
        print(f"    fold mean={np.mean(fold_f1):.4f}  elapsed={time.time()-t0:.0f}s")

    if not all_results:
        raise RuntimeError("No results")

    macros = np.array([r["macro_f1"] for r in all_results])
    fold_means = {}
    for r in all_results:
        fold_means.setdefault(r["fold_id"], []).append(r["macro_f1"])

    return {
        "mean_f1": float(macros.mean()),
        "std_f1": float(macros.std()),
        "worst_fold": float(min(np.mean(v) for v in fold_means.values())),
        "sell_f1": float(np.mean([r["sell_f1"] for r in all_results])),
        "hold_f1": float(np.mean([r["hold_f1"] for r in all_results])),
        "buy_f1": float(np.mean([r["buy_f1"] for r in all_results])),
        "n_train_seqs": len([r for r in all_results if r["fold_id"] == 1]) and sum(
            len(build_sequences(np.zeros((14001, INPUT_DIM), dtype=np.float32),
                np.ones(14001, dtype=np.int64), 0, 14000)[1]) for _ in [1]
        ),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(lstm: dict) -> None:
    delta = lstm["mean_f1"] - BASELINE_F1
    lines = [
        "# SBER H1 -- LSTM Sequence Model -- 2026-06-03", "",
        "## Hypothesis",
        "LSTM with 32-step OHLCV sequences models temporal dependencies that ExtraTrees",
        "cannot learn even with explicit lag features. The hidden state accumulates",
        "pattern memory across the 32-hour window.", "",
        "## Architecture",
        f"- Input: (batch, {SEQ_LEN}, {INPUT_DIM}) — 14-dim per timestep (9 OHLCV + 4 time + 1 EMA-dist)",
        f"- LSTM({HIDDEN_SIZE}, layers={NUM_LAYERS}, dropout={DROPOUT})",
        "- Head: Linear(128->64) -> ReLU -> Dropout(0.3) -> Linear(64->3)",
        "- Loss: CrossEntropyLoss | Optim: Adam(lr=0.001) + CosineAnnealingLR",
        f"- Early stopping: patience={PATIENCE} on val macro-F1",
        f"- Batch: {BATCH_SIZE} | Max epochs: {MAX_EPOCHS}", "",
        "## Input features (14 per timestep)",
        "- ret_1h, ret_3h: price momentum",
        "- body, range_, upper_shadow, lower_shadow: candle shape",
        "- close_pos: close position within candle range",
        "- vol_ratio, vol_z: volume anomaly detection",
        "- hour_sin, hour_cos, dow_sin, dow_cos: time-of-day/week (62% ET importance!)",
        "- ema_dist: close distance from EMA-8 (trend position)", "",
        "## Results", "",
        "| Model | F1 mean+-std | Worst fold | SELL | HOLD | BUY | Delta vs ET baseline |",
        "|-------|-------------|------------|------|------|-----|---------------------|",
        f"| ExtraTrees baseline | 0.4738+-0.0217 | 0.4377 | 0.4204 | 0.5815 | 0.4195 | -- |",
        f"| LSTM ({SEQ_LEN}-step) | {lstm['mean_f1']:.4f}+-{lstm['std_f1']:.4f} | {lstm['worst_fold']:.4f} | {lstm['sell_f1']:.4f} | {lstm['hold_f1']:.4f} | {lstm['buy_f1']:.4f} | {delta:+.4f} |",
        "", "## Conclusion", "",
    ]

    if delta > 0.02:
        verdict = (
            f"LSTM significantly outperforms ExtraTrees: delta={delta:+.4f}.\n"
            f"Sequence modelling captures temporal patterns invisible to flat feature models.\n"
            f"Recommendation: adopt LSTM as new primary model. Tune architecture further:\n"
            f"  - Try bidirectional LSTM\n"
            f"  - Add attention mechanism\n"
            f"  - Combine with continuous_regime features (hybrid model)"
        )
    elif delta > 0:
        verdict = (
            f"LSTM modestly outperforms ExtraTrees: delta={delta:+.4f}.\n"
            f"Sequence structure adds signal but gains are moderate.\n"
            f"Next: hybrid model (LSTM + ExtraTrees features), larger hidden size, attention."
        )
    else:
        verdict = (
            f"LSTM did not outperform ExtraTrees (delta={delta:+.4f}).\n"
            f"Possible reasons: MOEX 1H triple-barrier target has high noise at 3h horizon;\n"
            f"LSTM may be underfitting on CPU with small batch. Consider:\n"
            f"  - Longer horizon target (h=6 or h=12)\n"
            f"  - Larger architecture or attention\n"
            f"  - Pre-training on unsupervised next-candle prediction\n"
            f"  - Longer sequence (SEQ_LEN=64)"
        )
    lines.append(verdict)

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to {OUTPUT_MD}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading SBER 1H data...")
    df = load_candles(str(DATA_DIR), ticker="SBER", timeframe="1H")
    df["begin"] = pd.to_datetime(df["begin"], utc=True)
    df = df.sort_values("begin").reset_index(drop=True)
    labels = make_research_action_targets(df, TARGET_SPEC).labels
    n_trainval = int(len(df) * 0.85)
    print(f"  {len(df)} candles, first {n_trainval} rows (test excluded)")

    print("Building per-step features...")
    feat_mat = build_per_step_features(df)
    print(f"  Feature matrix: {feat_mat.shape}")

    # Quick sanity: check sequence build
    X_test, y_test = build_sequences(feat_mat, labels, 0, 500)
    print(f"  Sample sequences: {X_test.shape}, labels: {np.unique(y_test)}")

    print(f"\n{'='*60}")
    print(f"LSTM walk-forward experiment ({WF_N_SPLITS} folds x {len(SEEDS)} seeds)")
    print(f"{'='*60}")
    results = run_walk_forward(feat_mat, labels, n_trainval)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print("="*60)
    print(f"  LSTM macro-F1:    {results['mean_f1']:.4f} +- {results['std_f1']:.4f}  (worst={results['worst_fold']:.4f})")
    print(f"  ET baseline:      0.4738 +- 0.0217  (worst=0.4377)")
    print(f"  Delta:            {results['mean_f1'] - BASELINE_F1:+.4f}")
    print(f"  SELL={results['sell_f1']:.4f}  HOLD={results['hold_f1']:.4f}  BUY={results['buy_f1']:.4f}")

    write_report(results)


if __name__ == "__main__":
    main()
