"""
Transformer sequence model for SBER H1 triple-barrier prediction.

Compares against LSTM v2 baseline (WF macro-F1=0.4778, conf>0.50 Sharpe=6.38).

Key difference from LSTM:
    Self-attention dynamically weights which of the 32 past candles matter most
    for each prediction. LSTM processes sequentially (equal weight to all steps).
    Transformers typically outperform LSTM on tasks where distant context matters.

Architecture:
    Input:  (batch, 32, 14) — same 14-dim features as LSTM v2
    Proj:   Linear(14 -> 64)
    PosEmb: nn.Embedding(32, 64) — learned positional encoding
    Encoder: TransformerEncoder(d_model=64, nhead=4, ffn=256, layers=2, dropout=0.1)
    Pool:   mean over time dimension
    Head:   Linear(64->32) -> ReLU -> Dropout(0.1) -> Linear(32->3)

Same walk-forward protocol as LSTM for fair comparison:
    4 folds, initial_train=12000, val=2000, seeds=[7,42,100]

Results saved to: ml/docs/research/sber_h1_transformer_results_YYYYMMDD_HHMMSS.json
This JSON file should be sent to the project maintainer for analysis.
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
from src.models.lstm_model import build_per_step_features

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data" / "raw"
RESULTS_DIR = ML_DIR / "docs" / "research"

TARGET_SPEC = ActionTargetSpec(
    mode="triple_barrier", barrier_horizon=3, barrier_vol_window=12,
    barrier_up_k=1.25, barrier_down_k=1.25,
)

SEQ_LEN = 32
INPUT_DIM = 14
D_MODEL = 64
NHEAD = 4
NUM_ENCODER_LAYERS = 2
FFN_DIM = 256
DROPOUT = 0.1

BATCH_SIZE = 256
MAX_EPOCHS = 60
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10

SEEDS = [7, 42, 100]
WF_INITIAL_TRAIN = 12_000
WF_VAL_SIZE = 2_000
WF_N_SPLITS = 4

LSTM_BASELINE_WF_F1 = 0.4778   # for comparison in output
ET_BASELINE_WF_F1 = 0.4738

DEVICE = torch.device("cpu")


# ── Transformer model ─────────────────────────────────────────────────────────

class CandleTransformer(nn.Module):
    """Transformer encoder with learned positional embeddings for candle sequences."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_ENCODER_LAYERS,
        ffn_dim: int = FFN_DIM,
        dropout: float = DROPOUT,
        seq_len: int = SEQ_LEN,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        b, t, _ = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        x = self.input_proj(x) + self.pos_emb(pos)
        x = self.encoder(x)              # (batch, seq_len, d_model)
        x = x.mean(dim=1)               # mean pool over time
        return self.head(x)


# ── Data helpers ──────────────────────────────────────────────────────────────

def build_sequences(feat_mat, labels, start, end):
    X, y = [], []
    for t in range(start + SEQ_LEN, end):
        if labels[t] == -1:
            continue
        X.append(feat_mat[t - SEQ_LEN: t])
        y.append(labels[t])
    if not X:
        return np.empty((0, SEQ_LEN, INPUT_DIM), np.float32), np.empty(0, np.int64)
    return np.stack(X).astype(np.float32), np.array(y, np.int64)


def normalize_seqs(X_tr, X_va):
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    m, s = flat.mean(0), flat.std(0)
    s = np.where(s < 1e-12, 1.0, s)
    return ((X_tr - m) / s).astype(np.float32), ((X_va - m) / s).astype(np.float32)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_fold(X_tr, y_tr, X_va, y_va, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr_n, X_va_n = normalize_seqs(X_tr, X_va)
    tr_ds = TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(y_tr).long())
    va_ds = TensorDataset(torch.from_numpy(X_va_n), torch.from_numpy(y_va).long())
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE * 4)

    model = CandleTransformer().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    crit = nn.CrossEntropyLoss()

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

    # Restore best, collect full metrics + probabilities for confidence analysis
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_trues, all_proba = [], [], []
    with torch.no_grad():
        for xb, yb in DataLoader(va_ds, batch_size=BATCH_SIZE * 4):
            logits = model(xb.to(DEVICE))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_trues.extend(yb.numpy())
            all_proba.append(proba)

    proba_mat = np.vstack(all_proba)
    conf = proba_mat.max(axis=1)
    pc = f1_score(all_trues, all_preds, average=None, labels=[0, 1, 2], zero_division=0)

    return {
        "macro_f1": float(best_f1),
        "sell_f1": float(pc[0]),
        "hold_f1": float(pc[1]),
        "buy_f1": float(pc[2]),
        "epochs_run": epoch + 1,
        "conf_gt_035": float((conf > 0.35).mean()),
        "conf_gt_040": float((conf > 0.40).mean()),
        "conf_gt_045": float((conf > 0.45).mean()),
        "conf_gt_050": float((conf > 0.50).mean()),
    }


# ── Walk-forward ──────────────────────────────────────────────────────────────

def run_walk_forward(feat_mat, labels, n_trainval):
    folds = walk_forward_ranges(
        n_trainval, n_splits=WF_N_SPLITS,
        initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE,
    )
    fold_records = []

    for fold in folds:
        X_tr, y_tr = build_sequences(feat_mat, labels, fold.train_start, fold.train_end)
        X_va, y_va = build_sequences(feat_mat, labels, fold.val_start, fold.val_end)
        if len(y_tr) < 200 or len(y_va) < 50:
            print(f"  fold {fold.fold_id}: skipping (too few sequences)")
            continue

        print(f"  fold {fold.fold_id}: {len(y_tr)} train seqs, {len(y_va)} val seqs")
        t0 = time.time()
        for seed in SEEDS:
            metrics = train_fold(X_tr, y_tr, X_va, y_va, seed)
            fold_records.append({
                "fold_id": fold.fold_id,
                "seed": seed,
                **metrics,
            })
            print(f"    seed={seed}: macro={metrics['macro_f1']:.4f}  "
                  f"S={metrics['sell_f1']:.3f} H={metrics['hold_f1']:.3f} B={metrics['buy_f1']:.3f}  "
                  f"conf>0.50={metrics['conf_gt_050']:.1%}  epochs={metrics['epochs_run']}")
        elapsed = time.time() - t0
        fold_f1s = [r["macro_f1"] for r in fold_records if r["fold_id"] == fold.fold_id]
        print(f"    fold mean={float(np.mean(fold_f1s)):.4f}  elapsed={elapsed:.0f}s")

    return fold_records


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate(fold_records):
    if not fold_records:
        return {}
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
        "mean_conf_gt_045": float(np.mean([r["conf_gt_045"] for r in fold_records])),
        "delta_vs_lstm": float(macros.mean() - LSTM_BASELINE_WF_F1),
        "delta_vs_et": float(macros.mean() - ET_BASELINE_WF_F1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"sber_h1_transformer_results_{timestamp}.json"

    print("=" * 60)
    print("Transformer experiment — SBER H1 triple-barrier")
    print(f"PyTorch {torch.__version__}  |  Device: {DEVICE}")
    print(f"Output: {output_path}")
    print("=" * 60)

    print("\nLoading SBER 1H data...")
    df = load_candles(str(DATA_DIR), ticker="SBER", timeframe="1H")
    df["begin"] = pd.to_datetime(df["begin"], utc=True)
    df = df.sort_values("begin").reset_index(drop=True)
    labels = make_research_action_targets(df, TARGET_SPEC).labels
    n_trainval = int(len(df) * 0.85)
    print(f"  {len(df)} candles, first {n_trainval} rows used (test excluded)")

    print("Building per-step features...")
    feat_mat = build_per_step_features(df)
    print(f"  Feature matrix: {feat_mat.shape}")

    print(f"\nWalk-forward ({WF_N_SPLITS} folds x {len(SEEDS)} seeds)...")
    fold_records = run_walk_forward(feat_mat, labels, n_trainval)

    agg = aggregate(fold_records)
    total_time = time.time() - run_start

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Transformer macro-F1: {agg['mean_macro_f1']:.4f} +- {agg['std_macro_f1']:.4f}  (worst={agg['worst_fold_f1']:.4f})")
    print(f"  LSTM v2 baseline:     {LSTM_BASELINE_WF_F1:.4f}")
    print(f"  Delta vs LSTM:        {agg['delta_vs_lstm']:+.4f}")
    print(f"  Conf > 0.50:          {agg['mean_conf_gt_050']:.1%}  (LSTM was ~1%)")
    print(f"  Total time:           {total_time/60:.1f} min")

    result = {
        "experiment": "sber_h1_transformer",
        "timestamp": timestamp,
        "git_branch": "ml-expirement",
        "system": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "config": {
            "target": str(TARGET_SPEC.label),
            "seq_len": SEQ_LEN,
            "input_dim": INPUT_DIM,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_encoder_layers": NUM_ENCODER_LAYERS,
            "ffn_dim": FFN_DIM,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "seeds": SEEDS,
            "wf_initial_train": WF_INITIAL_TRAIN,
            "wf_val_size": WF_VAL_SIZE,
            "wf_n_splits": WF_N_SPLITS,
        },
        "baselines": {
            "et_wf_macro_f1": ET_BASELINE_WF_F1,
            "lstm_v2_wf_macro_f1": LSTM_BASELINE_WF_F1,
            "lstm_v2_conf050_sharpe": 6.38,
            "lstm_v2_conf050_trades_per_16mo": 78,
        },
        "aggregate": agg,
        "fold_records": fold_records,
        "total_training_seconds": round(total_time, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {output_path}")
    print("Send this file to the project maintainer for analysis.")


if __name__ == "__main__":
    main()
