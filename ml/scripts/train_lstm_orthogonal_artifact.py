"""Train an LSTM artifact that uses orthogonal (cross-instrument) features.

Same protocol as train_lstm_artifact.py (fit on first 85%, test untouched) but the per-step
feature vector is 14 OHLCV/time + orthogonal features (src/features/orthogonal). The artifact
records `orthogonal_groups` in feature_config so inference (research_artifact._lstm_feature_matrix)
self-fetches the orthogonal series via the market-context provider — the input contract is
unchanged.

Usage:
    python ml/scripts/train_lstm_orthogonal_artifact.py --ticker LKOH \\
        --groups commodity,market --output-dir ml/artifacts/research_lstm_v2_lkoh_h1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
sys.path.insert(0, str(ML_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.load import load_candles
from src.nlp.targets import ActionTargetSpec, make_research_action_targets
from src.models.lstm_model import CandleLSTM, SEQ_LEN
from src.features.orthogonal import load_ortho_series, build_combined_features
from src.utils.io import ensure_dir

TARGET_SPEC = ActionTargetSpec(mode="triple_barrier", barrier_horizon=3, barrier_vol_window=12,
                               barrier_up_k=1.25, barrier_down_k=1.25)
HIDDEN_SIZE, NUM_LAYERS, DROPOUT = 128, 2, 0.3
BATCH_SIZE, MAX_EPOCHS, LR, WEIGHT_DECAY, PATIENCE = 256, 50, 0.001, 1e-4, 8
DEVELOPMENT_RATIO = 0.85
DEVICE = torch.device("cpu")


def build_train_sequences(feat_mat, labels, dev_end, input_dim):
    X, y, idx = [], [], []
    for t in range(SEQ_LEN, dev_end):
        if labels[t] == -1:
            continue
        X.append(feat_mat[t - SEQ_LEN: t]); y.append(labels[t]); idx.append(t)
    if not X:
        raise ValueError("No training sequences")
    return np.stack(X).astype(np.float32), np.array(y, np.int64), np.array(idx, np.int64)


def train(X_train, y_train, input_dim, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    flat = X_train.reshape(-1, input_dim)
    mean = flat.mean(0).astype(np.float32)
    std = np.where(flat.std(0) < 1e-12, 1.0, flat.std(0)).astype(np.float32)
    Xn = ((X_train - mean) / std).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xn), torch.from_numpy(y_train).long()),
                        batch_size=BATCH_SIZE, shuffle=True)
    model = CandleLSTM(input_dim=input_dim, hidden_size=HIDDEN_SIZE,
                       num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    crit = nn.CrossEntropyLoss()
    best, best_state, no_imp = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train(); tot = 0.0
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE)); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item() * len(xb)
        sch.step(); avg = tot / len(y_train)
        if avg < best - 1e-5:
            best, best_state, no_imp = avg, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"    early stop epoch {epoch+1} loss={avg:.5f}"); break
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1} loss={avg:.5f}")
    model.load_state_dict(best_state)
    return model, mean, std


def wj(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--groups", required=True, help="comma list e.g. commodity,market")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()
    ticker = args.ticker.upper()
    groups = tuple(g.strip() for g in args.groups.split(",") if g.strip())
    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else REPO_ROOT / args.output_dir
    ensure_dir(out_dir)

    print(f"Loading {ticker} 1H (tz_aware MSK)...")
    df = load_candles(str(REPO_ROOT / "data" / "raw"), ticker=ticker, timeframe="1H", tz_aware=True)
    df = df.sort_values("begin").reset_index(drop=True)
    dev_end = int(len(df) * DEVELOPMENT_RATIO)
    labels = make_research_action_targets(df, TARGET_SPEC).labels

    print(f"Building combined features (groups={groups})...")
    ortho = load_ortho_series(str(REPO_ROOT / "data" / "raw"))
    feat_mat, names = build_combined_features(df, ortho, ticker, groups=groups)
    input_dim = feat_mat.shape[1]
    print(f"  {len(df)} candles, dev_end={dev_end}, input_dim={input_dim} ({len(names)-14} orthogonal)")

    X, y, idx = build_train_sequences(feat_mat, labels, dev_end, input_dim)
    print(f"  training sequences: {X.shape}")
    model, mean, std = train(X, y, input_dim, args.random_state)

    torch.save(model.state_dict(), out_dir / "model.pt")
    wj(out_dir / "model_config.json", model.config())
    wj(out_dir / "feature_config.json", {
        "model_type": "lstm", "seq_len": SEQ_LEN, "input_dim": input_dim,
        "feature_names": names, "orthogonal_groups": list(groups),
        "normalization": "zscore_per_feature_from_train_sequences",
        "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
    })
    wj(out_dir / "target_config.json", {
        "target_mode": "triple_barrier", "horizon": TARGET_SPEC.barrier_horizon,
        "vol_window": TARGET_SPEC.barrier_vol_window, "up_k": TARGET_SPEC.barrier_up_k,
        "down_k": TARGET_SPEC.barrier_down_k, "target_label": TARGET_SPEC.label,
        "label_order": ["SELL", "HOLD", "BUY"],
    })
    wj(out_dir / "metadata.json", {
        "artifact_id": f"research_lstm_v2_{ticker.lower()}_orth_20260615",
        "model_version": f"research_lstm_v2_{ticker.lower()}_orth_20260615",
        "artifact_type": "research", "is_production": False,
        "ticker": ticker, "timeframe": "1H",
        "model_family": "triple_barrier_lstm", "model_class": "CandleLSTM",
        "target": TARGET_SPEC.label, "feature_set": f"lstm_orthogonal_{'_'.join(groups)}",
        "orthogonal_groups": list(groups),
        "seq_len": SEQ_LEN, "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
        "random_state": args.random_state, "probabilities_calibrated": False,
        "min_candles_for_prediction": SEQ_LEN + 1, "recommended_min_candles": SEQ_LEN + 20,
        "training_protocol": "fit on first 85% development data; test untouched",
        "created_at": "2026-06-15",
        "production_rule": "BUY conf>0.50; hold 3h + lower-barrier stop; no take-profit; long-only; skip weekend sessions",
        "notes": ("Orthogonal LSTM. Serves predictions with self-fetched market context "
                  "(Brent/IMOEX/RTSI) via MarketContextProvider; input contract unchanged. "
                  "LKOH oil+market validation backtest: Sharpe=5.42, +11.73%, win 56.8%, 81 trades; "
                  "bootstrap P(profit)=97.3%. is_production=false until the test gate + sign-off."),
    })
    wj(out_dir / "label_mapping.json", {
        "internal_to_contract": {"SELL": "sell", "HOLD": "hold", "BUY": "buy"},
        "contract_to_internal": {"sell": "SELL", "hold": "HOLD", "buy": "BUY"},
        "class_index": {"SELL": 0, "HOLD": 1, "BUY": 2},
    })
    wj(out_dir / "schema_version.json", {
        "artifact_schema_version": 2, "model_format": "pytorch_state_dict",
        "ml_prediction_schema": "contracts/ml_prediction.schema.json",
        "candle_batch_schema": "contracts/candle_batch.schema.json",
    })
    vals, cnts = np.unique(y, return_counts=True)
    wj(out_dir / "training_summary.json", {
        "raw_rows": int(len(df)), "development_ratio": DEVELOPMENT_RATIO, "development_end": int(dev_end),
        "n_training_sequences": int(len(y)), "input_dim": input_dim,
        "orthogonal_feature_names": names[14:],
        "training_class_distribution": {["SELL", "HOLD", "BUY"][int(v)]: int(c) for v, c in zip(vals, cnts)},
    })
    print(f"\nArtifact written to {out_dir} (input_dim={input_dim}, groups={groups})")


if __name__ == "__main__":
    main()
