"""
Backtest: LSTM v2 vs ExtraTrees vs Buy&Hold on SBER H1 val periods.

Strategy:
    - Signal at close of candle t (using only data up to t, no lookahead)
    - If max(proba) > threshold AND signal != HOLD:
        enter position at close[t], exit at close[t+1]
    - Long (BUY signal): earn (close[t+1] - close[t]) / close[t] - 2*fee
    - Short (SELL signal): earn -(close[t+1] - close[t]) / close[t] - 2*fee
    - No position otherwise

Walk-forward: same 4 folds as classification experiments.
For each fold, train on [train_start:train_end], predict on [val_start:val_end].
Then concatenate all fold val predictions and backtest the combined equity curve.

Thresholds tested: [0.35, 0.40, 0.45, 0.50]
Fee: 0.05% one-way (0.1% round-trip) — standard MOEX retail fee.

Metrics:
    - Total return
    - Annualised Sharpe (1H bars, MOEX ~7h/day × ~250 days = 1750h/year)
    - Max drawdown
    - Win rate (% profitable trades)
    - Trade count
    - Action rate (% candles with position)

Result saved to: ml/docs/research/sber_h1_backtest_2026-06-03.md
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

from src.data.load import load_candles
from src.data.split import walk_forward_ranges
from src.nlp.action_features import make_continuous_past_features
from src.nlp.targets import ActionTargetSpec, make_research_action_targets

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_MD = ML_DIR / "docs" / "research" / "sber_h1_backtest_2026-06-03.md"

TARGET_SPEC = ActionTargetSpec(
    mode="triple_barrier", barrier_horizon=3, barrier_vol_window=12,
    barrier_up_k=1.25, barrier_down_k=1.25,
)

SEQ_LEN = 32
INPUT_DIM = 14
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

FEE = 0.0005          # 0.05% one-way
THRESHOLDS = [0.35, 0.40, 0.45, 0.50]
HOURS_PER_YEAR = 1750  # ~250 trading days × 7h MOEX session

DEVICE = torch.device("cpu")


# ── LSTM (reuse from sber_lstm_research.py) ───────────────────────────────────

class CandleLSTM(nn.Module):
    def __init__(self, input_dim, hidden, layers, dropout, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            dropout=dropout if layers > 1 else 0.0, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


def build_per_step_features(df):
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    v = df["volume"].astype(float).values
    safe_o = np.where(np.abs(o) < 1e-12, np.nan, o)
    c_prev = np.roll(c, 1); c_prev[0] = np.nan
    c_prev3 = np.roll(c, 3); c_prev3[:3] = np.nan
    safe_hl = np.where((h - l) < 1e-12, np.nan, h - l)
    ret_1h = (c - c_prev) / np.where(np.abs(c_prev) < 1e-12, np.nan, c_prev)
    ret_3h = (c - c_prev3) / np.where(np.abs(c_prev3) < 1e-12, np.nan, c_prev3)
    body = (c - o) / safe_o
    range_ = (h - l) / safe_o
    upper_shadow = (h - np.maximum(o, c)) / safe_o
    lower_shadow = (np.minimum(o, c) - l) / safe_o
    close_pos = (c - l) / safe_hl
    v_s = pd.Series(v)
    vol_mean = v_s.shift(1).rolling(20, min_periods=4).mean().values
    vol_std  = v_s.shift(1).rolling(20, min_periods=4).std().values
    vol_ratio = v / np.where(np.abs(vol_mean) < 1e-12, np.nan, vol_mean)
    vol_z = (v - vol_mean) / np.where(np.abs(vol_std) < 1e-12, 1.0, vol_std)
    if "begin" in df.columns:
        begin = pd.to_datetime(df["begin"])
        hour = begin.dt.hour.astype(float).values
        dow  = begin.dt.dayofweek.astype(float).values
    else:
        hour = dow = np.zeros(len(df))
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin  = np.sin(2 * np.pi * dow  / 7)
    dow_cos  = np.cos(2 * np.pi * dow  / 7)
    c_s = pd.Series(c)
    ema8 = c_s.ewm(span=8, adjust=False).mean().values
    safe_ema8 = np.where(np.abs(ema8) < 1e-12, np.nan, ema8)
    ema_dist = (c - ema8) / safe_ema8
    mat = np.column_stack([ret_1h, ret_3h, body, range_, upper_shadow, lower_shadow,
                           close_pos, vol_ratio, vol_z, hour_sin, hour_cos, dow_sin, dow_cos, ema_dist])
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_sequences_with_indices(feat_mat, labels, start, end):
    """Like build_sequences but also returns the original candle index t for each sample."""
    X_list, y_list, idx_list = [], [], []
    for t in range(start + SEQ_LEN, end):
        if labels[t] == -1:
            continue
        X_list.append(feat_mat[t - SEQ_LEN: t])
        y_list.append(labels[t])
        idx_list.append(t)
    if not X_list:
        return (np.empty((0, SEQ_LEN, INPUT_DIM), np.float32),
                np.empty(0, np.int64), np.empty(0, np.int64))
    return (np.stack(X_list).astype(np.float32),
            np.array(y_list, np.int64),
            np.array(idx_list, np.int64))


def normalize_seqs(X_tr, X_va):
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    m, s = flat.mean(0), flat.std(0)
    s = np.where(s < 1e-12, 1.0, s)
    return ((X_tr - m) / s).astype(np.float32), ((X_va - m) / s).astype(np.float32)


def train_lstm_fold(X_tr, y_tr, X_va, y_va, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr_n, X_va_n = normalize_seqs(X_tr, X_va)
    tr_ds = TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(y_tr).long())
    va_ds = TensorDataset(torch.from_numpy(X_va_n), torch.from_numpy(y_va).long())
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE * 4)
    model = CandleLSTM(INPUT_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
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
    model.load_state_dict(best_state)
    model.eval()
    all_proba = []
    with torch.no_grad():
        for xb, yb in DataLoader(va_ds, batch_size=BATCH_SIZE * 4):
            logits = model(xb.to(DEVICE))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            all_proba.append(proba)
    return np.vstack(all_proba), best_f1


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(close: np.ndarray, proba: np.ndarray, candle_indices: np.ndarray,
                 threshold: float) -> dict:
    """
    close:          full price series, shape (N,)
    proba:          (M, 3) probabilities [SELL, HOLD, BUY] for M val candles
    candle_indices: the candle index t in close for each val prediction
    threshold:      min confidence to enter trade

    Returns dict with equity curve and summary metrics.
    """
    equity = 1.0
    equity_curve = [equity]
    trade_returns = []
    n_buy = n_sell = n_hold_skip = n_low_conf = 0

    for i, t in enumerate(candle_indices):
        if t + 1 >= len(close):
            equity_curve.append(equity)
            continue

        conf = proba[i].max()
        signal = int(proba[i].argmax())   # 0=SELL, 1=HOLD, 2=BUY

        if conf < threshold or signal == 1:
            n_low_conf += 1
            equity_curve.append(equity)
            continue

        raw_ret = (close[t + 1] - close[t]) / close[t]

        if signal == 2:   # BUY → long
            trade_ret = raw_ret - 2 * FEE
            n_buy += 1
        else:             # SELL → short
            trade_ret = -raw_ret - 2 * FEE
            n_sell += 1

        equity *= (1 + trade_ret)
        trade_returns.append(trade_ret)
        equity_curve.append(equity)

    eq = np.array(equity_curve)
    n_trades = len(trade_returns)

    if n_trades == 0:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "n_trades": 0, "action_rate": 0.0,
                "n_buy": 0, "n_sell": 0, "equity_curve": eq}

    tr_arr = np.array(trade_returns)
    sharpe = (tr_arr.mean() / (tr_arr.std() + 1e-9)) * np.sqrt(HOURS_PER_YEAR)
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.min(eq / np.where(peak < 1e-12, 1.0, peak)) - 1)
    win_rate = float((tr_arr > 0).mean())
    action_rate = float(n_trades / len(candle_indices))

    return {
        "total_return": float(eq[-1] - 1),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": win_rate,
        "n_trades": n_trades,
        "action_rate": action_rate,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "equity_curve": eq,
    }


# ── Walk-forward prediction collection ───────────────────────────────────────

def collect_lstm_predictions(df, lstm_feat, labels, n_trainval):
    folds = walk_forward_ranges(n_trainval, n_splits=WF_N_SPLITS,
                                initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE)
    all_proba, all_true, all_idx = [], [], []

    for fold in folds:
        X_tr, y_tr, _ = build_sequences_with_indices(lstm_feat, labels,
                                                      fold.train_start, fold.train_end)
        X_va, y_va, va_idx = build_sequences_with_indices(lstm_feat, labels,
                                                           fold.val_start, fold.val_end)
        if len(y_tr) < 200 or len(y_va) < 50:
            continue

        print(f"  LSTM fold {fold.fold_id}: {len(y_tr)} train, {len(y_va)} val")
        # Average predictions over seeds
        seed_probas = []
        for seed in SEEDS:
            proba, best_f1 = train_lstm_fold(X_tr, y_tr, X_va, y_va, seed)
            seed_probas.append(proba)
            print(f"    seed={seed}: F1={best_f1:.4f}")
        avg_proba = np.mean(seed_probas, axis=0)
        all_proba.append(avg_proba)
        all_true.append(y_va)
        all_idx.append(va_idx)

    return (np.vstack(all_proba), np.concatenate(all_true), np.concatenate(all_idx))


def collect_et_predictions(df, cont_feat, labels, n_trainval):
    folds = walk_forward_ranges(n_trainval, n_splits=WF_N_SPLITS,
                                initial_train_size=WF_INITIAL_TRAIN, val_size=WF_VAL_SIZE)
    all_proba, all_true, all_idx = [], [], []

    for fold in folds:
        tr = np.arange(fold.train_start, fold.train_end)
        va = np.arange(fold.val_start, fold.val_end)
        tr_v = tr[labels[tr] != -1]; va_v = va[labels[va] != -1]
        if len(tr_v) < 500 or len(va_v) < 50:
            continue

        mean = cont_feat[tr_v].mean(0); std = cont_feat[tr_v].std(0)
        std = np.where(std < 1e-12, 1.0, std)
        X_tr = np.nan_to_num((cont_feat[tr_v] - mean) / std)
        X_va = np.nan_to_num((cont_feat[va_v] - mean) / std)

        seed_probas = []
        for seed in SEEDS:
            m = ExtraTreesClassifier(random_state=seed, n_estimators=300,
                                     max_depth=None, min_samples_leaf=20,
                                     max_features="sqrt", n_jobs=-1)
            m.fit(X_tr, labels[tr_v])
            seed_probas.append(m.predict_proba(X_va))
        all_proba.append(np.mean(seed_probas, axis=0))
        all_true.append(labels[va_v])
        all_idx.append(va_v)

    return (np.vstack(all_proba), np.concatenate(all_true), np.concatenate(all_idx))


# ── Buy & Hold ────────────────────────────────────────────────────────────────

def buy_and_hold(close, indices):
    if len(indices) == 0:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    returns = np.diff(close[indices]) / close[indices[:-1]]
    equity = np.cumprod(1 + returns)
    equity = np.concatenate([[1.0], equity])
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / np.where(peak < 1e-12, 1.0, peak)) - 1)
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(HOURS_PER_YEAR)
    return {"total_return": float(equity[-1] - 1), "sharpe": float(sharpe), "max_drawdown": max_dd}


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(bh, et_results, lstm_results, macro_f1_et, macro_f1_lstm):
    lines = [
        "# SBER H1 -- Backtest -- 2026-06-03", "",
        "## Strategy",
        "- Enter at close of signal candle t, exit at close of t+1 (1-hour hold)",
        f"- Fee: {FEE:.2%} one-way ({2*FEE:.2%} round-trip, standard MOEX retail)",
        "- Long if signal=BUY and conf>threshold; Short if signal=SELL and conf>threshold",
        f"- Walk-forward: {WF_N_SPLITS} folds, proba averaged over seeds {SEEDS}", "",
        "## Classification F1 (walk-forward, threshold-free)",
        f"- ExtraTrees macro-F1: {macro_f1_et:.4f}",
        f"- LSTM v2 macro-F1:    {macro_f1_lstm:.4f}", "",
        "## Buy & Hold baseline",
        f"- Total return: {bh['total_return']:+.2%}",
        f"- Sharpe:       {bh['sharpe']:.3f}",
        f"- Max drawdown: {bh['max_drawdown']:.2%}", "",
        "## ExtraTrees — Backtest by Threshold", "",
        "| Threshold | Total return | Sharpe | Max DD | Win rate | Trades | Action rate |",
        "|-----------|-------------|--------|--------|----------|--------|-------------|",
    ]
    for thr in THRESHOLDS:
        r = et_results[thr]
        lines.append(f"| {thr:.2f} | {r['total_return']:+.2%} | {r['sharpe']:.3f} | "
                     f"{r['max_drawdown']:.2%} | {r['win_rate']:.1%} | {r['n_trades']} | "
                     f"{r['action_rate']:.1%} |")

    lines += ["", "## LSTM v2 — Backtest by Threshold", "",
              "| Threshold | Total return | Sharpe | Max DD | Win rate | Trades | Action rate |",
              "|-----------|-------------|--------|--------|----------|--------|-------------|"]
    for thr in THRESHOLDS:
        r = lstm_results[thr]
        lines.append(f"| {thr:.2f} | {r['total_return']:+.2%} | {r['sharpe']:.3f} | "
                     f"{r['max_drawdown']:.2%} | {r['win_rate']:.1%} | {r['n_trades']} | "
                     f"{r['action_rate']:.1%} |")

    # Find best configs
    best_et = max(THRESHOLDS, key=lambda t: et_results[t]["sharpe"])
    best_lstm = max(THRESHOLDS, key=lambda t: lstm_results[t]["sharpe"])

    lines += ["", "## Conclusion", ""]
    et_best_sharpe = et_results[best_et]["sharpe"]
    lstm_best_sharpe = lstm_results[best_lstm]["sharpe"]

    if lstm_best_sharpe > 0.3 and lstm_best_sharpe > et_best_sharpe:
        verdict = (
            f"LSTM v2 is PRODUCTION-READY from a financial perspective.\n"
            f"Best threshold: {best_lstm:.2f} → Sharpe={lstm_best_sharpe:.3f}, "
            f"return={lstm_results[best_lstm]['total_return']:+.2%}\n\n"
            f"Recommendation: package LSTM v2 as new primary artifact.\n"
            f"Set risk_manager confidence threshold to {best_lstm:.2f}."
        )
    elif et_best_sharpe > 0.3:
        verdict = (
            f"ExtraTrees is PRODUCTION-READY. Best threshold: {best_et:.2f} → "
            f"Sharpe={et_best_sharpe:.3f}\n"
            f"LSTM v2 Sharpe={lstm_best_sharpe:.3f} — "
            f"{'better' if lstm_best_sharpe > et_best_sharpe else 'similar'}.\n\n"
            f"Recommendation: use LSTM v2 for directional calls, ET artifact for contract interface."
        )
    elif max(et_best_sharpe, lstm_best_sharpe) > 0:
        winner = "LSTM v2" if lstm_best_sharpe >= et_best_sharpe else "ExtraTrees"
        best_sharpe = max(et_best_sharpe, lstm_best_sharpe)
        verdict = (
            f"Both models show positive but weak Sharpe ({best_sharpe:.3f}).\n"
            f"{winner} is marginally better at threshold "
            f"{best_lstm if winner == 'LSTM v2' else best_et:.2f}.\n\n"
            f"Not yet production-ready by Sharpe criterion (target: > 0.5).\n"
            f"Next: improve model quality (Transformer, multi-ticker) OR lower the Sharpe bar\n"
            f"if team accepts that a positive-expectation model with high variance is acceptable."
        )
    else:
        verdict = (
            "Neither model shows positive Sharpe at any threshold.\n"
            "1H triple-barrier prediction does not yet translate to profitable trading.\n"
            "Root cause: high noise, low confidence signals, 1h hold period too short.\n\n"
            "Recommendations:\n"
            "1. Hold position for 3h (matching barrier horizon) instead of 1h\n"
            "2. Portfolio approach: trade only the highest-confidence signals\n"
            "3. Improve model quality (Transformer, more data)"
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
    close = df["close"].astype(float).values
    n_trainval = int(len(df) * 0.85)
    print(f"  {len(df)} candles, using first {n_trainval}")

    print("\nBuilding features...")
    lstm_feat = build_per_step_features(df)
    cont_feat, _ = make_continuous_past_features(df)
    print(f"  LSTM: {lstm_feat.shape}  ET: {cont_feat.shape}")

    # ── Collect predictions ───────────────────────────────────────────────────
    print(f"\n{'='*60}\nCollecting ExtraTrees predictions...")
    t0 = time.time()
    et_proba, et_true, et_idx = collect_et_predictions(df, cont_feat, labels, n_trainval)
    macro_f1_et = f1_score(et_true, et_proba.argmax(1), average="macro")
    print(f"  ET done in {time.time()-t0:.0f}s  |  macro-F1={macro_f1_et:.4f}  |  {len(et_idx)} val predictions")

    print(f"\n{'='*60}\nCollecting LSTM v2 predictions...")
    t0 = time.time()
    lstm_proba, lstm_true, lstm_idx = collect_lstm_predictions(df, lstm_feat, labels, n_trainval)
    macro_f1_lstm = f1_score(lstm_true, lstm_proba.argmax(1), average="macro")
    print(f"  LSTM done in {time.time()-t0:.0f}s  |  macro-F1={macro_f1_lstm:.4f}  |  {len(lstm_idx)} val predictions")

    # ── Backtest ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nRunning backtests...")

    # Buy & hold over all val periods
    bh = buy_and_hold(close, et_idx)
    print(f"  Buy&Hold: return={bh['total_return']:+.2%}  Sharpe={bh['sharpe']:.3f}  DD={bh['max_drawdown']:.2%}")

    et_results, lstm_results = {}, {}
    for thr in THRESHOLDS:
        et_results[thr] = run_backtest(close, et_proba, et_idx, thr)
        lstm_results[thr] = run_backtest(close, lstm_proba, lstm_idx, thr)
        print(f"  thr={thr:.2f}  ET: Sharpe={et_results[thr]['sharpe']:.3f} ret={et_results[thr]['total_return']:+.2%} "
              f"n={et_results[thr]['n_trades']}  |  "
              f"LSTM: Sharpe={lstm_results[thr]['sharpe']:.3f} ret={lstm_results[thr]['total_return']:+.2%} "
              f"n={lstm_results[thr]['n_trades']}")

    print(f"\n{'='*60}\nSUMMARY — Best Sharpe")
    best_et_thr = max(THRESHOLDS, key=lambda t: et_results[t]["sharpe"])
    best_lstm_thr = max(THRESHOLDS, key=lambda t: lstm_results[t]["sharpe"])
    print(f"  ET best:   thr={best_et_thr:.2f}  Sharpe={et_results[best_et_thr]['sharpe']:.3f}  return={et_results[best_et_thr]['total_return']:+.2%}")
    print(f"  LSTM best: thr={best_lstm_thr:.2f}  Sharpe={lstm_results[best_lstm_thr]['sharpe']:.3f}  return={lstm_results[best_lstm_thr]['total_return']:+.2%}")
    print(f"  Buy&Hold:           Sharpe={bh['sharpe']:.3f}  return={bh['total_return']:+.2%}")

    write_report(bh, et_results, lstm_results, macro_f1_et, macro_f1_lstm)


if __name__ == "__main__":
    main()
