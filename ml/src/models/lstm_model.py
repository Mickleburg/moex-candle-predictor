"""LSTM model for SBER H1 triple-barrier prediction (v2, 14-feature per-step input)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


SEQ_LEN = 32
INPUT_DIM = 14
FEATURE_NAMES = [
    "ret_1h", "ret_3h", "body", "range_", "upper_shadow", "lower_shadow",
    "close_pos", "vol_ratio", "vol_z",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "ema_dist_8",
]


class CandleLSTM(nn.Module):
    """Two-layer LSTM with linear head for BUY/HOLD/SELL classification."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_classes = n_classes

        self.lstm = nn.LSTM(
            input_dim, hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "n_classes": self.n_classes,
            "seq_len": SEQ_LEN,
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "CandleLSTM":
        return cls(
            input_dim=cfg["input_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            n_classes=cfg["n_classes"],
        )


def build_per_step_features(df: pd.DataFrame) -> np.ndarray:
    """Build 14-dim scale-invariant feature vector per timestep (past-only).

    All features are dimensionless ratios — no absolute price levels.
    Returns ndarray of shape (N, 14), NaN rows at window start filled with 0.
    """
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
        hour = np.zeros(len(df))
        dow  = np.zeros(len(df))

    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    dow_sin  = np.sin(2.0 * np.pi * dow  / 7.0)
    dow_cos  = np.cos(2.0 * np.pi * dow  / 7.0)

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
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
