"""Download SBER H1 candles and compare implemented model families.

The downloader mirrors the backend MOEX ISS request contract and writes the
same raw Parquet schema consumed by the ML pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.data import clean_candles, load_candles, time_split
from src.evaluation.metrics import compute_classification_metrics
from src.features import CandleTokenizer, compute_all_indicators, resolve_feature_columns
from src.models import LGBMClassifier, LogisticRegressionBaseline, MajorityClassifier, MarkovClassifier
from src.utils.io import ensure_dir, write_json


RAW_SCHEMA = [
    "ticker",
    "timeframe",
    "begin",
    "end",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "value",
    "source",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: dict[str, Any]

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        suffix = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({suffix})"


def fetch_moex_candles(
    ticker: str,
    timeframe: str,
    date_from: str,
    date_to: str,
    page_size: int = 500,
) -> tuple[pd.DataFrame, str]:
    """Fetch all pages from MOEX ISS candles endpoint."""
    interval_by_timeframe = {"1M": 1, "10M": 10, "1H": 60, "1D": 24}
    interval = interval_by_timeframe[timeframe.upper()]
    endpoint = (
        "https://iss.moex.com/iss/engines/stock/markets/shares/boards/"
        f"TQBR/securities/{ticker}/candles.json"
    )
    headers = {
        "Accept": "application/json",
        "User-Agent": "moex-candle-predictor-backend/0.1",
    }

    rows: list[dict[str, Any]] = []
    first_url = ""
    start = 0

    while True:
        params = {
            "iss.meta": "off",
            "iss.only": "candles",
            "from": date_from,
            "till": date_to,
            "interval": str(interval),
            "start": str(start),
        }
        request_url = endpoint + "?" + urllib.parse.urlencode(params)
        if not first_url:
            first_url = request_url

        request = urllib.request.Request(request_url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)

        block = payload.get("candles", {})
        columns = block.get("columns", [])
        data = block.get("data", [])
        if not data:
            break

        index = {column: i for i, column in enumerate(columns)}
        for item in data:
            rows.append(
                {
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "begin": item[index["begin"]],
                    "end": item[index["end"]],
                    "open": item[index["open"]],
                    "high": item[index["high"]],
                    "low": item[index["low"]],
                    "close": item[index["close"]],
                    "volume": item[index["volume"]],
                    "value": item[index["value"]],
                    "source": "moex",
                }
            )

        start += page_size
        if len(data) < page_size:
            break
        if start % 5000 == 0:
            print(f"Fetched {start} rows...")
        time.sleep(0.03)

    if not rows:
        raise ValueError("MOEX returned no candle rows")

    df = pd.DataFrame(rows, columns=RAW_SCHEMA)
    df["begin"] = pd.to_datetime(df["begin"])
    df["end"] = pd.to_datetime(df["end"])
    df = df.drop_duplicates(subset=["begin"], keep="last").sort_values("begin")
    return df.reset_index(drop=True), first_url


def save_raw_backend_contract(df: pd.DataFrame, raw_dir: Path) -> Path:
    """Write candles to backend-compatible raw Parquet file."""
    ensure_dir(raw_dir)
    ticker = df["ticker"].iloc[0]
    timeframe = df["timeframe"].iloc[0]
    begin = df["begin"].iloc[0].strftime("%Y%m%dT%H%M")
    end = df["begin"].iloc[-1].strftime("%Y%m%dT%H%M")
    path = raw_dir / f"{ticker}_{timeframe}_{begin}_{end}.parquet"
    df.to_parquet(path, index=False)
    return path


def build_tabular_windows_with_indices(
    features_df: pd.DataFrame,
    tokens: np.ndarray,
    window_size: int,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = features_df[feature_cols].values
    X_list, y_list, idx_list = [], [], []
    for start in range(len(features_df) - window_size + 1):
        target_idx = start + window_size - 1
        y_value = tokens[target_idx]
        if y_value == -1:
            continue
        X_list.append(matrix[start:start + window_size].flatten())
        y_list.append(y_value)
        idx_list.append(target_idx)
    return np.array(X_list), np.array(y_list), np.array(idx_list)


def build_token_windows_with_indices(
    tokens: np.ndarray,
    window_size: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list, y_list, idx_list = [], [], []
    for target_idx in range(window_size + horizon - 1, len(tokens)):
        context_end = target_idx - horizon + 1
        context_start = context_end - window_size
        window_tokens = tokens[context_start:context_end]
        y_value = tokens[target_idx]
        if y_value == -1 or np.any(window_tokens == -1):
            continue
        X_list.append(window_tokens)
        y_list.append(y_value)
        idx_list.append(target_idx)
    return np.array(X_list), np.array(y_list), np.array(idx_list)


def token_signal(tokens: np.ndarray, n_classes: int) -> np.ndarray:
    mid = n_classes // 2
    return np.where(tokens > mid, 1, np.where(tokens < mid, -1, 0))


def future_returns(df: pd.DataFrame, indices: np.ndarray, horizon: int) -> np.ndarray:
    close = df["close"].to_numpy()
    out = []
    for idx in indices:
        future_idx = idx + horizon
        if future_idx >= len(close):
            out.append(np.nan)
        else:
            out.append((close[future_idx] - close[idx]) / close[idx])
    return np.array(out)


def trading_summary(
    pred_tokens: np.ndarray,
    true_tokens: np.ndarray,
    realized_returns: np.ndarray,
    n_classes: int,
    commission: float = 0.0005,
) -> dict[str, float]:
    pred = token_signal(pred_tokens, n_classes)
    true = token_signal(true_tokens, n_classes)
    valid = ~np.isnan(realized_returns)
    pred = pred[valid]
    true = true[valid]
    returns = realized_returns[valid]

    trade_costs = np.abs(np.diff(pred, prepend=0)) * commission
    strategy_returns = pred * returns - trade_costs
    equity = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(equity) if len(equity) else np.array([1.0])
    drawdown = (running_max - equity) / running_max if len(equity) else np.array([0.0])
    active = pred != 0

    return {
        "action_accuracy": float((pred == true).mean()) if len(true) else 0.0,
        "active_direction_hit_rate": float((np.sign(returns[active]) == pred[active]).mean()) if active.any() else 0.0,
        "active_share": float(active.mean()) if len(active) else 0.0,
        "total_return": float(equity[-1] - 1) if len(equity) else 0.0,
        "sharpe_per_trade": float(strategy_returns.mean() / strategy_returns.std()) if strategy_returns.std() > 0 else 0.0,
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "n_trades": int(np.sum(np.abs(np.diff(pred, prepend=0)) > 0)),
    }


def make_model(spec: ModelSpec, n_classes: int):
    if spec.name == "majority":
        return MajorityClassifier(n_classes=n_classes)
    if spec.name == "markov":
        return MarkovClassifier(n_classes=n_classes, order=spec.params.get("order", 1))
    if spec.name == "logistic":
        return LogisticRegressionBaseline(n_classes=n_classes, random_state=42)
    if spec.name == "lgbm":
        return LGBMClassifier(n_classes=n_classes, random_state=42, verbose=-1, **spec.params)
    raise ValueError(f"Unsupported model: {spec.name}")


def evaluate_spec(
    spec: ModelSpec,
    n_classes: int,
    horizon: int,
    window_size: int,
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> dict[str, Any]:
    tokenizer = CandleTokenizer(n_bins=n_classes, horizon=horizon, random_state=42)
    train_tokens = tokenizer.fit_transform(train_features)
    val_tokens = tokenizer.transform(val_features)
    test_tokens = tokenizer.transform(test_features)

    if spec.name == "markov":
        X_train, y_train, _ = build_token_windows_with_indices(train_tokens, window_size, horizon)
        X_val, y_val, val_idx = build_token_windows_with_indices(val_tokens, window_size, horizon)
        X_test, y_test, test_idx = build_token_windows_with_indices(test_tokens, window_size, horizon)
    else:
        feature_cols = resolve_feature_columns(train_features)
        X_train, y_train, _ = build_tabular_windows_with_indices(train_features, train_tokens, window_size, feature_cols)
        X_val, y_val, val_idx = build_tabular_windows_with_indices(val_features, val_tokens, window_size, feature_cols)
        X_test, y_test, test_idx = build_tabular_windows_with_indices(test_features, test_tokens, window_size, feature_cols)

    model = make_model(spec, n_classes)
    if spec.name == "lgbm":
        model.fit(X_train, y_train, X_val, y_val)
    else:
        model.fit(X_train, y_train)

    def split_metrics(X, y, idx, features):
        pred = model.predict(X)
        proba = model.predict_proba(X)
        cls = compute_classification_metrics(y, pred, proba)
        trade = trading_summary(pred, y, future_returns(features, idx, horizon), n_classes)
        return cls, trade

    val_cls, val_trade = split_metrics(X_val, y_val, val_idx, val_features)
    test_cls, test_trade = split_metrics(X_test, y_test, test_idx, test_features)

    return {
        "model": spec.name,
        "model_label": spec.label,
        "model_params": spec.params,
        "K": n_classes,
        "horizon": horizon,
        "window_size": window_size,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "val": {**val_cls, **{f"trade_{k}": v for k, v in val_trade.items()}},
        "test": {**test_cls, **{f"trade_{k}": v for k, v in test_trade.items()}},
    }


def model_grid() -> list[ModelSpec]:
    return [
        ModelSpec("majority", {}),
        ModelSpec("markov", {"order": 1}),
        ModelSpec("markov", {"order": 2}),
        ModelSpec("logistic", {}),
        ModelSpec(
            "lgbm",
            {
                "n_estimators": 160,
                "max_depth": 3,
                "learning_rate": 0.04,
                "num_leaves": 15,
                "min_child_samples": 80,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
            },
        ),
        ModelSpec(
            "lgbm",
            {
                "n_estimators": 220,
                "max_depth": 4,
                "learning_rate": 0.025,
                "num_leaves": 31,
                "min_child_samples": 50,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.05,
                "reg_lambda": 0.5,
            },
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-date", default="2020-01-01")
    parser.add_argument("--to-date", default=date.today().isoformat())
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    raw_dir = REPO_ROOT / "data" / "raw"
    reports_dir = REPO_ROOT / "data" / "reports"
    ensure_dir(reports_dir)

    source_url = None
    raw_path = None
    if not args.skip_download:
        candles, source_url = fetch_moex_candles("SBER", "1H", args.from_date, args.to_date)
        raw_path = save_raw_backend_contract(candles, raw_dir)
        print(f"Saved {len(candles)} candles to {raw_path}")

    df = load_candles(raw_dir, ticker="SBER", timeframe="1H", format="parquet")
    df = clean_candles(df, drop_invalid=True)
    train_df, val_df, test_df = time_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, min_train_size=1000)
    train_features = compute_all_indicators(train_df)
    val_features = compute_all_indicators(val_df)
    test_features = compute_all_indicators(test_df)

    if args.quick:
        classes = [3, 5]
        horizons = [1, 3]
        windows = [16, 32]
        specs = [spec for spec in model_grid() if spec.name in {"majority", "markov", "lgbm"}]
    else:
        classes = [3, 5, 7]
        horizons = [1, 3, 6]
        windows = [16, 32, 64]
        specs = model_grid()

    results = []
    for n_classes in classes:
        for horizon in horizons:
            for window_size in windows:
                for spec in specs:
                    if spec.name == "logistic" and window_size == 64:
                        continue
                    print(f"Evaluating {spec.label}, K={n_classes}, h={horizon}, L={window_size}")
                    try:
                        results.append(
                            evaluate_spec(
                                spec,
                                n_classes,
                                horizon,
                                window_size,
                                train_features,
                                val_features,
                                test_features,
                            )
                        )
                    except Exception as exc:
                        results.append(
                            {
                                "model": spec.name,
                                "model_label": spec.label,
                                "model_params": spec.params,
                                "K": n_classes,
                                "horizon": horizon,
                                "window_size": window_size,
                                "error": str(exc),
                            }
                        )
                        print(f"  failed: {exc}")

    successful = [item for item in results if "error" not in item]
    successful.sort(
        key=lambda item: (
            item["val"].get("macro_f1", -1),
            item["val"].get("trade_action_accuracy", -1),
            item["test"].get("macro_f1", -1),
        ),
        reverse=True,
    )
    best = successful[0] if successful else None

    report = {
        "dataset": {
            "ticker": "SBER",
            "timeframe": "1H",
            "requested_from": args.from_date,
            "requested_to": args.to_date,
            "source_url": source_url,
            "raw_path": str(raw_path) if raw_path else None,
            "rows_after_cleaning": int(len(df)),
            "begin": df["begin"].min().isoformat(),
            "end": df["begin"].max().isoformat(),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "selection": {
            "criterion": "validation macro_f1, then validation action accuracy, then test macro_f1",
            "best": best,
        },
        "top_10": successful[:10],
        "all_results": results,
    }

    output_path = reports_dir / "sber_h1_model_research_20260503.json"
    write_json(report, output_path)
    print(f"Research report saved to {output_path}")
    if best:
        print("Best:", json.dumps(best, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
