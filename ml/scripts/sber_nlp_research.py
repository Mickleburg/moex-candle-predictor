"""Run paper-inspired candle-language research on SBER H1 candles."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
if str(ML_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ML_DIR / "scripts"))

from sber_hourly_research import fetch_moex_candles, save_raw_backend_contract
from src.data import clean_candles
from src.nlp import ClassifierSpec, ClusterSpec, ExperimentConfig, VectorizerSpec, run_experiment
from src.utils.io import ensure_dir, write_json


def find_latest_raw(raw_dir: Path, ticker: str, timeframe: str) -> Path | None:
    pattern = f"{ticker}_{timeframe}_*.parquet"
    files = sorted(raw_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_or_fetch_data(args: argparse.Namespace) -> tuple[pd.DataFrame, Path, str]:
    raw_dir = REPO_ROOT / args.raw_dir
    data_path = Path(args.data).resolve() if args.data else find_latest_raw(raw_dir, args.ticker, args.timeframe)
    source_url = ""

    if args.refresh_data or data_path is None:
        date_to = args.date_to or date.today().isoformat()
        print(f"Fetching {args.ticker} {args.timeframe} candles from MOEX: {args.date_from}..{date_to}")
        df_raw, source_url = fetch_moex_candles(args.ticker, args.timeframe, args.date_from, date_to)
        data_path = save_raw_backend_contract(df_raw, raw_dir)
    if data_path is None:
        raise FileNotFoundError("No raw parquet data found and refresh_data is disabled")

    df = pd.read_parquet(data_path)
    df = clean_candles(df)
    return df, data_path, source_url


def build_grid(args: argparse.Namespace) -> list[ExperimentConfig]:
    commission = args.commission
    shape_variants = ["ohlc", "shape"]
    horizons = [1, 3] if args.quick else [1, 3, 6]
    windows = [16, 32] if args.quick else [8, 16, 32]

    cluster_specs = [
        ClusterSpec("kmeans", {"n_clusters": 12, "init": "k-means++", "n_init": 20}),
        ClusterSpec("kmeans", {"n_clusters": 20, "init": "k-means++", "n_init": 20}),
        ClusterSpec("kmeans", {"n_clusters": 20, "init": "random", "n_init": 20}),
        ClusterSpec("kmeans", {"n_clusters": 32, "init": "k-means++", "n_init": 20}),
        ClusterSpec("minibatch_kmeans", {"n_clusters": 32, "n_init": 10, "batch_size": 512}),
        ClusterSpec("agglomerative", {"n_clusters": 20, "linkage": "ward"}),
        ClusterSpec("gmm", {"n_components": 20, "covariance_type": "diag", "reg_covar": 1e-6}),
        ClusterSpec("dbscan", {"eps": 0.38, "min_samples": 25}),
        ClusterSpec("hdbscan", {"min_cluster_size": 80, "min_samples": 15}),
    ]
    if not args.quick:
        cluster_specs.extend(
            [
                ClusterSpec("gmm", {"n_components": 32, "covariance_type": "diag", "reg_covar": 1e-6}),
                ClusterSpec("hdbscan", {"min_cluster_size": 160, "min_samples": 30}),
            ]
        )

    vector_specs = [
        VectorizerSpec("tfidf", {"ngram_range": (1, 2), "min_df": 2, "max_features": 3000}),
        VectorizerSpec("tfidf_svd", {"ngram_range": (1, 3), "min_df": 2, "max_features": 5000, "n_components": 32}),
        VectorizerSpec(
            "cooccurrence_svd",
            {"embedding_dim": 24, "context_window": 2, "pool": "mean+std+last", "include_histogram": True},
        ),
    ]
    if not args.quick:
        vector_specs.append(
            VectorizerSpec(
                "cooccurrence_svd",
                {"embedding_dim": 32, "context_window": 4, "pool": "mean+std+max+last", "include_histogram": True},
            )
        )

    classifier_specs = [
        ClassifierSpec("ridge", {"alpha": 1.0}),
        ClassifierSpec("linear_svc", {"C": 0.5}),
        ClassifierSpec("lightgbm", {"n_estimators": 250, "learning_rate": 0.03, "num_leaves": 31}),
        ClassifierSpec("extra_trees", {"n_estimators": 250, "max_depth": 12, "min_samples_leaf": 3}),
    ]
    if not args.quick:
        classifier_specs.extend(
            [
                ClassifierSpec("logreg", {"C": 0.5, "max_iter": 1200}),
                ClassifierSpec("mlp", {"hidden_layer_sizes": (96, 32), "alpha": 0.001, "max_iter": 180}),
            ]
        )

    grid = [
        ExperimentConfig(
            shape_variant=shape_variant,
            horizon=horizon,
            window_size=window,
            commission=commission,
            cluster=cluster,
            vectorizer=vectorizer,
            classifier=classifier,
        )
        for shape_variant in shape_variants
        for horizon in horizons
        for window in windows
        for cluster in cluster_specs
        for vectorizer in vector_specs
        for classifier in classifier_specs
    ]
    if args.limit and args.limit < len(grid):
        grid = evenly_sample_grid(grid, args.limit)
    return grid


def evenly_sample_grid(grid: list[ExperimentConfig], limit: int) -> list[ExperimentConfig]:
    """Sample a deterministic spread across the full grid."""

    if limit <= 0 or limit >= len(grid):
        return grid
    if limit == 1:
        return [grid[0]]

    max_index = len(grid) - 1
    selected_indices: list[int] = []
    seen: set[int] = set()
    for pos in range(limit):
        idx = round(pos * max_index / (limit - 1))
        if idx not in seen:
            selected_indices.append(idx)
            seen.add(idx)

    candidate = 0
    while len(selected_indices) < limit and candidate < len(grid):
        if candidate not in seen:
            selected_indices.append(candidate)
            seen.add(candidate)
        candidate += 1

    return [grid[idx] for idx in sorted(selected_indices)]


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    cfg = result["config"]
    return {
        "status": result.get("status", "ok"),
        "shape": cfg["shape_variant"],
        "horizon": cfg["horizon"],
        "window": cfg["window_size"],
        "cluster": cfg["cluster"]["name"],
        "cluster_params": json.dumps(cfg["cluster"]["params"], ensure_ascii=False, sort_keys=True),
        "vectorizer": cfg["vectorizer"]["name"],
        "classifier": cfg["classifier"]["name"],
        "val_macro_f1": result.get("metrics", {}).get("val", {}).get("macro_f1"),
        "val_accuracy": result.get("metrics", {}).get("val", {}).get("accuracy"),
        "test_macro_f1": result.get("metrics", {}).get("test", {}).get("macro_f1"),
        "test_accuracy": result.get("metrics", {}).get("test", {}).get("accuracy"),
        "test_trade_return": result.get("trading", {}).get("test", {}).get("total_return"),
        "test_trade_rate": result.get("trading", {}).get("test", {}).get("trade_rate"),
        "n_words": result.get("cluster", {}).get("n_words"),
        "noise_ratio": result.get("cluster", {}).get("noise_ratio"),
        "silhouette": result.get("cluster", {}).get("silhouette"),
        "duration_sec": result.get("duration_sec"),
        "error": result.get("error", ""),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_grid(df: pd.DataFrame, grid: list[ExperimentConfig], random_state: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(grid)
    for idx, config in enumerate(grid, start=1):
        started = time.perf_counter()
        print(f"[{idx:03d}/{total:03d}] {config.label}")
        try:
            result = run_experiment(df, config, random_state=random_state)
            result["status"] = "ok"
            val_f1 = result["metrics"]["val"]["macro_f1"]
            test_f1 = result["metrics"]["test"]["macro_f1"]
            print(f"    val_macro_f1={val_f1:.4f} test_macro_f1={test_f1:.4f}")
        except Exception as exc:
            result = {
                "status": "error",
                "label": config.label,
                "config": {
                    "shape_variant": config.shape_variant,
                    "horizon": config.horizon,
                    "window_size": config.window_size,
                    "commission": config.commission,
                    "cluster": {"name": config.cluster.name, "params": dict(config.cluster.params)},
                    "vectorizer": {"name": config.vectorizer.name, "params": dict(config.vectorizer.params)},
                    "classifier": {"name": config.classifier.name, "params": dict(config.classifier.params)},
                },
                "error": str(exc),
                "duration_sec": float(time.perf_counter() - started),
            }
            print(f"    ERROR: {exc}")
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--timeframe", default="1H")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data", default="")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--date-from", default="2020-01-01")
    parser.add_argument("--date-to", default="")
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--quick", action="store_true", help="Run a compact but diverse grid")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-json", default="data/reports/sber_h1_nlp_research_20260504.json")
    parser.add_argument("--output-csv", default="data/reports/sber_h1_nlp_research_20260504.csv")
    args = parser.parse_args()

    df, data_path, source_url = load_or_fetch_data(args)
    print(f"Loaded {len(df)} candles from {data_path}")
    if source_url:
        print(f"MOEX source: {source_url}")

    grid = build_grid(args)
    print(f"Running {len(grid)} experiments")
    results = run_grid(df, grid, random_state=args.random_state)
    ok_results = [result for result in results if result.get("status") == "ok"]
    rows = [result_row(result) for result in results]

    payload = {
        "data_path": str(data_path),
        "rows": len(df),
        "ticker": args.ticker,
        "timeframe": args.timeframe,
        "quick": bool(args.quick),
        "limit": int(args.limit),
        "random_state": int(args.random_state),
        "results": results,
    }
    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    write_json(payload, output_json)
    write_csv(rows, output_csv)

    if ok_results:
        best = max(
            ok_results,
            key=lambda item: (
                item["metrics"]["val"]["macro_f1"],
                item["metrics"]["val"]["accuracy"],
                item["metrics"]["test"]["macro_f1"],
            ),
        )
        print("Best by validation macro_f1:")
        print(best["label"])
        print(json.dumps(best["metrics"], ensure_ascii=False, indent=2))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
