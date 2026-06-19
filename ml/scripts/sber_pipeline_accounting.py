"""Print candle-language pipeline accounting for SBER H1 data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.nlp.accounting import (
    accounting_console_summary,
    build_nlp_accounting_report,
    load_raw_frame,
    raw_files_from_path,
)
from src.nlp.clustering import ClusterSpec


def find_latest_raw(raw_dir: Path, ticker: str, timeframe: str) -> Path:
    files = sorted(
        raw_dir.glob(f"{ticker}_{timeframe}_*.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No raw files found for {ticker} {timeframe} in {raw_dir}")
    return files[0]


def matching_raw_files(raw_dir: Path, ticker: str, timeframe: str) -> list[Path]:
    return sorted([*raw_dir.glob(f"{ticker}_{timeframe}_*.parquet"), *raw_dir.glob(f"{ticker}_{timeframe}_*.csv")])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--timeframe", default="1H")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--data", default="")
    parser.add_argument("--shape-variant", default="shape", choices=["ohlc", "shape", "ohlc_shape"])
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--cluster", default="gmm", choices=["kmeans", "gmm", "minibatch_kmeans"])
    parser.add_argument("--n-clusters", type=int, default=20)
    parser.add_argument("--include-calendar-diagnostics", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    raw_dir = REPO_ROOT / args.raw_dir
    data_path = Path(args.data).resolve() if args.data else find_latest_raw(
        raw_dir,
        args.ticker,
        args.timeframe,
    )
    raw_df = load_raw_frame(data_path)
    raw_files = raw_files_from_path(data_path)
    raw_candidates = matching_raw_files(raw_dir, args.ticker, args.timeframe)

    if "ticker" in raw_df.columns:
        raw_df = raw_df[raw_df["ticker"] == args.ticker]
    if "timeframe" in raw_df.columns:
        raw_df = raw_df[raw_df["timeframe"] == args.timeframe]

    cluster_params: dict[str, object]
    if args.cluster == "gmm":
        cluster_params = {"n_components": args.n_clusters, "covariance_type": "diag", "reg_covar": 1e-6}
    else:
        cluster_params = {"n_clusters": args.n_clusters}
        if args.cluster in {"kmeans", "minibatch_kmeans"}:
            cluster_params["n_init"] = 10

    report = build_nlp_accounting_report(
        raw_df,
        shape_variant=args.shape_variant,
        horizon=args.horizon,
        window_size=args.window_size,
        commission=args.commission,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        cluster=ClusterSpec(args.cluster, cluster_params),
        include_calendar_diagnostics=args.include_calendar_diagnostics,
        raw_files=raw_candidates or raw_files,
    )
    report["data_path"] = str(data_path)
    report["raw_file_selection"] = {
        "loaded_path": str(data_path),
        "candidate_count_for_ticker_timeframe": int(len(raw_candidates)),
        "candidate_paths": [str(path) for path in raw_candidates],
        "uses_latest_file_when_data_not_specified": not bool(args.data),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(accounting_console_summary(report))
    print(text)
    if args.output_json:
        output_path = REPO_ROOT / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
