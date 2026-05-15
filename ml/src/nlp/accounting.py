"""Accounting diagnostics for candle-language data flow."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.clean import clean_candles
from .candles import candle_shape_matrix, label_distribution, make_action_labels, make_sentence_samples, split_ranges
from .clustering import CandleClusterer, ClusterSpec


def build_nlp_accounting_report(
    raw_df: pd.DataFrame,
    *,
    shape_variant: str = "shape",
    horizon: int = 1,
    window_size: int = 32,
    commission: float = 0.0005,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    cluster: ClusterSpec | None = None,
    random_state: int = 42,
    include_calendar_diagnostics: bool = False,
    raw_files: list[Path] | None = None,
) -> dict[str, Any]:
    """Return counts and alignment checks for the candle-language pipeline."""

    raw_df = raw_df.copy()
    raw_summary = _candle_frame_summary(raw_df)
    cleaned_df = clean_candles(raw_df, drop_invalid=True)
    cleaned_summary = _candle_frame_summary(cleaned_df)

    df = cleaned_df.sort_values("begin").reset_index(drop=True) if "begin" in cleaned_df.columns else cleaned_df
    ranges = split_ranges(len(df), train_ratio=train_ratio, val_ratio=val_ratio)
    shape_matrix, shape_columns = candle_shape_matrix(df, variant=shape_variant)
    labels, future_returns, threshold = make_action_labels(
        df,
        horizon=horizon,
        commission=commission,
    )

    cluster = cluster or ClusterSpec("gmm", {"n_components": 20, "covariance_type": "diag", "reg_covar": 1e-6})
    train_start, train_end = ranges["train"]
    clusterer = CandleClusterer(cluster, random_state=random_state)
    clusterer.fit(shape_matrix[train_start:train_end])

    word_ids = np.empty(len(df), dtype=int)
    word_ids[train_start:train_end] = clusterer.train_labels_
    for split_name in ("val", "test"):
        split_start, split_end = ranges[split_name]
        word_ids[split_start:split_end] = clusterer.predict(shape_matrix[split_start:split_end])
    word_tokens = clusterer.labels_to_words(word_ids)

    split_reports: dict[str, Any] = {}
    for split_name, (split_start, split_end) in ranges.items():
        samples = make_sentence_samples(
            word_tokens,
            labels,
            future_returns,
            split_start,
            split_end,
            window_size,
            horizon,
        )
        split_reports[split_name] = _split_accounting(
            df,
            samples.target_indices,
            labels,
            split_start,
            split_end,
            window_size,
            horizon,
        ) | {
            "sample_count": samples.size,
            "target_distribution": label_distribution(samples.y),
        }

    expected_valid_labels = max(0, len(df) - horizon)
    report = {
        "parameters": {
            "shape_variant": shape_variant,
            "horizon": int(horizon),
            "window_size": int(window_size),
            "commission": float(commission),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "cluster": {"name": cluster.name, "params": dict(cluster.params)},
        },
        "raw": raw_summary,
        "cleaned": cleaned_summary,
        "losses": {
            "cleaning": int(len(raw_df) - len(df)),
            "nan_or_invalid_labels_total": int(len(df) - np.count_nonzero(labels >= 0)),
            "expected_horizon_label_loss_total": int(len(df) - expected_valid_labels),
        },
        "split_ranges": {name: [int(start), int(end)] for name, (start, end) in ranges.items()},
        "shape_matrix": {
            "rows": int(shape_matrix.shape[0]),
            "columns": shape_columns,
            "nonfinite_values": int(np.size(shape_matrix) - np.count_nonzero(np.isfinite(shape_matrix))),
        },
        "word_assignment": {
            "rows": int(len(word_tokens)),
            "n_words": int(clusterer.n_words_),
            "cluster_quality": clusterer.quality_,
            "missing_words": int(sum(token is None for token in word_tokens)),
        },
        "labels": {
            "rows": int(len(labels)),
            "valid_rows": int(np.count_nonzero(labels >= 0)),
            "invalid_rows": int(np.count_nonzero(labels < 0)),
            "threshold": float(threshold),
            "distribution": label_distribution(labels[labels >= 0]),
        },
        "splits": split_reports,
        "checks": _checks(
            df,
            ranges,
            split_reports,
            len(raw_df),
            len(cleaned_df),
            shape_matrix,
            word_tokens,
            labels,
            horizon,
            cleaned_summary,
        ),
    }
    if include_calendar_diagnostics:
        report["calendar"] = build_calendar_diagnostics(df)
    if raw_files is not None:
        report["raw_files"] = build_raw_file_report(raw_files)
    return report


def load_raw_frame(path: str | Path) -> pd.DataFrame:
    """Load a raw parquet/csv file or all parquet files in a directory."""

    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def raw_files_from_path(path: str | Path) -> list[Path]:
    """Return source raw files that would be loaded from a file or directory."""

    path = Path(path)
    if path.is_dir():
        return sorted([*path.glob("*.parquet"), *path.glob("*.csv")])
    return [path]


def build_raw_file_report(files: list[Path]) -> dict[str, Any]:
    """Return per-file coverage and interval overlap diagnostics."""

    file_reports = []
    intervals = []
    for file in files:
        if not file.exists():
            file_reports.append({"path": str(file), "exists": False})
            continue
        df = pd.read_csv(file) if file.suffix.lower() == ".csv" else pd.read_parquet(file)
        item: dict[str, Any] = {
            "path": str(file),
            "exists": True,
            "rows": int(len(df)),
            "columns": list(df.columns),
        }
        if "begin" in df.columns and len(df):
            begin = pd.to_datetime(df["begin"])
            begin_min = begin.min()
            begin_max = begin.max()
            item.update(
                {
                    "begin_min": begin_min.isoformat(),
                    "begin_max": begin_max.isoformat(),
                    "duplicate_begin_rows": int(begin.duplicated().sum()),
                }
            )
            intervals.append((begin_min, begin_max, file))
        file_reports.append(item)

    intervals.sort(key=lambda item: item[0])
    overlaps = []
    gaps = []
    for prev, curr in zip(intervals, intervals[1:]):
        prev_start, prev_end, prev_file = prev
        curr_start, curr_end, curr_file = curr
        if curr_start <= prev_end:
            overlaps.append(
                {
                    "previous_file": str(prev_file),
                    "current_file": str(curr_file),
                    "overlap_start": curr_start.isoformat(),
                    "overlap_end": min(prev_end, curr_end).isoformat(),
                }
            )
        else:
            gaps.append(
                {
                    "previous_file": str(prev_file),
                    "current_file": str(curr_file),
                    "from": prev_end.isoformat(),
                    "to": curr_start.isoformat(),
                    "delta": str(curr_start - prev_end),
                }
            )

    return {
        "file_count": int(len(files)),
        "files": file_reports,
        "overlaps": overlaps,
        "gaps_between_files": gaps,
    }


def build_calendar_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """Describe trading-date, hour, and gap structure of candle timestamps."""

    if "begin" not in df.columns or df.empty:
        return {}

    begin = pd.to_datetime(df["begin"]).sort_values().reset_index(drop=True)
    first = begin.iloc[0]
    last = begin.iloc[-1]
    all_dates = pd.date_range(first.normalize(), last.normalize(), freq="D")
    weekdays = all_dates[all_dates.dayofweek < 5]
    date_counts = begin.dt.normalize().value_counts().sort_index()
    candles_per_day = date_counts.astype(int)

    naive_24h = int(((last.floor("h") - first.floor("h")) / pd.Timedelta(hours=1)) + 1)
    weekday_24h = int(sum(24 for day in all_dates if day.dayofweek < 5))
    observed = int(len(begin))

    return {
        "date_range": {
            "first_begin": first.isoformat(),
            "last_begin": last.isoformat(),
            "total_calendar_days": int(len(all_dates)),
            "weekdays_in_range": int(len(weekdays)),
            "unique_trading_dates": int(candles_per_day.size),
            "dates_with_at_least_one_candle": int(candles_per_day.size),
            "timestamp_note": "begin is treated as timezone-naive local exchange timestamp",
        },
        "expectations": {
            "naive_24_7_expected_bars": naive_24h,
            "weekday_only_24h_expected_bars": weekday_24h,
            "observed_candles": observed,
            "observed_to_naive_ratio": float(observed / naive_24h) if naive_24h else 0.0,
            "observed_to_weekday_24h_ratio": float(observed / weekday_24h) if weekday_24h else 0.0,
        },
        "candles_per_day": _candles_per_day_stats(candles_per_day),
        "weekend_trading_dates": _weekend_trading_dates(candles_per_day),
        "hour_distribution": _hour_distribution(begin),
        "gap_analysis": _gap_analysis(begin),
        "per_day_continuity": _per_day_continuity(begin),
    }


def accounting_console_summary(report: dict[str, Any]) -> str:
    """Build a compact human-readable summary for CLI output."""

    lines = [
        "Pipeline accounting summary",
        f"  raw rows: {report.get('raw', {}).get('rows')}",
        f"  cleaned rows: {report.get('cleaned', {}).get('rows')}",
        f"  cleaning loss: {report.get('losses', {}).get('cleaning')}",
        f"  split ranges: {report.get('split_ranges')}",
        f"  checks: {report.get('checks')}",
    ]
    calendar = report.get("calendar")
    if calendar:
        expectations = calendar["expectations"]
        date_range = calendar["date_range"]
        continuity = calendar["per_day_continuity"]
        gaps = calendar["gap_analysis"]
        lines.extend(
            [
                "Calendar summary",
                f"  first/last: {date_range['first_begin']} .. {date_range['last_begin']}",
                (
                    "  observed vs naive 24/7: "
                    f"{expectations['observed_candles']} / {expectations['naive_24_7_expected_bars']} "
                    f"({expectations['observed_to_naive_ratio']:.3f})"
                ),
                (
                    "  observed vs weekday 24h: "
                    f"{expectations['observed_candles']} / {expectations['weekday_only_24h_expected_bars']} "
                    f"({expectations['observed_to_weekday_24h_ratio']:.3f})"
                ),
                f"  unique trading dates: {date_range['unique_trading_dates']}",
                (
                    "  weekend dates with candles: "
                    f"sat={calendar['weekend_trading_dates']['saturday_dates_with_candles']}, "
                    f"sun={calendar['weekend_trading_dates']['sunday_dates_with_candles']}"
                ),
                f"  hours present: {calendar['hour_distribution']['hours_present']}",
                f"  days with intraday gaps >1h: {continuity['days_with_intraday_gaps_gt_1h']}",
                f"  gap categories: {gaps['category_counts']}",
            ]
        )
    raw_files = report.get("raw_files")
    if raw_files:
        lines.extend(
            [
                "Raw file coverage",
                f"  file count: {raw_files['file_count']}",
                f"  overlaps: {len(raw_files['overlaps'])}",
                f"  gaps between files: {len(raw_files['gaps_between_files'])}",
            ]
        )
    return "\n".join(lines)


def _candle_frame_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": int(len(df)), "columns": list(df.columns)}
    if "begin" not in df.columns:
        return summary

    begin = pd.to_datetime(df["begin"])
    summary.update(
        {
            "begin_min": begin.min().isoformat() if len(begin) else None,
            "begin_max": begin.max().isoformat() if len(begin) else None,
            "duplicate_begin_rows": int(begin.duplicated().sum()),
            "is_monotonic_increasing": bool(begin.is_monotonic_increasing),
            "null_begin_rows": int(begin.isna().sum()),
            "time_delta_counts_top": _time_delta_counts(begin),
            "gaps_gt_1h": _gap_examples(begin, pd.Timedelta(hours=1)),
        }
    )

    ohlcv = [column for column in ("open", "high", "low", "close", "volume") if column in df.columns]
    if ohlcv:
        summary["missing_ohlcv_rows"] = int(df[ohlcv].isna().any(axis=1).sum())
    if {"open", "high", "low", "close"}.issubset(df.columns):
        invalid = (
            (df["high"] < df["low"])
            | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
            | (df["high"] < df[["open", "close"]].max(axis=1))
            | (df["low"] > df[["open", "close"]].min(axis=1))
        )
        summary["invalid_ohlc_rows"] = int(invalid.sum())
    return summary


def _candles_per_day_stats(candles_per_day: pd.Series) -> dict[str, Any]:
    quantiles = candles_per_day.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    top = candles_per_day.sort_values(ascending=False).head(10)
    bottom = candles_per_day.sort_values(ascending=True).head(10)
    return {
        "min": int(candles_per_day.min()),
        "max": int(candles_per_day.max()),
        "mean": float(candles_per_day.mean()),
        "median": float(candles_per_day.median()),
        "quantiles": {str(key): float(value) for key, value in quantiles.items()},
        "top_dates_by_count": _date_count_rows(top),
        "bottom_dates_by_count": _date_count_rows(bottom),
    }


def _weekend_trading_dates(candles_per_day: pd.Series) -> dict[str, Any]:
    weekend = candles_per_day[candles_per_day.index.dayofweek >= 5]
    saturday = weekend[weekend.index.dayofweek == 5]
    sunday = weekend[weekend.index.dayofweek == 6]
    return {
        "saturday_dates_with_candles": int(len(saturday)),
        "sunday_dates_with_candles": int(len(sunday)),
        "weekend_dates_with_candles": int(len(weekend)),
        "first_20_weekend_dates": _date_count_rows(weekend.head(20)),
    }


def _date_count_rows(series: pd.Series) -> list[dict[str, Any]]:
    return [{"date": idx.date().isoformat(), "count": int(value)} for idx, value in series.items()]


def _hour_distribution(begin: pd.Series) -> dict[str, Any]:
    counts = begin.dt.hour.value_counts().sort_index()
    return {
        "hours_present": [int(hour) for hour in counts.index.tolist()],
        "counts_by_hour": {str(int(hour)): int(count) for hour, count in counts.items()},
        "missing_hours_0_23": [hour for hour in range(24) if hour not in set(counts.index.tolist())],
    }


def _gap_analysis(begin: pd.Series) -> dict[str, Any]:
    deltas = begin.diff()
    category_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for pos in range(1, len(begin)):
        prev = begin.iloc[pos - 1]
        curr = begin.iloc[pos]
        delta = curr - prev
        category = _gap_category(prev, curr, delta)
        category_counts[category] += 1
        if delta > pd.Timedelta(hours=1):
            examples.append({"from": prev.isoformat(), "to": curr.isoformat(), "delta": str(delta), "category": category})

    largest_positions = np.argsort(deltas.fillna(pd.Timedelta(0)).to_numpy())[-10:][::-1]
    largest = []
    for pos in largest_positions:
        if pos == 0:
            continue
        prev = begin.iloc[pos - 1]
        curr = begin.iloc[pos]
        delta = curr - prev
        largest.append(
            {
                "from": prev.isoformat(),
                "to": curr.isoformat(),
                "delta": str(delta),
                "category": _gap_category(prev, curr, delta),
            }
        )

    return {
        "category_counts": {key: int(value) for key, value in sorted(category_counts.items())},
        "gaps_gt_1h_count": int(sum(1 for delta in deltas.dropna() if delta > pd.Timedelta(hours=1))),
        "gap_examples": examples[:20],
        "largest_gaps": largest,
    }


def _gap_category(prev: pd.Timestamp, curr: pd.Timestamp, delta: pd.Timedelta) -> str:
    if delta == pd.Timedelta(hours=1):
        return "exactly_1h"
    if prev.normalize() == curr.normalize():
        if delta <= pd.Timedelta(hours=3):
            return "intraday_2_3h"
        return "intraday_gt3h"
    if delta >= pd.Timedelta(days=7):
        return "very_large"
    if _contains_weekend(prev.normalize(), curr.normalize()):
        return "weekend_or_holiday"
    return "overnight_or_holiday"


def _contains_weekend(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    for day in pd.date_range(start, end, freq="D"):
        if day.dayofweek >= 5:
            return True
    return False


def _per_day_continuity(begin: pd.Series) -> dict[str, Any]:
    frame = pd.DataFrame({"begin": begin})
    frame["date"] = frame["begin"].dt.normalize()
    days = []
    for date_value, group in frame.groupby("date", sort=True):
        ordered = group["begin"].sort_values().reset_index(drop=True)
        deltas = ordered.diff().dropna()
        gap_positions = np.flatnonzero(deltas > pd.Timedelta(hours=1))
        if len(gap_positions):
            gaps = []
            for pos in gap_positions[:10]:
                gaps.append(
                    {
                        "from": ordered.iloc[pos].isoformat(),
                        "to": ordered.iloc[pos + 1].isoformat(),
                        "delta": str(ordered.iloc[pos + 1] - ordered.iloc[pos]),
                    }
                )
            span_expected = int(((ordered.iloc[-1] - ordered.iloc[0]) / pd.Timedelta(hours=1)) + 1)
            days.append(
                {
                    "date": date_value.date().isoformat(),
                    "first": ordered.iloc[0].isoformat(),
                    "last": ordered.iloc[-1].isoformat(),
                    "candles": int(len(ordered)),
                    "span_expected_hourly_bars": span_expected,
                    "missing_inside_observed_span": int(span_expected - len(ordered)),
                    "gaps": gaps,
                }
            )

    return {
        "days_with_intraday_gaps_gt_1h": int(len(days)),
        "examples": days[:20],
    }


def _time_delta_counts(begin: pd.Series, top_n: int = 8) -> list[dict[str, Any]]:
    deltas = begin.sort_values().diff().dropna()
    counts = Counter(str(delta) for delta in deltas)
    return [{"delta": delta, "count": int(count)} for delta, count in counts.most_common(top_n)]


def _gap_examples(begin: pd.Series, threshold: pd.Timedelta, limit: int = 10) -> dict[str, Any]:
    ordered = begin.sort_values().reset_index(drop=True)
    deltas = ordered.diff()
    positions = np.flatnonzero(deltas > threshold)
    examples = []
    for pos in positions[:limit]:
        examples.append(
            {
                "from": ordered.iloc[pos - 1].isoformat(),
                "to": ordered.iloc[pos].isoformat(),
                "delta": str(deltas.iloc[pos]),
            }
        )
    return {"count": int(len(positions)), "examples": examples}


def _split_accounting(
    df: pd.DataFrame,
    target_indices: np.ndarray,
    labels: np.ndarray,
    split_start: int,
    split_end: int,
    window_size: int,
    horizon: int,
) -> dict[str, Any]:
    split_len = split_end - split_start
    expected_samples = max(0, split_len - window_size - horizon + 1)
    first_target = int(split_start + window_size - 1)
    last_target_inclusive = int(split_end - horizon - 1)
    invalid_labels_inside_usable_range = 0
    if first_target <= last_target_inclusive:
        usable_labels = labels[first_target : last_target_inclusive + 1]
        invalid_labels_inside_usable_range = int(np.count_nonzero(usable_labels < 0))

    result: dict[str, Any] = {
        "rows": int(split_len),
        "expected_samples_without_invalids": int(expected_samples),
        "lost_to_window_warmup": int(min(window_size - 1, split_len)),
        "lost_to_horizon_tail": int(min(horizon, max(0, split_len - (window_size - 1)))),
        "invalid_labels_inside_usable_range": invalid_labels_inside_usable_range,
        "first_target_index": int(target_indices[0]) if len(target_indices) else None,
        "last_target_index": int(target_indices[-1]) if len(target_indices) else None,
        "min_allowed_target_index": first_target if expected_samples else None,
        "max_allowed_target_index": last_target_inclusive if expected_samples else None,
    }

    if len(target_indices) and "begin" in df.columns:
        result["first_target_begin"] = df["begin"].iloc[int(target_indices[0])].isoformat()
        result["last_target_begin"] = df["begin"].iloc[int(target_indices[-1])].isoformat()
    return result


def _checks(
    df: pd.DataFrame,
    ranges: dict[str, tuple[int, int]],
    split_reports: dict[str, Any],
    raw_rows: int,
    cleaned_rows: int,
    shape_matrix: np.ndarray,
    word_tokens: list[str],
    labels: np.ndarray,
    horizon: int,
    cleaned_summary: dict[str, Any],
) -> dict[str, bool]:
    split_rows_sum = sum(end - start for start, end in ranges.values())
    windows_aligned = True
    for report in split_reports.values():
        if report["sample_count"] != report["expected_samples_without_invalids"] - report["invalid_labels_inside_usable_range"]:
            windows_aligned = False
        first = report["first_target_index"]
        last = report["last_target_index"]
        if first is not None and first != report["min_allowed_target_index"]:
            windows_aligned = False
        if last is not None and last != report["max_allowed_target_index"]:
            windows_aligned = False

    invalid_label_positions = np.flatnonzero(labels < 0)
    expected_invalid_positions = np.arange(max(0, len(labels) - horizon), len(labels), dtype=int)
    ranges_ordered = [ranges["train"], ranges["val"], ranges["test"]]

    return {
        "cleaning_never_increases_rows": cleaned_rows <= raw_rows,
        "cleaned_begin_sorted": bool(cleaned_summary.get("is_monotonic_increasing", True)),
        "cleaned_has_no_duplicate_begin": int(cleaned_summary.get("duplicate_begin_rows", 0)) == 0,
        "cleaned_has_no_invalid_ohlc": int(cleaned_summary.get("invalid_ohlc_rows", 0)) == 0,
        "cleaned_has_no_missing_ohlcv": int(cleaned_summary.get("missing_ohlcv_rows", 0)) == 0,
        "split_rows_sum_to_cleaned_rows": split_rows_sum == len(df),
        "split_ranges_do_not_overlap": all(ranges_ordered[i][1] <= ranges_ordered[i + 1][0] for i in range(len(ranges_ordered) - 1)),
        "shape_rows_equal_cleaned_rows": shape_matrix.shape[0] == len(df),
        "word_rows_equal_cleaned_rows": len(word_tokens) == len(df),
        "valid_labels_match_len_minus_horizon": int(np.count_nonzero(labels >= 0)) == max(0, len(df) - horizon),
        "invalid_labels_are_tail_only": np.array_equal(invalid_label_positions, expected_invalid_positions),
        "sentence_windows_aligned": windows_aligned,
    }
