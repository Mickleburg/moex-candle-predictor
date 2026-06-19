"""Assemble the frozen `feature_bundle` contract at a decision as_of: [quant + news] per ticker.

This is the join the whole H2/H3 pipeline runs on. Quant features come from price (past-only at
as_of, cross-sectionally z-scored); news features come from the LLM/news block's per-(ticker,
as_of) table, looked up with no-lookahead (news as_of <= decision, within the trailing window).
Output validates against contracts/feature_bundle.schema.json and feeds the decision model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cross_sectional import UNIVERSE
from .decision_grid import NEWS_WINDOW_DAYS

REPO_ROOT = Path(__file__).resolve().parents[3]
QUANT_FEATURES = ["mom20", "vol20", "ma20_dist"]


def _quant_row(panel: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, np.ndarray]:
    """Past-only quant features at as_of, cross-sectionally z-scored across the universe."""
    sub = panel.loc[:as_of]
    if len(sub) < 21:
        raise ValueError("not enough history at as_of for quant features")
    last = sub.iloc[-1]
    feats = {
        "mom20": last / sub.iloc[-21] - 1.0,
        "vol20": sub.pct_change().iloc[-20:].std(),
        "ma20_dist": last / sub.iloc[-20:].mean() - 1.0,
    }
    out = {}
    for name, s in feats.items():
        z = (s - s.mean()) / (s.std() + 1e-9)
        out[name] = z
    return out


def assemble_feature_bundle(
    panel: pd.DataFrame,
    news: pd.DataFrame | None,
    as_of: pd.Timestamp,
    *,
    news_cols: list[str] = ("sentiment", "impact_score"),
    universe: list[str] = UNIVERSE,
    window_days: int = NEWS_WINDOW_DAYS,
    is_production: bool = False,
) -> dict[str, Any]:
    """Build a dict compatible with feature_bundle.schema.json for one decision as_of.

    `news`: tidy table [ticker, as_of (tz-aware), <news_cols...>] or None (zeros -> price-only).
    No-lookahead: only news with as_of_news in (decision-window, decision] is used (latest wins).
    """
    news_cols = list(news_cols)
    qz = _quant_row(panel, as_of)
    universe = [t for t in universe if t in panel.columns]

    by_ticker = {}
    if news is not None and len(news):
        floor = as_of - pd.Timedelta(days=window_days)
        vis = news[(news["as_of"] <= as_of) & (news["as_of"] > floor)].sort_values("as_of")
        by_ticker = {t: g for t, g in vis.groupby("ticker")}

    entries = []
    for t in universe:
        quant = [float(qz[f].get(t, np.nan)) for f in QUANT_FEATURES]
        if t in by_ticker:
            row = by_ticker[t].iloc[-1]
            newsf = [float(row.get(c, 0.0)) for c in news_cols]
        else:
            newsf = [0.0] * len(news_cols)
        valid = all(np.isfinite(v) for v in quant)
        entries.append({"ticker": t,
                        "quant": [round(v, 6) if np.isfinite(v) else 0.0 for v in quant],
                        "news": [round(v, 6) for v in newsf],
                        "valid": bool(valid)})

    return {
        "as_of": as_of.isoformat(),
        "timeframe": "1D",
        "universe": list(universe),
        "feature_spec": {
            "quant_features": QUANT_FEATURES,
            "news_features": news_cols,
            "cross_sectional_normalized": True,
        },
        "entries": entries,
        "is_production": bool(is_production),
    }


def validate_against_schema(bundle: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError:
        return
    schema = json.loads(
        (REPO_ROOT / "contracts" / "feature_bundle.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(bundle)
