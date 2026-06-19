"""Smoke test for the V2 news-feature block.

Runs the deterministic baseline (no LLM, no network) on real extracted SBER
disclosures and asserts the output is a schema-valid, no-lookahead llm_analysis with
no buy/hold/sell leakage. Requires data/news/edisclosure/SBER.parquet.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

LLM_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LLM_ROOT / "src"))

from llm_ta.analyzer import NewsFeatureService  # noqa: E402
from llm_ta.validator import DEFAULT_SCHEMA_PATH, validate_analysis  # noqa: E402

AS_OF = "2024-06-03T18:00:00+03:00"


def main() -> int:
    service = NewsFeatureService(provider=None)  # deterministic baseline
    result = service.analyze({"ticker": "SBER", "as_of": AS_OF, "timeframe": "1H"})

    # schema + no-lookahead (validate_analysis enforces both)
    validate_analysis(result, DEFAULT_SCHEMA_PATH)

    # V2 invariants
    assert "probabilities" not in result and "probabilities" not in result["features"], \
        "must not carry buy/hold/sell"
    f = result["features"]
    for key in ("sentiment", "impact_score", "novelty", "event_type", "news_count"):
        assert key in f, f"missing feature {key}"
    assert -1.0 <= f["sentiment"] <= 1.0
    assert 0.0 <= f["impact_score"] <= 1.0
    assert 0.0 <= f["novelty"] <= 1.0
    assert result["is_production"] is False

    as_of_dt = dt.datetime.fromisoformat(AS_OF)
    for src in result["sources"]:
        assert dt.datetime.fromisoformat(src["published_at"]) <= as_of_dt, "lookahead source"

    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
