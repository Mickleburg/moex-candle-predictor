"""CLI: produce an llm_analysis (news features) for a ticker at an as_of time.

Examples:
  # deterministic baseline (no LLM, free):
  python -m llm_ta.cli --ticker SBER --as-of 2024-06-03T12:00:00+03:00
  # refine with a local Ollama model (free) or DeepSeek (cheap):
  python -m llm_ta.cli --ticker SBER --as-of 2024-06-03T12:00:00 --provider ollama
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzer import DEFAULT_PROMPT_PATH, NewsFeatureService
from .features import DEFAULT_WINDOW_HOURS
from .providers import provider_from_name
from .validator import DEFAULT_SCHEMA_PATH, validate_analysis


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract llm_analysis news features for a ticker.")
    p.add_argument("--ticker", required=True)
    p.add_argument("--as-of", required=True, help="ISO-8601 decision time (no-lookahead boundary).")
    p.add_argument("--timeframe", default="1H")
    p.add_argument("--window-hours", type=int, default=DEFAULT_WINDOW_HOURS)
    p.add_argument("--provider", default="baseline",
                   help="baseline (no LLM) | ollama | openai-compatible | deepseek")
    p.add_argument("--prompt", default=DEFAULT_PROMPT_PATH, type=Path)
    p.add_argument("--schema", default=DEFAULT_SCHEMA_PATH, type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = NewsFeatureService(
        provider=provider_from_name(args.provider),
        prompt_path=args.prompt,
        schema_path=args.schema,
        window_hours=args.window_hours,
    )
    result = service.analyze({
        "ticker": args.ticker,
        "as_of": args.as_of,
        "timeframe": args.timeframe,
        "window_hours": args.window_hours,
    })
    validate_analysis(result, args.schema)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
