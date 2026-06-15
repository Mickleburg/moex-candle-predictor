from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .analyzer import DEFAULT_PROMPT_PATH, DEFAULT_SCHEMA_PATH, TechnicalAnalysisService
from .providers import provider_from_name
from .validator import validate_analysis


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM technical analysis for a snapshot.")
    parser.add_argument("--input", required=True, type=Path, help="Path to input technical snapshot JSON.")
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "openai-compatible"],
        help="LLM provider adapter to use.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_PATH, type=Path, help="Prompt template path.")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA_PATH, type=Path, help="Output schema path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request_payload = load_json(args.input)
    provider = provider_from_name(args.provider)
    service = TechnicalAnalysisService(provider=provider, prompt_path=args.prompt, schema_path=args.schema)
    result = service.analyze(request_payload)
    validate_analysis(result, args.schema)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
