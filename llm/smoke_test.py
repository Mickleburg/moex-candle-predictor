from __future__ import annotations

import json
import sys
from pathlib import Path


LLM_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LLM_ROOT / "src"))

from llm_ta.analyzer import DEFAULT_SCHEMA_PATH, TechnicalAnalysisService  # noqa: E402
from llm_ta.providers import MockProvider  # noqa: E402
from llm_ta.validator import validate_analysis  # noqa: E402


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    input_path = LLM_ROOT / "examples" / "sber_1h_input.json"
    expected_path = LLM_ROOT / "examples" / "sber_1h_expected_output.json"
    request_payload = load_json(input_path)
    expected = load_json(expected_path)

    service = TechnicalAnalysisService(provider=MockProvider())
    result = service.analyze(request_payload)
    validate_analysis(result, DEFAULT_SCHEMA_PATH)

    if result != expected:
        raise AssertionError("mock output does not match examples/sber_1h_expected_output.json")

    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
