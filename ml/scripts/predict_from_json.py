"""Build an ML prediction contract response from a candle_batch JSON file.

This CLI is intentionally contract-first. Without ``--artifact-dir`` it writes
a valid ``ml_prediction`` JSON with ``diagnostics.artifact_missing=true``. With
a complete research artifact bundle it runs real ``predict_proba`` inference
for integration testing. The artifact is still research-only, not production.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.service.contracts import (  # noqa: E402
    CURRENT_RESEARCH_DEFAULT,
    build_artifact_missing_response,
    candle_batch_to_dataframe,
    load_candle_batch_json,
)
from src.service.research_artifact import (  # noqa: E402
    build_artifact_prediction_response,
    load_research_artifact,
)
from src.utils.io import ensure_dir  # noqa: E402


def predict_contract_from_json(input_json: str | Path, artifact_dir: str | Path | None = None) -> dict[str, Any]:
    """Load a candle batch and produce an ml_prediction contract response."""

    batch = load_candle_batch_json(input_json)
    df = candle_batch_to_dataframe(batch)

    if artifact_dir is None:
        return build_artifact_missing_response(
            batch=batch,
            df=df,
            artifact_dir=ML_DIR / "artifacts" / "research_triple_barrier_sber_h1",
            metadata=CURRENT_RESEARCH_DEFAULT,
        )

    artifact = load_research_artifact(artifact_dir)
    return build_artifact_prediction_response(batch=batch, df=df, artifact=artifact)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, help="Path to candle_batch JSON")
    parser.add_argument("--output-json", required=True, help="Path to write ml_prediction JSON")
    parser.add_argument(
        "--artifact-dir",
        default="",
        help="Optional research artifact bundle directory. If omitted, returns artifact_missing=true.",
    )
    args = parser.parse_args()

    try:
        response = predict_contract_from_json(args.input_json, args.artifact_dir or None)
    except Exception as exc:
        print(f"Failed to build ML prediction JSON: {exc}", file=sys.stderr)
        return 2
    output_path = REPO_ROOT / args.output_json if not Path(args.output_json).is_absolute() else Path(args.output_json)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote ML prediction contract JSON: {output_path}")
    if response.get("diagnostics", {}).get("artifact_missing"):
        print("Artifact missing mode: no fitted production artifact was used.")
    else:
        print("Research artifact inference mode: probabilities came from predict_proba.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
