"""Build an ML prediction contract response from a candle_batch JSON file.

This CLI is intentionally contract-first. The current best triple-barrier
research candidate is documented, but no fitted production artifact bundle is
available yet. In that case the command writes a valid ``ml_prediction`` JSON
with ``diagnostics.artifact_missing=true`` instead of pretending to forecast.
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
from src.utils.io import ensure_dir  # noqa: E402


def artifact_bundle_available(path: Path) -> bool:
    """Return true only for the future research artifact bundle shape."""

    required = ["model.pkl", "metadata.json"]
    return all((path / item).exists() for item in required)


def predict_contract_from_json(input_json: str | Path, artifact_dir: str | Path) -> dict[str, Any]:
    """Load a candle batch and produce an ml_prediction contract response."""

    batch = load_candle_batch_json(input_json)
    df = candle_batch_to_dataframe(batch)
    artifact_path = Path(artifact_dir)

    if not artifact_bundle_available(artifact_path):
        return build_artifact_missing_response(
            batch=batch,
            df=df,
            artifact_dir=artifact_path,
            metadata=CURRENT_RESEARCH_DEFAULT,
        )

    return build_artifact_missing_response(
        batch=batch,
        df=df,
        artifact_dir=artifact_path,
        metadata=CURRENT_RESEARCH_DEFAULT,
        message=(
            "Artifact bundle files exist, but triple_barrier_extra_trees contract inference "
            "is not implemented until feature_pipeline/metadata protocol is finalized."
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, help="Path to candle_batch JSON")
    parser.add_argument("--output-json", required=True, help="Path to write ml_prediction JSON")
    parser.add_argument(
        "--artifact-dir",
        default=str(ML_DIR / "artifacts" / "research_triple_barrier_extra_trees"),
        help="Future research artifact bundle directory",
    )
    args = parser.parse_args()

    response = predict_contract_from_json(args.input_json, args.artifact_dir)
    output_path = REPO_ROOT / args.output_json if not Path(args.output_json).is_absolute() else Path(args.output_json)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote ML prediction contract JSON: {output_path}")
    if response.get("diagnostics", {}).get("artifact_missing"):
        print("Artifact missing mode: no fitted production artifact was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
