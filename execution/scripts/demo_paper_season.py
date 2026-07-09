"""Demo: replay the illustrative H9 run-up season through the paper simulator.

    python execution/scripts/demo_paper_season.py

Shows the daily reconciliation building the position from flat -> long inside the -12/-2 window ->
flat again before the ex-gap, with a full audit trail. No real orders.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from execution.src.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main([
        "paper-season",
        "--season", str(REPO_ROOT / "execution" / "examples" / "risk_book_season.example.json"),
    ]))
