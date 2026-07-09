"""Demo: dry-run the real risk_book example into delta LIMIT orders (sends nothing).

    python execution/scripts/demo_dry_run.py

Reads contracts/examples/risk_book.example.json + execution/examples/prices.example.json, reconciles
against an empty book (so every leg is an entry), and prints the delta orders. No broker is contacted.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from execution.src.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main([
        "dry-run",
        "--risk-book", str(REPO_ROOT / "contracts" / "examples" / "risk_book.example.json"),
        "--prices", str(REPO_ROOT / "execution" / "examples" / "prices.example.json"),
    ]))
