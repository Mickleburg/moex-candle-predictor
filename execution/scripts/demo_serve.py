"""Demo: the orchestrator seam. Feed the example request envelope into `serve`, print the result.

    python execution/scripts/demo_serve.py

Equivalent to:
    python -m execution.src.cli serve --mode paper < execution/examples/serve_request.example.json
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Isolate the dedupe ledger / audit so this demo is repeatable (the fixed 2025-06-02 date would
# otherwise be treated as the same already-submitted intent on a second run).
_tmp = tempfile.mkdtemp(prefix="exec_demo_serve_")
os.environ["EXECUTION_STATE_DIR"] = str(Path(_tmp) / "state")
os.environ["EXECUTION_AUDIT_DIR"] = str(Path(_tmp) / "audit")

from execution.src.cli import main  # noqa: E402

REQUEST = REPO_ROOT / "execution" / "examples" / "serve_request.example.json"

if __name__ == "__main__":
    sys.stdin = io.StringIO(REQUEST.read_text(encoding="utf-8"))
    out = io.StringIO()
    real_stdout, sys.stdout = sys.stdout, out
    try:
        main(["serve", "--mode", "paper"])
    finally:
        sys.stdout = real_stdout
    result = json.loads(out.getvalue())
    print(f"orders: {len(result['orders'])}  reports: {len(result['reports'])}  "
          f"rejected: {len(result['rejected'])}  is_production={result['is_production']}")
    for o in result["orders"]:
        print(f"  {o['side']:4} {o['ticker']:7} {o['quantity_lots']:>10} lots @ LIMIT {o['limit_price']:>12,.4f}")
    print("resulting book:")
    for p in result["positions"]:
        print(f"  {p['ticker']:7} {p['lots']:>10} lots  hedge={p['is_hedge']}  last={p['last_price']}")
