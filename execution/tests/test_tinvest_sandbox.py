"""Opt-in networked sandbox wire-test (T-Invest). DEFAULT-SKIPPED.

Runs the full order_request <-> execution_report e2e against the T-Invest SANDBOX (open account,
fund, quote, marketable fill, cancel, duplicate-protection). Needs network + a sandbox TINVEST_TOKEN
in .env. To run it locally:

    EXECUTION_WIRE_TEST=1  ml/.venv-win/Scripts/python.exe -m pytest execution/tests/test_tinvest_sandbox.py

It is skipped in the normal suite so `pytest execution/tests` stays hermetic and green. The same flow
is runnable directly: `python execution/scripts/wire_test_tinvest_sandbox.py`.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("EXECUTION_WIRE_TEST") != "1",
    reason="set EXECUTION_WIRE_TEST=1 (+ TINVEST_TOKEN in .env) to run the live-sandbox wire test")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "wire_test_tinvest_sandbox.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("wire_test_tinvest_sandbox", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sandbox_wire_e2e():
    # main() opens/funds a sandbox account, fills a marketable BUY, cancels a passive order, and
    # asserts duplicate-protection; it returns 0 on success (asserts raise on any mismatch).
    assert _load_script().main() == 0
