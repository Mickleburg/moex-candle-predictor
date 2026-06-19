"""Focused triple-barrier research entrypoint.

This wrapper keeps the broad target/feature CLI reusable while making the
triple-barrier research command explicit. It injects ``--target-modes
triple_barrier`` when the caller did not provide target modes.
"""

from __future__ import annotations

import sys

from sber_action_target_feature_research import main


def _ensure_triple_barrier_default(argv: list[str]) -> list[str]:
    if any(arg == "--target-modes" or arg.startswith("--target-modes=") for arg in argv):
        return argv
    return [argv[0], "--target-modes", "triple_barrier", *argv[1:]]


if __name__ == "__main__":
    sys.argv = _ensure_triple_barrier_default(sys.argv)
    raise SystemExit(main())
