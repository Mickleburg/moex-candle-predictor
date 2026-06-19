"""Thin contract validation for the orchestrator.

Every JSON object that crosses a block seam (sleeve_signal, risk_book, order_request,
execution_report, agent_cycle_result) is validated here before the agent acts on it. If
`jsonschema` is installed (it is on the VDS / in requirements) we validate against the real
schema with $ref resolution; otherwise we fall back to a structural required-keys check so
the stdlib-only core still guards the seams. Mirrors scripts/validate_contracts.py's
optional-dependency pattern.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO_ROOT / "contracts"


class ContractError(ValueError):
    """Raised when a payload does not satisfy its contract."""


@lru_cache(maxsize=None)
def _schema(name: str) -> dict[str, Any]:
    path = CONTRACTS_DIR / f"{name}.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _has_jsonschema() -> bool:
    try:
        import jsonschema  # noqa: F401
        return True
    except ImportError:
        return False


def validate(payload: dict[str, Any], contract: str) -> dict[str, Any]:
    """Validate `payload` against `contract` (e.g. 'risk_book'). Returns it on success."""
    schema = _schema(contract)
    if _has_jsonschema():
        import jsonschema
        from jsonschema import Draft202012Validator, RefResolver

        store = {f"{contract}.schema.json": schema}
        resolver = RefResolver(
            base_uri=f"file:///{CONTRACTS_DIR.as_posix()}/", referrer=schema, store=store
        )
        try:
            jsonschema.Draft202012Validator(schema, resolver=resolver).validate(payload)
        except jsonschema.ValidationError as exc:  # pragma: no cover - exercised via live deps
            raise ContractError(f"{contract}: {exc.message}") from exc
        return payload

    # stdlib fallback: required top-level keys only (good enough to catch seam breakage).
    missing = [k for k in schema.get("required", []) if k not in payload]
    if missing:
        raise ContractError(f"{contract}: missing required keys {missing}")
    return payload


def is_research_artifact(payload: dict[str, Any]) -> bool:
    """True while is_production=false (the invariant the whole pipeline must keep until sign-off)."""
    return payload.get("is_production", False) is False
