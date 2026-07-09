"""Roundtrip + verification tests for the forward-snapshot export/import scripts.

Hermetic: a fake repo data root under tmp (module REPO_ROOT monkeypatched) so nothing touches the
real store. The dividend-feed verifier is exercised separately (it needs raw parquet + pandas); here
it is legitimately "skipped" because no feed is bundled — the SHA256 + shadow-track no-lookahead
checks are what these assert.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import export_forward_snapshot as exp  # noqa: E402
import import_forward_snapshot as imp  # noqa: E402
from agent.src.state_store import StateStore  # noqa: E402

_GOOD_SHADOW = [
    {"trade_date": "2026-07-20", "as_of": "2026-07-20T19:05:00+03:00", "sleeve_pnl": {}},
    {"trade_date": "2026-07-21", "as_of": "2026-07-21T19:05:00+03:00", "sleeve_pnl": {}},
]


def _seed_repo(fake_repo: Path, shadow_records: list[dict]) -> Path:
    """Build a fake data/ tree + an agent config pointing at it; return the config path."""
    (fake_repo / "data" / "reports").mkdir(parents=True)
    (fake_repo / "data" / "news").mkdir(parents=True)
    (fake_repo / "data" / "agent").mkdir(parents=True)

    state_db = fake_repo / "data" / "agent" / "state.sqlite"
    StateStore(state_db).set_flag("seed", True)             # a real, openable SQLite db

    shadow = fake_repo / "data" / "agent" / "shadow_pnl.jsonl"
    shadow.write_text("".join(json.dumps(r) + "\n" for r in shadow_records), encoding="utf-8")

    (fake_repo / "data" / "reports" / "data_integrity_status.json").write_text(
        json.dumps({"status": "OK", "reasons": []}), encoding="utf-8")

    cfg = {"mode": "paper", "block_mode": "mock",
           "paths": {"state_db": str(state_db), "shadow_log": str(shadow),
                     "cycle_results_dir": str(fake_repo / "data" / "agent" / "cycles"),
                     "log_dir": str(fake_repo / "data" / "agent" / "logs")}}
    cfg_path = fake_repo / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_path


def _wire(monkeypatch, fake_repo: Path) -> None:
    monkeypatch.setattr(exp, "REPO_ROOT", fake_repo)
    monkeypatch.setattr(imp, "REPO_ROOT", fake_repo)
    monkeypatch.setattr(imp, "SNAPSHOTS_ROOT", fake_repo / "data" / "forward_snapshots")


def test_export_import_roundtrip(tmp_path, monkeypatch):
    fake_repo = tmp_path / "repo"
    cfg_path = _seed_repo(fake_repo, _GOOD_SHADOW)
    _wire(monkeypatch, fake_repo)

    out = tmp_path / "snap"
    manifest = exp.export(out, str(cfg_path))
    assert manifest["snapshot_type"] == "forward_accrual" and manifest["read_only"] is True
    assert "NOT" in manifest["not_for_tuning"].upper()       # the no-tuning banner is stamped
    assert manifest["as_of_range"] == {"min": _GOOD_SHADOW[0]["as_of"],
                                       "max": _GOOD_SHADOW[1]["as_of"], "n_cycles": 2}
    names = {f["name"] for f in manifest["files"]}
    assert {"state.sqlite", "shadow_pnl.jsonl", "reports/data_integrity_status.json"} <= names

    res = imp.import_snapshot(out, force=False)
    assert res["status"] == "ok"
    assert res["feed_nolookahead"] == "skipped"              # no feed bundled -> skipped, not failed
    landed = Path(res["landed"])
    assert landed.name == "2026-07-21" and (landed / "manifest.json").exists()
    assert not os.access(landed / "manifest.json", os.W_OK)  # landed read-only

    # already-imported guard, then --force replaces it
    with pytest.raises(imp.VerifyError):
        imp.import_snapshot(out, force=False)
    assert imp.import_snapshot(out, force=True)["status"] == "ok"


def test_import_fails_on_sha256_tamper(tmp_path, monkeypatch):
    fake_repo = tmp_path / "repo"
    cfg_path = _seed_repo(fake_repo, _GOOD_SHADOW)
    _wire(monkeypatch, fake_repo)
    out = tmp_path / "snap"
    exp.export(out, str(cfg_path))

    (out / "reports" / "data_integrity_status.json").write_text("TAMPERED", encoding="utf-8")
    with pytest.raises(imp.VerifyError, match="sha256 mismatch"):
        imp.import_snapshot(out, force=False)


def test_import_fails_on_shadow_lookahead(tmp_path, monkeypatch):
    rewound = [_GOOD_SHADOW[1], _GOOD_SHADOW[0]]              # as_of goes BACKWARDS (rewind)
    fake_repo = tmp_path / "repo"
    cfg_path = _seed_repo(fake_repo, rewound)
    _wire(monkeypatch, fake_repo)
    out = tmp_path / "snap"
    exp.export(out, str(cfg_path))
    with pytest.raises(imp.VerifyError, match="rewind"):
        imp.import_snapshot(out, force=False)


def test_export_refuses_prod_store(tmp_path, monkeypatch):
    fake_repo = tmp_path / "repo"
    _seed_repo(fake_repo, _GOOD_SHADOW)
    _wire(monkeypatch, fake_repo)
    # --out inside the production store (data/agent) must be refused by the CLI guard
    rc = exp.main(["--out", str(fake_repo / "data" / "agent" / "x"),
                   "--config", str(fake_repo / "cfg.json")])
    assert rc == 2
