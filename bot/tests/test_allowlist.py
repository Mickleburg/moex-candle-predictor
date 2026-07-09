"""Managed allowlist store: atomic add/remove, fresh reads, fail-safe on missing/corrupt."""

from __future__ import annotations

import json
from pathlib import Path

from bot.src.allowlist import AllowlistStore


def test_add_persists_entry_and_metadata(tmp_path: Path):
    store = AllowlistStore(tmp_path / "bot" / "allowlist.json")
    assert store.add(444, note="alice", added_by=999) is True
    entries = store.entries()
    assert set(entries) == {444}
    assert entries[444]["note"] == "alice"
    assert entries[444]["added_by"] == 999
    assert entries[444]["added_at"]  # timestamp recorded
    # on-disk keys are strings (JSON), parsed back to int
    raw = json.loads((tmp_path / "bot" / "allowlist.json").read_text(encoding="utf-8"))
    assert list(raw) == ["444"]


def test_add_is_idempotent(tmp_path: Path):
    store = AllowlistStore(tmp_path / "allowlist.json")
    assert store.add(444) is True
    assert store.add(444) is False     # already present
    assert store.ids() == {444}


def test_remove(tmp_path: Path):
    store = AllowlistStore(tmp_path / "allowlist.json")
    store.add(444)
    assert store.remove(444) is True
    assert store.remove(444) is False  # already gone
    assert store.ids() == set()


def test_missing_file_is_empty(tmp_path: Path):
    store = AllowlistStore(tmp_path / "nope.json")
    assert store.entries() == {}
    assert store.ids() == set()


def test_corrupt_file_is_empty_failsafe(tmp_path: Path):
    p = tmp_path / "allowlist.json"
    p.write_text("{not json", encoding="utf-8")
    assert AllowlistStore(p).entries() == {}


def test_reads_are_fresh_across_instances(tmp_path: Path):
    p = tmp_path / "allowlist.json"
    AllowlistStore(p).add(444)
    # a second, independent reader sees it immediately (no in-process cache to go stale)
    assert AllowlistStore(p).ids() == {444}
