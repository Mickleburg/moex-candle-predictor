"""Dead-man's-switch: verdict logic, alert deduplication, and the EOD reference date.

Offline: no APScheduler, no network. Drives `deadman_tick` against a real StateStore + a capturing
notifier, so the delivery/persistence rules are exercised, not just the pure predicate.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from agent.src.scheduler import DEADMAN_FLAG, _reference_eod_date, deadman_tick
from agent.src.state_store import StateStore
from agent.tests.conftest import CapturingNotifier

MSK = ZoneInfo("Europe/Moscow")
REF = "2026-07-09"                      # a Thursday; the stale bring-up cycle in the incident
T0 = datetime(2026, 7, 10, 2, 8, tzinfo=MSK)
REPEAT_H = 6.0
GRACE_M = 90


class FailingNotifier(CapturingNotifier):
    def send(self, subject: str, body: str) -> bool:
        super().send(subject, body)
        return False                    # e.g. Telegram proxy down


def _store(tmp_path) -> StateStore:
    return StateStore(tmp_path / "s.sqlite")


def _seed_cycle(store: StateStore, status: str, *, started_at: str | None = None) -> None:
    store.begin_cycle(REF, "eod", mode="paper", block_mode="mock", as_of=f"{REF}T19:05:00+03:00")
    if status != "running":
        store.finish_cycle(REF, "eod", status)
    if started_at is not None:
        with store._tx() as c:
            c.execute("UPDATE cycle_runs SET started_at=? WHERE trade_date=? AND phase=?",
                      (started_at, REF, "eod"))


def _drop_cycle(store: StateStore) -> None:
    with store._tx() as c:
        c.execute("DELETE FROM cycle_runs WHERE trade_date=? AND phase=?", (REF, "eod"))


def _tick(store, notifier, now) -> bool:
    return deadman_tick(store, notifier, REF, now,
                        repeat_hours=REPEAT_H, running_grace_minutes=GRACE_M)


# --- deduplication (D1) ------------------------------------------------------------------

def test_same_incident_alerts_once_inside_window(tmp_path):
    # the reported spam: identical ref+status every 30 min must produce exactly ONE alert
    store, note = _store(tmp_path), CapturingNotifier()
    _seed_cycle(store, "failed")
    assert _tick(store, note, T0) is True
    assert _tick(store, note, T0 + timedelta(minutes=30)) is False
    assert _tick(store, note, T0 + timedelta(minutes=60)) is False
    assert len(note.messages) == 1
    assert "did not complete (status=failed)" in note.messages[0][1]


def test_reminder_resumes_after_repeat_hours(tmp_path):
    # a genuinely dead agent must keep reminding — dedup must not silence it forever
    store, note = _store(tmp_path), CapturingNotifier()
    _seed_cycle(store, "failed")
    assert _tick(store, note, T0) is True
    assert _tick(store, note, T0 + timedelta(hours=REPEAT_H, minutes=-1)) is False
    assert _tick(store, note, T0 + timedelta(hours=REPEAT_H, minutes=1)) is True
    assert len(note.messages) == 2


def test_new_status_alerts_immediately(tmp_path):
    # a DIFFERENT incident (failed -> missing) is never suppressed by the open quiet window
    store, note = _store(tmp_path), CapturingNotifier()
    _seed_cycle(store, "failed")
    assert _tick(store, note, T0) is True
    _drop_cycle(store)                                  # the cycle row is gone -> status 'missing'
    assert _tick(store, note, T0 + timedelta(minutes=30)) is True
    assert len(note.messages) == 2
    assert "status=missing" in note.messages[1][1]


def test_healthy_clears_flag_then_new_failure_alerts_immediately(tmp_path):
    # a stale suppression timestamp must never swallow the alert for a LATER failure
    store, note = _store(tmp_path), CapturingNotifier()
    _seed_cycle(store, "failed")
    assert _tick(store, note, T0) is True
    store.finish_cycle(REF, "eod", "completed")
    assert _tick(store, note, T0 + timedelta(minutes=30)) is False
    assert store.get_flag(DEADMAN_FLAG) is None          # flag cleared, not left armed

    store.finish_cycle(REF, "eod", "failed")             # fails again, well inside repeat_hours
    assert _tick(store, note, T0 + timedelta(minutes=60)) is True
    assert len(note.messages) == 2


# --- 'running' grace (D2 / D3) -----------------------------------------------------------

def test_running_within_grace_is_quiet(tmp_path):
    # a slow-but-alive EOD on 1 vCPU: 'running' younger than the grace is working, not stuck
    store, note = _store(tmp_path), CapturingNotifier()
    _seed_cycle(store, "running", started_at=(T0 - timedelta(minutes=5)).isoformat())
    assert _tick(store, note, T0) is False
    assert note.messages == []


def test_running_beyond_grace_alerts_stuck(tmp_path):
    # a hard crash mid-cycle leaves 'running' forever -> a real incident, with its own wording
    store, note = _store(tmp_path), CapturingNotifier()
    started = (T0 - timedelta(hours=3)).isoformat()
    _seed_cycle(store, "running", started_at=started)
    assert _tick(store, note, T0) is True
    subject, body = note.messages[0]
    assert "stuck" in subject.lower()
    assert "stuck in `running`" in body and started in body
    assert "did not complete" not in body
    # and it is deduplicated like any other incident
    assert _tick(store, note, T0 + timedelta(minutes=30)) is False


# --- delivery (persist only on a successful send) ------------------------------------------

def test_failed_send_does_not_arm_the_quiet_window(tmp_path):
    store = _store(tmp_path)
    _seed_cycle(store, "failed")
    failing = FailingNotifier()
    assert _tick(store, failing, T0) is False            # attempted, not delivered
    assert store.get_flag(DEADMAN_FLAG) is None          # window NOT armed

    note = CapturingNotifier()                           # next tick retries and gets through
    assert _tick(store, note, T0 + timedelta(minutes=30)) is True
    assert store.get_flag(DEADMAN_FLAG)["key"] == f"{REF}:failed"


# --- reference EOD date (regression: behaviour unchanged) ----------------------------------

def test_reference_eod_date_before_threshold_is_previous_trading_day():
    # 19:05 + 15 min grace: before 19:20 today's EOD is not yet due -> reference is yesterday
    now = datetime(2026, 7, 10, 19, 0, tzinfo=MSK)       # Friday
    assert _reference_eod_date(now, 19, 5) == "2026-07-09"


def test_reference_eod_date_after_threshold_is_today():
    now = datetime(2026, 7, 10, 19, 25, tzinfo=MSK)      # Friday, past 19:20
    assert _reference_eod_date(now, 19, 5) == "2026-07-10"


def test_reference_eod_date_on_weekend_is_last_trading_day():
    now = datetime(2026, 7, 11, 12, 0, tzinfo=MSK)       # Saturday
    assert _reference_eod_date(now, 19, 5) == "2026-07-10"
