"""Regression tests for the Telegram alert notifier.

The operational alert bodies carry '/', '_' and '*' (ticker keys like BR_CONT, reasons like
freshness/SBER/1H). Under parse_mode=Markdown those are unbalanced entities and Telegram rejects
the whole sendMessage with HTTP 400 — the alert is silently dropped. These tests pin the fix:
the notifier sends PLAIN TEXT (no parse_mode) and preserves the raw operational characters.
"""

from __future__ import annotations

import json
import urllib.parse

from agent.src.notifier import TelegramNotifier


class _FakeResp:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _CapturingOpener:
    """Stands in for the urllib opener; records the Request instead of hitting the network."""

    def __init__(self, payload: dict | None = None):
        self.last_req = None
        self._payload = json.dumps(payload or {"ok": True}).encode()

    def open(self, req, timeout=None):
        self.last_req = req
        return _FakeResp(self._payload)


def _sent_params(notifier: TelegramNotifier) -> dict[str, str]:
    body = notifier._opener.last_req.data.decode()
    return dict(urllib.parse.parse_qsl(body))


def test_alert_is_plain_text_no_parse_mode():
    """A HALT body full of '/' and '_' must send without parse_mode (else Telegram 400s)."""
    notifier = TelegramNotifier("token", "123")
    notifier._opener = _CapturingOpener()

    ok = notifier.send(
        "DATA HALT — not trading",
        "reasons: freshness/SBER/1H, sync/BR_CONT/1H lags newest required series",
    )

    assert ok is True
    params = _sent_params(notifier)
    assert "parse_mode" not in params                      # the regression guard
    assert params["chat_id"] == "123"
    # raw operational characters survive verbatim (no Markdown escaping / stripping)
    assert "BR_CONT" in params["text"]
    assert "freshness/SBER/1H" in params["text"]
    assert params["text"].startswith("DATA HALT — not trading\n")


def test_send_returns_false_on_not_ok_payload():
    notifier = TelegramNotifier("token", "123")
    notifier._opener = _CapturingOpener({"ok": False, "description": "bad"})
    assert notifier.send("subj", "body") is False


def test_send_never_raises_on_transport_error():
    """Alerts must never crash the cycle — a raising opener yields False, not an exception."""
    class _Boom:
        def open(self, req, timeout=None):
            raise OSError("network down")

    notifier = TelegramNotifier("token", "123")
    notifier._opener = _Boom()
    assert notifier.send("subj", "body") is False
