"""Alert channel for the daily digest, data-HALT alerts and the dead-man's-switch.

Telegram is the recommended channel (token + chat id from the environment, never the file);
stdout is the default so dev / paper / CI runs need no secrets. The Telegram backend uses
stdlib urllib so the core carries no extra dependency.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Protocol

from .config import AgentConfig


class Notifier(Protocol):
    def send(self, subject: str, body: str) -> bool:
        """Deliver a message. Returns True on success. Must never raise."""
        ...


class StdoutNotifier:
    """Prints alerts to stdout (no secrets needed). Default for dev/paper/CI."""

    def send(self, subject: str, body: str) -> bool:
        print(f"\n=== ALERT: {subject} ===\n{body}\n", file=sys.stdout, flush=True)
        return True


class TelegramNotifier:
    """Sends alerts to a Telegram chat via the Bot API (stdlib urllib)."""

    def __init__(self, bot_token: str, chat_id: str, timeout: float = 10.0):
        self._token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout

    def send(self, subject: str, body: str) -> bool:
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        text = f"*{subject}*\n{body}"[:4000]
        data = urllib.parse.urlencode(
            {"chat_id": self._chat_id, "text": text, "parse_mode": "Markdown"}
        ).encode()
        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode())
                return bool(payload.get("ok"))
        except Exception as exc:  # noqa: BLE001 - alerts must never crash the cycle
            print(f"[telegram alert failed: {exc}] subject={subject}", file=sys.stderr)
            return False


def build_notifier(config: AgentConfig) -> Notifier:
    """Pick the notifier from config + environment. Falls back to stdout if Telegram is
    selected but its secrets are absent (so a misconfigured VDS still logs, never crashes)."""
    if config.alerts.channel == "telegram":
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if token and chat_id:
            return TelegramNotifier(token, chat_id)
        print("[notifier] alerts.channel=telegram but TELEGRAM_BOT_TOKEN/CHAT_ID unset; "
              "falling back to stdout", file=sys.stderr)
    return StdoutNotifier()
