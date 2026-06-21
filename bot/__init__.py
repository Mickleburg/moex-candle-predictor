"""Telegram monitoring bot for the MOEX V3 trading agent (read-only observer).

This block ONLY reads the agent's durable state + regenerable reports and renders them to a
whitelisted Telegram owner. It never trades, and by default exposes no control actions (the
kill-switch is intentionally out of v1 — see bot/README.md). The agent's notifier owns PUSH
alerts (sendMessage); this bot is the single getUpdates POLLER for the same token.
"""
