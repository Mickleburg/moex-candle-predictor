"""Pure text-rendering helpers for the bot's replies (no I/O, no Telegram types).

Messages are compact HTML (Telegram parse_mode=HTML — safer than Markdown for arbitrary text,
no need to escape '.', '-', '(' that appear in tickers/reasons). live and shadow capital are
ALWAYS rendered as separate, labelled sections so the two tracks are never conflated.
"""

from __future__ import annotations

from html import escape
from typing import Any


def money(value: float | None) -> str:
    """Compact RUB formatting: 1_234_567 -> '1.23M', 12_345 -> '12.3k'."""
    if value is None:
        return "n/a"
    v = float(value)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1_000_000:
        return f"{sign}{a / 1_000_000:.2f}M"
    if a >= 1_000:
        return f"{sign}{a / 1_000:.1f}k"
    return f"{sign}{a:.0f}"


def signed(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _b(text: Any) -> str:
    return f"<b>{escape(str(text))}</b>"


def _section(title: str) -> str:
    return f"\n{_b(title)}"


def fmt_status(d: dict) -> str:
    ks = "🔴 ENGAGED" if d["kill_switch"] else "🟢 off"
    lc = d.get("last_cycle")
    if lc:
        last = f"{lc.get('trade_date')} {lc.get('phase')} → {lc.get('status')} ({lc.get('at', '')[:19]})"
    else:
        last = "none yet"
    lines = [
        _b("Agent status"),
        f"mode: {escape(d['mode'])} / block={escape(d['block_mode'])} / live_enabled={d['live_enabled']}",
        f"kill-switch: {ks}",
        f"last cycle: {escape(last)}",
        f"gross (directional) — live {money(d['live_gross']['directional'])} | "
        f"shadow {money(d['shadow_gross']['directional'])}",
        f"  +hedge — live {money(d['live_gross']['hedge'])} | "
        f"shadow {money(d['shadow_gross']['hedge'])}",
    ]
    return "\n".join(lines)


def _positions_block(rows: list[dict], capital_rub: float) -> list[str]:
    if not rows:
        return ["  (none)"]
    out = []
    for p in rows:
        last = p.get("last_price") or p.get("avg_price") or 0.0
        notional = abs(int(p.get("lots", 0)) * float(last))
        weight = notional / capital_rub if capital_rub else 0.0
        hedge = " [hedge]" if p.get("is_hedge") else ""
        out.append(
            f"  {escape(p['ticker'])} {p.get('sector', '')}: {int(p.get('lots', 0))} lots, "
            f"w={pct(weight)}{hedge}"
        )
    return out


def fmt_positions(live: list[dict], shadow: list[dict], capital_rub: float) -> str:
    lines = [_b("Positions"), _section("LIVE")]
    lines += _positions_block(live, capital_rub)
    lines.append(_section("SHADOW (gated out — not real capital)"))
    lines += _positions_block(shadow, capital_rub)
    return "\n".join(lines)


def _pnl_block(rows: list[dict]) -> list[str]:
    if not rows:
        return ["  (none)"]
    out = []
    for r in rows:
        realized = r.get("realized") or 0.0
        unreal = r.get("unrealized") or 0.0
        out.append(
            f"  {escape(r['sleeve'])}: realized {money(realized)}, unreal {money(unreal)}, "
            f"total {money(realized + unreal)}"
        )
    return out


def fmt_pnl(live: list[dict], shadow: list[dict]) -> str:
    lines = [_b("P&L by sleeve"), _section("LIVE")]
    lines += _pnl_block(live)
    lines.append(_section("SHADOW (forward-shadow track — gate, not real P&L)"))
    lines += _pnl_block(shadow)
    return "\n".join(lines)


def fmt_prices(prices: list[tuple[str, float | None]], timeframe: str) -> str:
    lines = [_b(f"Last close ({escape(timeframe)})")]
    if not prices:
        lines.append("  (no tickers)")
    for ticker, price in prices:
        val = f"{price:.2f}" if price is not None else "no data"
        lines.append(f"  {escape(ticker)}: {val}")
    return "\n".join(lines)


def fmt_gate(gate: dict, shadow_forward: list[dict]) -> str:
    is_prod = gate.get("is_production", False)
    lines = [_b("Shadow gate (H9 dividend run-up)"),
             f"is_production: {'true' if is_prod else 'false'}"]
    if "met" in gate:
        lines.append(f"verdict: {'✅ MET' if gate['met'] else '❌ NOT MET'}")
    elif not gate.get("found", False):
        lines.append("verdict: report not found — standing invariant is_production=false")
    if "forward_n" in gate:
        lines.append(
            f"forward: n={gate['forward_n']}, net {signed(gate.get('forward_net'), 4)}, "
            f"%pos {pct(gate.get('forward_pct_pos'))}"
        )
    if shadow_forward:
        lines.append(_section("shadow forward P&L (state-store)"))
        for r in shadow_forward:
            fwd = (r.get("realized") or 0.0) + (r.get("unrealized") or 0.0)
            lines.append(f"  {escape(r['sleeve'])}: {money(fwd)}")
    return "\n".join(lines)


def fmt_cycle(cycle: dict | None) -> str:
    if not cycle:
        return f"{_b('Last EOD cycle')}\n  no cycle on record yet"
    result = cycle.get("result") or {}
    rs = result.get("risk_summary") or {}
    orders = result.get("selected_orders") or []
    lines = [
        _b("Last EOD cycle"),
        f"{escape(str(cycle.get('trade_date')))} → {escape(str(cycle.get('status')))} "
        f"(mode={escape(str(result.get('mode', cycle.get('mode'))))})",
        f"orders: {len(orders)}",
    ]
    binding = rs.get("binding_limits") or []
    lines.append(f"binding limits: {escape(', '.join(map(str, binding))) if binding else 'none'}")
    if cycle.get("halt_reason"):
        lines.append(f"halt: {escape(str(cycle['halt_reason']))}")
    for g in (rs.get("gating") or []):
        lines.append(f"  gate[{escape(str(g.get('sleeve')))}]: "
                     f"{escape(str(g.get('capital_state')))} ({escape(str(g.get('reason')))})")
    for o in orders[:12]:
        lines.append(f"  {escape(str(o.get('side')))} {o.get('quantity_lots')} "
                     f"{escape(str(o.get('ticker')))} @ {o.get('limit_price')}")
    return "\n".join(lines)


def fmt_integrity(report: dict | None) -> str:
    if report is None:
        return f"{_b('Data integrity')}\n  report not found — cannot confirm freshness"
    status = report.get("status", "?")
    icon = "🟢" if status == "OK" else "🔴"
    lines = [
        _b("Data integrity"),
        f"{icon} {escape(str(status))} (ref {escape(str(report.get('reference_date')))}, "
        f"{report.get('n_fail', 0)} fail / {report.get('n_warn', 0)} warn)",
    ]
    for reason in (report.get("reasons") or [])[:10]:
        lines.append(f"  ⛔ {escape(str(reason))}")
    for warn in (report.get("warnings") or [])[:5]:
        lines.append(f"  ⚠️ {escape(str(warn))}")
    return "\n".join(lines)


def fmt_shadowlog(records: list[dict]) -> str:
    """Render the tail of the forward-shadow track: per cycle, sleeves + shadow P&L by sleeve."""
    lines = [_b("Forward-shadow log (newest last)")]
    if not records:
        lines.append("  no shadow-log entries yet")
        return "\n".join(lines)
    for rec in records:
        td = rec.get("trade_date", "?")
        sleeves = rec.get("sleeves") or []
        sleeve_str = ", ".join(map(str, sleeves)) if sleeves else "—"
        lines.append(f"\n{_b(escape(str(td)))}  sleeves: {escape(sleeve_str)}")
        sleeve_pnl = rec.get("sleeve_pnl") or {}
        if not sleeve_pnl:
            lines.append("  shadow P&L: (flat — no holdings)")
        for sleeve, vals in sleeve_pnl.items():
            unreal = (vals or {}).get("unrealized", 0.0)
            lines.append(f"  {escape(str(sleeve))}: shadow P&L {money(unreal)}")
    return "\n".join(lines)


def fmt_users(admins, seed, managed: dict[int, dict]) -> str:
    """Render the allowlist: admins (👑 immutable), env seed, managed (note + who added)."""
    admins = set(admins)
    seed = set(seed)
    lines = [_b("Allowlist"), _section("Admins (👑 immutable)")]
    lines += [f"  👑 {a}" for a in sorted(admins)] or ["  (none)"]
    lines.append(_section("Env seed (BOT_ALLOWED_CHAT_IDS)"))
    seed_only = sorted(seed - admins)
    lines += [f"  {s}" for s in seed_only] or ["  (none)"]
    lines.append(_section("Managed (/allow)"))
    mg = {cid: info for cid, info in managed.items() if cid not in admins and cid not in seed}
    if mg:
        for cid in sorted(mg):
            info = mg[cid] or {}
            extra = []
            if info.get("note"):
                extra.append(escape(str(info["note"])))
            if info.get("added_by"):
                extra.append(f"by {info['added_by']}")
            suffix = f" ({', '.join(extra)})" if extra else ""
            lines.append(f"  {cid}{suffix}")
    else:
        lines.append("  (none)")
    return "\n".join(lines)


def unauthorized_text(chat_id: Any) -> str:
    cid = chat_id if chat_id is not None else "unknown"
    return (
        "⛔ You don't have access to this bot.\n"
        f"Your chat id: <code>{escape(str(cid))}</code>\n"
        "Send this id to an administrator to request access."
    )


def admin_only_text() -> str:
    return "⛔ Admin-only command."


def unknown_command_text() -> str:
    return "Unknown command — try /help."


_HELP_READ = (
    f"{_b('MOEX agent monitor')} (read-only)\n"
    "/status — mode, kill-switch, last cycle, live/shadow gross\n"
    "/positions — live + shadow book (lots, weight, sector)\n"
    "/pnl — P&L by sleeve (live vs shadow)\n"
    "/prices [TICKERS] — last close (default: universe)\n"
    "/gate — shadow gate: is_production, MET/NOT MET, forward P&L\n"
    "/shadowlog [N] — last N forward-shadow cycles (default 5)\n"
    "/cycle — last EOD result: orders, binding limits, alerts\n"
    "/integrity — data gate OK/HALT + reasons\n"
    "/help — this message"
)
_HELP_ADMIN = (
    f"\n{_b('Admin')}\n"
    "/users — show the allowlist (admins / seed / managed)\n"
    "/allow <chat_id> [note] — grant read access\n"
    "/deny <chat_id> — revoke a managed id"
)


def fmt_help(is_admin: bool = False) -> str:
    return _HELP_READ + (_HELP_ADMIN if is_admin else "")
