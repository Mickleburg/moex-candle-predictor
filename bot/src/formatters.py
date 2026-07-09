"""Pure text-rendering helpers for the bot's replies (no I/O, no Telegram types).

Visual grammar (kept consistent across every fmt_*):
  * one accent glyph (ACCENT) on the report title — nothing else competes with it;
  * status glyphs only for state: OK_GLYPH / BAD_GLYPH / WARN_GLYPH;
  * live and shadow capital are ALWAYS separate, labelled sections (never conflated);
  * tabular output is aligned in a monospace <pre> block.

HTML: replies are sent parse_mode="HTML", whose allowed tag set is small (<b> <i> <u> <s>
<code> <pre> <a> <tg-spoiler> <blockquote>) and reserved chars <, >, & must be escaped. So
ALL dynamic text (tickers, notes, reasons) is html.escape'd — including inside <pre>. The table
builder pads columns by RAW visible width, then escapes the whole assembled body once: HTML
entities (&amp;) render as a single glyph in Telegram, so alignment survives the escape.
"""

from __future__ import annotations

from html import escape
from typing import Any

# --- design tokens -------------------------------------------------------------------------
ACCENT = "📊"          # single accent glyph from the bot avatar (candles) — report titles only
OK_GLYPH = "🟢"
BAD_GLYPH = "🔴"
WARN_GLYPH = "⚠️"


def money(value: float | None, *, signed: bool = False) -> str:
    """Compact RUB: 1_234_567 -> '1.23M ₽', 12_345 -> '12.3k ₽', -450 -> '-450 ₽'.

    signed=True prefixes '+' on POSITIVE values — used ONLY for P&L columns, where the sign is
    the colour-independent (dark/light) way to tell profit from loss at a glance. Zero is 'flat'
    (never '+0'/'−0'), and unsigned gross/notional exposure keeps signed=False.
    """
    if value is None:
        return "n/a"
    v = float(value)
    if v < 0:
        sign = "-"
    elif signed and v > 0:
        sign = "+"
    else:
        sign = ""
    a = abs(v)
    if a >= 1_000_000:
        return f"{sign}{a / 1_000_000:.2f}M ₽"
    if a >= 1_000:
        return f"{sign}{a / 1_000:.1f}k ₽"
    return f"{sign}{a:.0f} ₽"


def signed(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


# --- html + layout helpers -----------------------------------------------------------------
def _title(text: Any) -> str:
    """Report header: the one accent glyph + bold escaped title."""
    return f"{ACCENT} <b>{escape(str(text))}</b>"


def _label(text: Any) -> str:
    """Sub-section label (LIVE / SHADOW / Orders / …) — bold, no glyph."""
    return f"<b>{escape(str(text))}</b>"


def _price(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def _truncate(rows: list, cap: int) -> tuple[list, int]:
    """Cap a row list for the 4096-char limit; return (shown, n_hidden)."""
    if len(rows) <= cap:
        return rows, 0
    return rows[:cap], len(rows) - cap


def _table(rows: list[list[Any]], aligns: list[str], header: list[str] | None = None) -> str:
    """Aligned monospace table in a single <pre>. Cells are RAW; the whole body is escaped once.

    aligns[i] = 'r' right-justifies (numbers), anything else left-justifies (text). Columns are
    padded with spaces (never tabs) to the max raw width, so entities render as one glyph and the
    columns still line up after escaping.
    """
    all_rows = ([header] if header else []) + list(rows)
    if not all_rows:
        return ""
    ncol = max(len(r) for r in all_rows)
    widths = [0] * ncol
    for r in all_rows:
        for i in range(ncol):
            widths[i] = max(widths[i], len(str(r[i])) if i < len(r) else 0)

    def render(r: list[Any]) -> str:
        cells = []
        for i in range(ncol):
            cell = str(r[i]) if i < len(r) else ""
            al = aligns[i] if i < len(aligns) else "l"
            cells.append(cell.rjust(widths[i]) if al == "r" else cell.ljust(widths[i]))
        return "  ".join(cells).rstrip()

    lines = []
    if header:
        lines.append(render(header))
        lines.append("  ".join("─" * widths[i] for i in range(ncol)))
    lines += [render(r) for r in rows]
    return f"<pre>{escape(chr(10).join(lines))}</pre>"


def _with_more(table: str, hidden: int) -> str:
    return table + (f"\n…+{hidden} more" if hidden else "")


# --- reports -------------------------------------------------------------------------------
def fmt_status(d: dict) -> str:
    ks = (f"{BAD_GLYPH} kill-switch: ENGAGED" if d["kill_switch"]
          else f"{OK_GLYPH} kill-switch: off")
    lc = d.get("last_cycle")
    last = (f"{lc.get('trade_date')} {lc.get('phase')} → {lc.get('status')}" if lc else "none yet")
    lg, sg = d["live_gross"], d["shadow_gross"]
    table = _table(
        [["directional", money(lg["directional"]), money(sg["directional"])],
         ["hedge", money(lg["hedge"]), money(sg["hedge"])]],
        aligns=["l", "r", "r"], header=["gross", "live", "shadow"],
    )
    return "\n".join([
        _title("Agent status"),
        ks,
        f"mode {escape(str(d['mode']))} · block {escape(str(d['block_mode']))} · "
        f"live {'on' if d['live_enabled'] else 'off'}",
        f"last cycle: {escape(last)}",
        "",
        table,
    ])


def _positions_block(rows: list[dict], capital_rub: float) -> str:
    if not rows:
        return "  book empty"
    shown, hidden = _truncate(rows, 30)
    trows = []
    for p in shown:
        last = p.get("last_price") or p.get("avg_price") or 0.0
        weight = abs(int(p.get("lots", 0)) * float(last)) / capital_rub if capital_rub else 0.0
        trows.append([p["ticker"], p.get("sector", ""), str(int(p.get("lots", 0))),
                      pct(weight), "hedge" if p.get("is_hedge") else "dir"])
    table = _table(trows, aligns=["l", "l", "r", "r", "l"],
                   header=["ticker", "sector", "lots", "wt", "type"])
    return _with_more(table, hidden)


def fmt_positions(live: list[dict], shadow: list[dict], capital_rub: float) -> str:
    return "\n".join([
        _title("Positions"),
        "",
        _label("LIVE"),
        _positions_block(live, capital_rub),
        "",
        _label("SHADOW (gated out — not real capital)"),
        _positions_block(shadow, capital_rub),
    ])


def _pnl_block(rows: list[dict]) -> str:
    if not rows:
        return "  (nothing recorded)"
    trows = []
    for r in rows:
        realized = r.get("realized") or 0.0
        unreal = r.get("unrealized") or 0.0
        trows.append([r["sleeve"], money(realized, signed=True), money(unreal, signed=True),
                      money(realized + unreal, signed=True)])
    return _table(trows, aligns=["l", "r", "r", "r"],
                  header=["sleeve", "realized", "unreal", "total"])


def fmt_pnl(live: list[dict], shadow: list[dict]) -> str:
    return "\n".join([
        _title("P&L by sleeve"),
        "",
        _label("LIVE"),
        _pnl_block(live),
        "",
        _label("SHADOW (forward-shadow track — gate, not real P&L)"),
        _pnl_block(shadow),
    ])


def fmt_prices(prices: list[tuple[str, float | None]], timeframe: str) -> str:
    if not prices:
        return "\n".join([_title(f"Last close ({timeframe})"), "  no tickers"])
    shown, hidden = _truncate(prices, 30)
    trows = [[ticker, (f"{price:.2f}" if price is not None else "no data")]
             for ticker, price in shown]
    table = _table(trows, aligns=["l", "r"], header=["ticker", "close ₽"])
    return "\n".join([_title(f"Last close ({timeframe})"), _with_more(table, hidden)])


def fmt_gate(gate: dict, shadow_forward: list[dict]) -> str:
    is_prod = gate.get("is_production", False)
    lines = [_title("Shadow gate — H9 dividend run-up"),
             f"is_production: {'true' if is_prod else 'false'}"]
    if "met" in gate:
        glyph = OK_GLYPH if gate["met"] else BAD_GLYPH
        lines.append(f"{glyph} verdict: {'MET' if gate['met'] else 'NOT MET'}")
    elif not gate.get("found", False):
        lines.append(f"{WARN_GLYPH} verdict: report not found — is_production stays false")
    if "forward_n" in gate:
        lines.append(f"forward: n={gate['forward_n']} · net {signed(gate.get('forward_net'), 4)} "
                     f"· %pos {pct(gate.get('forward_pct_pos'))}")
    if shadow_forward:
        trows = [[r["sleeve"], money((r.get("realized") or 0.0) + (r.get("unrealized") or 0.0),
                                     signed=True)]
                 for r in shadow_forward]
        lines += ["", _label("Shadow forward P&L"),
                  _table(trows, aligns=["l", "r"], header=["sleeve", "fwd P&L"])]
    return "\n".join(lines)


def fmt_cycle(cycle: dict | None) -> str:
    if not cycle:
        return f"{_title('Last EOD cycle')}\n  no cycle yet"
    result = cycle.get("result") or {}
    rs = result.get("risk_summary") or {}
    orders = result.get("selected_orders") or []
    binding = rs.get("binding_limits") or []
    lines = [
        _title("Last EOD cycle"),
        f"{escape(str(cycle.get('trade_date')))} → {escape(str(cycle.get('status')))} "
        f"(mode {escape(str(result.get('mode', cycle.get('mode'))))})",
        f"orders: {len(orders)} · binding: "
        f"{escape(', '.join(map(str, binding))) if binding else 'none'}",
    ]
    if cycle.get("halt_reason"):
        lines.append(f"{BAD_GLYPH} halt: {escape(str(cycle['halt_reason']))}")
    if orders:
        shown, hidden = _truncate(orders, 12)
        trows = [[str(o.get("side")), str(o.get("quantity_lots")), str(o.get("ticker")),
                  _price(o.get("limit_price"))] for o in shown]
        lines += ["", _label("Orders"),
                  _with_more(_table(trows, aligns=["l", "r", "l", "r"],
                                    header=["side", "qty", "ticker", "limit"]), hidden)]
    gating = rs.get("gating") or []
    if gating:
        lines += ["", _label("Gating")]
        for g in gating:
            lines.append(f"  {escape(str(g.get('sleeve')))}: "
                         f"{escape(str(g.get('capital_state')))} ({escape(str(g.get('reason')))})")
    return "\n".join(lines)


def fmt_integrity(report: dict | None) -> str:
    if report is None:
        return f"{_title('Data integrity')}\n  {WARN_GLYPH} report not found — cannot confirm freshness"
    status = report.get("status", "?")
    glyph = OK_GLYPH if status == "OK" else BAD_GLYPH
    lines = [
        _title("Data integrity"),
        f"{glyph} {escape(str(status))} — ref {escape(str(report.get('reference_date')))} · "
        f"{report.get('n_fail', 0)} fail / {report.get('n_warn', 0)} warn",
    ]
    for reason in (report.get("reasons") or [])[:10]:
        lines.append(f"  {BAD_GLYPH} {escape(str(reason))}")
    for warn in (report.get("warnings") or [])[:5]:
        lines.append(f"  {WARN_GLYPH} {escape(str(warn))}")
    return "\n".join(lines)


def fmt_shadowlog(records: list[dict]) -> str:
    lines = [_title("Forward-shadow log (newest last)")]
    if not records:
        return "\n".join(lines + ["  no shadow-log entries yet"])
    trows: list[list[Any]] = []
    for rec in records:
        td = str(rec.get("trade_date", "?"))
        sleeves = rec.get("sleeves") or []
        sleeve_pnl = rec.get("sleeve_pnl") or {}
        if not sleeve_pnl:
            trows.append([td, ", ".join(map(str, sleeves)) if sleeves else "—", "flat"])
        else:
            for sleeve, vals in sleeve_pnl.items():
                trows.append([td, str(sleeve),
                              money((vals or {}).get("unrealized", 0.0), signed=True)])
    return "\n".join(lines + ["", _table(trows, aligns=["l", "l", "r"],
                                         header=["date", "sleeve", "shadow P&L"])])


def fmt_users(admins, seed, managed: dict[int, dict]) -> str:
    admins, seed = set(admins), set(seed)
    lines = [_title("Allowlist"), "", _label("Admins (👑 immutable)")]
    lines += [f"  👑 {a}" for a in sorted(admins)] or ["  (none)"]
    lines += ["", _label("Env seed (BOT_ALLOWED_CHAT_IDS)")]
    lines += [f"  {s}" for s in sorted(seed - admins)] or ["  (none)"]
    lines += ["", _label("Managed (/allow)")]
    mg = {cid: info for cid, info in managed.items() if cid not in admins and cid not in seed}
    if mg:
        for cid in sorted(mg):
            info = mg[cid] or {}
            extra = []
            if info.get("note"):
                extra.append(escape(str(info["note"])))
            if info.get("added_by"):
                extra.append(f"by {info['added_by']}")
            lines.append(f"  {cid}" + (f" ({', '.join(extra)})" if extra else ""))
    else:
        lines.append("  (none)")
    return "\n".join(lines)


# --- static notices ------------------------------------------------------------------------
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


def fmt_start() -> str:
    """Warm greeting (distinct from /help): what this is, one-line capability, read-only, access."""
    return "\n".join([
        _title("MOEX Agent Monitor"),
        "Read-only monitor for the MOEX V3 multi-strategy trading agent.",
        "I report status, positions, P&amp;L, data integrity and the shadow gate — I never trade.",
        "",
        "Send /help for the full command list.",
        "Access is allowlist-controlled. Not allowed yet? /status shows your chat id to give an admin.",
    ])


_HELP_READ = "\n".join([
    _title("MOEX Agent Monitor — commands"),
    "Read-only. I observe the agent; I never trade.",
    "",
    _label("Monitor"),
    "/status — mode, kill-switch, last cycle, gross",
    "/positions — live + shadow book",
    "/pnl — P&amp;L by sleeve (live vs shadow)",
    "/prices [TICKERS] — last close",
    "/cycle — last EOD result",
    "/integrity — data freshness gate",
    "",
    _label("Research"),
    "/gate — H9 shadow gate (is_production, forward P&amp;L)",
    "/shadowlog [N] — forward-shadow track (last N)",
    "",
    _label("General"),
    "/start — about this bot",
    "/help — this message",
])
_HELP_ADMIN = "\n".join([
    _label("Admin"),
    "/users — show the allowlist",
    "/allow &lt;chat_id&gt; [note] — grant read access",
    "/deny &lt;chat_id&gt; — revoke a managed id",
])


def fmt_help(is_admin: bool = False) -> str:
    return _HELP_READ + (f"\n\n{_HELP_ADMIN}" if is_admin else "")
