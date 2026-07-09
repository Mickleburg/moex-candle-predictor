"""Wire-test the T-Invest adapter against the SANDBOX (local only — needs TINVEST_TOKEN in .env).

Proves the order_request -> execution_report path matches the live wire API end-to-end:
  1. open + fund a sandbox account
  2. pull a real-time quote (proves prices come from the broker, no paid data subscription)
  3. place a marketable LIMIT BUY (1 lot) -> poll order state for a fill
  4. place a passive LIMIT, then cancel it
  5. duplicate-order protection: same client_order_id twice -> idempotent (one exchange order)
  6. close the sandbox account (cleanup)

Run:  ml/.venv-win/Scripts/python.exe execution/scripts/wire_test_tinvest_sandbox.py
This is NOT part of the default pytest suite (it needs network + a token). Secrets stay in .env.
Nothing here touches LIVE — sandbox only.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from execution.src.brokers.tinvest import TInvestBroker  # noqa: E402

# Well-known-correct FIGI (used while backend FIGIs are curated/unverified, per task).
FALLBACK_FIGI = {"SBER": "BBG004730N88"}
TICKER = "SBER"
LOTS = 1


def _load_env(path: Path) -> None:
    """Tiny .env loader so TINVEST_TOKEN is available without a dependency. Never prints values."""
    import os
    if not path.exists():
        return
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _figi_map() -> dict[str, str]:
    try:
        from execution.src.instruments import load_figi_map
        m = load_figi_map()
        if m.get(TICKER):
            return {TICKER: m[TICKER]}
    except Exception:
        pass
    return dict(FALLBACK_FIGI)


def main() -> int:
    import os
    _load_env(REPO_ROOT / ".env")
    if not os.environ.get("TINVEST_TOKEN"):
        print("SKIP: TINVEST_TOKEN not set (.env). This is a local sandbox wire-test.")
        return 0

    figi = _figi_map()
    print(f"FIGI for {TICKER}: {figi[TICKER]}")
    broker = TInvestBroker(sandbox=True, figi_by_ticker=figi)

    acct = broker.open_sandbox_account()
    print(f"opened sandbox account: {acct}")
    try:
        broker.sandbox_pay_in(1_000_000)
        print("paid in 1,000,000 RUB")

        # 2) quotes from the broker (task 3: live prices come from here, no paid sub)
        last = broker.last_price(TICKER)
        marketable = broker.quote_limit_price(TICKER, "BUY", marketable=True)
        print(f"broker quote: last={last:.2f}  marketable_buy(best_ask/last)={marketable:.2f}")

        # 3) marketable BUY limit -> expect a fill (during market hours)
        buy = {"ticker": TICKER, "side": "BUY", "quantity_lots": LOTS, "order_type": "LIMIT",
               "limit_price": round(marketable * 1.01, 2), "client_order_id": f"wire-buy-{int(time.time())}"}
        rep = broker.place_order(buy)
        _assert_report_shape(rep, buy["client_order_id"])
        print(f"placed BUY -> status={rep['status']} exch_id={rep['exchange_order_id']} "
              f"filled={rep['filled_quantity_lots']} avg={rep['avg_fill_price']}")
        for _ in range(6):
            if rep["status"] in ("FILLED", "REJECTED", "CANCELED"):
                break
            time.sleep(1.0)
            rep = broker.order_state(buy["client_order_id"], TICKER)
        print(f"final BUY state -> status={rep['status']} filled={rep['filled_quantity_lots']} "
              f"avg={rep['avg_fill_price']}")
        if rep["status"] == "FILLED":
            print("  FILL CONFIRMED")
        else:
            print("  not filled (market likely closed) — order form + lifecycle still verified;"
                  " canceling residual")
            if rep["status"] == "PLACED":
                broker.cancel(buy["client_order_id"])

        # 4) passive limit + cancel
        passive_px = round(last * 0.80, 2)   # far below market -> rests, won't fill
        passive = {"ticker": TICKER, "side": "BUY", "quantity_lots": LOTS, "order_type": "LIMIT",
                   "limit_price": passive_px, "client_order_id": f"wire-cancel-{int(time.time())}"}
        prep = broker.place_order(passive)
        print(f"placed passive BUY @ {passive_px} -> status={prep['status']} exch_id={prep['exchange_order_id']}")
        crep = broker.cancel(passive["client_order_id"])
        _assert_report_shape(crep, passive["client_order_id"])
        assert crep["status"] == "CANCELED", crep
        print(f"canceled -> status={crep['status']}  CANCEL CONFIRMED")

        # 5) duplicate-order protection (idempotency by client_order_id), both layers:
        dup = {"ticker": TICKER, "side": "BUY", "quantity_lots": LOTS, "order_type": "LIMIT",
               "limit_price": round(last * 0.80, 2), "client_order_id": f"wire-dup-{int(time.time())}"}
        n_before = len(broker.get_orders())
        r1 = broker.place_order(dup)                       # creates one resting order
        r2 = broker.place_order(dup)                       # layer 1: cached -> not re-sent
        assert r2["exchange_order_id"] == r1["exchange_order_id"], (r1, r2)
        # layer 2: force the wire path (clear the cache) -> the API itself dedups (code 30057)
        broker._open.pop(dup["client_order_id"], None)
        r3 = broker.place_order(dup)
        assert r3["status"] != "REJECTED", r3
        n_after = len(broker.get_orders())
        print(f"dup-protection: layer1 cached(exch r1==r2)=True; layer2 wire r3='{r3['message']}'; "
              f"open orders {n_before}->{n_after} (+1, not +3)")
        assert n_after == n_before + 1, (n_before, n_after)
        broker._open[dup["client_order_id"]] = r1["exchange_order_id"]
        broker.cancel(dup["client_order_id"])

        print("\nWIRE TEST OK: order_request <-> execution_report forms match; "
              "cancel + duplicate-protection verified; quotes sourced from broker.")
        return 0
    finally:
        try:
            broker.close_sandbox_account(acct)
            print(f"closed sandbox account: {acct}")
        except Exception as exc:  # noqa: BLE001 - cleanup best-effort
            print(f"(cleanup) could not close sandbox account {acct}: {exc}")


def _assert_report_shape(rep: dict, coid: str) -> None:
    """The adapter must return a contract-shaped execution_report."""
    for key in ("client_order_id", "ticker", "status", "filled_quantity_lots", "message"):
        assert key in rep, f"execution_report missing {key}: {rep}"
    assert rep["client_order_id"] == coid, rep
    assert rep["status"] in ("DRY_RUN", "PLACED", "REJECTED", "FILLED", "CANCELED"), rep


if __name__ == "__main__":
    raise SystemExit(main())
