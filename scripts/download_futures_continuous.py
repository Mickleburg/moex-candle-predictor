"""Build a continuous front-month series for a MOEX FORTS monthly future (Brent BR, gas NG).

MOEX futures roll monthly and the live securities list only shows active contracts, so we
ENUMERATE historical contracts SECID = <ASSET><monthcode><yeardigit> (e.g. BRG0 = Brent Feb 2020),
fetch each contract's 1H candles, and stitch a front-month series.

Roll handling without back-adjustment math: we store a synthetic `close` = cumulative product of
WITHIN-CONTRACT 1h returns (return is set to 0 at each roll boundary). Downstream features compute
pct_change on this close -> they recover the clean within-contract returns with no roll jumps.
Absolute price level is not preserved (irrelevant for dimensionless return/vol features).

Usage:
    python scripts/download_futures_continuous.py --asset BR --from 2020 --to 2026
    python scripts/download_futures_continuous.py --asset NG --from 2020 --to 2026
Saves: data/raw/<ASSET>_CONT_1H_<range>.parquet  (ticker=<ASSET>_CONT)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
MOEX_ISS_BASE = "https://iss.moex.com"

# Futures month codes -> month number
MONTH_CODE = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
              "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}
CODE_BY_MONTH = {v: k for k, v in MONTH_CODE.items()}


def fetch_contract_candles(session, secid: str) -> pd.DataFrame:
    """Fetch all 1H candles for one contract (ISS returns its trading window)."""
    rows_all = []
    start = 0
    while True:
        r = session.get(
            f"{MOEX_ISS_BASE}/iss/engines/futures/markets/forts/securities/{secid}/candles.json",
            params={"iss.meta": "off", "iss.only": "candles", "interval": 60,
                    "from": "2019-11-01", "till": "2027-02-01", "start": start},
            timeout=30)
        r.raise_for_status()
        b = r.json().get("candles", {})
        cols, rows = b.get("columns", []), b.get("data", [])
        if not rows:
            break
        ci = {c: i for i, c in enumerate(cols)}
        for row in rows:
            rows_all.append((row[ci["begin"]], float(row[ci["close"]])))
        if len(rows) < 500:
            break
        start += 500
        time.sleep(0.2)
    if not rows_all:
        return pd.DataFrame(columns=["begin", "close"])
    df = pd.DataFrame(rows_all, columns=["begin", "close"])
    df["begin"] = pd.to_datetime(df["begin"])
    return df.drop_duplicates("begin").sort_values("begin").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", required=True, help="BR (Brent) or NG (natural gas)")
    ap.add_argument("--from", dest="y_from", type=int, default=2020)
    ap.add_argument("--to", dest="y_to", type=int, default=2026)
    args = ap.parse_args()
    asset = args.asset.upper()

    session = requests.Session()
    session.headers["User-Agent"] = "moex-futures-continuous/0.1"

    # Enumerate contracts (asset + monthcode + yeardigit), compute approx expiry (1st of month).
    contracts = []
    for ydig in range(args.y_from % 10, (args.y_to % 10) + 1):
        year = 2020 + ydig
        for mnum in range(1, 13):
            secid = f"{asset}{CODE_BY_MONTH[mnum]}{ydig}"
            expiry = pd.Timestamp(year=year, month=mnum, day=1)
            contracts.append((secid, expiry))
    contracts.sort(key=lambda x: x[1])

    print(f"Fetching {len(contracts)} {asset} contracts...")
    segments = []
    prev_expiry = pd.Timestamp("2019-01-01")
    for secid, expiry in contracts:
        df = fetch_contract_candles(session, secid)
        if df.empty:
            prev_expiry = expiry
            continue
        # front-month window: (prev_expiry, expiry]
        seg = df[(df["begin"] > prev_expiry) & (df["begin"] <= expiry)].copy()
        if not seg.empty:
            seg["ret"] = seg["close"].pct_change().fillna(0.0)  # 0 at roll boundary
            segments.append(seg[["begin", "ret"]])
            print(f"  {secid}: {len(seg)} front bars  ({seg['begin'].min()} .. {seg['begin'].max()})")
        prev_expiry = expiry
        time.sleep(0.1)

    if not segments:
        print("No data assembled."); return 1

    cont = pd.concat(segments).drop_duplicates("begin").sort_values("begin").reset_index(drop=True)
    cont["ret"] = cont["ret"].clip(-0.5, 0.5)               # guard against bad ticks
    cont["close"] = 100.0 * (1.0 + cont["ret"]).cumprod()  # synthetic continuous close
    cont["ticker"] = f"{asset}_CONT"
    cont["timeframe"] = "1H"
    out = cont[["ticker", "timeframe", "begin", "close"]]

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    b0 = out["begin"].min().strftime("%Y%m%dT%H%M"); b1 = out["begin"].max().strftime("%Y%m%dT%H%M")
    path = DATA_RAW / f"{asset}_CONT_1H_{b0}_{b1}.parquet"
    out.to_parquet(path, index=False, engine="pyarrow")
    print(f"Saved {len(out)} continuous bars: {out['begin'].min()} .. {out['begin'].max()}")
    print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
