r"""REAL LLM news-feature extraction on the WEEKLY decision grid (content-hypothesis test).

For every (ticker, as_of) on data/features/decision_grid_weekly.csv we take the disclosures in
the trailing 7-day window (no-lookahead, pub_date <= as_of) and ask the internal Positive LLM
(Gemma) for sentiment / impact_score / novelty / event_type from the actual disclosure content.
news_count / recency_minutes stay deterministic. Cells with no news skip the LLM (emit zeros).

Resumable: a content-keyed cache (data/news/.llm_weekly_cache.json) means a rerun re-issues no
calls for already-seen disclosure sets. Throttled to respect RPM 30 (provider also retries 429).

Output: data/news/llm_weekly_features.csv  (same tidy schema as baseline_features.csv)
  ticker, as_of (verbatim grid string), sentiment, impact_score, novelty, event_type,
  news_count, recency_minutes

Run: $env:PYTHONPATH="llm\src"; & "ml\.venv-win\Scripts\python.exe" llm\scripts\gen_llm_weekly_features.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "llm" / "src"))

from llm_ta import features as feat              # noqa: E402
from llm_ta.providers import provider_from_name  # noqa: E402
from llm_ta.validator import parse_strict_json   # noqa: E402

GRID_CSV = REPO_ROOT / "data" / "features" / "decision_grid_weekly.csv"
OUT_CSV = REPO_ROOT / "data" / "news" / "llm_weekly_features.csv"
CACHE_JSON = REPO_ROOT / "data" / "news" / ".llm_weekly_cache.json"
PROMPT_PATH = REPO_ROOT / "llm" / "prompts" / "news_features_prompt.txt"

WINDOW_HOURS = 7 * 24
THROTTLE_S = 2.1            # >= 1 req / 2s keeps us under RPM 30
PROVIDER_NAME = "positive"  # Gemma (positive-llm-chat)

UNIVERSE = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK",
            "TATN", "MGNT", "MTSS", "SNGS", "CHMF", "ALRS"]


def _clamp(v, lo, hi, default):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, f))


def _content_key(ticker: str, titles: str) -> str:
    return hashlib.sha1(f"{ticker}\n{titles}".encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=UNIVERSE)
    ap.add_argument("--limit", type=int, default=0, help="limit as_of points (0 = all); smoke only")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()
    tickers = args.tickers

    provider = provider_from_name(PROVIDER_NAME)
    template = PROMPT_PATH.read_text(encoding="utf-8")

    cache: dict[str, dict] = {}
    if CACHE_JSON.exists():
        cache = json.loads(CACHE_JSON.read_text(encoding="utf-8"))
        print(f"loaded cache: {len(cache)} entries")

    raw_as_of = pd.read_csv(GRID_CSV)["as_of"].astype(str).tolist()
    if args.limit:
        raw_as_of = raw_as_of[:args.limit]
    # parse for no-lookahead: naive weekly stamps are UTC -> aware; output keeps verbatim string
    aware = []
    for s in raw_as_of:
        ts = pd.Timestamp(s)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        aware.append(ts.to_pydatetime())
    print(f"weekly grid: {len(raw_as_of)} as_of  ({raw_as_of[0]} .. {raw_as_of[-1]})")

    rows: list[dict] = []
    calls = hits = fails = nonews = 0
    t0 = time.time()

    for ticker in tickers:
        disclosures = feat.load_disclosures(ticker)
        for s, as_of in zip(raw_as_of, aware):
            feats, window = feat.compute_features(disclosures, as_of, window_hours=WINDOW_HOURS)
            row = {"ticker": ticker, "as_of": s,
                   "sentiment": 0.0, "impact_score": 0.0, "novelty": 0.0,
                   "event_type": "none",
                   "news_count": feats["news_count"], "recency_minutes": feats["recency_minutes"]}

            if feats["news_count"] == 0:
                nonews += 1
                rows.append(row)
                continue

            titles = "\n".join(f"- [{d.pub_date.date()}] {d.event_name}" for d in window[-40:])
            key = _content_key(ticker, titles)

            if key in cache:
                refined = cache[key]
                hits += 1
            else:
                prompt = template.replace("{{TICKER}}", ticker).replace("{{DISCLOSURES}}", titles)
                try:
                    parsed = parse_strict_json(provider.generate(prompt=prompt, request_payload={}))
                    refined = {
                        "sentiment": _clamp(parsed.get("sentiment"), -1, 1, feats["sentiment"]),
                        "impact_score": _clamp(parsed.get("impact_score"), 0, 1, feats["impact_score"]),
                        "novelty": _clamp(parsed.get("novelty"), 0, 1, feats["novelty"]),
                        "event_type": str(parsed.get("event_type") or feats["event_type"]),
                    }
                    cache[key] = refined
                except Exception as exc:  # provider exhausted retries -> fall back to baseline
                    fails += 1
                    refined = {"sentiment": feats["sentiment"], "impact_score": feats["impact_score"],
                               "novelty": feats["novelty"], "event_type": feats["event_type"]}
                    print(f"  ! {ticker} {s}: LLM failed ({type(exc).__name__}: {str(exc)[:80]}) -> baseline")
                calls += 1
                if calls % 25 == 0:
                    CACHE_JSON.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                    rate = calls / max(1e-6, (time.time() - t0)) * 60
                    print(f"  ... {calls} calls ({hits} cache hits, {fails} fails), "
                          f"{rate:.0f} calls/min")
                time.sleep(THROTTLE_S)

            row.update({"sentiment": round(float(refined["sentiment"]), 4),
                        "impact_score": round(float(refined["impact_score"]), 4),
                        "novelty": round(float(refined["novelty"]), 4),
                        "event_type": refined["event_type"]})
            rows.append(row)
        print(f"  [{ticker}] done (calls so far={calls}, hits={hits}, nonews={nonews})", flush=True)

    CACHE_JSON.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")

    mins = (time.time() - t0) / 60
    print(f"\nwrote {args.out}  rows={len(df)}  in {mins:.1f} min")
    print(f"  LLM calls={calls}, cache hits={hits}, failures={fails}, no-news cells={nonews}")
    print(f"  sentiment nonzero: {(df['sentiment'] != 0).mean():.1%}; "
          f"mean|sentiment|={df['sentiment'].abs().mean():.3f}; "
          f"event_type=none: {(df['event_type'] == 'none').mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
