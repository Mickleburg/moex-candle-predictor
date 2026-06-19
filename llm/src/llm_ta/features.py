"""V2 news-feature extraction for the LLM block.

Turns e-disclosure corporate disclosures (data/news/edisclosure/{TICKER}.parquet,
produced by llm/scripts/edisc_extract.py) into the frozen `llm_analysis` contract:
structured NEWS FEATURES for one ticker at one decision time — NOT a buy/hold/sell call.

Hard rules (see llm/CLAUDE.md, contracts/llm_analysis.schema.json):
  * No-lookahead by PUBLICATION time: only disclosures with pub_date <= as_of contribute.
    We use `pub_date` (publication), never `event_date` (when the event happened).
  * Output is deterministic and schema-valid; mock/baseline => is_production=false.

The deterministic baseline here needs no LLM and no network (free, reproducible). An
optional LLM provider can refine sentiment/novelty later via the same contract — the
decision model only ever sees the contract fields, not how they were produced.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
DISCLOSURES_DIR = REPO_ROOT / "data" / "news" / "edisclosure"

# e-disclosure pubDate and MOEX candle timestamps are Moscow wall-clock. We anchor all
# times to MSK so as_of (which may be naive or carry an offset) compares cleanly.
MSK = dt.timezone(dt.timedelta(hours=3))


def to_msk(value: dt.datetime) -> dt.datetime:
    """Return a tz-aware MSK datetime (naive inputs are assumed to be Moscow time)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=MSK)
    return value.astimezone(MSK)

BASELINE_MODEL_VERSION = "edisc_rule_baseline_v0"
DEFAULT_WINDOW_HOURS = 72        # news lookback for a 1H cross-sectional decision
NOVELTY_PRIOR_DAYS = 30          # window used to judge how "new" an event type is
MAX_SOURCES = 10                 # cap sources[] in the emitted contract

# --- event_type taxonomy: map a Russian disclosure title to a coarse event class ----
# Order matters: first matching rule wins. Keys are lowercase substrings of event_name.
_EVENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("dividend",     ("дивиденд", "выплаченные доход", "начисленные доход", "выплат", "причитающиеся")),
    ("earnings",     ("отчётност", "отчетност", "бухгалтерск", "мсфо", "финансовые результат", "годовой отчёт", "годовой отчет")),
    ("meeting",      ("общего собрания", "общем собрании", "собрани", "акционеров", "решения единственного", "повестк")),
    ("m_and_a",      ("реорганизац", "присоединен", "слияни", "приобретен", "контролирующ", "взаимозависим", "существенной сделк", "заинтересованност")),
    ("listing",      ("допущен", "список ценных бумаг", "размещени", "выпуск", "эмисси", "облигац", "регистрац", "делистинг", "исключение")),
    ("rating",       ("рейтинг",)),
    ("distress",     ("банкротств", "дефолт", "ликвидац", "несостоятельн")),
    ("sanctions",    ("санкц", "ограничительн")),
    ("guidance",     ("прогноз", "стратеги", "план")),
    ("price_impact", ("стоимость или котировк", "стоимост его эмиссионн", "существенное влияние")),
    ("management",   ("совет директоров", "единоличн", "правлен", "руководител", "избран", "досрочн")),
]

# coarse sentiment / impact priors per event class (deterministic baseline only;
# the real signal is meant to come from the optional LLM layer). Conservative on purpose.
_CLASS_PRIOR: dict[str, tuple[float, float]] = {
    # event_type: (sentiment, impact_score)
    "dividend":     (0.30, 0.55),
    "earnings":     (0.00, 0.65),
    "meeting":      (0.00, 0.30),
    "m_and_a":      (0.05, 0.70),
    "listing":      (0.05, 0.45),
    "rating":       (0.00, 0.55),
    "distress":     (-0.60, 0.85),
    "sanctions":    (-0.50, 0.80),
    "guidance":     (0.05, 0.50),
    "price_impact": (0.00, 0.60),
    "management":   (0.00, 0.40),
    "other":        (0.00, 0.35),
}


def classify_event(event_name: str | None) -> str:
    name = (event_name or "").lower()
    for label, needles in _EVENT_RULES:
        if any(n in name for n in needles):
            return label
    return "other"


@dataclass(frozen=True)
class Disclosure:
    pub_date: dt.datetime
    event_name: str
    event_type: str
    pseudo_guid: str | None
    agency: str | None


def _parse_dt(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value
    try:
        return dt.datetime.fromisoformat(str(value))
    except ValueError:
        return None


def load_disclosures(ticker: str, disclosures_dir: Path = DISCLOSURES_DIR) -> list[Disclosure]:
    """Load all disclosures for a ticker (unfiltered by time)."""
    import pandas as pd

    path = disclosures_dir / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no disclosure file for {ticker}: {path}")
    df = pd.read_parquet(path, columns=["pub_date", "event_name", "pseudo_guid", "agency"])
    out: list[Disclosure] = []
    for rec in df.itertuples(index=False):
        pub = _parse_dt(rec.pub_date)
        if pub is None:
            continue
        out.append(Disclosure(
            pub_date=to_msk(pub),
            event_name=str(rec.event_name),
            event_type=classify_event(rec.event_name),
            pseudo_guid=(None if rec.pseudo_guid is None else str(rec.pseudo_guid)),
            agency=(None if rec.agency is None else str(rec.agency)),
        ))
    out.sort(key=lambda d: d.pub_date)
    return out


def _dominant(values: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    if not counts:
        return "none"
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def compute_features(
    disclosures: list[Disclosure],
    as_of: dt.datetime,
    window_hours: int = DEFAULT_WINDOW_HOURS,
) -> tuple[dict[str, Any], list[Disclosure]]:
    """Deterministic baseline features for `as_of`. Returns (features, window_items).

    No-lookahead: only disclosures with pub_date <= as_of are considered.
    """
    past = [d for d in disclosures if d.pub_date <= as_of]
    window_start = as_of - dt.timedelta(hours=window_hours)
    window = [d for d in past if d.pub_date >= window_start]

    news_count = len(window)
    if news_count == 0:
        return ({
            "sentiment": 0.0,
            "impact_score": 0.0,
            "novelty": 0.0,
            "event_type": "none",
            "news_count": 0,
            "recency_minutes": float(window_hours * 60),
        }, [])

    dom = _dominant(d.event_type for d in window)
    sent = sum(_CLASS_PRIOR[d.event_type][0] for d in window) / news_count
    # impact: max single-item impact, nudged up by volume (more disclosures -> more salient)
    base_impact = max(_CLASS_PRIOR[d.event_type][1] for d in window)
    impact = min(1.0, base_impact + 0.03 * (news_count - 1))

    # novelty: how rare the dominant event_type was in the trailing NOVELTY_PRIOR_DAYS
    prior_start = window_start - dt.timedelta(days=NOVELTY_PRIOR_DAYS)
    prior = [d for d in past if prior_start <= d.pub_date < window_start]
    prior_same = sum(1 for d in prior if d.event_type == dom)
    novelty = 1.0 / (1.0 + prior_same)

    most_recent = max(d.pub_date for d in window)
    recency_minutes = max(0.0, (as_of - most_recent).total_seconds() / 60.0)

    features = {
        "sentiment": round(max(-1.0, min(1.0, sent)), 4),
        "impact_score": round(impact, 4),
        "novelty": round(novelty, 4),
        "event_type": dom,
        "news_count": news_count,
        "recency_minutes": round(recency_minutes, 1),
    }
    return features, window


def build_analysis(
    ticker: str,
    as_of: dt.datetime,
    timeframe: str = "1H",
    window_hours: int = DEFAULT_WINDOW_HOURS,
    disclosures: list[Disclosure] | None = None,
    model_version: str = BASELINE_MODEL_VERSION,
    is_production: bool = False,
) -> dict[str, Any]:
    """Assemble a schema-valid llm_analysis object from disclosures (deterministic)."""
    if disclosures is None:
        disclosures = load_disclosures(ticker)
    as_of = to_msk(as_of)
    features, window = compute_features(disclosures, as_of, window_hours)

    sources = [
        {"source": (d.agency or "edisclosure"),
         "url": (f"https://www.e-disclosure.ru/portal/event.aspx?eventid={d.pseudo_guid}"
                 if d.pseudo_guid else "https://www.e-disclosure.ru/"),
         "published_at": d.pub_date.isoformat()}
        for d in sorted(window, key=lambda x: x.pub_date, reverse=True)[:MAX_SOURCES]
    ]

    return {
        "as_of": as_of.isoformat(),
        "ticker": ticker,
        "timeframe": timeframe,
        "features": features,
        "affected_tickers": [ticker],
        "sources": sources,
        "model_version": model_version,
        "is_production": is_production,
    }
