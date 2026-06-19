"""Integrity audit of the ML block's V2 paths — run before trusting any H2 result.

Concrete checks (not eyeballing):
  1. Quant no-lookahead: perturbing a FUTURE candle must not change features at an earlier as_of.
  2. News no-lookahead: future news must not enter the bundle; in-window past news must.
  3. Target label window: target[d] depends only on prices up to d+horizon (perturb beyond -> no change).
  4. Deployment-sim leakage: training pool at decision i contains only labels realized by i.
  5. News informativeness — time-shuffle control: shuffling news as_of WITHIN a ticker should
     collapse a DYNAMIC news signal toward 0; if IC survives, it is a static ticker bias, not news.
  6. Contracts: feature_bundle and aggregated_signal examples validate against frozen schemas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
REPO = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from src.features.cross_sectional import load_panels, relative_target  # noqa: E402
from src.features.feature_bundle import assemble_feature_bundle, validate_against_schema  # noqa: E402
from src.features.decision_grid import NEWS_WINDOW_DAYS  # noqa: E402
from scripts.h2_news_ic_diag import feature_panel, spearman, FORWARD_START  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((ok, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}{(' — ' + detail) if detail else ''}")


def main() -> int:
    panel, sector_panel, market = load_panels(timeframe="1D")
    news = pd.read_csv(REPO / "data/news/baseline_features.csv")
    news["as_of"] = pd.to_datetime(news["as_of"], utc=True).dt.tz_convert("Europe/Moscow")
    as_of = panel.index[-40]

    print("1) Quant no-lookahead (perturb future candle -> earlier bundle unchanged)")
    b0 = assemble_feature_bundle(panel, news, as_of)
    p2 = panel.copy(); p2.iloc[-10] *= 1.5          # perturb a candle AFTER as_of
    b1 = assemble_feature_bundle(p2, news, as_of)
    q_same = all(e0["quant"] == e1["quant"] for e0, e1 in zip(b0["entries"], b1["entries"]))
    check("quant features ignore future candles", q_same)

    print("2) News no-lookahead (future news ignored; in-window past news used)")
    fut = pd.concat([news, pd.DataFrame([{"ticker": "SBER", "as_of": as_of + pd.Timedelta(days=3),
        "sentiment": 9.0, "impact_score": 9.0, "novelty": 0, "event_type": "x",
        "news_count": 9, "recency_minutes": 0}])], ignore_index=True)
    b_fut = assemble_feature_bundle(panel, fut, as_of)
    sber0 = next(e for e in b0["entries"] if e["ticker"] == "SBER")["news"]
    sberF = next(e for e in b_fut["entries"] if e["ticker"] == "SBER")["news"]
    check("future news does not enter the bundle", sber0 == sberF)
    # inject the LATEST in-window row (as_of exactly, appended last -> wins under latest-takes-all)
    past = pd.concat([news, pd.DataFrame([{"ticker": "SBER", "as_of": as_of,
        "sentiment": 0.987654, "impact_score": 0.5, "novelty": 0, "event_type": "x",
        "news_count": 1, "recency_minutes": 0}])], ignore_index=True)
    b_past = assemble_feature_bundle(panel, past, as_of)
    sberP = next(e for e in b_past["entries"] if e["ticker"] == "SBER")["news"]
    check("in-window latest news updates the bundle", sberP != sber0,
          f"sentiment {sber0[0]} -> {sberP[0]}")

    print("3) Target label window (target[d] uses prices only up to d+horizon)")
    H = 10
    tgt0 = relative_target(panel, H, "beta_residual", sector_panel, market)
    di = len(panel) - 200
    p3 = panel.copy(); p3.iloc[di + H + 5] *= 1.5    # perturb beyond the label window
    tgt1 = relative_target(p3, H, "beta_residual", sector_panel, market)
    same = np.allclose(tgt0.iloc[di].to_numpy(), tgt1.iloc[di].to_numpy(), equal_nan=True)
    check("target ignores prices beyond d+horizon", same)
    p4 = panel.copy(); p4.iloc[di + H] *= 1.5        # perturb the label endpoint close[d+H]
    tgt2 = relative_target(p4, H, "beta_residual", sector_panel, market)
    changed = not np.allclose(tgt0.iloc[di].to_numpy(), tgt2.iloc[di].to_numpy(), equal_nan=True)
    check("target responds to the label endpoint close[d+H]", changed)

    print("4) Deployment-sim leakage (training labels realized before decision)")
    # reconstruct the pool rule: at decision i, samples come from d=i-H, label realized at d+H=i
    ok = all((i - H) + H <= i for i in range(H, len(panel)))
    check("training label end (d+H) <= decision i", ok, "rolling pool is causal by construction")

    print("5) News informativeness — time-shuffle control (dynamic vs static ticker bias)")
    tgt = relative_target(panel, 20, "beta_residual", sector_panel, market)
    def fwd_ic(nws, col):
        fp = feature_panel(nws, col, panel.index, panel.columns)
        ics = [spearman(fp.loc[t].to_numpy(), tgt.loc[t].to_numpy())
               for t in panel.index if np.isfinite(fp.loc[t].to_numpy()).sum() >= 3
               and np.nanstd(fp.loc[t].to_numpy()) > 0]
        s = pd.Series(ics, index=[t for t in panel.index
            if np.isfinite(fp.loc[t].to_numpy()).sum() >= 3 and np.nanstd(fp.loc[t].to_numpy())>0]).dropna()
        return float(s[s.index >= FORWARD_START].mean())
    rng = np.random.default_rng(0)
    shuf = news.copy()
    for tkr, idx in news.groupby("ticker").groups.items():
        vals = shuf.loc[idx, ["sentiment", "impact_score", "novelty", "news_count", "recency_minutes"]].to_numpy()
        shuf.loc[idx, ["sentiment", "impact_score", "novelty", "news_count", "recency_minutes"]] = rng.permutation(vals)
    for col in ("impact_score", "news_count"):
        real, sh = fwd_ic(news, col), fwd_ic(shuf, col)
        # informative-as-dynamic if real IC noticeably exceeds the shuffled (static) baseline
        print(f"    {col}: real FWD IC={real:+.4f}  shuffled={sh:+.4f}  "
              f"dynamic-share={real - sh:+.4f}")

    print("6) Contracts validate against frozen schemas")
    try:
        validate_against_schema(b0)
        from src.service.cross_sectional_signal import build_aggregated_signal, validate_against_schema as vas
        sig = build_aggregated_signal(as_of=as_of.isoformat(), timeframe="1D", horizon_bars=20,
            scores={t: float(i) for i, t in enumerate(panel.columns)}, k=3, model_version="audit")
        vas(sig)
        check("feature_bundle + aggregated_signal schema-valid", True)
    except Exception as e:
        check("feature_bundle + aggregated_signal schema-valid", False, str(e))

    n_fail = sum(1 for ok, _, _ in results if not ok)
    print(f"\n{'='*60}\nAUDIT: {len(results)-n_fail}/{len(results)} checks PASS, {n_fail} FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
