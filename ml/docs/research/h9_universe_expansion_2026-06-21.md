# H9 dividend universe expansion — a-priori spec (fixed 2026-06-21)

> **Purpose.** Grow the H9 dividend-sleeve universe so the **July-2026 wave produces more forward
> events**, to push the realized-P&L shadow gate toward **n ≥ 25 in one accrual** instead of waiting
> for autumn. This document **FIXES the inclusion criteria and the expanded list BEFORE any expanded
> backtest** — that ordering is the whole discipline. Selection is by liquidity + dividend history,
> never by "which names make the backtest prettier."
>
> **Gate is unchanged.** n ≥ 25 forward events AND net > 0 AND %pos > 0.5. Not lowered. The burned
> 2025-09→2026-06 split is not touched. The expanded universe must re-pass the placebo/time-shuffle
> control (run-up must vanish on shuffled dates) or it is reverted.

## 0. Current state (empirical, 2026-06-21)

- Candle panel (`data/raw/*_1H_*.parquet`) and dividend history (`data/raw/dividends.csv`) both cover
  **exactly the same 16 ordinary lines**: SBER GAZP LKOH GMKN ROSN NVTK TATN MGNT MTSS SNGS CHMF ALRS
  VTBR MAGN NLMK PLZL. No preferred lines, no new issuers — **zero data** for any candidate below.
- `dividends.csv`: 283 events, **range 2013-08-09 → 2025-07-20** (16 tickers).
- Forward feed (`data/news/dividend_calendar_upcoming.csv`): **only the 7 current-universe July-2026
  records** (MTSS/ROSN 07-09, PLZL 07-13, TATN 07-15, SNGS 07-16, SBER/VTBR 07-20).
- Shadow gate today: **forward n = 12** (all 2025), NOT MET.

**Consequence:** the expanded IS study, the placebo control, and the final serving update are
**hard-gated on data** that backend (candles + ISS history) and llm (forward feed) must deliver. This
doc delivers everything that is *not* gated: the fixed criteria, the fixed list, the per-name
justification, the data request, and the forward-n forecast.

## 1. A-priori inclusion criteria (FIXED — apply uniformly to every candidate)

A **tradeable line** (an issuer's ordinary OR preferred share, treated as **separate lines**) enters
the H9 universe **iff all** hold:

1. **Liquidity / capacity.** MOEX Broad-Market constituent with trailing-12m median daily turnover
   comfortably above the sleeve's per-name capacity need. H9 capacity is ~130–190 M₽ at full book, so
   require **ADTV ≥ ~300 M₽** (single-name slice stays a small ADV fraction; slippage can't eat the
   edge). Screened by backend from the 1H panel once candles land.
2. **Regular payer.** A positive cash dividend in **≥ 4 of the last 6 fiscal years** (≥4y/6).
   Preferred lines qualify on the **issuer's** payment record. Excludes one-off / just-resumed payers
   whose forward behavior is unestablished.
3. **No forward structural disqualifier.** Not under an active dividend suspension or blocked-payment
   regime (sanctions / redomiciliation-in-progress) that makes the forward run-up untradeable.
4. **Continuous MOEX listing ≥ IS window.** The line has traded continuously on MOEX since **≥ 2021**
   (so it has IS event history). Newly-redomiciled lines with < ~1.5y continuous history are excluded
   until they accrue their own track.

> Criteria 1–4 are evaluated for **all** candidates the same way. Names are not added or dropped on
> their backtest contribution. This list is frozen by this commit prior to the expanded IS study.

## 2. Candidate evaluation (a-priori; per-name history to be CONFIRMED when backend delivers)

Calls below use public MOEX facts; ✅ = include, ⛔ = exclude. "July?" = whether the line typically has
a **summer/July record** (relevant to the July-2026 accrual, *not* an inclusion criterion).

| Line  | Type | Liquidity | Payer history (≥4/6) | July record? | Verdict | Reason |
|-------|------|-----------|----------------------|--------------|---------|--------|
| **SBERP** | pref | very high | yes (= SBER) | **yes** (~07-20) | ✅ | separate high-liquidity line, same record as SBER |
| **SNGSP** | pref | very high | yes — SNGS records **every July** (12/12 in history) | **yes** (~07-16) | ✅ | the SNGS dividend *story* is the pref; very liquid |
| **TATNP** | pref | high | yes (= TATN, ≥4/6) | **yes** (~07-15) | ✅ | liquid pref, same record as TATN |
| **SIBN** | ord | high | yes (Gazprom Neft, interim+final) | likely (summer final) | ✅ | regular payer, confirm July reco via feed |
| **PHOR** | ord | high | yes (PhosAgro, ~quarterly) | maybe | ✅ | regular payer; July record not guaranteed |
| **RTKMP** | pref | medium* | yes (Rostelecom) | ~summer | ✅* | borderline liquidity — drop if ADTV screen fails |
| **MOEX** | ord | high | yes (annual) | **no** (record ~May) | ✅ | valid member; does **not** add a July event |
| **BSPB** | ord | medium* | yes (recent, regular) | spring/summer | ✅* | borderline liquidity — confirm ADTV |
| **X5** (ex-FIVE) | ord | high now | **no** — disrupted by redomicile, resumed 2025 w/ special | n/a | ⛔ | fails crit 2 & 4 (<1.5y continuous MOEX, just-resumed/special) |
| **AFLT** | ord | high | **no** dividends (suspended) | — | ⛔ | fails crit 2 |
| **MTLR/MTLRP** | ord/pref | medium | irregular/suspended | — | ⛔ | fails crit 2 & 3 |

\* RTKMP, BSPB are **provisional**: included pending backend's ADTV confirmation (criterion 1). Drop if
they fail the 300 M₽ screen.

## 3. FIXED expanded universe (frozen by this doc)

```
Existing (16): SBER GAZP LKOH GMKN ROSN NVTK TATN MGNT MTSS SNGS CHMF ALRS VTBR MAGN NLMK PLZL
Added (8):     SBERP SNGSP TATNP SIBN PHOR RTKMP* MOEX BSPB*    (* provisional on ADTV)
-> 24 lines (22 firm + 2 provisional)
```

Note the cleanest, lowest-risk additions are the three **prefs of issuers already in the universe**
(SBERP, SNGSP, TATNP): their record dates are already confirmed (the ordinary is recommending), they
just need the pref **price** series + the pref **dividend value** history (pref payouts differ from
ordinary, so both are required).

## 4. Forward-event forecast for the July-2026 wave

Per-line July-2026 increment (record 07-09…07-20), high→low confidence:

| Source | Lines | July-2026 events |
|--------|-------|------------------|
| Current universe (already in feed) | MTSS ROSN PLZL TATN SNGS SBER VTBR | **7** |
| Prefs of in-feed issuers (near-certain) | SBERP SNGSP TATNP | **+3** |
| New ordinaries with likely summer record | SIBN (+ maybe PHOR / RTKMP) | **+1 to +3** |
| Spring-record members (no July add) | MOEX, BSPB | +0 |

**Forecast of total forward n after July-2026 closes:**

```
prior 2025 closed         12
+ current-universe July    7   -> 19
+ expanded July lines     +3 to +5
= 22 to 24   (optimistic 25-26 only if SIBN+PHOR+RTKMP all land July records)
```

**Verdict (honest):** universe expansion **materially helps but probably lands ~22–24 — just short of
n ≥ 25 from July alone.** Reliably clearing 25 in the July wave needs the optimistic case. Plan for
**a small autumn (Sep–Oct interim) top-up** to safely clear 25, or accept a re-run after the September
records.

### 4b. Higher-leverage lever — fill the 10-month history hole (recommended, stacks with expansion)

The dividend calendar **ends 2025-07-20**: the entire **2025-08 → 2026-06** window is absent (only the
7 July-2026 feed rows exist past that date). Those records have **already occurred** — backfilling them
is pure past-data completion (no lookahead) and yields **CLOSED forward events immediately**, without
waiting for July and **independent of universe size**. Rough estimate for the 16 current names over
~10 months: **~8–14 events**. This single backfill could clear n ≥ 25 on its own. **The two levers
stack** — recommend backend/llm prioritize the backfill in parallel with the universe expansion.

## 5. Data dependencies (coordinate — NOT done in the ML block)

**backend chat** (`scripts/download_candles.py`, ISS dividend history):
- Download **1H candles from 2020-01** for: **SBERP SNGSP TATNP SIBN PHOR RTKMP MOEX BSPB**.
- Pull **ISS dividend history** for the same 8 lines (pref payouts differ from ordinary — fetch the
  pref series explicitly).
- **Backfill `dividends.csv` for ALL universe lines from 2025-08 to present** (close the 10-month gap §4b).
- Run the **ADTV ≥ 300 M₽** screen (criterion 1) and report; drop RTKMP/BSPB if they fail.

**llm chat** (`llm/scripts/refresh_dividend_feed.py`, e-disclosure by INN):
- Extend `data/news/dividend_calendar_upcoming.csv` to the new lines' **2026 recommended records**,
  esp. July (**SBERP SNGSP TATNP SIBN**; + any autumn interims), keyed by issuer INN.
- Keep the independent no-lookahead verify + anchor sverka in the refresh path.

## 6. ML steps — queued, DATA-GATED (run once §5 lands; nothing tuned until then)

1. **Expanded IS study** — `h9_dividend_research.py` on the 24-line universe, **seed-aggregated, no
   cherry-pick**. The edge must **persist**; if the new lines carry no run-up, report it honestly and
   keep only the lines that do (decided by the control, not by P&L).
2. **Placebo / time-shuffle control** (invariant #8 analog for a calendar event study): the existing
   `placebo_test` (run-up on random non-dividend dates) must show the expanded run-up **vanishing on
   shuffled/random dates** — else revert the expansion.
3. **Re-run the realized-P&L shadow gate** (`h9_shadow_pnl.py`) and report the actual July-2026 n.
4. **Update the serving universe** — promote a single source of truth (the `UNIVERSE` constant
   `h9_dividend_research.py` imports into the gate, plus the panel the sleeve is fed) to the validated
   list. Keep contracts + `ml/test_smoke.py` green. **`is_production` stays false** until the gate is
   MET on accrued forward events + sign-off.

## Acceptance status (this commit)

- [x] A-priori criteria + expanded list **fixed** (this doc, before any expanded backtest).
- [x] Per-name include/exclude justification.
- [x] July-2026 forward-n forecast + autumn/backfill verdict.
- [x] Data dependencies written for backend + llm.
- [ ] IS-edge re-confirmed + placebo/time-shuffle ok — **DATA-GATED** (needs §5).
- [ ] Serving universe updated — **DATA-GATED** (after IS-confirmation).
