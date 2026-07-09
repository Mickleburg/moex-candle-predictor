# Risk Manager Block — V3 sleeve combiner + portfolio risk layer

`risk_manager/` строит из **слабо-коррелированных стратегийных сливов** одну книгу и накладывает общий
РИСК-СЛОЙ. Это комбинатор портфеля V3 (`docs/ARCHITECTURE_V3.md`), не пер-тикерный допуск.

## Конвейер (один шаг решения)

```
sleeve inputs ─┐
               ├─ normalize     (sleeve_signal позиции / aggregated_signal ранги → directional веса)
               ├─ SHADOW GATE   слив live ТОЛЬКО если is_production И forward-гейт MET (инв. #9/#4);
               │                иначе → shadow-книга, 0 живого капитала
               ├─ NET           (один тикер из разных live-сливов → одна нетто-позиция, атрибуция)
               │                + кап на сливовый гросс (кап корреляции по sleeve-id)
               ├─ LIMITS        имя ≤ cap · сектор-гросс ≤ cap (= кап корреляции) · гросс ≤ cap
               ├─ RISK          vol-targeting (H4 пер-тикер vol) × режимный гейт (H5 exposure_scalar)
               ├─ HEDGE         по СЕКТОРУ (предпочт. для H9 run-up) / по рынку IMOEX / нет
               └─ EMIT          risk_book {net_positions=LIVE, shadow_positions=paper, gating}
                                + пер-имя risk_decision[] (только LIVE) для execution
```

Пустой ввод/полный режимный срез → пустая книга (не падает). `is_production` книги = true только
если есть LIVE-слив и все live-сливы production; иначе false.

## Shadow-гейт (инварианты #9 + #4) — КОРРЕКТНОСТЬ

Слив получает **живой капитал** только пройдя свой гейт:
1. блок подписал слив (`is_production=true`), И
2. forward shadow-гейт **MET** — нет отрицательной forward-P&L атрибуции.

Иначе слив идёт в книгу как **shadow** (`shadow_positions`, **0 живого капитала**) — тречится для
атрибуции, но не рискуется. `net_positions` (живой капитал, execution сайзит в лоты) содержит ТОЛЬКО
live-сливы; `to_risk_decisions` рендерит только их → execution не ставит реальных ордеров за
неподтверждённый эдж.

**H9 сейчас:** `is_production=false`, shadow-гейт **NOT MET** (`ml/scripts/h9_shadow_pnl.py`: forward
n=12, net −0.93%, dose-инвертирована) → H9 в книге **shadow-only, directional_gross=0**. Полный
риск-слой (vol-target × режим × лимиты × сектор-хедж) применяется к shadow-книге → faithful paper-трек.

forward-P&L статус подаётся опциональным `sleeve_status={sleeve: {gate|forward_pnl}}` (из state-store
agent) — даже у production-слива отрицательный forward возвращает его в shadow. По умолчанию гейт = по
`is_production`.

## Входы

| Вход | Контракт | Источник |
|------|----------|----------|
| Слив-позиции (S3 H9) | `sleeve_signal` (новый) | `ml/src/service/dividend_sleeve.py::build_sleeve_signal` |
| Слив-ранги (S1/S2, форма) | `aggregated_signal` | решающая модель (закрыты, но форма поддержана) |
| Риск-аналитика H4/H5 | `risk_analytics` | `ml/src/service/risk_analytics.py` |

## Выход

- **`risk_book`** (новый контракт) — `net_positions` (LIVE, живой капитал) + `shadow_positions`
  (paper, 0 live) + `gating` (live/shadow вердикт по сливу) + применённые скаляры (vol/exposure/gross,
  + `shadow_gross`) + аудит лимитов + хедж-ноги. Портфельный аналог `risk_decision`.
- **`risk_decision[]`** — пер-инструмент, **только LIVE-книга**, валидно против
  `contracts/risk_decision.schema.json` (`order_intent=null`: лоты считает execution по капиталу/цене).

## Замороженный entry point (оркестратор зовёт in-process)

```python
risk_manager.src.combine(
    sleeve_signals: list[dict],          # sleeve_signal и/или aggregated_signal
    risk_analytics: dict | None = None,  # H4 vol + H5 режим (ML)
    config: CombinerConfig | None = None,
    as_of: str | None = None,
    *, sleeve_status: dict | None = None,  # {sleeve: {gate|forward_pnl}} из P&L-атрибуции agent
) -> RiskBook                            # .to_dict() валидно против risk_book.schema.json
```

`agent/src/adapters/live.py::LiveCombiner` зовёт это напрямую — shadow-гейт включён по умолчанию, без
изменений в agent. Опц. процессный шов (как у ML): `scripts/predict_risk_book.py` (чистый stdlib —
risk_analytics приходит JSON-файлом от ML-CLI).

## Решения дизайна

- **Хедж по СЕКТОРУ по умолчанию.** P0-анализ H9: сектор-хедж Sharpe +0.92 / DD −0.105 vs рыночный
  beta=1 IMOEX +0.54 / −0.173 — run-up это эффект бумага-vs-сектор. Хедж строится на уровне книги;
  собственная IMOEX-нога слива игнорируется (комбинатор сам выбирает хедж). Для S4 core: `hedge_mode="none"`.
- **Кап корреляции = кап на сектор-гросс.** Без полной ковариации коррелированные имена кластеризуются
  по сектору; кап на сектор-гросс ограничивает коррелированную экспозицию.
- **Лимиты — последний шаг.** vol-targeting может масштабировать ВВЕРХ и пере-нарушить кап → имя→сектор→гросс
  накладываются к фикспоинту в конце (каждый кап только ужимает → сходится).
- **Чистый Python** (без numpy/pandas) — лёгкий портфельный слой; весь I/O — dict против схем `contracts/`.

## Режимный гейт доказан (реальный 2022)

`exposure_scalar` (H5) реально срезает гросс при `novel=true`. На живых данных
(`demo_combine_h9.py --as-of 2022-04-15`): novel=True, exposure_scalar≈0.26 → directional gross
0.118 → 0.030. Пик шока фев–мар 2022: exposure_scalar≈0.01 (почти полный срез). Юнит-тест
`test_regime_gate_cuts_gross_when_novel` фиксирует механику детерминированно.

## Файлы

```
src/combiner.py   — combine(), to_risk_decisions(), shadow_gate_status(), CombinerConfig, RiskBook
src/sleeves.py    — normalize(): sleeve_signal / aggregated_signal → directional веса
src/sectors.py    — SECTOR_MAP (зеркало ml, не импорт) + хедж-инструменты
scripts/predict_risk_book.py — CLI-шов (чистый stdlib): sleeve_signal[+risk_analytics] JSON → risk_book JSON
scripts/demo_combine_h9.py   — живой хендшейк с ML (build_sleeve_signal + risk_analytics), иначе canned
tests/test_combiner.py       — нетинг, shadow-гейт, режимный гейт, лимиты, сливовый кап, хедж, рендер, живой 2022
```

## Запуск

```powershell
$PY = "ml\.venv-win\Scripts\python.exe"
& $PY risk_manager\scripts\demo_combine_h9.py                 # живой хендшейк (H9 → shadow, 0 live)
& $PY risk_manager\scripts\demo_combine_h9.py --canned        # только примеры контрактов
& $PY risk_manager\scripts\demo_combine_h9.py --force-live    # снять гейт, показать live-книгу
& $PY risk_manager\scripts\predict_risk_book.py --sleeves contracts\examples\sleeve_signal.example.json `
      --risk-analytics contracts\examples\risk_analytics.example.json --out -   # CLI-шов
& $PY -m pytest risk_manager\tests\test_combiner.py -q
& $PY scripts\validate_contracts.py                           # включая sleeve_signal + risk_book
```

## Статус

Комбинатор + риск-слой + **shadow-гейт** реализованы и протестированы. `is_production=false` сквозь все
артефакты до forward-гейта + sign-off. **H9 = shadow-only (0 живого капитала)** пока гейт NOT MET. Хедж
по сектору (P0), режимный гейт (H5), лимиты и кап корреляции по сливу применяются и проверены.
