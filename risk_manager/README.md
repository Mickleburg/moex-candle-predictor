# Risk Manager Block — V3 sleeve combiner + portfolio risk layer

`risk_manager/` строит из **слабо-коррелированных стратегийных сливов** одну книгу и накладывает общий
РИСК-СЛОЙ. Это комбинатор портфеля V3 (`docs/ARCHITECTURE_V3.md`), не пер-тикерный допуск.

## Конвейер (один шаг решения)

```
sleeve inputs ─┐
               ├─ normalize  (sleeve_signal позиции / aggregated_signal ранги → directional веса)
               ├─ NET        (один тикер из разных сливов → одна нетто-позиция, атрибуция по сливам)
               ├─ LIMITS     имя ≤ cap · сектор-гросс ≤ cap (= кап корреляции) · гросс ≤ cap
               ├─ RISK       vol-targeting (H4 пер-тикер vol) × режимный гейт (H5 exposure_scalar)
               ├─ HEDGE      по СЕКТОРУ (предпочт. для H9 run-up) / по рынку IMOEX / нет
               └─ EMIT       risk_book + пер-имя risk_decision[] для execution
```

Пустой ввод/полный режимный срез → пустая книга (не падает). `is_production` книги = false, пока
**каждый** слив не production.

## Входы

| Вход | Контракт | Источник |
|------|----------|----------|
| Слив-позиции (S3 H9) | `sleeve_signal` (новый) | `ml/src/service/dividend_sleeve.py::build_sleeve_signal` |
| Слив-ранги (S1/S2, форма) | `aggregated_signal` | решающая модель (закрыты, но форма поддержана) |
| Риск-аналитика H4/H5 | `risk_analytics` | `ml/src/service/risk_analytics.py` |

## Выход

- **`risk_book`** (новый контракт) — нетто-книга + применённые скаляры (vol/exposure/gross) + аудит
  лимитов + хедж-ноги. Портфельный аналог `risk_decision`.
- **`risk_decision[]`** — пер-инструмент, валидно против `contracts/risk_decision.schema.json`
  (`order_intent=null`: лоты считает execution по капиталу/цене).

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
src/combiner.py   — combine(), to_risk_decisions(), CombinerConfig, RiskBook (риск-слой + эмиссия)
src/sleeves.py    — normalize(): sleeve_signal / aggregated_signal → directional веса
src/sectors.py    — SECTOR_MAP (зеркало ml, не импорт) + хедж-инструменты
scripts/demo_combine_h9.py — живой хендшейк с ML (build_sleeve_signal + risk_analytics), иначе canned
tests/test_combiner.py     — нетинг, режимный гейт, лимиты, хедж, рендер контрактов, живой 2022
```

## Запуск

```powershell
$PY = "ml\.venv-win\Scripts\python.exe"
& $PY risk_manager\scripts\demo_combine_h9.py                 # живой хендшейк (или canned fallback)
& $PY risk_manager\scripts\demo_combine_h9.py --canned        # только примеры контрактов
& $PY -m pytest risk_manager\tests\test_combiner.py -q
& $PY scripts\validate_contracts.py                           # включая sleeve_signal + risk_book
```

## Статус

Комбинатор + риск-слой реализованы и протестированы. `is_production=false` сквозь все артефакты до
forward-гейта + sign-off. Хедж по сектору (P0), режимный гейт (H5) и лимиты применяются и проверены.
