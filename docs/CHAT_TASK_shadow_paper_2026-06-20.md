# Задачи: execution + agent — допилить shadow-гейт до зелёного (2026-06-20)

> Выдано лид-чатом после интеграционной проверки. risk_manager закоммитил shadow-гейт (`6b47279`):
> слив без `is_production=true` + пройденного forward-гейта → **0 живого капитала**, позиции уходят в
> `shadow_positions`. Проверено e2e: оркестратор в paper-цикле ставит **0 LIVE-ордеров** по H9 (было 11).
> Это корректный фикс (инварианты #9/#4). Но он вскрыл шов — починить, чтобы дерево было зелёным.

## Почему сломалось (контекст)
Канонический `contracts/examples/risk_book.example.json` теперь `is_production=false` → `net_positions: []`
(всё в `shadow_positions`). Это ВЫНУЖДЕННО правильно: валидатор запрещает `is_production=true` в примерах,
а non-production слив обязан быть shadow. Значит «живого» примера для реконсиляции в принципе быть не может —
**в paper/dry-run надо реконсилить SHADOW-книгу.** 2 execution-теста падают, потому что реконсилили старый
пример ради 5 живых ордеров.

---

## EXECUTION (приоритет 🔴 — блокирует зелёное дерево)
**Решение лида (форсировано контрактом):** режимы execution трактуют книгу так:
- **dry-run / paper:** эффективная **paper-книга = `net_positions` ∪ `shadow_positions`** (+ слить
  `hedge` и `shadow_hedge`). Paper-режим ПАПЕР-ТОРГУЕТ shadow-книгу → форвард-shadow трек копится через
  реальный execution-путь (это и есть смысл paper-трека: точная симуляция того, что делал бы live).
- **live:** реконсилить ТОЛЬКО `net_positions` (+ live `hedge`) → по shadow-сливам 0 ордеров (прод-безопасно).

**Задачи:**
1. `reconcile()` (execution/src/reconcile.py): в non-live режимах строить целевую книгу из
   `net_positions` + `shadow_positions`, хедж из `hedge` + `shadow_hedge`; в live — только `net_positions`+`hedge`.
2. Починить 2 падающих теста (`test_reconcile.py::test_real_example_risk_book_reconciles`,
   `test_contract_conformance.py::test_generated_orders_conform_to_order_request`): они гоняют DRY_RUN →
   теперь реконсилят shadow-книгу канонического примера → снова 5 ордеров (3 BUY + 2 SELL).
3. **Добавить тест прод-безопасности:** LIVE-режим на том же (all-shadow) примере → **0 live-ордеров**.
4. `serve`-конверт не менять (envelope `{risk_book, positions, prices, capital, mode, …}` принят как канон —
   он верен, согласован с агентом; обнови только формулировку в README, если нужно).

**Приёмка:** `pytest execution/tests` зелёный (вкл. новый live-0-orders тест); `ml/test_smoke.py` 19/19;
`validate_contracts.py` зелёный. Коммить ТОЛЬКО `execution/…`. После твоего коммита лид перепроверит
e2e: paper-цикл папер-торгует H9 shadow-книгу, live-цикл — 0 ордеров.

---

## AGENT (приоритет 🟠 — замкнуть forward-P&L гейт + атрибуцию)
1. **Включить forward-P&L гейт в живом цикле (1 строка):** `LiveCombiner` должен передавать
   `store.pnl_by_sleeve()` в `combine(..., sleeve_status=...)` (risk_manager заморозил сигнатуру и отдал
   seam). Без этого гейт работает только по `is_production`, но не отзывает production-слив при
   отрицательном forward-P&L. Сейчас по умолчанию H9 и так shadow (is_production=false) — но seam надо замкнуть.
2. **Атрибуция shadow-филлов:** когда execution начнёт папер-торговать shadow-книгу (см. задачу execution),
   убедись, что эти paper-филлы идут в **shadow-трек / `shadow_pnl.jsonl`**, а НЕ считаются как live-P&L по
   сливу. Per-sleeve P&L-атрибуция должна различать live vs shadow капитал.
3. **Относительный путь интерпретатора** в `_comment` примере `sleeve.command` — заменить на абсолютный /
   `sys.executable` (относительный exe падает на Windows; на Linux-VDS ок, но пример должен быть кросс-платформенным).

**Приёмка:** `pytest agent/tests` зелёный; forward-P&L gate замкнут (тест: production-слив с отрицательным
forward → shadow); shadow paper-P&L отделён от live. Коммить ТОЛЬКО `agent/…`.

---

## Дисциплина
Каждый чат коммитит ТОЛЬКО свои файлы. `is_production=false` сквозь артефакты; live за двойным флагом.
Лид перепроверит интеграцию (все тесты + оркестратор e2e) перед тем, как считать волну закрытой.
