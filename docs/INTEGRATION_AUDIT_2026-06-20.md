# Интеграционный аудит после параллельной волны 5 чатов — 2026-06-20

> Все блоки доставили вторую волну параллельно. Этот аудит проверяет, что параллельная работа НЕ
> сломала интеграцию на текущем HEAD (а не только на момент каждого коммита). Зонтик —
> `docs/VDS_AUTONOMOUS_PLAN.md`. Следующие задачи — `docs/CHAT_TASKS_FOLLOWUP_2026-06-20.md` (обновить).

## История коммитов (порядок волны)
`aa3d62f` ML шаг1 → `a2a5d54` ML шаг5 → `c5e98a2` LLM фид-рефреш → `ab1f523` agent live-блоки →
`3a354bf` backend api+метаданные → `4f32eae` execution serve-CLI. Ветка чиста, even с origin; не-трекнуты
только `data/features/`, `data/news/edisclosure_bodies/` (регенерируемые кэши, ничьи).

## Вердикт: ИНТЕГРАЦИЯ ЦЕЛА ✅
- **206 тестов зелёные на HEAD вместе:** 19 ml-smoke + 29 agent + 40 execution + 33 backend +
  15 risk_manager + 70 llm. Контракты (`validate_contracts.py`) зелёные.
- **Оркестратор гоняет end-to-end с ЖИВЫМИ блоками** (sleeve+combiner+execution live, backend mock без
  сети) на 2026-07-06: ML-слив CLI → risk_manager-комбинатор → execution-engine → 11 дельта-ордеров
  (дивидендные BUY-лонги + сектор-хедж SELL MOEXFN/MM/OG/TL), режимный гейт×vol-target×лимиты применены,
  paper-исполнено, state персистнут, `calendar:"backend"` (единый канон). Полный V3-цикл работает.
- **ML шаг1 цел:** `dividend_sleeve.py` использует `trading_days_between` (не `np.busday_count`).
- **ML шаг2 закрыт:** backend освежил store до 2026-06-19; панель свежая, live inverse-vol больше НЕ
  клампится (`vol_pos_clamped=False`); слив сайзит на актуальных vol (06-25: 3 лонга, 07-06: 7 лонгов).

## Параллельной порчи нет. Мелкие координационные находки (не блокеры):
1. **Относительный путь интерпретатора в примере agent-конфига.** `_comment` для `sleeve.command`
   предлагает `["ml/.venv-win/Scripts/python.exe", …]` — относительный exe-путь. На Windows
   `CreateProcess` НЕ резолвит его от cwd сабпроцесса → `FileNotFoundError`. На Linux-VDS (POSIX execvp,
   путь со слешем резолвится от cwd) — работает. **agent:** в примере дать абсолютный путь / `sys.executable`.
2. **agent обходит замороженный `backend.api`.** agent импортит напрямую `backend.ingest/integrity/store/
   trading_calendar` (коммитнул ДО появления `api.py`). Работает (модули стабильны), но обесценивает
   заморозку. **agent (и опц. ML):** мигрировать потребителей на `backend.api`.
3. **Два шва execution.** agent зовёт execution IN-PROCESS (`ExecutionEngine`), а execution построил ещё
   `serve`-CLI (`4f32eae`). Оба работают; дефолт — in-process. **agent/execution:** выбрать канонический шов.
4. **Дефолт agent-конфига почти весь mock.** По умолчанию backend/sleeve/combiner = mock, только
   execution = live(paper-sim), `llm.refresh_cmd=null`, `sleeve.command=null`. Адаптеры разведены, но
   РЕАЛЬНЫЙ конвейер по умолчанию НЕ гоняется. **agent:** отдать «paper»-профиль конфига (все блоки live,
   абс. пути, `llm.refresh_cmd` задан, backend live), mock оставить дефолтом для тестов.
5. **Устаревшая заметка backend:** «ML-слив всё ещё на np.busday_count» — НЕВЕРНО (`aa3d62f` это починил).
   Действий не требует. (backend предлагает ML брать `backend.api.trading_days_between` — это та же
   функция через ре-экспорт; ML может выровняться на api для гигиены, функционально без разницы.)
6. **risk_manager: путаница git** («мои файлы в a2a5d54») — мисрид; их файлы в `8282224`, дерево чисто.

## Состояние H9 / проекта
Конвейер собран и проверен интегрированно. `is_production=false`. Единственный нерешённый
**корректностный** долг — risk_manager НЕ гейтит слив по shadow-статусу: комбинатор даёт H9 полный риск,
хотя `is_production=false` и shadow-гейт NOT MET. Это инвариант #9/#4 — топ-приоритет (см. следующие задачи).

## ML-сторона H9 — ПОЛНОСТЬЮ ЗАКРЫТА
Шаги 1-5 готовы (календарь, свежий панель, realized-P&L shadow-гейт, сверка якоря, serving-CLI). Дальше
ML-кода для H9 нет; остаётся сезонное накопление shadow-трека (запуск `h9_shadow_pnl.py` по мере закрытия
июльских событий) + paper-сезон + sign-off. Опц. ML-гигиена: импорт календаря через `backend.api`.
