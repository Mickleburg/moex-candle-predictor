# Контекст и задача для НОВОГО ЧАТА (Telegram-бот мониторинга) — блок `bot/`

> Скопируй весь файл в новый чат. Самодостаточен. Новый блок `bot/` — коммить ТОЛЬКО `bot/` (+ свои
> строки в top-level `requirements.txt` и `.env.example`). НЕ трогай ml/llm/risk_manager/agent/execution/backend.

## Проект
MOEX мульти-стратегийный торговый агент, ветка `change-strategy`. Автономный сервис на Linux-VDS. Источник
правды: `docs/ARCHITECTURE_V3.md`, `docs/VDS_AUTONOMOUS_PLAN.md`, корневой `README.md`. Стек — Python.
Торговля по умолчанию paper; `is_production=false`. **Деплой на VDS делаем ПОЗЖЕ** — сейчас строим и тестируем
локально (против засеянной тестовой БД состояния).

## Задача
Интерактивный **Telegram-бот для мониторинга** портфеля/состояния агента: запрос по требованию (позиции,
цены, P&L, статус гейта/режима/данных) + читаемые дайджесты. **Read-only по умолчанию** (наблюдение, не
торговля). Бот НИКОГДА не выставляет сделки.

## Источник данных (НЕ реализовывать заново — читать существующее)
Агент уже ведёт состояние; бот его ЧИТАЕТ:
- **SQLite state-store агента** (`data/agent/state.sqlite`, путь из `agent/config/agent_config.json` →
  `paths.state_db`): позиции (с `capital_state` live|shadow), открытые ордера, `pnl_by_sleeve` (live vs
  shadow), kill-switch, последний/последний успешный цикл. Методы — в `agent/src/state_store.py` (читай
  его публичный интерфейс; открывай БД read-only).
- **Отчёты** (gitignored, регенерируемые): `data/reports/data_integrity_status.json` (OK/HALT данных),
  `data/agent/shadow_pnl.jsonl` (forward-shadow трек), `data/agent/cycles/` (результаты циклов).
- Цены — последние из стора свечей (`backend.store.load_ticker` / последний бар) ИЛИ из state-store, если
  агент их кэширует. Не ходи в сеть сам — читай то, что собрал backend.

## Команды (read-only)
`/status` (сводка: режим, kill-switch, последний цикл, live/shadow гросс) · `/positions` (live + shadow,
по имени, лоты/вес/сектор) · `/pnl` (по сливам, **live отдельно от shadow**) · `/prices [тикеры]` · `/gate`
(shadow-гейт: is_production, MET/NOT_MET, forward-P&L) · `/cycle` (последний EOD-результат: ордера, биндящие
лимиты, алерты) · `/integrity` (HALT/OK + причины) · `/help`.

## Управление (ОСТОРОЖНО — обсуди прежде чем включать)
Kill-switch через бота — control-действие. По умолчанию **НЕ включать** в первой версии (бот read-only). Если
делать — только из whitelisted chat_id, с **двойным подтверждением** (`/killswitch on` → запрос «подтвердите»),
писать в `state_store.set_kill_switch`. Реальные ордера через бота — НИКОГДА.

## Безопасность (обязательно)
- **Whitelist `chat_id`** (только владелец): список разрешённых из `.env` (`BOT_ALLOWED_CHAT_IDS`). Любой
  чужой апдейт — игнор + лог. Без whitelist бот не отвечает никому.
- Токен бота — из `.env` (`TELEGRAM_BOT_TOKEN`), НИКОГДА в git. Дополни `.env.example` (без значений).

## Координация с агентским нотифаером (важно — один токен)
Агент уже умеет слать PUSH-алерты (`agent/src/notifier.py`, Telegram со stdout-фолбэком) — это `sendMessage`,
без polling. Твой бот — **поллер** (`getUpdates`/long-poll) для команд. Telegram разрешает ТОЛЬКО ОДИН
getUpdates-консьюмер на токен: поллер — это твой бот; агент только шлёт. **Конфликта нет, если бот —
единственный, кто поллит.** Если используете webhook вместо polling — тем более ок. Зафиксируй модель
(polling) в README и не запускай второй поллер.

## Технически
- Библиотека: `python-telegram-bot` (async) или `aiogram` — выбери, **запинь версию** в top-level
  `requirements.txt`. Long-lived процесс (на VDS позже — systemd/docker, координируется с infra; СЕЙЧАС —
  локальный запуск + тест).
- Форматирование: компактные сообщения (Markdown/HTML), числа округлены, live/shadow явно разделены.
- Деградация: если state-store/отчёт отсутствует — внятное «нет данных», не падать.

## Приёмка
- Бот стартует с токеном из `.env`, отвечает ТОЛЬКО whitelisted chat_id.
- Против засеянной тестовой `state.sqlite` команды отдают корректные срезы; live и shadow P&L разделены.
- `/integrity` показывает HALT/OK; `/gate` показывает `is_production=false` + NOT_MET (текущее состояние H9).
- Тесты бота зелёные (мок Telegram API / парсинг состояния); `ml/test_smoke.py` 19/19 не сломан.
- Read-only; никаких сделок; токен/whitelist вне git; `is_production=false` сквозь артефакты.

## Дисциплина
Коммить ТОЛЬКО `bot/` (+ свои строки в `requirements.txt`/`.env.example`). Деплой — отдельным шагом позже.
Лид проверит интеграцию (бот против реального state-store) перед закрытием. Отчитайся коммитом.
