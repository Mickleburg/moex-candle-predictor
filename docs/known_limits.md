# Known Limits And Verification Status

## Что Проверено Этой Сверкой

Эта сверка подтверждает на уровне code review:

- структуру репозитория;
- текущие entrypoints;
- наличие backend, ML, shared schemas и data directories;
- форму backend<->ML контракта;
- фактический raw candle contract по коду;
- наличие checked-in artifacts;
- наличие исторических decision log записей.

Также отдельно были прочитаны приложенные PDF:

- `backend_integration.pdf`
- `Архитектура гибридного торгового агента  LLM + алгоритмический предиктор.pdf`

## Что Не Проверено Запуском

Эта сверка не подтверждает:

- свежий успешный backend build;
- свежий успешный запуск ML inference service;
- свежий успешный training run;
- свежий успешный end-to-end backend -> ML path;
- совместимость checked-in artifacts с текущим кодом после последних правок pipeline.

Поэтому documentation intentionally избегает формулировок вроде:

- production-ready;
- end-to-end verified;
- artifacts are known-good;
- model is confirmed up to date.

## Environment Reproducibility Limits

- В репозитории нет зафиксированного воспроизводимого dev environment для всех частей системы.
- `ml/.venv` закоммичен в репозиторий, но не является надежным переносимым окружением.
- Содержимое checked-in `.venv` указывает на platform-specific packages и не должно использоваться как source of truth для clean setup.
- Root `.env.example` сейчас пуст и не документирует runtime requirements.

## Checked-In `.venv` And Artifacts Caveats

### `.venv`

`ml/.venv` не следует считать:

- актуальным для текущей машины;
- переносимым между ОС;
- достаточным доказательством, что ML service можно поднять без пересоздания окружения.

### Artifacts

Наличие этих файлов:

- `ml/artifacts/model.pkl`
- `ml/artifacts/tokenizer.pkl`
- `ml/artifacts/metadata.json`

не гарантирует:

- train/serve consistency;
- актуальность по отношению к текущему feature pipeline;
- отсутствие shape mismatch;
- корректность stored metrics после последующих изменений кода.

## Leakage Fix Vs Not-Yet-Retrained Artifacts

По контексту проекта leakage-related правки в pipeline уже вносились.

Но documentation должна явно сохранять границу:

- исправления в коде не равны автоматически переобученным artifacts;
- checked-in artifacts нельзя считать подтвержденно переобученными после этих исправлений;
- без отдельного retrain и verification нельзя утверждать, что модель синхронна с текущим кодом.

## Contract / Docs Drift Risks

Есть несколько мест, где drift risk повышен:

- `shared/schemas/` покрывают только `/predict`, а не raw contract и не `/health`;
- backend PDF использует camelCase examples, а текущий код работает в snake_case;
- часть config-полей существует шире, чем реально используется кодом;
- в кодовой базе есть hybrid-oriented pieces, которые легко принять за current architecture, хотя их нужно трактовать как experimental/planned.

## Historical Evidence Of Train/Serve Drift Risk

`data/reports/decision_log.jsonl` содержит исторические записи, в которых виден runtime error вида:

- `X has 800 features, but LGBMClassifier is expecting 832 features as input`

Это не доказывает текущее состояние сервиса, но честно показывает, что в проекте уже был зафиксирован риск несовместимости между inference preprocessing и ожиданиями модели.

## Data Availability Limits

- `data/raw/` сейчас пуст, кроме `.gitkeep`.
- Репозиторий не содержит зафиксированного reproducible training dataset.
- Из-за этого training path можно документировать точно по коду, но нельзя подтверждать его повторное выполнение на checked-in data.

## Future Architecture Boundary

LLM/hybrid architecture из отдельного PDF относится к planned/future scope.

Даже если в backend уже есть decision/LLM/risk code:

- это не делает hybrid architecture текущей основной runtime-архитектурой;
- документация обязана явно отделять implemented scope от future extension.
