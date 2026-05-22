# MOEX Candle Predictor

`moex-candle-predictor` оформлен как модульная research-платформа для будущего торгового агента на MOEX. Сейчас рабочим и проверенным блоком остается ML research; остальные блоки являются архитектурным scaffold без live trading и без production-интеграций.

## Текущий статус

- `ml/` - активный Python research-блок для свечей MOEX, candle-language экспериментов и action classification.
- `backend/` - существующий Go backend для загрузки/хранения свечей и HTTP-интеграции с ML service.
- `contracts/` - JSON Schema контрактов между будущими блоками агента.
- `llm/`, `aggregator/`, `risk_manager/`, `execution/`, `agent/` - scaffold будущей агентной архитектуры.

Лучший validation-only research candidate на момент создания ветки:

```text
target:       triple_barrier:h3:w12:up1.25:down1.25
features:     continuous_regime
model:        extra_trees:depth=none:leaf=20:maxfeat=sqrt
class_weight: none
mean macro-F1: 0.4695
```

Это не production artifact. Test не используется для подбора новых candidates. Перед production-независимыми выводами нужны seed robustness, frozen evaluation protocol, backtest и paper trading.

## Блоки

```text
backend/data -> ml prediction
backend/data -> llm technical analysis
ml + llm -> aggregator
aggregator + portfolio -> risk manager
risk manager -> execution
execution -> agent logs
```

- `backend/` - исторические и будущие свежие свечи, raw storage, batch validation.
- `ml/` - ML forecast block: текущий рабочий research/inference слой.
- `llm/` - будущий LLM technical-analysis block; не имеет права торговать.
- `aggregator/` - будущий late-fusion слой для ML и LLM сигналов.
- `risk_manager/` - будущий слой ограничений, лимитов и запрета рискованных действий.
- `execution/` - будущий dry-run/paper/live adapter к broker/MOEX API.
- `agent/` - будущий orchestrator полного цикла.
- `contracts/` - shared JSON contracts.
- `config/` - shared конфигурация, включая demo-supported тикеры.

## Safety

- Реальная торговля по умолчанию запрещена.
- Первый режим исполнения - dry-run/paper.
- Risk manager сильнее ML, LLM и aggregator: он может заблокировать любой сигнал.
- LLM не отправляет orders и не принимает финальное торговое решение.
- Test split нельзя использовать для research tuning.

## Текущие ML проверки

```powershell
python -m compileall -q ml\src ml\scripts ml\test_smoke.py
python ml\test_smoke.py
```

ML-блок также имеет contract-compatible CLI:

```powershell
python ml\scripts\predict_from_json.py `
  --input-json contracts\examples\candle_batch.example.json `
  --output-json data\reports\ml_prediction_example.json
```

Без artifact CLI возвращает `diagnostics.artifact_missing=true` и не имитирует реальный прогноз. На ветке `ml-expirement` добавлен research-only artifact path: локально можно обучить frozen triple-barrier artifact и передать его через `--artifact-dir`, чтобы получить реальные `predict_proba` probabilities по `ml_prediction` contract. Это все еще не production/live trading artifact.

## Проверка contracts и backend

```powershell
python scripts\validate_contracts.py
```

Если менялся Go backend:

```powershell
Set-Location backend
go test ./...
Set-Location ..
```

## Документация

- ML research reports: `ml/docs/research/`
- ML block overview: `ml/README.md`
- Architecture contracts: `contracts/README.md`
- Existing backend: `backend/README.md`

Корневой `docs/` оставлен для общей технической документации проекта, которая не является SBER H1 research report.
