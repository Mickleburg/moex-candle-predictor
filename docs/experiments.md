# Experiments And Evaluation

Этот документ описывает только текущий training/evaluation scope, который реально присутствует в кодовой базе `ml/`.

## Что Реально Исполняется

Основной исполняемый training path:

```powershell
Set-Location C:\Users\ancha\Projects\MOEX\moex-candle-predictor\ml
python -m src.models.train --config-dir configs
```

Этот pipeline:

- читает raw candles;
- очищает данные;
- делает time split;
- считает признаки;
- fit-ит tokenizer на train split;
- обучает выбранную модель;
- считает classification metrics для val/test;
- сохраняет artifacts.

## Что Не Исполняется Автоматически

Хотя в коде есть evaluation helpers, основной training entrypoint автоматически не делает:

- полноценный backtest report;
- автоматическую запись online evaluation report;
- walk-forward эксперимент как обязательную часть run;
- автоматический экспорт детальных experiment artifacts в `data/reports/`.

Следующие модули существуют как helper scope, но не wired в стандартный `train.py` flow:

- `ml/src/evaluation/backtest.py`
- `ml/src/evaluation/online_eval.py`

## Supported Model Variants

Код training pipeline поддерживает:

- `majority`
- `markov`
- `logistic`
- `lgbm`

`rnn` присутствует в кодовой базе как future/stub implementation и не должен трактоваться как рабочий automated experiment path.

## Конфиги Экспериментов

Training code загружает:

- `configs/data.yaml`
- `configs/features.yaml`
- `configs/train.yaml`
- `configs/eval.yaml`

Но важно:

- не все поля из этих конфигов гарантированно участвуют в текущем кодовом пути;
- часть параметров остается резервной или документальной;
- наличие поля в config не означает, что оно реально влияет на training result.

## Outputs Training Run

При успешном run training pipeline пишет:

- `ml/artifacts/model.pkl`
- `ml/artifacts/tokenizer.pkl`
- `ml/artifacts/metadata.json`

Сохранение metrics сейчас происходит внутри `metadata.json`:

- `validation_metrics`
- `test_metrics`
- train/val/test periods
- artifact version

## Как Трактовать Stored Metrics And Artifacts

Stored metrics и checked-in artifacts нужно читать осторожно:

- это след конкретного прошлого training run;
- они не подтверждают, что этот run воспроизводим из текущего репозитория;
- они не подтверждают отсутствие train/serve drift;
- они не подтверждают актуальность после последующих изменений pipeline.

Особенно важно:

- leakage-related исправления уже вносились в pipeline;
- checked-in artifacts не следует автоматически считать переобученными после этих изменений;
- historical decision log уже содержит следы feature mismatch между runtime inference и ожиданиями модели.

## Что Полезно Считать Минимальным Honest Experiment Loop

1. Подготовить reproducible raw dataset в `data/raw/`.
2. Запустить `python -m src.models.train --config-dir configs`.
3. Проверить, что artifacts действительно обновились.
4. Поднять inference service.
5. Проверить `GET /health`.
6. Проверить `POST /predict` на данных того же контракта.
7. Сверить feature expectations текущего кода и свежих artifacts.

Без этих шагов stored metrics и stored artifacts остаются только артефактами прошлого run, а не подтверждением текущего рабочего состояния.
