# Training And Experiments

Документ описывает, что реально поддерживает текущий training code.

## Запуск

```bash
cd ml
python -m src.models.train --config-dir configs
```

## Что требуется

- raw candles в `data/raw/`;
- формат данных совместим с loader и cleaner;
- установлен Python runtime с пакетами из `ml/requirements.txt`.

## Какие конфиги реально используются

- `configs/data.yaml`
- `configs/features.yaml`
- `configs/train.yaml`
- `configs/eval.yaml`

Важно:

- не все поля из конфигов влияют на поведение кода;
- часть параметров в `features.yaml` и `eval.yaml` сейчас документальная или резервная, а не исполняемая.

## Поддерживаемые модели

- `majority`
- `markov`
- `logistic`
- `lgbm`

`rnn` не интегрирован в pipeline и не должен указываться как рабочий вариант.

## Выходы pipeline

После успешного run pipeline пишет:

- `ml/artifacts/model.pkl`
- `ml/artifacts/tokenizer.pkl`
- `ml/artifacts/metadata.json`

Код сейчас не пишет полноценный backtest report автоматически, несмотря на наличие helper-функций в `ml/src/evaluation/`.

## Что не стоит предполагать

- наличие `metadata.json` не доказывает корректность train/serve consistency;
- сохранённые метрики не эквивалентны production validation;
- examples в старой документации с “ready for production” и конкретными результатами нельзя считать подтверждёнными.

## Рекомендуемый минимальный экспериментальный цикл

1. Положить reproducible raw dataset в `data/raw/`.
2. Запустить training pipeline.
3. Проверить, что артефакты обновились.
4. Поднять `uvicorn src.service.api:app`.
5. Проверить `/health` и `/predict`.
6. Сравнить структуру признаков и поведение inference с metadata и текущим training code.
