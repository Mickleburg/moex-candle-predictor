# SBER H1 NLP Candle-Language Research, 2026-05-04

## Цель

Перенести ML-логику ближе к идее из приложенных статей:

- текущая свеча кодируется как «слово» через кластеризацию нормализованной формы свечи;
- последовательность последних свечей становится «предложением»;
- предложение векторизуется NLP-методом;
- классификатор предсказывает действие `SELL` / `HOLD` / `BUY` для будущего горизонта.

Важное отличие от предыдущего бейзлайна: токен больше не является квантилем будущей доходности. Токен теперь описывает только форму уже известной свечи, а будущая доходность используется только как supervised target.

## Реализованный пайплайн

Добавлен отдельный модуль `ml/src/nlp/`, чтобы не ломать существующий inference-контракт:

- `candles.py`: нормализация свечей относительно `open`, derived shape-признаки, разметка `SELL/HOLD/BUY` по будущему return и комиссии.
- `clustering.py`: `KMeans`, `MiniBatchKMeans`, `AgglomerativeClustering`, `GaussianMixture`, `DBSCAN`, `HDBSCAN`.
- `vectorizers.py`: `TF-IDF`, `TF-IDF + TruncatedSVD`, `PPMI cooccurrence + SVD` с pooling `mean/std/last` как детерминированная Word2Vec-подобная альтернатива.
- `classifiers.py`: `RidgeClassifier`, `LinearSVC`, `LightGBM`, `ExtraTrees`; в полной сетке также доступны `LogisticRegression` и `MLP`.
- `pipeline.py`: единый train/validation/test прогон без leakage.
- `scripts/sber_nlp_research.py`: перебор конфигураций, запись JSON/CSV результатов.

Конфигурация исследования описана в `ml/configs/nlp_research.yaml`.

## Данные

- Инструмент: `SBER`
- Таймфрейм: `1H`
- Файл: `data/raw/SBER_1H_20200103T0900_20260503T1800.parquet`
- Свечей после очистки: `24613`
- Split по времени: train `[0, 17229)`, validation `[17229, 20921)`, test `[20921, 24613)`
- Комиссия: `0.05%` на сторону, порог действия: `2 * commission = 0.10%`
- Основные горизонты исследования: `h=1`, `h=3`
- Окна: `16`, `32`

Границы split строгие: и контекстное окно, и future label остаются внутри своего split. Кластеры и векторизаторы fit-ятся только на train; выбор параметров идет по validation macro-F1; test не используется для подбора.

## Запуск

```powershell
python ml\scripts\sber_nlp_research.py --quick --limit 72 --output-json data/reports/sber_h1_nlp_research_20260504.json --output-csv data/reports/sber_h1_nlp_research_20260504.csv
```

`--limit` берет равномерную детерминированную выборку по всей сетке, а не первые N конфигураций. Полный перебор доступен без `--limit`; расширенная сетка без `--quick` добавляет `h=6`, окно `8`, дополнительные `GMM/HDBSCAN`, `LogisticRegression` и `MLP`.

## Лучшие результаты

Отбор ниже выполнен честно по validation macro-F1.

| Rank | Shape | Horizon | Window | Clustering | Vectorizer | Classifier | Val macro-F1 | Test macro-F1 | Test acc. |
| ---: | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | shape | 1 | 32 | GMM, 20 diag components | cooccurrence-SVD | Ridge | 0.4067 | 0.3695 | 0.5557 |
| 2 | ohlc | 1 | 16 | GMM, 20 diag components | TF-IDF | LinearSVC | 0.4003 | 0.3723 | 0.5593 |
| 3 | ohlc | 1 | 16 | KMeans, k=32, k-means++ | TF-IDF | Ridge | 0.3983 | 0.3667 | 0.5582 |
| 4 | shape | 1 | 32 | MiniBatchKMeans, k=32 | cooccurrence-SVD | Ridge | 0.3963 | 0.3483 | 0.5309 |
| 5 | shape | 1 | 16 | KMeans, k=20, k-means++ | TF-IDF-SVD | LightGBM | 0.3932 | 0.3668 | 0.5433 |

Лучшая test macro-F1 в прогоне была у второй строки (`0.3723`), но она не выбрана как финальная, потому что критерий подбора не должен смотреть на test.

## Агрегация

Средние значения по группам:

| Группа | Лучшее наблюдение |
| --- | --- |
| Кластеризация | `GMM` в среднем лучше по validation macro-F1 (`0.3789`), затем `MiniBatchKMeans` (`0.3749`) и `KMeans` (`0.3647`). |
| Плотностные методы | `DBSCAN` и `HDBSCAN` заметно хуже: у HDBSCAN средний noise ratio около `0.75`, validation macro-F1 около `0.184`. |
| Векторизация | В среднем `TF-IDF` стабильнее (`0.3398` val macro-F1), но лучший одиночный результат дал `cooccurrence-SVD`. |
| Классификатор | В среднем лидировал `LightGBM` (`0.3516` val macro-F1), но лучший одиночный результат дал `Ridge` на cooccurrence-SVD. |
| Горизонт | `h=1` лучше `h=3`: `0.3405` против `0.3110` по среднему validation macro-F1. |
| Shape-признаки | Derived `shape` немного лучше `ohlc` по validation macro-F1 (`0.3331` против `0.3183`). |

## Финальная конфигурация

```yaml
shape_variant: shape
horizon: 1
window_size: 32
commission: 0.0005
cluster:
  name: gmm
  params:
    n_components: 20
    covariance_type: diag
    reg_covar: 1e-6
vectorizer:
  name: cooccurrence_svd
  params:
    embedding_dim: 24
    context_window: 2
    pool: mean+std+last
    include_histogram: true
classifier:
  name: ridge
  params:
    alpha: 1.0
```

Для этой конфигурации:

- validation: accuracy `0.4385`, macro-F1 `0.4067`;
- test: accuracy `0.5557`, macro-F1 `0.3695`;
- test distribution: `SELL=831`, `HOLD=2030`, `BUY=799`;
- кластеризация: `20` слов, silhouette `0.1416`, Davies-Bouldin `1.4437`;
- простой long/short/flat backtest после комиссии отрицательный: test total return `-0.7854`, trade rate `0.1902`.

## Выводы

Идея из статей реализована и проверена: свеча превращается в слово, последние свечи - в NLP-предложение, а классификация отделена от кластеризации формы. На SBER H1 лучшая честно выбранная конфигурация дает test macro-F1 около `0.37`; это полезный исследовательский сигнал, но не готовая торговая стратегия.

По качеству классификации новый NLP-пайплайн пока не превзошел предыдущий табличный бейзлайн на этом же инструменте, но он логически чище для проверки гипотезы candle-language и лучше расширяется: можно добавлять walk-forward validation, multi-stock корпус, online дообучение словаря, Transformer/TCN поверх word IDs и отдельный risk layer.

## Исправления по ходу работы

- Убрана практическая ошибка `ml/test_smoke.py`: Unicode-галочки ломали запуск в Windows cp1251-консоли.
- Smoke test теперь проверяет `src.nlp` и загружает конфиги из `ml/configs`.
- `data/reports/*.csv` добавлен в `.gitignore`, так как CSV/JSON результатов являются воспроизводимыми локальными research-артефактами.
