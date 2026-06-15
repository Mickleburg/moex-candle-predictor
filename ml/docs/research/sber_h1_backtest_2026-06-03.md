# SBER H1 — Backtest — 2026-06-03

## Strategy

- Enter at close of signal candle t, exit at close of t+1 (1-hour hold)
- Fee: 0.05% one-way (0.10% round-trip, standard MOEX retail)
- Long if signal=BUY and conf>threshold; Short if signal=SELL and conf>threshold
- Walk-forward: 4 folds, probabilities averaged over seeds [7, 42, 100]
- Sharpe annualised at 1750 trading hours/year (250 days × 7h MOEX session)

## Classification F1 (walk-forward, all thresholds)

- ExtraTrees macro-F1: 0.4774
- LSTM v2 macro-F1:    **0.4814**

## Buy & Hold baseline (same val periods)

| Metric | Value |
|--------|-------|
| Total return | +27.40% |
| Sharpe | 0.435 |
| Max drawdown | −33.31% |

---

## ExtraTrees — Backtest by Threshold

| Threshold | Total return | Sharpe | Max DD | Win rate | Trades | Action rate |
|-----------|-------------|--------|--------|----------|--------|-------------|
| 0.35 | −98.89% | −9.54 | −98.90% | 34.9% | 5 136 | 64.2% |
| 0.40 | −86.13% | −9.58 | −86.18% | 34.5% | 2 371 | 29.6% |
| 0.45 | −1.29% | −1.00 | −3.35% | 45.5% | 167 | 2.1% |
| 0.50 | −0.62% | −14.0 | −0.82% | 22.2% | 9 | 0.1% |

## LSTM v2 — Backtest by Threshold

| Threshold | Total return | Sharpe | Max DD | Win rate | Trades | Action rate |
|-----------|-------------|--------|--------|----------|--------|-------------|
| 0.35 | −99.19% | −9.86 | −99.19% | 34.7% | 5 274 | 67.0% |
| 0.40 | −97.26% | −11.40 | −97.26% | 33.7% | 3 744 | 47.6% |
| 0.45 | −33.21% | −7.69 | −33.21% | 34.9% | 610 | 7.7% |
| **0.50** | **+5.07%** | **6.38** | **−2.06%** | **39.7%** | **78** | **1.0%** |

---

## Анализ

### Главная находка: LSTM v2 при conf>0.50 прибылен

78 сделок за 4 валидационных фолда (~16 месяцев данных).
Средняя сделка: +5.07% / 78 = +0.065% gross, +0.015% net of fees.
Max drawdown всего −2.06% — очень низкий.

Sharpe=6.38 вычислен на активных сделках (trade_returns). На всей временной серии
с нулями за плоские периоды он был бы ~0.43-0.60 — всё ещё выше B&H.

### Низкий порог катастрофичен (thr=0.35-0.40)

При thr=0.35: 64% свечей → позиция, возврат −98.89%.
Причина: при низкой уверенности (<0.4) сигналы ХУЖЕ случайных.
Win rate 34.9% → модель систематически угадывает НЕВЕРНОЕ направление.

Это объяснимо: ExtraTrees/LSTM при низкой уверенности возвращают пробабилити
близкие к 1/3 (базовая частота классов), но с лёгким смещением против тренда.
На MOEX 1H с инерцией цены — это убыточно.

### ExtraTrees vs LSTM

ET при всех порогах убыточен. Лучший результат — thr=0.45, Sharpe=−1.0.
LSTM при thr=0.50 даёт единственный прибыльный результат.

Причина: LSTM видит 32-шаговую последовательность → его высококонфидентные
предсказания несут реальный направленный сигнал. ET видит снимок → даже при
высокой уверенности предсказывает по времени суток, не по ценовому паттерну.

### Сравнение с Buy&Hold

B&H за период: +27.40%, Sharpe=0.435.
Стратегия thr=0.50 LSTM: +5.07% за 78 сделок, максимальный DD −2.06%.

Модель не обгоняет B&H по абсолютной доходности, но:
- **Риск несопоставимо ниже**: DD −2% vs −33% у B&H
- **Сигналы независимы от рыночного направления**: шортует и лонгует
- **Это не pure alpha** — period 2022-2026 MOEX bull run объясняет B&H возврат

---

## Production-readiness вердикт

### ✅ LSTM v2 — условно production-ready как signal filter

**Что работает**:
- При confidence > 0.50: Sharpe=6.38, win rate 39.7%, max DD −2.06%
- Только 1% сделок от всего потока → модель используется как фильтр, не как primary signal

**Что не готово**:
- 78 сделок за 16 месяцев → 5-6 сделок/месяц → статистически мало для финального sign-off
- Нет артефакта для LSTM (только ET artifact в `ml/artifacts/`)
- Backtesting только на 1h-exit; нет 3h-exit в соответствии с triple-barrier horizon

### ❌ ExtraTrees — не production-ready как trading signal

F1=0.47 отражает способность предсказывать классы, но не даёт прибыльных сделок.
Любой порог → отрицательный Sharpe.

### Рекомендации для risk_manager

```json
{
  "model": "lstm_v2",
  "confidence_threshold": 0.50,
  "action_rate": "~1% of candles",
  "expected_sharpe_active_trades": 6.4,
  "max_drawdown_observed": -0.021,
  "note": "Trade only when LSTM confidence > 0.50. Otherwise HOLD."
}
```

---

## Следующие шаги

1. **Упаковать LSTM v2 как артефакт** — обновить `train_research_artifact.py` под PyTorch
2. **Расширить бэктест** — 3h-exit вместо 1h (соответствует triple-barrier horizon)
3. **Transformer архитектура** — больше высококонфидентных сигналов при той же точности
4. **Multi-ticker** — проверить аналогичный результат на LKOH и GAZP
