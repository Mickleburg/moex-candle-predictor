# SBER H1 — Research Session Summary — 2026-06-03

## Что было сделано

Полная серия из 8 экспериментов для диагностики потолка производительности и улучшения
предсказательной способности ML-блока на данных SBER 1H.

---

## Результаты всех экспериментов

| # | Эксперимент | WF F1 | Δ vs ET | Вывод |
|---|-------------|-------|---------|-------|
| 1 | **Time features ablation** | 0.4097 без них | −0.064 | hour/dow — реальный MOEX-сигнал, 62% importance |
| 2 | **Calibration (isotonic + Platt)** | 0.4503 / 0.4520 | −0.017/−0.015 | ECE=0.045 уже OK, без изменений |
| 3 | **SVD co-occurrence W2V** | 0.4717 | −0.002 | gensim недоступен; SVD≠skip-gram |
| 4 | **Lag features + ExtraTrees** | 0.4711 | −0.003 | ET не умеет учить последовательности |
| 5 | **LSTM v1** (без time-фич) | 0.4591 | −0.015 | fold 4 обрушился без hour/dow |
| 6 | **LSTM v2** (14 фич + time) | **0.4778** | **+0.004** | Лучшая модель; SELL F1 +2.6% |
| 7 | **Target horizon** h=3..12 | 0.4738 (h=3 лучший) | — | h≥6: HOLD исчезает, F1 падает |
| 8 | **Horizon × barrier grid** | 0.4738 (h=3,k=1.25) | — | Baseline-таргет уже оптимален |

---

## Ключевые находки

### 1. Диагностика потолка
Оба метода (ET и LSTM) упираются в 0.47-0.48 walk-forward F1 при любых изменениях
фич, архитектуры или таргета.

**Причина**: 1H OHLCV данные имеют ограниченный предсказательный сигнал для 3h направления.
62% сигнала — это структурная сезонность торговой сессии MOEX (time-фичи),
а не ценовые паттерны.

### 2. LSTM лучше ExtraTrees
- Общий Δ=+0.004 (скромный, но позитивный)
- SELL F1: 0.446 vs ET 0.420 (+2.6%) — LSTM лучше видит медвежьи паттерны в окне 32ч
- Folds 2-3: 0.494-0.495 vs ET 0.488-0.477 — на достаточном количестве данных LSTM заметно сильнее
- Fold 4 (2025-2026): оба падают до ~0.440 — режимный сдвиг, не архитектурная проблема

### 3. Time-фичи обязательны для ВСЕХ моделей
Без hour_sin/cos/dow_sin/cos:
- ET: −0.064 (0.4738→0.4097)
- LSTM v1→v2: fold 4 0.396→0.440

MOEX имеет выраженную внутридневную сезонность (открытие 10:00, закрытие 18:45,
обеденный перерыв) — это реальный и сильный торговый сигнал.

### 4. h=3:k=1.25 — оптимальная настройка таргета
При h≥6 с k=1.25 класс HOLD исчезает (цена всегда пробивает барьер за 6 часов).
Калибровка барьеров под большие горизонты (h=6:k=2.0 и выше) всё равно даёт хуже.

---

## Текущее состояние

**Лучшая модель**: LSTM v2
- Walk-forward F1: 0.4778 ± 0.022
- Simple-split F1: ~0.59 (оценочно, аналогично ET 0.5792)
- Скрипт: `ml/scripts/sber_lstm_research.py`

**ET artifact**: существует, проходит smoke tests, F1=0.5792 (simple-split)
- Артефакт в `ml/artifacts/research_triple_barrier_sber_h1/`
- LSTM-артефакт ещё не упакован

---

## Следующие шаги (в порядке приоритета)

### 1. Backtest LSTM v2 (ближайшая задача, ~1 день)
**Почему важно**: F1=0.47 может быть прибыльным, если высококонфидентные предсказания
(conf>0.45) более точны. Sharpe ratio — это реальный production-readiness gate.

```python
# Логика бэктеста (см. SKILL.md):
# - Торгуем только когда max(proba) > 0.45
# - BUY → лонг, SELL → шорт, HOLD → без позиции
# - Метрики: Sharpe, max drawdown, total return, win rate
```

Если Sharpe > 0 → ML-блок готов к интеграции с risk_manager.

### 2. Transformer архитектура (~1-2 дня)
Self-attention нативно решает проблему "какие шаги из 32 важны". PyTorch установлен.

```python
# ml/scripts/sber_transformer_research.py
encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128)
transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
# + Linear(14→64) на входе + mean pooling + Linear(64→3) на выходе
```

Ожидаемый прирост: +0.02..0.05 над LSTM.

### 3. Multi-ticker joint training (~1 день)
SBER+LKOH+GAZP → 3× данных (75k свечей). Shared patterns across blue chips.
Это также делает модель более обобщённой для production.

### 4. Python 3.11 + gensim Word2Vec (~0.5 дня)
```powershell
python3.11 -m venv ml/.venv-311
ml/.venv-311/Scripts/pip install gensim scikit-learn pandas pyarrow
```
Затем переиспользовать `ml/scripts/sber_word2vec_research.py` с реальным Word2Vec.

### 5. LSTM v2 artifact packaging
Обновить `ml/scripts/train_research_artifact.py` для LSTM.
Обновить `ml/src/service/research_artifact.py` для PyTorch inference.

---

## Production-readiness checklist (обновлён)

- [ ] Положительный Sharpe ratio в бэктесте (не только F1)
- [ ] LSTM v2 упакован как артефакт (model.pt + feature_config + metadata)
- [ ] Smoke tests зелёные (`python -m pytest ml/test_smoke.py`)
- [ ] Contract validation passing (ml_prediction.schema.json)
- [ ] End-to-end test: candle_batch JSON → ml_prediction JSON
- [x] ET artifact существует и работает
- [x] ECE=0.045 (приемлемая калибровка)
- [x] Таргет h=3:k=1.25 подтверждён как оптимальный
