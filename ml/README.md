# ML Block

`ml/` - текущий основной рабочий research-блок проекта. Он отвечает за загрузку свечей, очистку, построение признаков, candle-language эксперименты, action target research и legacy FastAPI inference path.

## Что уже есть

- Legacy production inference path:
  - `ml/artifacts/model.pkl`;
  - `ml/artifacts/tokenizer.pkl`;
  - `ml/artifacts/metadata.json`;
  - `ml/src/service/api.py`;
  - `ml/src/service/predictor.py`.
- Candle-language research:
  - свеча как candle word;
  - sentence windows;
  - TF-IDF/cooccurrence/SVD;
  - n-gram/backoff word LM;
  - next-word continuation metrics.
- Action classification research:
  - return-threshold target;
  - LM-derived features;
  - continuous past-only features;
  - nested threshold calibration;
  - triple-barrier target.
- Diagnostics:
  - candle accounting;
  - calendar/raw coverage audit;
  - walk-forward validation;
  - target audit;
  - leakage/alignment smoke checks.

## Research-направления

- Legacy baseline для `SELL/HOLD/BUY`.
- Candle-word LM и sequence continuation.
- Return-threshold action target.
- Triple-barrier action target.
- Continuous past-only feature baseline.
- LM + continuous feature combinations.

## Текущий лучший validation-only research candidate

```text
target:       triple_barrier:h3:w12:up1.25:down1.25
features:     continuous_regime
model:        extra_trees:depth=none:leaf=20:maxfeat=sqrt
class_weight: none
mean macro-F1: 0.4695
```

Это не production artifact и не разрешение на торговлю. Test split не использовался для выбора этого candidate.

Перед production research artifact нужны:

1. seed robustness для ExtraTrees;
2. frozen candidate protocol;
3. одна честная final evaluation, если protocol разрешает;
4. backtest и paper trading;
5. явный risk layer.

## Команды проверки

```powershell
python -m compileall -q ml\src ml\scripts ml\test_smoke.py
python ml\test_smoke.py
```

## Документы

- `ml/docs/research/` - SBER H1 research reports.
- `ml/docs/README.md` - карта ML-документации.
