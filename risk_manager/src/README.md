# risk_manager/src — combiner package

V3 sleeve combiner + risk layer (см. `../README.md`).

| Модуль | Роль |
|--------|------|
| `combiner.py` | `combine(sleeve_signals, risk_analytics, config) -> RiskBook`; `to_risk_decisions(book)`; `CombinerConfig`. Нетинг, лимиты (имя/сектор/гросс к фикспоинту), vol-targeting (H4) × режимный гейт (H5), хедж (сектор/рынок/нет), эмиссия `risk_book` + `risk_decision[]`. |
| `sleeves.py` | `normalize(sig)` — `sleeve_signal` (позиции) и `aggregated_signal` (ранги) → directional веса + предложенный сливом хедж (комбинатор его заменяет). |
| `sectors.py` | `SECTOR_MAP` (зеркало `ml/src/features/cross_sectional.py`, локальная копия — не импортируем ml), `sector_of`, `is_index`. Сектор-индекс = инструмент сектор-хеджа. |

Чистый Python, без numpy/pandas. Весь I/O — dict против схем в `contracts/`.
