# Risk Manager Block

`risk_manager/` — scaffold портфельного слоя и safety checks (Python).

## Роль в V2 (ИЗМЕНИЛАСЬ)

В V2 risk_manager — **портфельный конструктор**, а не пер-тикерный допуск. На вход — кросс-секционный
ранг/скор по тикерам от решающей модели; на выход — **маркет-нейтральный портфель**:

- long top-k сильнейших / short bottom-k слабейших;
- балансировка long/short ноги (нейтральность по бете/деньгам);
- сайзинг позиций (в т.ч. по волатильности);
- лимиты: max position per ticker, max gross/net exposure, max daily loss, cooldown, min edge.

Risk manager имеет право **заблокировать любой сигнал** независимо от ML и LLM.

> Шорты в V2 — часть дизайна (маркет-нейтрал), а не «фаза 2». Старое «no short на первом этапе»
> относилось к проваленной направленной постановке и больше не действует.

## Учитывает

cash; текущие позиции; max position per ticker; max gross/net exposure; max daily loss; cooldown;
min expected edge; баланс long/short.

## Статус

Скаффолд, не реализован. Точный контракт входа (`aggregated_signal` → кросс-секционный ранг)
определяется на этапе заморозки контрактов V2.
