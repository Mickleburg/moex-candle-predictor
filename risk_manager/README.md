# Risk Manager Block

`risk_manager/` - scaffold будущего блока ограничений и safety checks.

Risk manager учитывает:

- cash;
- текущие позиции;
- max position per ticker;
- max total exposure;
- max daily loss;
- cooldown;
- min expected edge;
- no short на первом этапе.

Если позиции нет, допустимы только `BUY` или `HOLD`.

Если позиция есть, допустимы:

- `BUY_MORE`;
- `HOLD`;
- `SELL_PARTIAL`;
- `SELL_ALL`.

Risk manager имеет право заблокировать любой сигнал независимо от ML, LLM и aggregator.
