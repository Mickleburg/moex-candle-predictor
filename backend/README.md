# Backend / Data Block

`backend/` - существующий Go backend проекта. В demo-архитектуре он отвечает за data layer:

- загрузку исторических свечей MOEX;
- хранение raw candles;
- batch validation;
- поддержку нескольких тикеров;
- подготовку данных для ML research;
- будущий API layer для свежих market data.

Go-код уже находится внутри `backend/`, поэтому агрессивный move не выполнялся.

## Input

```json
{
  "ticker": "SBER",
  "timeframe": "1H",
  "from": "2020-01-01T00:00:00+03:00",
  "to": "2026-05-03T23:00:00+03:00",
  "source": "moex"
}
```

## Output

```json
{
  "ticker": "SBER",
  "timeframe": "1H",
  "candles_count": 24613,
  "raw_path": "ml/data/raw/SBER_1H_20200103T0900_20260503T1800.parquet",
  "quality": {
    "duplicates": 0,
    "invalid_ohlc": 0,
    "missing_ohlcv": 0
  }
}
```

## Current status

- Реализован Go HTTP backend.
- Есть MOEX клиент и storage layer.
- Backend может быть источником данных для ML research.
- Live trading/execution не относится к backend-блоку.

## Проверка

Если меняется Go-код:

```powershell
Set-Location backend
go test ./...
```
