# Execution Block

`execution/` - scaffold будущего adapter к broker/MOEX API.

Первый режим:

- dry-run;
- paper trading;
- no real orders by default.

Ограничения первого этапа:

- реальные orders запрещены без явного enable-флага;
- только limit orders;
- duplicate order protection;
- kill switch;
- audit log.

Live execution в demo-ветке не реализован.
