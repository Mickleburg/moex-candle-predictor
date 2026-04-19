package storage

// SQLite persistence is intentionally left out of the first backend MVP.
// The service writes decision audit logs to JSONL and raw market data to Parquet,
// which keeps the runtime simple while preserving traceability.
