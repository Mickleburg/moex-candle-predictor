package app

import (
	"fmt"

	"candle-predictor/internal/config"
	"candle-predictor/internal/storage"
)

func newFileStore(cfg *config.Config) (*storage.FileStore, error) {
	store, err := storage.NewFileStore(cfg.Storage.RawDir, cfg.Storage.DecisionLogPath)
	if err != nil {
		return nil, fmt.Errorf("build file store: %w", err)
	}
	return store, nil
}
