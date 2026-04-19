package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"candle-predictor/internal/models"
	parquet "github.com/parquet-go/parquet-go"
)

type RawCandleStore interface {
	SaveRawCandles(context.Context, []models.Candle) (string, error)
}

type DecisionLogger interface {
	AppendDecision(context.Context, models.DecisionResponse) error
}

type FileStore struct {
	rawDir          string
	decisionLogPath string
	mu              sync.Mutex
}

type rawCandleRow struct {
	Ticker    string    `parquet:"ticker"`
	Timeframe string    `parquet:"timeframe"`
	Begin     time.Time `parquet:"begin,timestamp(millisecond)"`
	End       time.Time `parquet:"end,timestamp(millisecond)"`
	Open      float64   `parquet:"open"`
	High      float64   `parquet:"high"`
	Low       float64   `parquet:"low"`
	Close     float64   `parquet:"close"`
	Volume    float64   `parquet:"volume"`
	Value     float64   `parquet:"value"`
	Source    string    `parquet:"source"`
}

func NewFileStore(rawDir, decisionLogPath string) (*FileStore, error) {
	if err := os.MkdirAll(rawDir, 0o755); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(decisionLogPath), 0o755); err != nil {
		return nil, err
	}
	return &FileStore{
		rawDir:          rawDir,
		decisionLogPath: decisionLogPath,
	}, nil
}

func (s *FileStore) SaveRawCandles(ctx context.Context, candles []models.Candle) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if len(candles) == 0 {
		return "", fmt.Errorf("no candles to save")
	}

	rows := make([]rawCandleRow, 0, len(candles))
	for _, candle := range candles {
		rows = append(rows, rawCandleRow{
			Ticker:    candle.Ticker,
			Timeframe: candle.Timeframe,
			Begin:     candle.Begin.UTC(),
			End:       candle.End.UTC(),
			Open:      candle.Open,
			High:      candle.High,
			Low:       candle.Low,
			Close:     candle.Close,
			Volume:    candle.Volume,
			Value:     candle.Value,
			Source:    candle.Source,
		})
	}

	name := fmt.Sprintf(
		"%s_%s_%s_%s.parquet",
		candles[0].Ticker,
		candles[0].Timeframe,
		candles[0].Begin.UTC().Format("20060102T1504"),
		candles[len(candles)-1].Begin.UTC().Format("20060102T1504"),
	)
	path := filepath.Join(s.rawDir, name)
	tempPath := path + ".tmp"

	if err := parquet.WriteFile(tempPath, rows); err != nil {
		return "", err
	}
	if err := os.Rename(tempPath, path); err != nil {
		return "", err
	}
	return path, nil
}

func (s *FileStore) AppendDecision(ctx context.Context, decision models.DecisionResponse) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	file, err := os.OpenFile(s.decisionLogPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()

	return json.NewEncoder(file).Encode(decision)
}
