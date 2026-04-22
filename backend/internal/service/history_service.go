package service

import (
	"context"
	"fmt"
	"sort"
	"time"

	"candle-predictor/internal/config"
	"candle-predictor/internal/models"
	"candle-predictor/internal/storage"
)

type HistoryService struct {
	store  storage.RawCandleStore
	market config.MarketConfig
}

func NewHistoryService(store storage.RawCandleStore, market config.MarketConfig) *HistoryService {
	return &HistoryService{store: store, market: market}
}

func (s *HistoryService) PrepareCandles(candles []models.Candle, source string) ([]models.Candle, error) {
	if len(candles) == 0 {
		return nil, fmt.Errorf("no candles provided")
	}

	prepared := make([]models.Candle, len(candles))
	for i, candle := range candles {
		defaultSource := s.market.DefaultSource
		if source != "" {
			defaultSource = source
		}

		candle = candle.WithDefaults(s.market.DefaultTicker, s.market.DefaultTimeframe, defaultSource)
		if candle.Begin.IsZero() {
			return nil, fmt.Errorf("candle %d has empty begin", i)
		}
		if candle.Open <= 0 || candle.High <= 0 || candle.Low <= 0 || candle.Close <= 0 {
			return nil, fmt.Errorf("candle %d contains non-positive OHLC", i)
		}
		if candle.High < candle.Low {
			return nil, fmt.Errorf("candle %d has high < low", i)
		}
		if candle.Volume <= 0 {
			return nil, fmt.Errorf("candle %d has non-positive volume", i)
		}
		prepared[i] = candle
	}

	sort.Slice(prepared, func(i, j int) bool {
		return prepared[i].Begin.Before(prepared[j].Begin)
	})

	ticker := prepared[0].Ticker
	timeframe := prepared[0].Timeframe
	seenBegins := make(map[string]struct{}, len(prepared))

	for i, candle := range prepared {
		if candle.Ticker != ticker {
			return nil, fmt.Errorf("mixed tickers in candle batch")
		}
		if candle.Timeframe != timeframe {
			return nil, fmt.Errorf("mixed timeframes in candle batch")
		}
		key := candle.Begin.UTC().Format("2006-01-02T15:04:05")
		if _, exists := seenBegins[key]; exists {
			return nil, fmt.Errorf("duplicate candle begin %s", key)
		}
		seenBegins[key] = struct{}{}
		if i > 0 {
			prev := prepared[i-1]
			if !prev.End.IsZero() && candle.Begin.Before(prev.End) {
				return nil, fmt.Errorf(
					"overlapping candles: previous [%s, %s) overlaps current begin %s",
					prev.Begin.UTC().Format(time.RFC3339),
					prev.End.UTC().Format(time.RFC3339),
					candle.Begin.UTC().Format(time.RFC3339),
				)
			}
		}
	}

	return prepared, nil
}

func (s *HistoryService) Save(ctx context.Context, candles []models.Candle, source string) (models.StoreCandlesResponse, error) {
	prepared, err := s.PrepareCandles(candles, source)
	if err != nil {
		return models.StoreCandlesResponse{}, err
	}

	path, err := s.store.SaveRawCandles(ctx, prepared)
	if err != nil {
		return models.StoreCandlesResponse{}, err
	}

	return models.StoreCandlesResponse{
		Count:      len(prepared),
		Ticker:     prepared[0].Ticker,
		Timeframe:  prepared[0].Timeframe,
		StoredPath: path,
		Begin:      prepared[0].Begin,
		End:        prepared[len(prepared)-1].End,
		Source:     prepared[0].Source,
	}, nil
}
