package moex

import (
	"fmt"
	"slices"
	"strings"
	"time"

	"candle-predictor/internal/models"
)

type candleResponse struct {
	Candles issBlock `json:"candles"`
}

type issBlock struct {
	Columns []string `json:"columns"`
	Data    [][]any  `json:"data"`
}

func (r candleResponse) toCandles(ticker, timeframe string) ([]models.Candle, error) {
	if len(r.Candles.Columns) == 0 {
		return nil, fmt.Errorf("moex response has no candle columns")
	}

	indexOf := func(column string) int {
		return slices.Index(r.Candles.Columns, column)
	}

	required := []string{"begin", "end", "open", "high", "low", "close", "value", "volume"}
	for _, field := range required {
		if indexOf(field) < 0 {
			return nil, fmt.Errorf("moex response missing %q column", field)
		}
	}

	candles := make([]models.Candle, 0, len(r.Candles.Data))
	for _, row := range r.Candles.Data {
		begin, err := parseISSDate(row[indexOf("begin")])
		if err != nil {
			return nil, err
		}
		end, err := parseISSDate(row[indexOf("end")])
		if err != nil {
			return nil, err
		}

		candles = append(candles, models.Candle{
			Ticker:    ticker,
			Timeframe: timeframe,
			Begin:     begin,
			End:       end,
			Open:      asFloat(row[indexOf("open")]),
			High:      asFloat(row[indexOf("high")]),
			Low:       asFloat(row[indexOf("low")]),
			Close:     asFloat(row[indexOf("close")]),
			Volume:    asFloat(row[indexOf("volume")]),
			Value:     asFloat(row[indexOf("value")]),
			Source:    "moex",
		})
	}

	return candles, nil
}

func IntervalFromTimeframe(timeframe string) (int, error) {
	switch strings.ToUpper(strings.TrimSpace(timeframe)) {
	case "1M":
		return 1, nil
	case "10M":
		return 10, nil
	case "1H":
		return 60, nil
	case "1D":
		return 24, nil
	default:
		return 0, fmt.Errorf("unsupported MOEX timeframe %q", timeframe)
	}
}

func parseISSDate(value any) (time.Time, error) {
	switch typed := value.(type) {
	case string:
		for _, layout := range []string{time.RFC3339, "2006-01-02 15:04:05", "2006-01-02"} {
			if parsed, err := time.Parse(layout, typed); err == nil {
				return parsed.UTC(), nil
			}
		}
	}
	return time.Time{}, fmt.Errorf("cannot parse MOEX datetime %v", value)
}

func asFloat(value any) float64 {
	switch typed := value.(type) {
	case float64:
		return typed
	case float32:
		return float64(typed)
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	case string:
		var parsed float64
		fmt.Sscanf(typed, "%f", &parsed)
		return parsed
	default:
		return 0
	}
}
