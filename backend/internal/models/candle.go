package models

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"
)

type Candle struct {
	Ticker    string    `json:"ticker,omitempty"`
	Timeframe string    `json:"timeframe,omitempty"`
	Begin     time.Time `json:"begin"`
	End       time.Time `json:"end,omitempty"`
	Open      float64   `json:"open"`
	High      float64   `json:"high"`
	Low       float64   `json:"low"`
	Close     float64   `json:"close"`
	Volume    float64   `json:"volume"`
	Value     float64   `json:"value,omitempty"`
	Source    string    `json:"source,omitempty"`
}

type PredictCandle struct {
	Begin     time.Time `json:"begin"`
	Open      float64   `json:"open"`
	High      float64   `json:"high"`
	Low       float64   `json:"low"`
	Close     float64   `json:"close"`
	Volume    float64   `json:"volume"`
	Ticker    string    `json:"ticker"`
	Timeframe string    `json:"timeframe"`
}

type CandlesIngestRequest struct {
	Candles []Candle `json:"candles"`
	Source  string   `json:"source,omitempty"`
}

func (c *Candle) UnmarshalJSON(data []byte) error {
	type rawCandle struct {
		Ticker    string  `json:"ticker,omitempty"`
		Timeframe string  `json:"timeframe,omitempty"`
		Begin     string  `json:"begin"`
		End       *string `json:"end,omitempty"`
		Open      float64 `json:"open"`
		High      float64 `json:"high"`
		Low       float64 `json:"low"`
		Close     float64 `json:"close"`
		Volume    float64 `json:"volume"`
		Value     float64 `json:"value,omitempty"`
		Source    string  `json:"source,omitempty"`
	}

	var raw rawCandle
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	begin, err := ParseFlexibleTime(raw.Begin)
	if err != nil {
		return err
	}

	var end time.Time
	if raw.End != nil && strings.TrimSpace(*raw.End) != "" {
		end, err = ParseFlexibleTime(*raw.End)
		if err != nil {
			return err
		}
	}

	*c = Candle{
		Ticker:    raw.Ticker,
		Timeframe: raw.Timeframe,
		Begin:     begin,
		End:       end,
		Open:      raw.Open,
		High:      raw.High,
		Low:       raw.Low,
		Close:     raw.Close,
		Volume:    raw.Volume,
		Value:     raw.Value,
		Source:    raw.Source,
	}
	return nil
}

func (c Candle) WithDefaults(defaultTicker, defaultTimeframe, defaultSource string) Candle {
	if c.Ticker == "" {
		c.Ticker = defaultTicker
	}
	if c.Timeframe == "" {
		c.Timeframe = defaultTimeframe
	}
	if c.Source == "" {
		c.Source = defaultSource
	}
	if c.Value == 0 {
		c.Value = c.Close * c.Volume
	}
	if c.End.IsZero() {
		if duration, err := ParseTimeframe(c.Timeframe); err == nil {
			c.End = c.Begin.Add(duration)
		}
	}
	return c
}

func (c Candle) ToPredictCandle() PredictCandle {
	return PredictCandle{
		Begin:     c.Begin,
		Open:      c.Open,
		High:      c.High,
		Low:       c.Low,
		Close:     c.Close,
		Volume:    c.Volume,
		Ticker:    c.Ticker,
		Timeframe: c.Timeframe,
	}
}

func ParseTimeframe(value string) (time.Duration, error) {
	normalized := strings.TrimSpace(strings.ToUpper(value))
	if len(normalized) < 2 {
		return 0, fmt.Errorf("unsupported timeframe %q", value)
	}

	unit := normalized[len(normalized)-1]
	amount, err := strconv.Atoi(normalized[:len(normalized)-1])
	if err != nil || amount <= 0 {
		return 0, fmt.Errorf("unsupported timeframe %q", value)
	}

	switch unit {
	case 'M':
		return time.Duration(amount) * time.Minute, nil
	case 'H':
		return time.Duration(amount) * time.Hour, nil
	case 'D':
		return time.Duration(amount) * 24 * time.Hour, nil
	default:
		return 0, fmt.Errorf("unsupported timeframe %q", value)
	}
}
