package moex

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"time"

	"candle-predictor/internal/config"
	"candle-predictor/internal/models"
)

type Client struct {
	cfg        config.MOEXConfig
	httpClient *http.Client
	logger     *slog.Logger
}

func NewClient(cfg config.MOEXConfig, logger *slog.Logger) *Client {
	return &Client{
		cfg:        cfg,
		httpClient: &http.Client{Timeout: cfg.Timeout()},
		logger:     logger,
	}
}

func (c *Client) FetchCandles(ctx context.Context, ticker, timeframe string, from, to time.Time) ([]models.Candle, string, error) {
	interval, err := IntervalFromTimeframe(timeframe)
	if err != nil {
		return nil, "", err
	}

	requestURL := fmt.Sprintf(
		"%s/iss/engines/%s/markets/%s/boards/%s/securities/%s/candles.json?iss.meta=off&iss.only=candles&from=%s&till=%s&interval=%d",
		c.cfg.BaseURL,
		url.PathEscape(c.cfg.Engine),
		url.PathEscape(c.cfg.Market),
		url.PathEscape(c.cfg.Board),
		url.PathEscape(ticker),
		from.UTC().Format("2006-01-02"),
		to.UTC().Format("2006-01-02"),
		interval,
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL, nil)
	if err != nil {
		return nil, "", err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, requestURL, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, requestURL, fmt.Errorf("moex returned %d: %s", resp.StatusCode, string(body))
	}

	var payload candleResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, requestURL, err
	}

	candles, err := payload.toCandles(ticker, timeframe)
	if err != nil {
		return nil, requestURL, err
	}

	c.logger.Info("fetched candles from moex", "ticker", ticker, "timeframe", timeframe, "count", len(candles))
	return candles, requestURL, nil
}
