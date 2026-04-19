package service

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"candle-predictor/internal/config"
	"candle-predictor/internal/models"
)

type PredictService struct {
	cfg        config.MLConfig
	market     config.MarketConfig
	httpClient *http.Client
	logger     *slog.Logger
}

func NewPredictService(cfg config.MLConfig, market config.MarketConfig, logger *slog.Logger) *PredictService {
	return &PredictService{
		cfg:        cfg,
		market:     market,
		httpClient: &http.Client{Timeout: cfg.Timeout()},
		logger:     logger,
	}
}

func (s *PredictService) Health(ctx context.Context) (models.MLHealth, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.joinURL(s.cfg.HealthPath), nil)
	if err != nil {
		return models.MLHealth{}, err
	}

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return models.MLHealth{
			Status:    "degraded",
			Timestamp: time.Now().UTC(),
		}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return models.MLHealth{
			Status:    "degraded",
			Timestamp: time.Now().UTC(),
		}, fmt.Errorf("ml health returned %d: %s", resp.StatusCode, string(body))
	}

	var raw struct {
		Status       string  `json:"status"`
		ModelLoaded  bool    `json:"model_loaded"`
		ModelVersion *string `json:"model_version,omitempty"`
		Timestamp    string  `json:"timestamp"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return models.MLHealth{}, err
	}

	timestamp, err := models.ParseFlexibleTime(raw.Timestamp)
	if err != nil {
		return models.MLHealth{}, err
	}

	return models.MLHealth{
		Status:       raw.Status,
		ModelLoaded:  raw.ModelLoaded,
		ModelVersion: raw.ModelVersion,
		Timestamp:    timestamp,
	}, nil
}

func (s *PredictService) PredictAlgoSignal(ctx context.Context, candles []models.Candle) (models.AlgoSignal, error) {
	if len(candles) < s.cfg.MinCandles {
		return models.AlgoSignal{}, fmt.Errorf("need at least %d candles for ML inference", s.cfg.MinCandles)
	}

	if s.cfg.StrictHealthCheck {
		health, err := s.Health(ctx)
		if err != nil {
			return models.AlgoSignal{}, fmt.Errorf("ml health check failed: %w", err)
		}
		if health.Status != "healthy" || !health.ModelLoaded {
			return models.AlgoSignal{}, fmt.Errorf("ml service is not healthy")
		}
	}

	reqBody := models.PredictRequest{
		Candles: make([]models.PredictCandle, 0, len(candles)),
	}
	for _, candle := range candles {
		reqBody.Candles = append(reqBody.Candles, candle.ToPredictCandle())
	}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return models.AlgoSignal{}, err
	}

	var lastErr error
	for attempt := 0; attempt <= s.cfg.RetryCount; attempt++ {
		response, err := s.doPredict(ctx, payload)
		if err == nil {
			return s.toAlgoSignal(response), nil
		}

		lastErr = err
		if !shouldRetry(err) || attempt == s.cfg.RetryCount {
			break
		}
		time.Sleep(s.cfg.RetryBackoff())
	}

	return models.AlgoSignal{}, lastErr
}

func (s *PredictService) doPredict(ctx context.Context, payload []byte) (models.PredictResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.joinURL(s.cfg.PredictPath), bytes.NewReader(payload))
	if err != nil {
		return models.PredictResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return models.PredictResponse{}, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return models.PredictResponse{}, err
	}

	if resp.StatusCode >= 400 {
		return models.PredictResponse{}, httpStatusError{
			StatusCode: resp.StatusCode,
			Message:    extractErrorMessage(body),
		}
	}

	var raw struct {
		PredictedToken int            `json:"predicted_token"`
		Probabilities  []float64      `json:"probabilities"`
		Action         string         `json:"action"`
		Confidence     float64        `json:"confidence"`
		ModelVersion   string         `json:"model_version"`
		Ticker         string         `json:"ticker"`
		Timeframe      string         `json:"timeframe"`
		Timestamp      string         `json:"timestamp"`
		NCandlesUsed   int            `json:"n_candles_used"`
		Diagnostics    map[string]any `json:"diagnostics,omitempty"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		return models.PredictResponse{}, err
	}

	timestamp, err := models.ParseFlexibleTime(raw.Timestamp)
	if err != nil {
		return models.PredictResponse{}, err
	}

	return models.PredictResponse{
		PredictedToken: raw.PredictedToken,
		Probabilities:  raw.Probabilities,
		Action:         raw.Action,
		Confidence:     raw.Confidence,
		ModelVersion:   raw.ModelVersion,
		Ticker:         raw.Ticker,
		Timeframe:      raw.Timeframe,
		Timestamp:      timestamp,
		NCandlesUsed:   raw.NCandlesUsed,
		Diagnostics:    raw.Diagnostics,
	}, nil
}

func (s *PredictService) toAlgoSignal(response models.PredictResponse) models.AlgoSignal {
	horizon := s.market.DecisionHorizon
	if raw, ok := response.Diagnostics["horizon"]; ok {
		horizon = fmt.Sprintf("%v", raw)
	}

	return models.AlgoSignal{
		Ticker:         response.Ticker,
		Horizon:        horizon,
		Action:         models.NormalizeAction(response.Action),
		Direction:      algoDirection(response),
		Confidence:     models.Clamp(response.Confidence, 0, 1),
		PredictedToken: response.PredictedToken,
		Probabilities:  response.Probabilities,
		ModelVersion:   response.ModelVersion,
		Timestamp:      response.Timestamp.UTC(),
		NCandlesUsed:   response.NCandlesUsed,
		Diagnostics:    response.Diagnostics,
	}
}

func algoDirection(response models.PredictResponse) float64 {
	if len(response.Probabilities) > 1 {
		mid := float64(len(response.Probabilities)-1) / 2
		if mid > 0 {
			var score float64
			for i, probability := range response.Probabilities {
				score += probability * ((float64(i) - mid) / mid)
			}
			return math.Max(-1, math.Min(1, score))
		}
	}

	switch strings.ToLower(strings.TrimSpace(response.Action)) {
	case "buy":
		return response.Confidence
	case "sell":
		return -response.Confidence
	default:
		return 0
	}
}

func shouldRetry(err error) bool {
	var statusErr httpStatusError
	if errorsAs(err, &statusErr) {
		return statusErr.StatusCode >= 500
	}

	var netErr net.Error
	if errorsAs(err, &netErr) {
		return netErr.Timeout() || netErr.Temporary()
	}

	var urlErr *url.Error
	if errorsAs(err, &urlErr) {
		return true
	}

	return false
}

func extractErrorMessage(body []byte) string {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err == nil {
		if detail, ok := payload["detail"].(string); ok && detail != "" {
			return detail
		}
		if message, ok := payload["error"].(string); ok && message != "" {
			return message
		}
	}
	return strings.TrimSpace(string(body))
}

func (s *PredictService) joinURL(path string) string {
	return strings.TrimRight(s.cfg.BaseURL, "/") + path
}

type httpStatusError struct {
	StatusCode int
	Message    string
}

func (e httpStatusError) Error() string {
	return fmt.Sprintf("http status %d: %s", e.StatusCode, e.Message)
}

func errorsAs[T error](err error, target *T) bool {
	return errors.As(err, target)
}
