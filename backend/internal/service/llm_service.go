package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strings"
	"time"

	"candle-predictor/internal/config"
	"candle-predictor/internal/models"
)

type LLMAnalyzer interface {
	Analyze(context.Context, models.DecisionRequest) (models.LLMSignal, error)
	Health(context.Context) models.ComponentHealth
}

type httpLLMAnalyzer struct {
	cfg        config.LLMConfig
	market     config.MarketConfig
	httpClient *http.Client
	logger     *slog.Logger
}

type heuristicLLMAnalyzer struct {
	provider string
	market   config.MarketConfig
}

type llmAnalyzePayload struct {
	Ticker     string                   `json:"ticker"`
	Horizon    string                   `json:"horizon"`
	Candles    []models.Candle          `json:"candles"`
	Indicators models.IndicatorSnapshot `json:"indicators"`
	News       []models.NewsItem        `json:"news,omitempty"`
	Portfolio  models.PortfolioSnapshot `json:"portfolio"`
}

func NewLLMAnalyzer(cfg config.LLMConfig, market config.MarketConfig, logger *slog.Logger) LLMAnalyzer {
	if strings.TrimSpace(cfg.BaseURL) != "" {
		return &httpLLMAnalyzer{
			cfg:        cfg,
			market:     market,
			httpClient: &http.Client{Timeout: cfg.Timeout()},
			logger:     logger,
		}
	}
	return &heuristicLLMAnalyzer{
		provider: strings.TrimSpace(cfg.Provider),
		market:   market,
	}
}

func (a *httpLLMAnalyzer) Analyze(ctx context.Context, req models.DecisionRequest) (models.LLMSignal, error) {
	payload := llmAnalyzePayload{
		Ticker:     resolveTicker(req.Candles, a.market.DefaultTicker),
		Horizon:    a.market.DecisionHorizon,
		Candles:    req.Candles,
		Indicators: req.Indicators,
		News:       req.News,
		Portfolio:  req.Portfolio,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return models.LLMSignal{}, err
	}

	url := strings.TrimRight(a.cfg.BaseURL, "/") + a.cfg.AnalyzePath
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return models.LLMSignal{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return models.LLMSignal{}, err
	}
	defer resp.Body.Close()

	responseBody, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return models.LLMSignal{}, err
	}
	if resp.StatusCode >= 400 {
		return models.LLMSignal{}, fmt.Errorf("llm analyzer returned %d: %s", resp.StatusCode, strings.TrimSpace(string(responseBody)))
	}

	var signal models.LLMSignal
	if err := json.Unmarshal(responseBody, &signal); err != nil {
		return models.LLMSignal{}, err
	}
	signal.Provider = a.cfg.Provider
	if signal.Timestamp.IsZero() {
		signal.Timestamp = time.Now().UTC()
	}
	return signal, nil
}

func (a *httpLLMAnalyzer) Health(ctx context.Context) models.ComponentHealth {
	url := strings.TrimRight(a.cfg.BaseURL, "/") + a.cfg.HealthPath
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return models.ComponentHealth{
			Status:    "degraded",
			Provider:  a.cfg.Provider,
			Message:   err.Error(),
			CheckedAt: time.Now().UTC(),
		}
	}

	resp, err := a.httpClient.Do(req)
	if err != nil {
		return models.ComponentHealth{
			Status:    "degraded",
			Provider:  a.cfg.Provider,
			Message:   err.Error(),
			CheckedAt: time.Now().UTC(),
		}
	}
	defer resp.Body.Close()

	status := "healthy"
	if resp.StatusCode >= 400 {
		status = "degraded"
	}
	return models.ComponentHealth{
		Status:    status,
		Provider:  a.cfg.Provider,
		CheckedAt: time.Now().UTC(),
	}
}

func (a *heuristicLLMAnalyzer) Analyze(_ context.Context, req models.DecisionRequest) (models.LLMSignal, error) {
	lastClose := 0.0
	if len(req.Candles) > 0 {
		lastClose = req.Candles[len(req.Candles)-1].Close
	}

	bullish := 0.0
	bearish := 0.0
	keyFactors := make([]string, 0, 5)
	riskFlags := make([]string, 0, 4)

	if req.Indicators.RSI > 0 {
		switch {
		case req.Indicators.RSI < 35:
			bullish += 0.35
			keyFactors = append(keyFactors, fmt.Sprintf("RSI %.1f указывает на перепроданность", req.Indicators.RSI))
		case req.Indicators.RSI > 65:
			bearish += 0.35
			keyFactors = append(keyFactors, fmt.Sprintf("RSI %.1f указывает на перекупленность", req.Indicators.RSI))
		}
		if req.Indicators.RSI < 20 || req.Indicators.RSI > 80 {
			riskFlags = append(riskFlags, "extreme_rsi")
		}
	}

	switch {
	case req.Indicators.MACDLine > req.Indicators.MACDSignal:
		bullish += 0.25
		keyFactors = append(keyFactors, "MACD выше сигнальной линии")
	case req.Indicators.MACDLine < req.Indicators.MACDSignal:
		bearish += 0.25
		keyFactors = append(keyFactors, "MACD ниже сигнальной линии")
	}

	switch {
	case req.Indicators.EMA20 > req.Indicators.EMA50:
		bullish += 0.25
		keyFactors = append(keyFactors, "EMA20 выше EMA50, краткосрочный тренд вверх")
	case req.Indicators.EMA20 < req.Indicators.EMA50:
		bearish += 0.25
		keyFactors = append(keyFactors, "EMA20 ниже EMA50, краткосрочный тренд вниз")
	}

	if req.Indicators.VolumeRatio >= 2.5 {
		riskFlags = append(riskFlags, "volume_spike")
		keyFactors = append(keyFactors, fmt.Sprintf("аномальный объём x%.2f", req.Indicators.VolumeRatio))
	}

	if nearestSupport, ok := nearestLevel(lastClose, req.Indicators.SupportLevels, true); ok && lastClose <= nearestSupport*1.01 {
		bullish += 0.15
		keyFactors = append(keyFactors, "цена у ближайшей поддержки")
	}
	if nearestResistance, ok := nearestLevel(lastClose, req.Indicators.ResistanceLevels, false); ok && lastClose >= nearestResistance*0.99 {
		bearish += 0.15
		keyFactors = append(keyFactors, "цена у ближайшего сопротивления")
	}

	for _, item := range req.News {
		title := strings.ToLower(item.Title + " " + item.Summary)
		switch {
		case containsAny(title, "beat", "рост", "дивиденд", "прибыль", "buyback"):
			bullish += 0.2
			keyFactors = append(keyFactors, "новостной фон умеренно позитивный")
		case containsAny(title, "санкц", "suspend", "halt", "убыт", "downgrade", "суд"):
			bearish += 0.25
			riskFlags = append(riskFlags, "negative_news")
			keyFactors = append(keyFactors, "обнаружены негативные корпоративные новости")
		}
	}

	if req.Portfolio.Exposure >= 0.9 {
		riskFlags = append(riskFlags, "high_portfolio_exposure")
	}

	diff := bullish - bearish
	direction := models.LLMNeutral
	if diff > 0.15 {
		direction = models.LLMBullish
	}
	if diff < -0.15 {
		direction = models.LLMBearish
	}

	strength := models.Clamp(math.Abs(diff), 0, 1)
	confidence := models.Clamp(0.2+0.3*math.Max(bullish, bearish)+0.3*strength, 0, 1)
	if direction == models.LLMNeutral {
		confidence = math.Min(confidence, 0.3)
	}

	if len(keyFactors) > 5 {
		keyFactors = keyFactors[:5]
	}
	if len(riskFlags) > 0 {
		riskFlags = uniqueStrings(riskFlags)
	}

	return models.LLMSignal{
		Direction:  direction,
		Strength:   strength,
		Confidence: confidence,
		KeyFactors: keyFactors,
		RiskFlags:  riskFlags,
		Horizon:    a.market.DecisionHorizon,
		Provider:   fallbackProvider(a.provider),
		Timestamp:  time.Now().UTC(),
	}, nil
}

func (a *heuristicLLMAnalyzer) Health(_ context.Context) models.ComponentHealth {
	return models.ComponentHealth{
		Status:    "healthy",
		Provider:  fallbackProvider(a.provider),
		Message:   "local heuristic fallback",
		CheckedAt: time.Now().UTC(),
	}
}

func resolveTicker(candles []models.Candle, fallback string) string {
	if len(candles) == 0 || candles[len(candles)-1].Ticker == "" {
		return fallback
	}
	return candles[len(candles)-1].Ticker
}

func nearestLevel(lastClose float64, levels []float64, support bool) (float64, bool) {
	if len(levels) == 0 || lastClose == 0 {
		return 0, false
	}
	best := 0.0
	bestDistance := math.MaxFloat64
	for _, level := range levels {
		if support && level > lastClose {
			continue
		}
		if !support && level < lastClose {
			continue
		}
		distance := math.Abs(lastClose - level)
		if distance < bestDistance {
			best = level
			bestDistance = distance
		}
	}
	if bestDistance == math.MaxFloat64 {
		return 0, false
	}
	return best, true
}

func containsAny(text string, keywords ...string) bool {
	for _, keyword := range keywords {
		if strings.Contains(text, keyword) {
			return true
		}
	}
	return false
}

func uniqueStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	return result
}

func fallbackProvider(provider string) string {
	if strings.TrimSpace(provider) != "" {
		return provider
	}
	return "heuristic-fallback"
}
