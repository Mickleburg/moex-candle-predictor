package models

import (
	"encoding/json"
	"math"
	"strings"
	"time"
)

const (
	ActionBuy  = "BUY"
	ActionSell = "SELL"
	ActionHold = "HOLD"

	LLMBullish = "bullish"
	LLMBearish = "bearish"
	LLMNeutral = "neutral"
)

type PredictRequest struct {
	Candles      []PredictCandle `json:"candles"`
	ModelVersion *string         `json:"model_version,omitempty"`
}

type PredictResponse struct {
	PredictedToken int            `json:"predicted_token"`
	Probabilities  []float64      `json:"probabilities"`
	Action         string         `json:"action"`
	Confidence     float64        `json:"confidence"`
	ModelVersion   string         `json:"model_version"`
	Ticker         string         `json:"ticker"`
	Timeframe      string         `json:"timeframe"`
	Timestamp      time.Time      `json:"timestamp"`
	NCandlesUsed   int            `json:"n_candles_used"`
	Diagnostics    map[string]any `json:"diagnostics,omitempty"`
}

type MLHealth struct {
	Status       string    `json:"status"`
	ModelLoaded  bool      `json:"model_loaded"`
	ModelVersion *string   `json:"model_version,omitempty"`
	Timestamp    time.Time `json:"timestamp"`
}

type AlgoSignal struct {
	Ticker         string         `json:"ticker"`
	Horizon        string         `json:"horizon"`
	Action         string         `json:"action"`
	Direction      float64        `json:"direction"`
	Confidence     float64        `json:"confidence"`
	PredictedToken int            `json:"predicted_token"`
	Probabilities  []float64      `json:"probabilities"`
	ModelVersion   string         `json:"model_version"`
	Timestamp      time.Time      `json:"timestamp"`
	NCandlesUsed   int            `json:"n_candles_used"`
	Diagnostics    map[string]any `json:"diagnostics,omitempty"`
}

type IndicatorSnapshot struct {
	RSI              float64   `json:"rsi"`
	MACDLine         float64   `json:"macd_line"`
	MACDSignal       float64   `json:"macd_signal"`
	EMA20            float64   `json:"ema20"`
	EMA50            float64   `json:"ema50"`
	VolumeRatio      float64   `json:"volume_ratio"`
	SupportLevels    []float64 `json:"support_levels,omitempty"`
	ResistanceLevels []float64 `json:"resistance_levels,omitempty"`
}

type NewsItem struct {
	Timestamp time.Time `json:"timestamp"`
	Title     string    `json:"title"`
	Summary   string    `json:"summary,omitempty"`
	Sentiment string    `json:"sentiment,omitempty"`
}

func (n *NewsItem) UnmarshalJSON(data []byte) error {
	type rawNewsItem struct {
		Timestamp string `json:"timestamp"`
		Title     string `json:"title"`
		Summary   string `json:"summary,omitempty"`
		Sentiment string `json:"sentiment,omitempty"`
	}

	var raw rawNewsItem
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	var timestamp time.Time
	if strings.TrimSpace(raw.Timestamp) != "" {
		parsed, err := ParseFlexibleTime(raw.Timestamp)
		if err != nil {
			return err
		}
		timestamp = parsed
	}

	*n = NewsItem{
		Timestamp: timestamp,
		Title:     raw.Title,
		Summary:   raw.Summary,
		Sentiment: raw.Sentiment,
	}
	return nil
}

func (l *LLMSignal) UnmarshalJSON(data []byte) error {
	type rawLLMSignal struct {
		Direction  string   `json:"direction"`
		Strength   float64  `json:"strength"`
		Confidence float64  `json:"confidence"`
		KeyFactors []string `json:"key_factors,omitempty"`
		RiskFlags  []string `json:"risk_flags,omitempty"`
		Horizon    string   `json:"horizon"`
		Provider   string   `json:"provider,omitempty"`
		Timestamp  string   `json:"timestamp"`
	}

	var raw rawLLMSignal
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	var timestamp time.Time
	if strings.TrimSpace(raw.Timestamp) != "" {
		parsed, err := ParseFlexibleTime(raw.Timestamp)
		if err != nil {
			return err
		}
		timestamp = parsed
	}

	*l = LLMSignal{
		Direction:  raw.Direction,
		Strength:   raw.Strength,
		Confidence: raw.Confidence,
		KeyFactors: raw.KeyFactors,
		RiskFlags:  raw.RiskFlags,
		Horizon:    raw.Horizon,
		Provider:   raw.Provider,
		Timestamp:  timestamp,
	}
	return nil
}

type PortfolioSnapshot struct {
	CurrentPosition      float64 `json:"current_position"`
	Exposure             float64 `json:"exposure"`
	DayPnL               float64 `json:"day_pnl"`
	Equity               float64 `json:"equity"`
	MaxPositionAbs       float64 `json:"max_position_abs,omitempty"`
	MaxPortfolioExposure float64 `json:"max_portfolio_exposure,omitempty"`
	DailyLossLimit       float64 `json:"daily_loss_limit,omitempty"`
}

type LLMSignal struct {
	Direction  string    `json:"direction"`
	Strength   float64   `json:"strength"`
	Confidence float64   `json:"confidence"`
	KeyFactors []string  `json:"key_factors,omitempty"`
	RiskFlags  []string  `json:"risk_flags,omitempty"`
	Horizon    string    `json:"horizon"`
	Provider   string    `json:"provider,omitempty"`
	Timestamp  time.Time `json:"timestamp"`
}

type DecisionRequest struct {
	Candles           []Candle          `json:"candles"`
	Indicators        IndicatorSnapshot `json:"indicators"`
	News              []NewsItem        `json:"news,omitempty"`
	Portfolio         PortfolioSnapshot `json:"portfolio"`
	LLMSignalOverride *LLMSignal        `json:"llm_signal_override,omitempty"`
	PersistRaw        *bool             `json:"persist_raw,omitempty"`
	Source            string            `json:"source,omitempty"`
}

type AggregationBreakdown struct {
	WeightAlgo       float64 `json:"weight_algo"`
	WeightLLM        float64 `json:"weight_llm"`
	AlgoComponent    float64 `json:"algo_component"`
	LLMComponent     float64 `json:"llm_component"`
	BuyThreshold     float64 `json:"buy_threshold"`
	SellThreshold    float64 `json:"sell_threshold"`
	RiskFlagOverride bool    `json:"risk_flag_override"`
}

type RiskAssessment struct {
	Approved          bool     `json:"approved"`
	BlockingReasons   []string `json:"blocking_reasons,omitempty"`
	RecommendedSize   float64  `json:"recommended_size"`
	ResultingPosition float64  `json:"resulting_position"`
}

type TradeDecision struct {
	Action      string               `json:"action"`
	Score       float64              `json:"score"`
	Confidence  float64              `json:"confidence"`
	Reason      string               `json:"reason"`
	Aggregation AggregationBreakdown `json:"aggregation"`
}

type DecisionResponse struct {
	Action        string               `json:"action"`
	Score         float64              `json:"score"`
	Confidence    float64              `json:"confidence"`
	Reason        string               `json:"reason"`
	Fallbacks     []string             `json:"fallbacks,omitempty"`
	StoredRawPath string               `json:"stored_raw_path,omitempty"`
	AlgoSignal    AlgoSignal           `json:"algo_signal"`
	LLMSignal     LLMSignal            `json:"llm_signal"`
	Aggregation   AggregationBreakdown `json:"aggregation"`
	Risk          RiskAssessment       `json:"risk"`
	Timestamp     time.Time            `json:"timestamp"`
}

type StoreCandlesResponse struct {
	Count      int       `json:"count"`
	Ticker     string    `json:"ticker"`
	Timeframe  string    `json:"timeframe"`
	StoredPath string    `json:"stored_path"`
	Begin      time.Time `json:"begin"`
	End        time.Time `json:"end"`
	Source     string    `json:"source"`
}

type ComponentHealth struct {
	Status    string    `json:"status"`
	Provider  string    `json:"provider,omitempty"`
	Message   string    `json:"message,omitempty"`
	CheckedAt time.Time `json:"checked_at"`
}

func NormalizeAction(action string) string {
	switch strings.ToUpper(strings.TrimSpace(action)) {
	case "BUY":
		return ActionBuy
	case "SELL":
		return ActionSell
	default:
		return ActionHold
	}
}

func DirectionFromLLM(direction string) float64 {
	switch strings.ToLower(strings.TrimSpace(direction)) {
	case LLMBullish:
		return 1
	case LLMBearish:
		return -1
	default:
		return 0
	}
}

func Clamp(value, min, max float64) float64 {
	return math.Min(max, math.Max(min, value))
}

func NeutralLLMSignal(horizon, provider string) LLMSignal {
	return LLMSignal{
		Direction:  LLMNeutral,
		Strength:   0,
		Confidence: 0.2,
		Horizon:    horizon,
		Provider:   provider,
		Timestamp:  time.Now().UTC(),
	}
}
