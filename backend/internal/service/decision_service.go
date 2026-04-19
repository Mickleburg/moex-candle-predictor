package service

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"time"

	"candle-predictor/internal/config"
	"candle-predictor/internal/models"
	"candle-predictor/internal/storage"
)

type DecisionService struct {
	cfg       *config.Config
	history   *HistoryService
	predict   *PredictService
	llm       LLMAnalyzer
	risk      *RiskService
	logger    storage.DecisionLogger
	appLogger *slog.Logger
}

func NewDecisionService(
	cfg *config.Config,
	history *HistoryService,
	predict *PredictService,
	llm LLMAnalyzer,
	risk *RiskService,
	logger storage.DecisionLogger,
	appLogger *slog.Logger,
) *DecisionService {
	return &DecisionService{
		cfg:       cfg,
		history:   history,
		predict:   predict,
		llm:       llm,
		risk:      risk,
		logger:    logger,
		appLogger: appLogger,
	}
}

func (s *DecisionService) Evaluate(ctx context.Context, req models.DecisionRequest) (models.DecisionResponse, error) {
	if len(req.Candles) == 0 {
		return models.DecisionResponse{}, fmt.Errorf("decision request requires candles")
	}

	candles, err := s.history.PrepareCandles(req.Candles, req.Source)
	if err != nil {
		return models.DecisionResponse{}, err
	}

	response := models.DecisionResponse{
		Action:    models.ActionHold,
		Reason:    "awaiting aggregation",
		Timestamp: time.Now().UTC(),
	}

	if persistRaw(req.PersistRaw, s.cfg.Storage.SaveRawOnDecision) {
		storeResult, err := s.history.Save(ctx, candles, req.Source)
		if err != nil {
			return models.DecisionResponse{}, fmt.Errorf("persist raw candles: %w", err)
		}
		response.StoredRawPath = storeResult.StoredPath
	}

	algoSignal, err := s.predict.PredictAlgoSignal(ctx, candles)
	if err != nil {
		response.Reason = "ml_unavailable_fallback"
		response.Fallbacks = append(response.Fallbacks, "ml_hold_fallback")
		response.LLMSignal = models.NeutralLLMSignal(s.cfg.Market.DecisionHorizon, "skipped")
		response.AlgoSignal = models.AlgoSignal{
			Ticker:       candles[len(candles)-1].Ticker,
			Horizon:      s.cfg.Market.DecisionHorizon,
			Action:       models.ActionHold,
			Direction:    0,
			Confidence:   0,
			ModelVersion: "unavailable",
			Timestamp:    time.Now().UTC(),
		}
		response.Aggregation = models.AggregationBreakdown{
			WeightAlgo:    s.cfg.Aggregator.WeightAlgo,
			WeightLLM:     s.cfg.Aggregator.WeightLLM,
			BuyThreshold:  s.cfg.Aggregator.BuyThreshold,
			SellThreshold: s.cfg.Aggregator.SellThreshold,
		}
		response.Risk = models.RiskAssessment{
			Approved:          false,
			BlockingReasons:   []string{err.Error()},
			RecommendedSize:   0,
			ResultingPosition: req.Portfolio.CurrentPosition,
		}
		s.logDecision(ctx, response)
		return response, nil
	}
	response.AlgoSignal = algoSignal

	llmSignal, llmFallback := s.resolveLLMSignal(ctx, req, candles)
	response.LLMSignal = llmSignal
	if llmFallback != "" {
		response.Fallbacks = append(response.Fallbacks, llmFallback)
	}

	tradeDecision := s.aggregate(algoSignal, llmSignal)
	response.Action = tradeDecision.Action
	response.Score = tradeDecision.Score
	response.Confidence = tradeDecision.Confidence
	response.Reason = tradeDecision.Reason
	response.Aggregation = tradeDecision.Aggregation

	risk := s.risk.Assess(tradeDecision, req.Portfolio)
	response.Risk = risk
	if !risk.Approved && response.Action != models.ActionHold {
		response.Action = models.ActionHold
		response.Reason = "risk_manager_block"
		response.Fallbacks = append(response.Fallbacks, "risk_manager_hold")
	}

	s.logDecision(ctx, response)
	return response, nil
}

func (s *DecisionService) resolveLLMSignal(ctx context.Context, req models.DecisionRequest, candles []models.Candle) (models.LLMSignal, string) {
	if req.LLMSignalOverride != nil {
		override := *req.LLMSignalOverride
		override.Provider = "request_override"
		if override.Timestamp.IsZero() {
			override.Timestamp = time.Now().UTC()
		}
		return override, ""
	}

	req.Candles = candles
	signal, err := s.llm.Analyze(ctx, req)
	if err != nil {
		s.appLogger.Warn("llm analysis fallback", "error", err.Error())
		return models.NeutralLLMSignal(s.cfg.Market.DecisionHorizon, "neutral_fallback"), "llm_neutral_fallback"
	}
	return signal, ""
}

func (s *DecisionService) aggregate(algo models.AlgoSignal, llm models.LLMSignal) models.TradeDecision {
	llmDirection := models.DirectionFromLLM(llm.Direction)
	algoComponent := s.cfg.Aggregator.WeightAlgo * algo.Direction * algo.Confidence
	llmComponent := s.cfg.Aggregator.WeightLLM * llmDirection * llm.Strength * llm.Confidence
	score := algoComponent + llmComponent

	breakdown := models.AggregationBreakdown{
		WeightAlgo:    s.cfg.Aggregator.WeightAlgo,
		WeightLLM:     s.cfg.Aggregator.WeightLLM,
		AlgoComponent: algoComponent,
		LLMComponent:  llmComponent,
		BuyThreshold:  s.cfg.Aggregator.BuyThreshold,
		SellThreshold: s.cfg.Aggregator.SellThreshold,
	}

	if s.cfg.Aggregator.RiskFlagsBlock && len(llm.RiskFlags) > 0 {
		breakdown.RiskFlagOverride = true
		return models.TradeDecision{
			Action:      models.ActionHold,
			Score:       score,
			Confidence:  math.Min(algo.Confidence, math.Max(llm.Confidence, 0.2)),
			Reason:      "llm_risk_flag_override",
			Aggregation: breakdown,
		}
	}

	action := models.ActionHold
	reason := "score inside hold band"
	switch {
	case score >= s.cfg.Aggregator.BuyThreshold:
		action = models.ActionBuy
		reason = "score above buy threshold"
	case score <= s.cfg.Aggregator.SellThreshold:
		action = models.ActionSell
		reason = "score below sell threshold"
	}

	return models.TradeDecision{
		Action:      action,
		Score:       score,
		Confidence:  math.Min(algo.Confidence, math.Max(llm.Confidence, 0.2)),
		Reason:      reason,
		Aggregation: breakdown,
	}
}

func (s *DecisionService) logDecision(ctx context.Context, response models.DecisionResponse) {
	if s.logger == nil {
		return
	}
	if err := s.logger.AppendDecision(ctx, response); err != nil {
		s.appLogger.Warn("append decision log", "error", err.Error())
	}
}

func persistRaw(requestValue *bool, defaultValue bool) bool {
	if requestValue == nil {
		return defaultValue
	}
	return *requestValue
}
