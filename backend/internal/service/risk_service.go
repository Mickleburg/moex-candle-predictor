package service

import (
	"math"

	"candle-predictor/internal/config"
	"candle-predictor/internal/models"
)

type RiskService struct {
	cfg config.RiskConfig
}

func NewRiskService(cfg config.RiskConfig) *RiskService {
	return &RiskService{cfg: cfg}
}

func (s *RiskService) Assess(decision models.TradeDecision, portfolio models.PortfolioSnapshot) models.RiskAssessment {
	if decision.Action == models.ActionHold {
		return models.RiskAssessment{
			Approved:          true,
			RecommendedSize:   0,
			ResultingPosition: portfolio.CurrentPosition,
		}
	}

	maxPosition := s.cfg.MaxPositionAbs
	if portfolio.MaxPositionAbs > 0 {
		maxPosition = portfolio.MaxPositionAbs
	}

	maxExposure := s.cfg.MaxPortfolioExposure
	if portfolio.MaxPortfolioExposure > 0 {
		maxExposure = portfolio.MaxPortfolioExposure
	}

	maxDailyLoss := s.cfg.MaxDailyLoss
	if portfolio.DailyLossLimit > 0 {
		maxDailyLoss = portfolio.DailyLossLimit
	}

	reasons := make([]string, 0, 4)
	if maxDailyLoss > 0 && portfolio.DayPnL <= -maxDailyLoss {
		reasons = append(reasons, "daily loss limit exceeded")
	}
	if maxExposure > 0 && math.Abs(portfolio.Exposure) >= maxExposure {
		reasons = append(reasons, "portfolio exposure limit exceeded")
	}

	size := s.cfg.BasePositionSize
	if size <= 0 && maxPosition > 0 {
		size = maxPosition * decision.Confidence
	}
	size = math.Max(0, size)

	resultingPosition := portfolio.CurrentPosition
	switch decision.Action {
	case models.ActionBuy:
		available := maxPosition - math.Abs(portfolio.CurrentPosition)
		if maxPosition > 0 {
			size = math.Min(size, math.Max(available, 0))
		}
		resultingPosition = portfolio.CurrentPosition + size
	case models.ActionSell:
		if !s.cfg.AllowShort && portfolio.CurrentPosition <= 0 {
			reasons = append(reasons, "short selling is disabled")
			size = 0
			break
		}
		if !s.cfg.AllowShort {
			size = math.Min(size, portfolio.CurrentPosition)
		}
		if maxPosition > 0 && s.cfg.AllowShort {
			available := maxPosition - math.Abs(portfolio.CurrentPosition)
			size = math.Min(size, math.Max(available, 0))
		}
		resultingPosition = portfolio.CurrentPosition - size
	}

	if size == 0 {
		reasons = append(reasons, "no available position size after limits")
	}

	return models.RiskAssessment{
		Approved:          len(reasons) == 0,
		BlockingReasons:   reasons,
		RecommendedSize:   size,
		ResultingPosition: resultingPosition,
	}
}
