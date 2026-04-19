package app

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"time"

	"candle-predictor/internal/config"
	"candle-predictor/internal/models"
	"candle-predictor/internal/moex"
	"candle-predictor/internal/service"
)

type App struct {
	cfg        *config.Config
	logger     *slog.Logger
	httpServer *http.Server
	history    *service.HistoryService
	predict    *service.PredictService
	decision   *service.DecisionService
	llm        service.LLMAnalyzer
	moex       *moex.Client
}

func New(cfg *config.Config) (*App, error) {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	store, err := newFileStore(cfg)
	if err != nil {
		return nil, err
	}

	history := service.NewHistoryService(store, cfg.Market)
	predict := service.NewPredictService(cfg.ML, cfg.Market, logger)
	llm := service.NewLLMAnalyzer(cfg.LLM, cfg.Market, logger)
	risk := service.NewRiskService(cfg.Risk)
	decision := service.NewDecisionService(cfg, history, predict, llm, risk, store, logger)
	moexClient := moex.NewClient(cfg.MOEX, logger)

	app := &App{
		cfg:      cfg,
		logger:   logger,
		history:  history,
		predict:  predict,
		decision: decision,
		llm:      llm,
		moex:     moexClient,
	}

	app.httpServer = &http.Server{
		Addr:         cfg.Server.Address,
		Handler:      app.routes(),
		ReadTimeout:  cfg.Server.ReadTimeout(),
		WriteTimeout: cfg.Server.WriteTimeout(),
	}

	return app, nil
}

func (a *App) Run(ctx context.Context) error {
	errCh := make(chan error, 1)

	go func() {
		a.logger.Info("backend started", "addr", a.httpServer.Addr)
		if err := a.httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
			return
		}
		errCh <- nil
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), a.cfg.Server.ShutdownTimeout())
		defer cancel()
		a.logger.Info("backend shutting down")
		return a.httpServer.Shutdown(shutdownCtx)
	case err := <-errCh:
		return err
	}
}

func (a *App) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", a.handleHealth)
	mux.HandleFunc("/api/v1/candles/store", a.handleStoreCandles)
	mux.HandleFunc("/api/v1/candles/fetch", a.handleFetchCandles)
	mux.HandleFunc("/api/v1/decisions/evaluate", a.handleEvaluateDecision)
	return a.loggingMiddleware(mux)
}

func (a *App) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		a.writeMethodNotAllowed(w, http.MethodGet)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()

	mlHealth, mlErr := a.predict.Health(ctx)
	llmHealth := a.llm.Health(ctx)

	status := "healthy"
	if mlErr != nil || mlHealth.Status != "healthy" {
		status = "degraded"
	}

	payload := map[string]any{
		"status":    status,
		"timestamp": time.Now().UTC(),
		"market": map[string]string{
			"ticker":    a.cfg.Market.DefaultTicker,
			"timeframe": a.cfg.Market.DefaultTimeframe,
		},
		"components": map[string]any{
			"ml":  mlHealth,
			"llm": llmHealth,
		},
	}
	if mlErr != nil {
		payload["ml_error"] = mlErr.Error()
	}

	code := http.StatusOK
	if status != "healthy" {
		code = http.StatusServiceUnavailable
	}
	a.writeJSON(w, code, payload)
}

func (a *App) handleStoreCandles(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		a.writeMethodNotAllowed(w, http.MethodPost)
		return
	}

	var req models.CandlesIngestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		a.writeError(w, http.StatusBadRequest, "invalid JSON body", err)
		return
	}

	result, err := a.history.Save(r.Context(), req.Candles, req.Source)
	if err != nil {
		a.writeError(w, http.StatusBadRequest, "failed to store candles", err)
		return
	}

	a.writeJSON(w, http.StatusOK, result)
}

func (a *App) handleFetchCandles(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		a.writeMethodNotAllowed(w, http.MethodGet)
		return
	}

	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		ticker = a.cfg.Market.DefaultTicker
	}
	timeframe := r.URL.Query().Get("timeframe")
	if timeframe == "" {
		timeframe = a.cfg.Market.DefaultTimeframe
	}

	from, err := parseDateTimeParam(r, "from")
	if err != nil {
		a.writeError(w, http.StatusBadRequest, "invalid from value", err)
		return
	}
	to, err := parseDateTimeParam(r, "to")
	if err != nil {
		a.writeError(w, http.StatusBadRequest, "invalid to value", err)
		return
	}

	candles, sourceURL, err := a.moex.FetchCandles(r.Context(), ticker, timeframe, from, to)
	if err != nil {
		a.writeError(w, http.StatusBadGateway, "failed to fetch candles from MOEX", err)
		return
	}

	result, err := a.history.Save(r.Context(), candles, "moex")
	if err != nil {
		a.writeError(w, http.StatusBadRequest, "failed to persist fetched candles", err)
		return
	}

	payload := map[string]any{
		"source_url": sourceURL,
		"storage":    result,
	}
	a.writeJSON(w, http.StatusOK, payload)
}

func (a *App) handleEvaluateDecision(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		a.writeMethodNotAllowed(w, http.MethodPost)
		return
	}

	var req models.DecisionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		a.writeError(w, http.StatusBadRequest, "invalid JSON body", err)
		return
	}

	response, err := a.decision.Evaluate(r.Context(), req)
	if err != nil {
		a.writeError(w, http.StatusBadRequest, "failed to evaluate trade decision", err)
		return
	}

	a.writeJSON(w, http.StatusOK, response)
}

func (a *App) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		a.logger.Info("request completed", "method", r.Method, "path", r.URL.Path, "duration", time.Since(start).String())
	})
}

func (a *App) writeMethodNotAllowed(w http.ResponseWriter, method string) {
	w.Header().Set("Allow", method)
	a.writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
}

func (a *App) writeError(w http.ResponseWriter, code int, message string, err error) {
	payload := map[string]any{"error": message}
	if err != nil {
		payload["detail"] = err.Error()
	}
	a.writeJSON(w, code, payload)
}

func (a *App) writeJSON(w http.ResponseWriter, code int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(payload)
}

func parseDateTimeParam(r *http.Request, name string) (time.Time, error) {
	value := r.URL.Query().Get(name)
	if value == "" {
		return time.Time{}, errors.New("missing required query parameter")
	}
	for _, layout := range []string{time.RFC3339, "2006-01-02"} {
		if parsed, err := time.Parse(layout, value); err == nil {
			return parsed.UTC(), nil
		}
	}
	return time.Time{}, strconv.ErrSyntax
}
