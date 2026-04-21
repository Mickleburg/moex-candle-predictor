package app

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
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
	mux.HandleFunc("/api/v1/moex/iss", a.handleMOEXISS)
	mux.HandleFunc("/api/v1/moex/security", a.handleMOEXSecurity)
	mux.HandleFunc("/api/v1/moex/candles", a.handleMOEXCandles)
	mux.HandleFunc("/api/v1/moex/orderbook", a.handleMOEXOrderBook)
	mux.HandleFunc("/api/v1/moex/trades", a.handleMOEXTrades)
	mux.HandleFunc("/api/v1/moex/sitenews", a.handleMOEXSiteNews)
	mux.HandleFunc("/api/v1/algopack/dataset", a.handleAlgoPackDataset)
	mux.HandleFunc("/api/v1/algopack/tradestats", a.handleAlgoPackTradeStats)
	mux.HandleFunc("/api/v1/algopack/orderstats", a.handleAlgoPackOrderStats)
	mux.HandleFunc("/api/v1/algopack/obstats", a.handleAlgoPackOBStats)
	mux.HandleFunc("/api/v1/algopack/hi2", a.handleAlgoPackHI2)
	mux.HandleFunc("/api/v1/algopack/futoi", a.handleAlgoPackFUTOI)
	mux.HandleFunc("/api/v1/algopack/realtime/candles", a.handleAlgoPackRealtimeCandles)
	mux.HandleFunc("/api/v1/algopack/realtime/orderbook", a.handleAlgoPackRealtimeOrderBook)
	mux.HandleFunc("/api/v1/algopack/realtime/trades", a.handleAlgoPackRealtimeTrades)
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

func (a *App) handleMOEXISS(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		a.writeMethodNotAllowed(w, http.MethodGet)
		return
	}

	resourcePath := strings.TrimSpace(r.URL.Query().Get("path"))
	if resourcePath == "" {
		a.writeError(w, http.StatusBadRequest, "missing path query parameter", errors.New("example: /iss/securities/SBER"))
		return
	}

	raw, requestURL, err := a.moex.FetchRawJSON(r.Context(), resourcePath, cloneQueryExcluding(r.URL.Query(), "path"))
	if err != nil {
		a.writeError(w, http.StatusBadGateway, "failed to query MOEX ISS resource", err)
		return
	}

	a.writeMOEXPayload(w, resourcePath, requestURL, raw)
}

func (a *App) handleMOEXSecurity(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		a.writeMethodNotAllowed(w, http.MethodGet)
		return
	}

	security := strings.TrimSpace(r.URL.Query().Get("security"))
	raw, requestURL, err := a.moex.FetchSecuritySpec(r.Context(), security)
	if err != nil {
		a.writeError(w, http.StatusBadGateway, "failed to fetch MOEX security data", err)
		return
	}

	resourcePath := "/iss/securities"
	if security != "" {
		resourcePath += "/" + security
	}
	a.writeMOEXPayload(w, resourcePath, requestURL, raw)
}

func (a *App) handleMOEXCandles(w http.ResponseWriter, r *http.Request) {
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

	engine, market, board := marketParamsOrDefault(r, a.cfg.MOEX)
	raw, requestURL, err := a.moex.FetchRawCandles(r.Context(), ticker, timeframe, from, to, engine, market, board)
	if err != nil {
		a.writeError(w, http.StatusBadGateway, "failed to fetch MOEX candles", err)
		return
	}

	a.writeMOEXPayload(w, pathForBoardResource(engine, market, board, ticker, "candles"), requestURL, raw)
}

func (a *App) handleMOEXOrderBook(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		a.writeMethodNotAllowed(w, http.MethodGet)
		return
	}

	security := strings.TrimSpace(r.URL.Query().Get("security"))
	if security == "" {
		a.writeError(w, http.StatusBadRequest, "missing security query parameter", errors.New("example: security=SBER"))
		return
	}

	depth, err := parseOptionalIntParam(r, "depth", 0)
	if err != nil {
		a.writeError(w, http.StatusBadRequest, "invalid depth value", err)
		return
	}

	engine, market, board := marketParamsOrDefault(r, a.cfg.MOEX)
	raw, requestURL, err := a.moex.FetchOrderBook(r.Context(), security, engine, market, board, depth)
	if err != nil {
		a.writeError(w, http.StatusBadGateway, "failed to fetch MOEX orderbook", err)
		return
	}

	a.writeMOEXPayload(w, pathForBoardResource(engine, market, board, security, "orderbook"), requestURL, raw)
}

func (a *App) handleMOEXTrades(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		a.writeMethodNotAllowed(w, http.MethodGet)
		return
	}

	security := strings.TrimSpace(r.URL.Query().Get("security"))
	if security == "" {
		a.writeError(w, http.StatusBadRequest, "missing security query parameter", errors.New("example: security=SBER"))
		return
	}

	start, err := parseOptionalIntParam(r, "start", 0)
	if err != nil {
		a.writeError(w, http.StatusBadRequest, "invalid start value", err)
		return
	}
	limit, err := parseOptionalIntParam(r, "limit", 0)
	if err != nil {
		a.writeError(w, http.StatusBadRequest, "invalid limit value", err)
		return
	}

	engine, market, board := marketParamsOrDefault(r, a.cfg.MOEX)
	raw, requestURL, err := a.moex.FetchTrades(r.Context(), security, engine, market, board, start, limit)
	if err != nil {
		a.writeError(w, http.StatusBadGateway, "failed to fetch MOEX trades", err)
		return
	}

	a.writeMOEXPayload(w, pathForBoardResource(engine, market, board, security, "trades"), requestURL, raw)
}

func (a *App) handleMOEXSiteNews(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		a.writeMethodNotAllowed(w, http.MethodGet)
		return
	}

	start, err := parseOptionalIntParam(r, "start", 0)
	if err != nil {
		a.writeError(w, http.StatusBadRequest, "invalid start value", err)
		return
	}
	limit, err := parseOptionalIntParam(r, "limit", 0)
	if err != nil {
		a.writeError(w, http.StatusBadRequest, "invalid limit value", err)
		return
	}

	raw, requestURL, err := a.moex.FetchSiteNews(r.Context(), start, limit)
	if err != nil {
		a.writeError(w, http.StatusBadGateway, "failed to fetch MOEX site news", err)
		return
	}

	a.writeMOEXPayload(w, "/iss/sitenews", requestURL, raw)
}

func (a *App) handleAlgoPackDataset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		a.writeMethodNotAllowed(w, http.MethodGet)
		return
	}

	dataset := strings.TrimSpace(r.URL.Query().Get("dataset"))
	if dataset == "" {
		a.writeError(w, http.StatusBadRequest, "missing dataset query parameter", errors.New("example: dataset=obstats"))
		return
	}

	market := strings.TrimSpace(r.URL.Query().Get("market"))
	if market == "" {
		market = "eq"
	}

	raw, requestURL, err := a.moex.FetchAlgoPackDataset(r.Context(), market, dataset, cloneQueryExcluding(r.URL.Query(), "market", "dataset"))
	if err != nil {
		a.writeError(w, http.StatusBadGateway, "failed to fetch ALGOPACK dataset", err)
		return
	}

	a.writeMOEXPayload(w, pathForAlgoPackDataset(market, dataset), requestURL, raw)
}

func (a *App) handleAlgoPackTradeStats(w http.ResponseWriter, r *http.Request) {
	a.handleNamedAlgoPackDataset(w, r, "tradestats")
}

func (a *App) handleAlgoPackOrderStats(w http.ResponseWriter, r *http.Request) {
	a.handleNamedAlgoPackDataset(w, r, "orderstats")
}

func (a *App) handleAlgoPackOBStats(w http.ResponseWriter, r *http.Request) {
	a.handleNamedAlgoPackDataset(w, r, "obstats")
}

func (a *App) handleAlgoPackHI2(w http.ResponseWriter, r *http.Request) {
	a.handleNamedAlgoPackDataset(w, r, "hi2")
}

func (a *App) handleAlgoPackFUTOI(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		a.writeMethodNotAllowed(w, http.MethodGet)
		return
	}

	raw, requestURL, err := a.moex.FetchFUTOI(r.Context(), cloneQueryExcluding(r.URL.Query()))
	if err != nil {
		a.writeError(w, http.StatusBadGateway, "failed to fetch ALGOPACK FUTOI", err)
		return
	}

	a.writeMOEXPayload(w, "/iss/analyticalproducts/futoi/securities", requestURL, raw)
}

func (a *App) handleAlgoPackRealtimeCandles(w http.ResponseWriter, r *http.Request) {
	a.handleMOEXCandles(w, r)
}

func (a *App) handleAlgoPackRealtimeOrderBook(w http.ResponseWriter, r *http.Request) {
	a.handleMOEXOrderBook(w, r)
}

func (a *App) handleAlgoPackRealtimeTrades(w http.ResponseWriter, r *http.Request) {
	a.handleMOEXTrades(w, r)
}

func (a *App) handleNamedAlgoPackDataset(w http.ResponseWriter, r *http.Request, dataset string) {
	if r.Method != http.MethodGet {
		a.writeMethodNotAllowed(w, http.MethodGet)
		return
	}

	market := strings.TrimSpace(r.URL.Query().Get("market"))
	if market == "" {
		market = "eq"
	}

	raw, requestURL, err := a.moex.FetchAlgoPackDataset(r.Context(), market, dataset, cloneQueryExcluding(r.URL.Query(), "market"))
	if err != nil {
		a.writeError(w, http.StatusBadGateway, "failed to fetch ALGOPACK dataset", err)
		return
	}

	a.writeMOEXPayload(w, pathForAlgoPackDataset(market, dataset), requestURL, raw)
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

func (a *App) writeMOEXPayload(w http.ResponseWriter, resourcePath, requestURL string, raw json.RawMessage) {
	var data any
	if err := json.Unmarshal(raw, &data); err != nil {
		a.writeError(w, http.StatusBadGateway, "failed to decode MOEX response", err)
		return
	}

	a.writeJSON(w, http.StatusOK, map[string]any{
		"path":        resourcePath,
		"request_url": requestURL,
		"data":        data,
	})
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

func parseOptionalIntParam(r *http.Request, name string, defaultValue int) (int, error) {
	value := strings.TrimSpace(r.URL.Query().Get(name))
	if value == "" {
		return defaultValue, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return 0, err
	}
	return parsed, nil
}

func cloneQueryExcluding(values url.Values, excludedKeys ...string) url.Values {
	excluded := make(map[string]struct{}, len(excludedKeys))
	for _, key := range excludedKeys {
		excluded[key] = struct{}{}
	}

	cloned := make(url.Values, len(values))
	for key, items := range values {
		if _, skip := excluded[key]; skip {
			continue
		}
		cloned[key] = append([]string(nil), items...)
	}
	return cloned
}

func marketParamsOrDefault(r *http.Request, cfg config.MOEXConfig) (string, string, string) {
	engine := strings.TrimSpace(r.URL.Query().Get("engine"))
	if engine == "" {
		engine = cfg.Engine
	}
	market := strings.TrimSpace(r.URL.Query().Get("market"))
	if market == "" {
		market = cfg.Market
	}
	board := strings.TrimSpace(r.URL.Query().Get("board"))
	if board == "" {
		board = cfg.Board
	}
	return engine, market, board
}

func pathForBoardResource(engine, market, board, security, resource string) string {
	return "/iss/engines/" + engine + "/markets/" + market + "/boards/" + board + "/securities/" + security + "/" + resource
}

func pathForAlgoPackDataset(market, dataset string) string {
	return "/iss/datashop/algopack/" + strings.ToLower(strings.TrimSpace(market)) + "/" + strings.ToLower(strings.TrimSpace(dataset))
}
