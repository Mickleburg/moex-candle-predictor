package moex

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"

	"candle-predictor/internal/config"
	"candle-predictor/internal/models"
)

type Client struct {
	cfg        config.MOEXConfig
	httpClient *http.Client
	logger     *slog.Logger
}

const moexCandlePageSize = 500

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

	baseQuery := url.Values{
		"iss.meta": {"off"},
		"iss.only": {"candles"},
		"from":     {from.UTC().Format("2006-01-02")},
		"till":     {to.UTC().Format("2006-01-02")},
		"interval": {strconv.Itoa(interval)},
	}

	var allCandles []models.Candle
	var firstRequestURL string

	for start := 0; ; start += moexCandlePageSize {
		if err := ctx.Err(); err != nil {
			return nil, firstRequestURL, err
		}

		query := cloneValues(baseQuery)
		if start > 0 {
			query.Set("start", strconv.Itoa(start))
		}

		raw, requestURL, err := c.FetchRawJSON(ctx, c.boardSecurityPath(ticker, "candles"), query)
		if firstRequestURL == "" {
			firstRequestURL = requestURL
		}
		if err != nil {
			return nil, requestURL, err
		}

		var payload candleResponse
		if err := json.Unmarshal(raw, &payload); err != nil {
			return nil, requestURL, err
		}

		candles, err := payload.toCandles(ticker, timeframe)
		if err != nil {
			return nil, requestURL, err
		}
		allCandles = append(allCandles, candles...)

		if len(candles) < moexCandlePageSize {
			break
		}
	}

	c.logger.Info("fetched candles from moex", "ticker", ticker, "timeframe", timeframe, "count", len(allCandles))
	return allCandles, firstRequestURL, nil
}

func (c *Client) FetchRawCandles(ctx context.Context, ticker, timeframe string, from, to time.Time, engine, market, board string) (json.RawMessage, string, error) {
	interval, err := IntervalFromTimeframe(timeframe)
	if err != nil {
		return nil, "", err
	}

	query := url.Values{
		"iss.meta": {"off"},
		"iss.only": {"candles"},
		"from":     {from.UTC().Format("2006-01-02")},
		"till":     {to.UTC().Format("2006-01-02")},
		"interval": {strconv.Itoa(interval)},
	}

	return c.FetchRawJSON(ctx, c.boardSecurityPathWith(engine, market, board, ticker, "candles"), query)
}

func (c *Client) FetchSecuritySpec(ctx context.Context, security string) (json.RawMessage, string, error) {
	query := url.Values{"iss.meta": {"off"}}
	resourcePath := "/iss/securities"
	if strings.TrimSpace(security) != "" {
		resourcePath = resourcePath + "/" + url.PathEscape(strings.TrimSpace(security))
	}
	return c.FetchRawJSON(ctx, resourcePath, query)
}

func (c *Client) FetchOrderBook(ctx context.Context, security, engine, market, board string, depth int) (json.RawMessage, string, error) {
	query := url.Values{"iss.meta": {"off"}}
	if depth > 0 {
		query.Set("depth", strconv.Itoa(depth))
	}
	return c.FetchRawJSON(ctx, c.boardSecurityPathWith(engine, market, board, security, "orderbook"), query)
}

func (c *Client) FetchTrades(ctx context.Context, security, engine, market, board string, start, limit int) (json.RawMessage, string, error) {
	query := url.Values{"iss.meta": {"off"}}
	if start > 0 {
		query.Set("start", strconv.Itoa(start))
	}
	if limit > 0 {
		query.Set("limit", strconv.Itoa(limit))
	}
	return c.FetchRawJSON(ctx, c.boardSecurityPathWith(engine, market, board, security, "trades"), query)
}

func (c *Client) FetchSiteNews(ctx context.Context, start, limit int) (json.RawMessage, string, error) {
	query := url.Values{"iss.meta": {"off"}}
	if start > 0 {
		query.Set("start", strconv.Itoa(start))
	}
	if limit > 0 {
		query.Set("limit", strconv.Itoa(limit))
	}
	return c.FetchRawJSON(ctx, "/iss/sitenews", query)
}

func (c *Client) FetchRawJSON(ctx context.Context, resourcePath string, query url.Values) (json.RawMessage, string, error) {
	requestURL, err := c.buildURL(c.cfg.BaseURL, resourcePath, query)
	if err != nil {
		return nil, "", err
	}
	return c.doJSONRequest(ctx, requestURL, resourcePath)
}

func (c *Client) FetchAlgoPackDataset(ctx context.Context, market, dataset string, query url.Values) (json.RawMessage, string, error) {
	market = strings.ToLower(strings.TrimSpace(market))
	if market == "" {
		market = "eq"
	}
	dataset = strings.ToLower(strings.TrimSpace(dataset))
	if dataset == "" {
		return nil, "", fmt.Errorf("empty algopack dataset")
	}

	resourcePath := "/iss/datashop/algopack/" + path.Join(market, dataset)
	requestURL, err := c.buildURL(c.algopackBaseURL(), resourcePath, query)
	if err != nil {
		return nil, "", err
	}
	return c.doJSONRequest(ctx, requestURL, resourcePath)
}

func (c *Client) FetchFUTOI(ctx context.Context, query url.Values) (json.RawMessage, string, error) {
	resourcePath := "/iss/analyticalproducts/futoi/securities"
	requestURL, err := c.buildURL(c.algopackBaseURL(), resourcePath, query)
	if err != nil {
		return nil, "", err
	}
	return c.doJSONRequest(ctx, requestURL, resourcePath)
}

func (c *Client) buildURL(baseURL, resourcePath string, query url.Values) (string, error) {
	base, err := url.Parse(strings.TrimRight(baseURL, "/"))
	if err != nil {
		return "", fmt.Errorf("invalid moex base_url: %w", err)
	}

	normalizedPath, err := normalizeISSJSONPath(resourcePath)
	if err != nil {
		return "", err
	}

	requestURL, err := base.Parse(normalizedPath)
	if err != nil {
		return "", err
	}

	values := requestURL.Query()
	for key, items := range query {
		for _, item := range items {
			values.Add(key, item)
		}
	}
	requestURL.RawQuery = values.Encode()
	return requestURL.String(), nil
}

func (c *Client) doJSONRequest(ctx context.Context, requestURL, resourcePath string) (json.RawMessage, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL, nil)
	if err != nil {
		return nil, requestURL, err
	}

	req.Header.Set("Accept", "application/json")
	if strings.TrimSpace(c.cfg.UserAgent) != "" {
		req.Header.Set("User-Agent", c.cfg.UserAgent)
	}
	c.applyAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, requestURL, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, requestURL, fmt.Errorf("moex returned %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, requestURL, err
	}
	if !json.Valid(raw) {
		contentType := strings.TrimSpace(resp.Header.Get("Content-Type"))
		finalURL := requestURL
		if resp.Request != nil && resp.Request.URL != nil {
			finalURL = resp.Request.URL.String()
		}

		preview := strings.TrimSpace(string(raw))
		preview = strings.ReplaceAll(preview, "\n", " ")
		preview = strings.ReplaceAll(preview, "\r", " ")
		if len(preview) > 240 {
			preview = preview[:240]
		}

		if strings.Contains(strings.ToLower(contentType), "text/html") || strings.HasPrefix(preview, "<!DOCTYPE") || strings.HasPrefix(preview, "<html") {
			return nil, requestURL, fmt.Errorf("moex returned html instead of json for %s (content-type=%q, final_url=%q, preview=%q); this usually means invalid auth, missing subscription, or redirect to login", resourcePath, contentType, finalURL, preview)
		}

		return nil, requestURL, fmt.Errorf("moex returned non-json response for %s (content-type=%q, final_url=%q, preview=%q)", resourcePath, contentType, finalURL, preview)
	}

	return json.RawMessage(raw), requestURL, nil
}

func (c *Client) applyAuth(req *http.Request) {
	if strings.TrimSpace(c.cfg.BearerToken) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(c.cfg.BearerToken))
	} else if strings.TrimSpace(c.cfg.Username) != "" || strings.TrimSpace(c.cfg.Password) != "" {
		req.SetBasicAuth(c.cfg.Username, c.cfg.Password)
	}

	if strings.TrimSpace(c.cfg.APIKeyValue) != "" {
		header := strings.TrimSpace(c.cfg.APIKeyHeader)
		if header == "" {
			header = "X-API-Key"
		}
		req.Header.Set(header, c.cfg.APIKeyValue)
	}

	if strings.TrimSpace(c.cfg.PassportCert) != "" {
		req.AddCookie(&http.Cookie{
			Name:  "MicexPassportCert",
			Value: strings.TrimSpace(c.cfg.PassportCert),
		})
	}
}

func (c *Client) boardSecurityPath(security, resource string) string {
	return c.boardSecurityPathWith(c.cfg.Engine, c.cfg.Market, c.cfg.Board, security, resource)
}

func (c *Client) boardSecurityPathWith(engine, market, board, security, resource string) string {
	engine = strings.TrimSpace(engine)
	if engine == "" {
		engine = c.cfg.Engine
	}
	market = strings.TrimSpace(market)
	if market == "" {
		market = c.cfg.Market
	}
	board = strings.TrimSpace(board)
	if board == "" {
		board = c.cfg.Board
	}

	parts := []string{
		"iss",
		"engines",
		engine,
		"markets",
		market,
		"boards",
		board,
		"securities",
		strings.TrimSpace(security),
	}
	if strings.TrimSpace(resource) != "" {
		parts = append(parts, strings.TrimSpace(resource))
	}

	return "/" + path.Join(parts...)
}

func normalizeISSJSONPath(resourcePath string) (string, error) {
	resourcePath = strings.TrimSpace(resourcePath)
	if resourcePath == "" {
		return "", fmt.Errorf("empty ISS path")
	}
	if strings.Contains(resourcePath, "?") {
		return "", fmt.Errorf("ISS path must not contain query string; pass query params separately")
	}

	if !strings.HasPrefix(resourcePath, "/") {
		resourcePath = "/" + resourcePath
	}
	resourcePath = strings.TrimRight(resourcePath, "/")
	if !strings.HasPrefix(resourcePath, "/iss/") && resourcePath != "/iss" {
		return "", fmt.Errorf("unsupported ISS path %q: only /iss/... resources are allowed", resourcePath)
	}

	switch {
	case strings.HasSuffix(resourcePath, ".json"):
		return resourcePath, nil
	case strings.HasSuffix(resourcePath, ".xml"), strings.HasSuffix(resourcePath, ".csv"), strings.HasSuffix(resourcePath, ".html"):
		return "", fmt.Errorf("only JSON ISS resources are supported by backend proxy")
	default:
		return resourcePath + ".json", nil
	}
}

func cloneValues(values url.Values) url.Values {
	cloned := make(url.Values, len(values))
	for key, items := range values {
		cloned[key] = append([]string(nil), items...)
	}
	return cloned
}

func (c *Client) algopackBaseURL() string {
	if strings.TrimSpace(c.cfg.AlgoPackBaseURL) != "" {
		return c.cfg.AlgoPackBaseURL
	}
	return c.cfg.BaseURL
}
