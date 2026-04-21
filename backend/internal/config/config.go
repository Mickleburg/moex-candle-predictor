package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server     ServerConfig     `yaml:"server"`
	Market     MarketConfig     `yaml:"market"`
	MOEX       MOEXConfig       `yaml:"moex"`
	ML         MLConfig         `yaml:"ml"`
	LLM        LLMConfig        `yaml:"llm"`
	Aggregator AggregatorConfig `yaml:"aggregator"`
	Risk       RiskConfig       `yaml:"risk"`
	Storage    StorageConfig    `yaml:"storage"`
}

type ServerConfig struct {
	Address                string `yaml:"address"`
	ReadTimeoutSeconds     int    `yaml:"read_timeout_seconds"`
	WriteTimeoutSeconds    int    `yaml:"write_timeout_seconds"`
	ShutdownTimeoutSeconds int    `yaml:"shutdown_timeout_seconds"`
}

type MarketConfig struct {
	DefaultTicker    string `yaml:"default_ticker"`
	DefaultTimeframe string `yaml:"default_timeframe"`
	DefaultSource    string `yaml:"default_source"`
	DecisionHorizon  string `yaml:"decision_horizon"`
}

type MOEXConfig struct {
	BaseURL         string `yaml:"base_url"`
	AlgoPackBaseURL string `yaml:"algopack_base_url"`
	Engine          string `yaml:"engine"`
	Market          string `yaml:"market"`
	Board           string `yaml:"board"`
	Username        string `yaml:"username"`
	Password        string `yaml:"password"`
	PassportCert    string `yaml:"passport_cert"`
	BearerToken     string `yaml:"bearer_token"`
	APIKeyHeader    string `yaml:"api_key_header"`
	APIKeyValue     string `yaml:"api_key_value"`
	UserAgent       string `yaml:"user_agent"`
	TimeoutSeconds  int    `yaml:"timeout_seconds"`
}

type MLConfig struct {
	BaseURL            string `yaml:"base_url"`
	HealthPath         string `yaml:"health_path"`
	PredictPath        string `yaml:"predict_path"`
	TimeoutSeconds     int    `yaml:"timeout_seconds"`
	RetryCount         int    `yaml:"retry_count"`
	RetryBackoffMillis int    `yaml:"retry_backoff_millis"`
	MinCandles         int    `yaml:"min_candles"`
	StrictHealthCheck  bool   `yaml:"strict_health_check"`
}

type LLMConfig struct {
	Provider       string `yaml:"provider"`
	BaseURL        string `yaml:"base_url"`
	HealthPath     string `yaml:"health_path"`
	AnalyzePath    string `yaml:"analyze_path"`
	TimeoutSeconds int    `yaml:"timeout_seconds"`
}

type AggregatorConfig struct {
	WeightAlgo     float64 `yaml:"weight_algo"`
	WeightLLM      float64 `yaml:"weight_llm"`
	BuyThreshold   float64 `yaml:"buy_threshold"`
	SellThreshold  float64 `yaml:"sell_threshold"`
	RiskFlagsBlock bool    `yaml:"risk_flags_block"`
}

type RiskConfig struct {
	MaxPositionAbs       float64 `yaml:"max_position_abs"`
	MaxPortfolioExposure float64 `yaml:"max_portfolio_exposure"`
	MaxDailyLoss         float64 `yaml:"max_daily_loss"`
	BasePositionSize     float64 `yaml:"base_position_size"`
	AllowShort           bool    `yaml:"allow_short"`
}

type StorageConfig struct {
	RawDir            string `yaml:"raw_dir"`
	DecisionLogPath   string `yaml:"decision_log_path"`
	SaveRawOnDecision bool   `yaml:"save_raw_on_decision"`
}

func Default() Config {
	return Config{
		Server: ServerConfig{
			Address:                ":8080",
			ReadTimeoutSeconds:     5,
			WriteTimeoutSeconds:    15,
			ShutdownTimeoutSeconds: 10,
		},
		Market: MarketConfig{
			DefaultTicker:    "SBER",
			DefaultTimeframe: "1H",
			DefaultSource:    "backend",
			DecisionHorizon:  "1h",
		},
		MOEX: MOEXConfig{
			BaseURL:         "https://iss.moex.com",
			AlgoPackBaseURL: "https://apim.moex.com",
			Engine:          "stock",
			Market:          "shares",
			Board:           "TQBR",
			APIKeyHeader:    "X-API-Key",
			UserAgent:       "moex-candle-predictor-backend/0.1",
			TimeoutSeconds:  10,
		},
		ML: MLConfig{
			BaseURL:            "http://localhost:8001",
			HealthPath:         "/health",
			PredictPath:        "/predict",
			TimeoutSeconds:     5,
			RetryCount:         1,
			RetryBackoffMillis: 300,
			MinCandles:         32,
			StrictHealthCheck:  true,
		},
		LLM: LLMConfig{
			Provider:       "heuristic-fallback",
			HealthPath:     "/health",
			AnalyzePath:    "/analyze",
			TimeoutSeconds: 8,
		},
		Aggregator: AggregatorConfig{
			WeightAlgo:     0.6,
			WeightLLM:      0.4,
			BuyThreshold:   0.35,
			SellThreshold:  -0.35,
			RiskFlagsBlock: true,
		},
		Risk: RiskConfig{
			MaxPositionAbs:       1.0,
			MaxPortfolioExposure: 1.0,
			MaxDailyLoss:         0,
			BasePositionSize:     0.25,
			AllowShort:           false,
		},
		Storage: StorageConfig{
			RawDir:            "../../data/raw",
			DecisionLogPath:   "../../data/reports/decision_log.jsonl",
			SaveRawOnDecision: true,
		},
	}
}

func Load(path string) (*Config, error) {
	cfg := Default()

	resolvedPath, err := resolveConfigPath(path)
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(resolvedPath)
	if err != nil {
		return nil, fmt.Errorf("read config %s: %w", resolvedPath, err)
	}

	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", resolvedPath, err)
	}

	cfg.applyEnvOverrides()
	cfg.resolveRelativePaths(filepath.Dir(resolvedPath))

	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	return &cfg, nil
}

func resolveConfigPath(path string) (string, error) {
	candidates := make([]string, 0, 3)
	if path != "" {
		candidates = append(candidates, path)
	}
	if envPath := os.Getenv("BACKEND_CONFIG"); envPath != "" {
		candidates = append(candidates, envPath)
	}
	candidates = append(candidates, "backend/config/config.yaml", "config/config.yaml")

	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}

	return "", fmt.Errorf("config file not found; tried %v", candidates)
}

func (c *Config) applyEnvOverrides() {
	if value := os.Getenv("PORT"); value != "" {
		c.Server.Address = ":" + value
	}
	if value := os.Getenv("BACKEND_ADDR"); value != "" {
		c.Server.Address = value
	}
	if value := os.Getenv("ML_BASE_URL"); value != "" {
		c.ML.BaseURL = value
	}
	if value := os.Getenv("MOEX_BASE_URL"); value != "" {
		c.MOEX.BaseURL = value
	}
	if value := os.Getenv("MOEX_ALGOPACK_BASE_URL"); value != "" {
		c.MOEX.AlgoPackBaseURL = value
	}
	if value := os.Getenv("MOEX_ENGINE"); value != "" {
		c.MOEX.Engine = value
	}
	if value := os.Getenv("MOEX_MARKET"); value != "" {
		c.MOEX.Market = value
	}
	if value := os.Getenv("MOEX_BOARD"); value != "" {
		c.MOEX.Board = value
	}
	if value := os.Getenv("MOEX_USERNAME"); value != "" {
		c.MOEX.Username = value
	}
	if value := os.Getenv("MOEX_PASSWORD"); value != "" {
		c.MOEX.Password = value
	}
	if value := os.Getenv("MOEX_PASSPORT_CERT"); value != "" {
		c.MOEX.PassportCert = value
	}
	if value := os.Getenv("MOEX_BEARER_TOKEN"); value != "" {
		c.MOEX.BearerToken = value
	}
	if value := os.Getenv("MOEX_API_KEY_HEADER"); value != "" {
		c.MOEX.APIKeyHeader = value
	}
	if value := os.Getenv("MOEX_API_KEY_VALUE"); value != "" {
		c.MOEX.APIKeyValue = value
	}
	if value := os.Getenv("MOEX_USER_AGENT"); value != "" {
		c.MOEX.UserAgent = value
	}
	if value := os.Getenv("LLM_BASE_URL"); value != "" {
		c.LLM.BaseURL = value
	}
	if value := os.Getenv("DEFAULT_TICKER"); value != "" {
		c.Market.DefaultTicker = value
	}
	if value := os.Getenv("DEFAULT_TIMEFRAME"); value != "" {
		c.Market.DefaultTimeframe = value
	}
	if value := os.Getenv("MAX_POSITION_ABS"); value != "" {
		if parsed, err := strconv.ParseFloat(value, 64); err == nil {
			c.Risk.MaxPositionAbs = parsed
		}
	}
}

func (c *Config) resolveRelativePaths(baseDir string) {
	if c.Storage.RawDir != "" && !filepath.IsAbs(c.Storage.RawDir) {
		c.Storage.RawDir = filepath.Clean(filepath.Join(baseDir, c.Storage.RawDir))
	}
	if c.Storage.DecisionLogPath != "" && !filepath.IsAbs(c.Storage.DecisionLogPath) {
		c.Storage.DecisionLogPath = filepath.Clean(filepath.Join(baseDir, c.Storage.DecisionLogPath))
	}
}

func (c *Config) Validate() error {
	if c.Server.Address == "" {
		return fmt.Errorf("server.address must not be empty")
	}
	if c.Market.DefaultTicker == "" {
		return fmt.Errorf("market.default_ticker must not be empty")
	}
	if c.Market.DefaultTimeframe == "" {
		return fmt.Errorf("market.default_timeframe must not be empty")
	}
	if c.ML.BaseURL == "" {
		return fmt.Errorf("ml.base_url must not be empty")
	}
	if c.ML.MinCandles <= 0 {
		return fmt.Errorf("ml.min_candles must be positive")
	}
	if c.Aggregator.WeightAlgo < 0 || c.Aggregator.WeightLLM < 0 {
		return fmt.Errorf("aggregator weights must be non-negative")
	}
	if c.Storage.RawDir == "" {
		return fmt.Errorf("storage.raw_dir must not be empty")
	}
	if c.Storage.DecisionLogPath == "" {
		return fmt.Errorf("storage.decision_log_path must not be empty")
	}
	return nil
}

func (c ServerConfig) ReadTimeout() time.Duration {
	return time.Duration(c.ReadTimeoutSeconds) * time.Second
}

func (c ServerConfig) WriteTimeout() time.Duration {
	return time.Duration(c.WriteTimeoutSeconds) * time.Second
}

func (c ServerConfig) ShutdownTimeout() time.Duration {
	return time.Duration(c.ShutdownTimeoutSeconds) * time.Second
}

func (c MLConfig) Timeout() time.Duration {
	return time.Duration(c.TimeoutSeconds) * time.Second
}

func (c MLConfig) RetryBackoff() time.Duration {
	return time.Duration(c.RetryBackoffMillis) * time.Millisecond
}

func (c LLMConfig) Timeout() time.Duration {
	return time.Duration(c.TimeoutSeconds) * time.Second
}

func (c MOEXConfig) Timeout() time.Duration {
	return time.Duration(c.TimeoutSeconds) * time.Second
}
