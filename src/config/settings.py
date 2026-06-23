from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Trading mode
    TRADING_MODE: str = "paper"   # "paper" or "notify"

    # Market filtering
    TICKER_SUFFIX: str = ".MI"
    TARGET_COUNTRY: str = "italy"

    @field_validator("TARGET_COUNTRY")
    @classmethod
    def validate_target_country(cls, v: str) -> str:
        return v.lower()

    # Paper trading initial balance (only used in paper mode)
    PAPER_INITIAL_BALANCE: float = 10000.0

    # Maximum risk per trade as a percentage of total portfolio value (e.g., 0.01 = 1%)
    RISK_PER_TRADE_PCT: float = 0.01

    @field_validator("RISK_PER_TRADE_PCT")
    @classmethod
    def validate_risk_per_trade_pct(cls, v: float) -> float:
        if not (0.0 < v <= 0.1):
            raise ValueError("RISK_PER_TRADE_PCT must be between 0.0 and 0.1")
        return v

    # Risk management check interval (seconds) – stop-loss/take-profit checks
    RISK_CHECK_INTERVAL_SECONDS: int = 30

    # Initial delay before first symbol evaluation (seconds)
    # Allows WebSocket and Telegram bot to initialize before first LLM call
    INITIAL_EVALUATION_DELAY_SECONDS: int = 15

    # Base currency
    BASE_CURRENCY: str = "EUR"

    # Benchmark symbol for relative strength and market trend (e.g., SPY, QQQ)
    BENCHMARK_SYMBOL: str = "FTSEMIB.MI"

    # Max symbols to trade simultaneously
    MAX_SYMBOLS: int = 10

    @field_validator("TRADING_MODE")
    @classmethod
    def validate_trading_mode(cls, v: str) -> str:
        if v not in ("paper", "notify"):
            raise ValueError("TRADING_MODE must be 'paper' or 'notify'")
        return v

    @field_validator("MAX_SYMBOLS")
    @classmethod
    def validate_max_symbols(cls, v: int) -> int:
        if v < 1:
            raise ValueError("MAX_SYMBOLS must be at least 1")
        return v

    # Symbol selection limits
    SYMBOL_SELECTION_TOP_VOLUME_LIMIT: int = 50
    SYMBOL_SELECTION_MAX_SYMBOLS: int = 100
    # Maximum number of candidate symbols to consider during symbol selection
    # (fetches tickers/OHLCV for this many top-volume symbols)
    SYMBOL_SELECTION_CANDIDATE_LIMIT: int = 200
    SYMBOL_SELECTION_MIN_SENTIMENT: float = -1.0   # -1.0 = disabled

    # Maximum number of candidates sent to the LLM for stock selection.
    # The engine pre‑ranks candidates by a composite score and keeps only the top N.
    LLM_STOCK_SELECTION_TOP_N: int = 30

    # ETFs that are always included in the candidate pool (if tradable),
    # regardless of volume or composite score.
    ALWAYS_INCLUDE_ETFS: list[str] = []

    # Additional base tickers (without suffix) that are always added to the discovery pool.
    # Useful for stocks that are not in the FTSE MIB or news feeds.
    ADDITIONAL_TICKERS: list[str] = []

    # Enable scraping the Euronext ISIN directory for Milan-listed tickers
    EURONEXT_TICKER_DISCOVERY_ENABLED: bool = True

    # Minimum composite score for a symbol to be used in the volume‑based fallback.
    # Symbols below this score are skipped even if they have high volume.
    FALLBACK_MIN_COMPOSITE_SCORE: float = 0.1
    FALLBACK_MIN_24H_VOLUME: float = 0.0
    EXCLUDED_SYMBOLS: list[str] = []

    # Maximum number of consecutive "keep paused" LLM decisions before the engine
    # force‑resumes trading with a reduced risk multiplier.
    PAUSE_MAX_CONSECUTIVE_KEEP: int = 2

    @field_validator("PAUSE_MAX_CONSECUTIVE_KEEP")
    @classmethod
    def validate_pause_max_consecutive_keep(cls, v: int) -> int:
        if v < 1:
            raise ValueError("PAUSE_MAX_CONSECUTIVE_KEEP must be at least 1")
        return v

    # Global risk multiplier applied when the engine force‑resumes after
    # PAUSE_MAX_CONSECUTIVE_KEEP consecutive "keep paused" decisions.
    PAUSE_FORCE_RESUME_RISK_MULTIPLIER: float = 0.5

    @field_validator("PAUSE_FORCE_RESUME_RISK_MULTIPLIER")
    @classmethod
    def validate_pause_force_resume_risk_multiplier(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("PAUSE_FORCE_RESUME_RISK_MULTIPLIER must be between 0.0 and 1.0")
        return v

    # Maximum number of consecutive partial take-profit reviews before force-executing
    MAX_PARTIAL_TP_REVIEWS: int = 10

    # Maximum number of consecutive dust sweep reviews before force-selling
    MAX_DUST_SWEEP_REVIEWS: int = 10

    # Minimum seconds between forced LLM evaluations triggered by the entry signal monitor.
    # Keeps the bot responsive without spamming the LLM.
    ENTRY_SIGNAL_COOLDOWN_SECONDS: int = 30

    # Minimum entry condition timeout as a multiple of the candle timeframe.
    # e.g., 2.0 means the timeout must be at least 2 × the candle period.
    ENTRY_CONDITION_MIN_TIMEOUT_MULT: float = 2.0

    @field_validator("ENTRY_CONDITION_MIN_TIMEOUT_MULT")
    @classmethod
    def validate_entry_condition_min_timeout_mult(cls, v: float) -> float:
        if v < 1.0:
            raise ValueError("ENTRY_CONDITION_MIN_TIMEOUT_MULT must be >= 1.0")
        return v

    # OHLCV timeframes for multi-timeframe analysis
    OHLCV_TIMEFRAMES: list[str] = ["1h", "1d"]

    # Market data download interval (seconds)
    MARKET_DATA_REFRESH_SECONDS: int = 300

    # OHLCV download staggering (delay between symbols)
    OHLCV_DOWNLOAD_SYMBOL_DELAY_SECONDS: float = 2.0

    # Maximum number of OHLCV candles to insert in a single backfill call.
    # Prevents memory exhaustion and timeouts when backfilling large ranges.
    BACKFILL_MAX_CANDLES_PER_CALL: int = 5000

    @field_validator("OHLCV_TIMEFRAMES")
    @classmethod
    def validate_ohlcv_timeframes(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list) or not all(isinstance(tf, str) for tf in v):
            raise ValueError("OHLCV_TIMEFRAMES must be a list of strings")
        allowed = {"1h", "1d"}
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(f"OHLCV_TIMEFRAMES contains unsupported short-term timeframes: {invalid}. Allowed: {allowed}")
        return v

    # Number of days of OHLCV data to retain and use for backtest / LLM analysis
    OHLCV_RETENTION_DAYS: int = 90

    @field_validator("OHLCV_RETENTION_DAYS")
    @classmethod
    def validate_ohlcv_retention_days(cls, v: int) -> int:
        if v < 7:
            raise ValueError("OHLCV_RETENTION_DAYS must be at least 7")
        return v

    # Yahoo Finance fallback for missing bid/ask quotes
    YAHOO_FINANCE_ENABLED: bool = True
    YAHOO_FINANCE_CACHE_SECONDS: int = 60




    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        if v not in ("ollama", "openai"):
            raise ValueError("LLM_PROVIDER must be 'ollama' or 'openai'")
        return v

    @model_validator(mode="after")
    def set_database_path(self):
        if "DATABASE_PATH" not in self.model_fields_set:
            if self.TRADING_MODE == "paper":
                self.DATABASE_PATH = "data/paper.db"
            else:
                self.DATABASE_PATH = "data/notify.db"
        return self

    # Ollama
    OLLAMA_BASE_URL: Optional[str] = None
    OLLAMA_MODEL: Optional[str] = None
    OLLAMA_API_KEY: Optional[str] = None

    # LLM Provider selection
    LLM_PROVIDER: str = "ollama"   # "ollama" or "openai"

    # OpenAI-compatible API
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_MODEL: Optional[str] = None

    # Mind model (complex reasoning: symbol selection, strategy generation)
    OLLAMA_MIND_MODEL: Optional[str] = None
    OPENAI_MIND_MODEL: Optional[str] = None

    # Actuator model (fast, time‑critical decisions: stop‑loss/take‑profit reviews, corrections)
    OLLAMA_ACTUATOR_MODEL: Optional[str] = None
    OPENAI_ACTUATOR_MODEL: Optional[str] = None

    # Per‑role provider overrides (empty = use global LLM_PROVIDER)
    LLM_MIND_PROVIDER: str = ""
    LLM_ACTUATOR_PROVIDER: str = ""

    # Per‑role OpenAI settings (empty or None = use global OPENAI_*)
    OPENAI_MIND_API_KEY: Optional[str] = None
    OPENAI_ACTUATOR_API_KEY: Optional[str] = None
    OPENAI_MIND_BASE_URL: Optional[str] = None
    OPENAI_ACTUATOR_BASE_URL: Optional[str] = None

    # Per‑role Ollama settings (empty or None = use global OLLAMA_*)
    OLLAMA_MIND_BASE_URL: Optional[str] = None
    OLLAMA_ACTUATOR_BASE_URL: Optional[str] = None
    OLLAMA_MIND_API_KEY: Optional[str] = None
    OLLAMA_ACTUATOR_API_KEY: Optional[str] = None

    # LLM temperature (applies to both providers)
    LLM_TEMPERATURE: float = 0.1

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def validate_llm_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError("LLM_TEMPERATURE must be between 0.0 and 2.0")
        return v

    # Per‑role temperature overrides (optional).
    # Can be a single float (e.g. "0.2") or a range "min-max" (e.g. "0.2-0.5").
    # If a range is given, the engine will pick a temperature inside it
    # based on prompt complexity (higher complexity → higher temperature).
    # If not set, the global LLM_TEMPERATURE is used.
    LLM_MIND_TEMPERATURE: Optional[str] = None
    LLM_ACTUATOR_TEMPERATURE: Optional[str] = None

    @field_validator("LLM_MIND_TEMPERATURE", "LLM_ACTUATOR_TEMPERATURE")
    @classmethod
    def validate_role_temperature(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v.strip() == "":
            return None
        Settings.parse_temperature_range(v)  # raises ValueError if invalid
        return v

    @staticmethod
    def parse_temperature_range(value: Optional[str]) -> Optional[tuple]:
        """Parse a temperature setting into (min, max) or None if not set.

        Returns None for unset; (val, val) for a single float; (min, max) for a range.
        """
        if value is None or value.strip() == "":
            return None
        value = value.strip()
        if "-" in value:
            parts = value.split("-", 1)
            try:
                lo = float(parts[0].strip())
                hi = float(parts[1].strip())
                if lo < 0.0 or hi > 2.0 or lo > hi:
                    raise ValueError
                return (lo, hi)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid temperature range: {value!r}")
        else:
            try:
                v = float(value)
                if not (0.0 <= v <= 2.0):
                    raise ValueError
                return (v, v)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid temperature value: {value!r}")

    # LLM timeout (seconds) for HTTP requests
    LLM_TIMEOUT: float = 60.0

    @field_validator("LLM_TIMEOUT")
    @classmethod
    def validate_llm_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("LLM_TIMEOUT must be positive")
        return v

    # Enforce the LLM's minimum profit per trade check.
    # Set to False to allow trades with very small expected profit.
    ENFORCE_MIN_PROFIT_PER_TRADE: bool = False

    # Order fill timeout (seconds) – used when LLM does not specify one
    ORDER_FILL_TIMEOUT_SECONDS: float = 60.0

    @field_validator("ORDER_FILL_TIMEOUT_SECONDS")
    @classmethod
    def validate_order_fill_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("ORDER_FILL_TIMEOUT_SECONDS must be positive")
        return v

    # Maximum allowed distance of a limit price from the current best bid/ask,
    # expressed as a fraction (e.g., 0.05 = 5%). Orders with a limit price
    # further away than this are rejected to avoid indefinite queuing.
    # Set to a higher value (e.g., 0.10) for paper trading if you want to allow
    # wider limit orders. Set to 0.0 to disable the check entirely.
    LIMIT_PRICE_MAX_DISTANCE_PCT: float = 0.05

    @field_validator("LIMIT_PRICE_MAX_DISTANCE_PCT")
    @classmethod
    def validate_limit_price_max_distance(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("LIMIT_PRICE_MAX_DISTANCE_PCT must be >= 0")
        return v

    # Maximum time (seconds) a queued limit order is allowed to stay open.
    # After this timeout the engine will cancel the order and free the capital.
    QUEUED_ORDER_TIMEOUT_SECONDS: float = 300.0   # 5 minutes

    @field_validator("QUEUED_ORDER_TIMEOUT_SECONDS")
    @classmethod
    def validate_queued_order_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("QUEUED_ORDER_TIMEOUT_SECONDS must be positive")
        return v

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Data directory for logs, database, etc.
    DATA_DIR: str = "data"

    # Database
    DATABASE_PATH: str = "data/trading_bot.db"

    # News
    NEWS_ENABLED: bool = False
    NEWS_UPDATE_INTERVAL_MINUTES: int = 60

    # Fast news refresh for currently tracked symbols (minutes)
    NEWS_FAST_UPDATE_INTERVAL_MINUTES: int = 5

    NEWS_API_KEY: Optional[str] = None       # for NewsAPI.org
    TWITTER_BEARER_TOKEN: Optional[str] = None
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: str = "trade-ledger/1.0"
    NEWS_MAX_ARTICLES_PER_SYMBOL: int = 5
    NEWS_CACHE_TTL_SECONDS: int = 1800       # 30 minutes
    NEWS_HTTP_TIMEOUT_SECONDS: float = 30.0   # timeout for each news source HTTP request
    NEWS_INITIAL_FETCH_TIMEOUT_SECONDS: float = 60.0   # max seconds for initial news fetch on startup
    NEWS_RETENTION_SECONDS: int = 86400   # delete articles older than 24 hours

    # News-driven symbol discovery
    NEWS_SYMBOL_DISCOVERY_ENABLED: bool = False
    NEWS_SYMBOL_DISCOVERY_MAX_SYMBOLS: int = 5
    NEWS_SYMBOL_DISCOVERY_MIN_SENTIMENT: float = 0.3
    NEWS_SYMBOL_DISCOVERY_MIN_ARTICLES: int = 3

    # RSS-based ticker discovery (scan RSS feeds for symbols with TICKER_SUFFIX)
    NEWS_TICKER_DISCOVERY_ENABLED: bool = False
    NEWS_TICKER_DISCOVERY_MAX_SYMBOLS: int = 10

    # Facebook (Graph API)
    FACEBOOK_PAGE_ACCESS_TOKEN: Optional[str] = None
    FACEBOOK_PAGE_ID: Optional[str] = None
    FACEBOOK_POST_LIMIT: int = 5

    # RSS Feeds
    RSS_FEEDS: list[str] = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
        "https://www.investing.com/rss/news.rss",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.bloomberg.com/feeds/podcasts/etf_report.xml",
    ]

    # YouTube Data API v3
    YOUTUBE_API_KEY: Optional[str] = None
    YOUTUBE_MAX_RESULTS: int = 5

    # Google News RSS (free, no API key)
    GOOGLE_NEWS_MAX_ARTICLES: int = 5

    # StockTwits API
    STOCKTWITS_API_KEY: Optional[str] = None
    STOCKTWITS_MAX_POSTS: int = 5


    # Rate limiting for news providers
    NEWS_RATE_LIMIT_ENABLED: bool = True
    NEWS_RATE_LIMIT_PER_SOURCE_SECONDS: float = 1.0   # minimum seconds between requests to the same source

    # Telegram
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

    # Web
    WEB_HOST: str = "0.0.0.0"
    WEB_PORT: int = 8083

    # Logging
    LOG_LEVEL: str = "INFO"

    # Notification log control
    NOTIFICATION_LOG_ENABLED: bool = True

    # Notification verbosity: "all", "errors_only", "trades_only", or "none"
    NOTIFICATION_VERBOSITY: str = "all"

    @field_validator("NOTIFICATION_VERBOSITY")
    @classmethod
    def validate_notification_verbosity(cls, v: str) -> str:
        allowed = {"all", "errors_only", "trades_only", "none"}
        if v not in allowed:
            raise ValueError(f"NOTIFICATION_VERBOSITY must be one of {allowed}")
        return v

    def reload(self):
        """Reload settings from .env file and environment variables."""
        new_settings = self.__class__()
        for field in self.__fields__:
            setattr(self, field, getattr(new_settings, field))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
