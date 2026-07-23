import json
import uuid
import logging
from pydantic import field_validator, model_validator, PrivateAttr
from pydantic_settings import BaseSettings, NoDecode
from typing import Annotated, Optional

class Settings(BaseSettings):
    # Private attribute to hold reload callbacks
    _reload_callbacks: list = PrivateAttr(default_factory=list)

    # Trading mode
    TRADING_MODE: str = "paper"   # "paper" or "notify"

    # Market filtering
    TICKER_SUFFIX: str = ".MI"
    TARGET_COUNTRY: str = "italy"
    ETF_ITALY_KEYWORDS: str = "FTSE MIB, Italy, BTP"

    @field_validator("TARGET_COUNTRY")
    @classmethod
    def validate_target_country(cls, v: str) -> str:
        return v.lower()

    def register_reload_callback(self, callback):
        """Register a callback to be invoked when settings are reloaded."""
        if callback not in self._reload_callbacks:
            self._reload_callbacks.append(callback)

    def unregister_reload_callback(self, callback):
        """Unregister a previously registered reload callback."""
        if callback in self._reload_callbacks:
            self._reload_callbacks.remove(callback)

    # Paper trading initial balance (only used in paper mode)
    PAPER_INITIAL_BALANCE: float = 10000.0
    PAPER_BALANCE_CHANGED: bool = False

    # Dividend reinvestment
    REINVEST_DIVIDENDS: bool = False

    # Risk management check interval (seconds) – stop-loss/take-profit checks
    RISK_CHECK_INTERVAL_SECONDS: int = 120

    # Risk check interval for very long timeframes (>= 1 year).
    # Overrides RISK_CHECK_INTERVAL_SECONDS for positions on 1Y/3Y/5Y timeframes.
    RISK_CHECK_INTERVAL_VERY_LONG_TF_SECONDS: int = 3600  # 1 hour

    @field_validator("RISK_CHECK_INTERVAL_VERY_LONG_TF_SECONDS")
    @classmethod
    def validate_risk_check_interval_very_long_tf(cls, v: int) -> int:
        if v < 300:
            raise ValueError("RISK_CHECK_INTERVAL_VERY_LONG_TF_SECONDS must be >= 300")
        return v

    # Main engine loop polling interval (seconds). For medium/long-term trading,
    # a longer interval reduces CPU usage while still processing symbols at their
    # designated strategy intervals.
    ENGINE_LOOP_INTERVAL_SECONDS: int = 60

    # Symbol re-evaluation interval (seconds) – how often the LLM re-selects symbols
    SYMBOL_REEVALUATION_INTERVAL: int = 43200  # 12 hours

    @field_validator("SYMBOL_REEVALUATION_INTERVAL")
    @classmethod
    def validate_symbol_reevaluation_interval(cls, v: int) -> int:
        if v < 300:
            raise ValueError("SYMBOL_REEVALUATION_INTERVAL must be >= 300")
        return v

    # Portfolio rebalance settings
    # Enabled by default for medium/long-term trading to maintain diversification and risk targets.
    PORTFOLIO_REBALANCE_ENABLED: bool = True
    PORTFOLIO_REBALANCE_INTERVAL_SECONDS: int = 7776000  # 90 days

    # Fallback strategy evaluation interval (seconds) when no timeframe or no symbols
    DEFAULT_STRATEGY_INTERVAL: int = 3600  # 1 hour

    @field_validator("DEFAULT_STRATEGY_INTERVAL")
    @classmethod
    def validate_default_strategy_interval(cls, v: int) -> int:
        if v < 60:
            raise ValueError("DEFAULT_STRATEGY_INTERVAL must be >= 60")
        return v

    # Minimum symbol re-evaluation interval (seconds) – prevents rapid toggling
    MIN_SYMBOL_REEVALUATION_INTERVAL: int = 3600  # 1 hour

    @field_validator("MIN_SYMBOL_REEVALUATION_INTERVAL")
    @classmethod
    def validate_min_symbol_reevaluation_interval(cls, v: int) -> int:
        if v < 300:
            raise ValueError("MIN_SYMBOL_REEVALUATION_INTERVAL must be >= 300")
        return v

    # Maximum number of trades to keep in memory (prevents unbounded growth)
    MAX_TRADES_IN_MEMORY: int = 1000

    @field_validator("MAX_TRADES_IN_MEMORY")
    @classmethod
    def validate_max_trades_in_memory(cls, v: int) -> int:
        if v < 100:
            raise ValueError("MAX_TRADES_IN_MEMORY must be >= 100")
        return v

    # Initial delay before first symbol evaluation (seconds)
    # Allows WebSocket and Telegram bot to initialize before first LLM call
    INITIAL_EVALUATION_DELAY_SECONDS: int = 15

    # Base currency
    BASE_CURRENCY: str = "EUR"

    # Benchmark symbol for relative strength and market trend (e.g., SPY, QQQ)
    BENCHMARK_SYMBOL: str = "FTSEMIB.MI"

    # Max symbols to trade simultaneously
    MAX_SYMBOLS: int = 10
    # Maximum number of simultaneously open positions
    MAX_OPEN_POSITIONS: int = 10

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

    @field_validator("MAX_OPEN_POSITIONS")
    @classmethod
    def validate_max_open_positions(cls, v: int) -> int:
        if v < 1:
            raise ValueError("MAX_OPEN_POSITIONS must be at least 1")
        return v

    # Minimum number of symbols the LLM must select (when not pausing).
    # Set to 0 to let the LLM decide freely.
    MIN_SYMBOLS: int = 3

    @field_validator("MIN_SYMBOLS")
    @classmethod
    def validate_min_symbols(cls, v: int) -> int:
        if v < 0:
            raise ValueError("MIN_SYMBOLS must be >= 0")
        return v

    # Number of candidate symbols per LLM chunk call during symbol re-evaluation.
    # All candidates are evaluated in chunks of this size, then a final LLM call
    # aggregates the results. Lower values = smaller prompts but more LLM calls.
    LLM_CHUNK_SIZE: int = 20

    @field_validator("LLM_CHUNK_SIZE")
    @classmethod
    def validate_llm_chunk_size(cls, v: int) -> int:
        if v < 5:
            raise ValueError("LLM_CHUNK_SIZE must be at least 5")
        return v

    # Symbol selection limits
    SYMBOL_SELECTION_MIN_SENTIMENT: float = -1.0   # -1.0 = disabled

    # ETFs that are always included in the candidate pool (if tradable),
    # regardless of volume or composite score.
    ALWAYS_INCLUDE_ETFS: Annotated[list[str], NoDecode] = []

    # Additional base tickers (without suffix) that are always added to the discovery pool.
    # Useful for stocks that are not in the FTSE MIB or news feeds.
    ADDITIONAL_TICKERS: Annotated[list[str], NoDecode] = []

    # Enable FinanceDatabase ticker discovery
    FINANCEDATABASE_TICKER_DISCOVERY_ENABLED: bool = True

    # When True, symbols for which yfinance cannot determine the country are
    # dropped from the tradable assets list.  When False (default), such
    # symbols are kept because they were already discovered from Italian
    # sources (Wikipedia, FinanceDatabase, news feeds, etc.).
    COUNTRY_FILTER_STRICT: bool = False

    # Minimum composite score for a symbol to be used in the volume‑based fallback.
    # Symbols below this score are skipped even if they have high volume.
    FALLBACK_MIN_COMPOSITE_SCORE: float = 0.1

    # Composite score weights for symbol selection
    COMPOSITE_TREND_WEIGHT: float = 0.6
    COMPOSITE_SENTIMENT_WEIGHT: float = 0.4

    FALLBACK_MIN_24H_VOLUME: float = 0.0
    EXCLUDED_SYMBOLS: Annotated[list[str], NoDecode] = []

    # Maximum number of consecutive "keep paused" LLM decisions before the engine
    # force‑resumes trading with a reduced risk multiplier.
    PAUSE_MAX_CONSECUTIVE_KEEP: int = 3

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

    # Maximum account drawdown (%) at which force-resume is blocked.
    # When drawdown exceeds this threshold, the engine keeps trading paused
    # even after PAUSE_MAX_CONSECUTIVE_KEEP consecutive "keep paused" decisions.
    PAUSE_FORCE_RESUME_MAX_DRAWDOWN_PCT: float = 15.0

    @field_validator("PAUSE_FORCE_RESUME_MAX_DRAWDOWN_PCT")
    @classmethod
    def validate_pause_force_resume_max_drawdown(cls, v: float) -> float:
        if v < 0 or v > 100:
            raise ValueError("PAUSE_FORCE_RESUME_MAX_DRAWDOWN_PCT must be between 0 and 100")
        return v

    # Portfolio-level cooldown after consecutive losses
    PORTFOLIO_COOLDOWN_MAX_CONSEC_LOSSES: int = 5

    @field_validator("PORTFOLIO_COOLDOWN_MAX_CONSEC_LOSSES")
    @classmethod
    def validate_portfolio_cooldown_max_consec_losses(cls, v: int) -> int:
        if v < 1:
            raise ValueError("PORTFOLIO_COOLDOWN_MAX_CONSEC_LOSSES must be >= 1")
        return v

    PORTFOLIO_COOLDOWN_SECONDS: int = 3600  # 1 hour

    @field_validator("PORTFOLIO_COOLDOWN_SECONDS")
    @classmethod
    def validate_portfolio_cooldown_seconds(cls, v: int) -> int:
        if v < 60:
            raise ValueError("PORTFOLIO_COOLDOWN_SECONDS must be >= 60")
        return v

    # Maximum daily loss as a fraction of initial balance (0.05 = 5%).
    # When daily realized losses exceed this fraction of the initial balance,
    # trading is paused until the next calendar day.
    MAX_DAILY_LOSS_PCT: float = 0.05

    @field_validator("MAX_DAILY_LOSS_PCT")
    @classmethod
    def validate_max_daily_loss_pct(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("MAX_DAILY_LOSS_PCT must be between 0.0 and 1.0")
        return v

    # Minimum LLM pause duration (seconds) – LLM cannot resume before this
    MIN_LLM_PAUSE_DURATION: int = 1800

    @field_validator("MIN_LLM_PAUSE_DURATION")
    @classmethod
    def validate_min_llm_pause_duration(cls, v: int) -> int:
        if v < 300:
            raise ValueError("MIN_LLM_PAUSE_DURATION must be >= 300")
        return v

    # Hard maximum unrealized loss percentage that forces immediate exit
    # regardless of LLM stop-loss review decisions (0.15 = 15%)
    HARD_MAX_LOSS_PCT: float = 0.15

    @field_validator("HARD_MAX_LOSS_PCT")
    @classmethod
    def validate_hard_max_loss_pct(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("HARD_MAX_LOSS_PCT must be between 0.0 and 1.0")
        return v

    # Timeframe-specific hard max loss overrides. If > 0, these take precedence
    # over HARD_MAX_LOSS_PCT for positions on the corresponding timeframe.
    HARD_MAX_LOSS_PCT_1H: float = 0.10
    HARD_MAX_LOSS_PCT_1D: float = 0.12
    HARD_MAX_LOSS_PCT_1W: float = 0.15
    HARD_MAX_LOSS_PCT_1M: float = 0.20
    HARD_MAX_LOSS_PCT_3M: float = 0.25
    HARD_MAX_LOSS_PCT_6M_1Y: float = 0.30

    @field_validator("HARD_MAX_LOSS_PCT_1H", "HARD_MAX_LOSS_PCT_1D", "HARD_MAX_LOSS_PCT_1W",
                     "HARD_MAX_LOSS_PCT_1M", "HARD_MAX_LOSS_PCT_3M", "HARD_MAX_LOSS_PCT_6M_1Y")
    @classmethod
    def validate_tf_hard_max_loss_pct(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Timeframe-specific HARD_MAX_LOSS_PCT must be between 0.0 and 1.0")
        return v

    # BTP-specific hard max loss (0.05 = 5%) — BTPs are lower volatility than stocks
    BTP_HARD_MAX_LOSS_PCT: float = 0.05

    @field_validator("BTP_HARD_MAX_LOSS_PCT")
    @classmethod
    def validate_btp_hard_max_loss_pct(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("BTP_HARD_MAX_LOSS_PCT must be between 0.0 and 1.0")
        return v

    # Timeframe-specific BTP hard max loss overrides. If > 0, these take precedence
    # over BTP_HARD_MAX_LOSS_PCT for positions on the corresponding timeframe.
    BTP_HARD_MAX_LOSS_PCT_1H: float = 0.03
    BTP_HARD_MAX_LOSS_PCT_1D: float = 0.04
    BTP_HARD_MAX_LOSS_PCT_1W: float = 0.05
    BTP_HARD_MAX_LOSS_PCT_1M: float = 0.06
    BTP_HARD_MAX_LOSS_PCT_3M: float = 0.08
    BTP_HARD_MAX_LOSS_PCT_6M_1Y: float = 0.10

    @field_validator("BTP_HARD_MAX_LOSS_PCT_1H", "BTP_HARD_MAX_LOSS_PCT_1D", "BTP_HARD_MAX_LOSS_PCT_1W",
                     "BTP_HARD_MAX_LOSS_PCT_1M", "BTP_HARD_MAX_LOSS_PCT_3M", "BTP_HARD_MAX_LOSS_PCT_6M_1Y")
    @classmethod
    def validate_tf_btp_hard_max_loss_pct(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Timeframe-specific BTP_HARD_MAX_LOSS_PCT must be between 0.0 and 1.0")
        return v

    # BTP-specific maximum take-profit percentage (0.03 = 3%) — BTPs trade in
    # narrow ranges, so take-profit targets should be much smaller than stocks.
    BTP_MAX_TAKE_PROFIT_PCT: float = 0.03

    @field_validator("BTP_MAX_TAKE_PROFIT_PCT")
    @classmethod
    def validate_btp_max_take_profit_pct(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("BTP_MAX_TAKE_PROFIT_PCT must be between 0.0 and 1.0")
        return v

    # BTP-specific maximum stop-loss percentage (0.03 = 3%) — BTPs trade in
    # narrow ranges, so stop-loss targets should be much smaller than stocks.
    BTP_MAX_STOP_LOSS_PCT: float = 0.03

    @field_validator("BTP_MAX_STOP_LOSS_PCT")
    @classmethod
    def validate_btp_max_stop_loss_pct(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("BTP_MAX_STOP_LOSS_PCT must be between 0.0 and 1.0")
        return v

    # Maximum LLM stop-loss reviews before force-selling
    MAX_STOP_LOSS_REVIEWS: int = 10

    @field_validator("MAX_STOP_LOSS_REVIEWS")
    @classmethod
    def validate_max_stop_loss_reviews(cls, v: int) -> int:
        if v < 1 or v > 50:
            raise ValueError("MAX_STOP_LOSS_REVIEWS must be between 1 and 50")
        return v

    # Maximum LLM take-profit reviews before force-selling
    MAX_TAKE_PROFIT_REVIEWS: int = 10

    @field_validator("MAX_TAKE_PROFIT_REVIEWS")
    @classmethod
    def validate_max_take_profit_reviews(cls, v: int) -> int:
        if v < 1 or v > 50:
            raise ValueError("MAX_TAKE_PROFIT_REVIEWS must be between 1 and 50")
        return v

    # Maximum LLM stop-loss reviews for long-term timeframes (>= 1 month).
    # Overrides MAX_STOP_LOSS_REVIEWS when the position timeframe is >= 1 month.
    LONG_TERM_MAX_STOP_LOSS_REVIEWS: int = 5

    @field_validator("LONG_TERM_MAX_STOP_LOSS_REVIEWS")
    @classmethod
    def validate_long_term_max_stop_loss_reviews(cls, v: int) -> int:
        if v < 1 or v > 50:
            raise ValueError("LONG_TERM_MAX_STOP_LOSS_REVIEWS must be between 1 and 50")
        return v

    # Maximum LLM stop-loss reviews for weekly timeframes (>= 1 week, < 1 month).
    # Overrides MAX_STOP_LOSS_REVIEWS when the position timeframe is >= 1 week.
    WEEKLY_MAX_STOP_LOSS_REVIEWS: int = 7

    @field_validator("WEEKLY_MAX_STOP_LOSS_REVIEWS")
    @classmethod
    def validate_weekly_max_stop_loss_reviews(cls, v: int) -> int:
        if v < 1 or v > 50:
            raise ValueError("WEEKLY_MAX_STOP_LOSS_REVIEWS must be between 1 and 50")
        return v

    # Timeframe threshold (in seconds) for reducing max stop-loss reviews
    # (2592000 = 30 days / 1 month). Positions on timeframes >= this value
    # get a reduced review cap to prevent excessive loss accumulation.
    LONG_TERM_TF_SECONDS: int = 2_592_000

    @field_validator("LONG_TERM_TF_SECONDS")
    @classmethod
    def validate_long_term_tf_seconds(cls, v: int) -> int:
        if v < 86400:
            raise ValueError("LONG_TERM_TF_SECONDS must be >= 86400 (1 day)")
        return v

    # Maximum number of consecutive partial take-profit reviews before force-executing
    MAX_PARTIAL_TP_REVIEWS: int = 10

    # Maximum number of consecutive dust sweep reviews before force-selling
    MAX_DUST_SWEEP_REVIEWS: int = 10

    # Maximum time (seconds) to wait for a native stop-loss order to fill
    # after the stop price is reached before falling back to a manual market sell.
    NATIVE_STOP_FILL_TIMEOUT_SECONDS: int = 300

    @field_validator("NATIVE_STOP_FILL_TIMEOUT_SECONDS")
    @classmethod
    def validate_native_stop_fill_timeout(cls, v: int) -> int:
        if v < 10:
            raise ValueError("NATIVE_STOP_FILL_TIMEOUT_SECONDS must be >= 10")
        return v

    # Maximum time (seconds) dust can be kept before auto-selling
    DUST_KEEP_TIMEOUT_SECONDS: int = 604800  # 7 days

    @field_validator("DUST_KEEP_TIMEOUT_SECONDS")
    @classmethod
    def validate_dust_keep_timeout(cls, v: int) -> int:
        if v < 3600:
            raise ValueError("DUST_KEEP_TIMEOUT_SECONDS must be >= 3600")
        return v

    # Maximum position age as a multiple of the original max_hold_time_seconds.
    # If the LLM keeps extending max_hold_time_seconds, the position is force-closed
    # once its age exceeds this multiplier × the original max hold time.
    # Set to 0 to disable the maximum position age safeguard.
    MAX_POSITION_AGE_MULTIPLIER: float = 2.0

    @field_validator("MAX_POSITION_AGE_MULTIPLIER")
    @classmethod
    def validate_max_position_age_multiplier(cls, v: float) -> float:
        if v < 0:
            raise ValueError("MAX_POSITION_AGE_MULTIPLIER must be >= 0")
        return v

    # Minimum seconds between condition-triggered re-evaluations
    TRIGGERED_REEVALUATION_COOLDOWN: int = 1800

    @field_validator("TRIGGERED_REEVALUATION_COOLDOWN")
    @classmethod
    def validate_triggered_reevaluation_cooldown(cls, v: int) -> int:
        if v < 300:
            raise ValueError("TRIGGERED_REEVALUATION_COOLDOWN must be >= 300")
        return v

    # Minimum entry condition timeout as a multiple of the candle timeframe.
    # e.g., 2.0 means the timeout must be at least 2 × the candle period.
    ENTRY_CONDITION_MIN_TIMEOUT_MULT: float = 2.0

    @field_validator("ENTRY_CONDITION_MIN_TIMEOUT_MULT")
    @classmethod
    def validate_entry_condition_min_timeout_mult(cls, v: float) -> float:
        if v < 1.0:
            raise ValueError("ENTRY_CONDITION_MIN_TIMEOUT_MULT must be >= 1.0")
        return v

    # Strategy evaluation interval multiplier (medium/long-term)
    # 1 means evaluate every candle period (e.g., 1w → every week, 1M → every month).
    STRATEGY_INTERVAL_MULTIPLIER: int = 1

    # Evaluation intervals per timeframe (seconds)
    EVAL_INTERVAL_1H: int = 3600      # 1 hour
    EVAL_INTERVAL_1D: int = 1800      # 30 minutes
    EVAL_INTERVAL_1W: int = 3600      # 1 hour
    EVAL_INTERVAL_1M: int = 86400     # 1 day
    EVAL_INTERVAL_3M: int = 172800    # 2 days
    EVAL_INTERVAL_6M_1Y: int = 604800 # 1 week
    EVAL_INTERVAL_DEFAULT: int = 3600 # 1 hour default

    # Maximum interval (seconds) to skip LLM evaluation before forcing a re-evaluation
    MAX_SKIP_INTERVAL_SECONDS: int = 604800  # 7 days

    @field_validator("MAX_SKIP_INTERVAL_SECONDS")
    @classmethod
    def validate_max_skip_interval(cls, v: int) -> int:
        if v < 3600:
            raise ValueError("MAX_SKIP_INTERVAL_SECONDS must be >= 3600")
        return v

    # Active period settings – during these windows, the bot evaluates more frequently
    # to catch opening/closing opportunities.
    MARKET_OPEN_ACTIVE_MINUTES: int = 60

    @field_validator("MARKET_OPEN_ACTIVE_MINUTES")
    @classmethod
    def validate_market_open_active_minutes(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MARKET_OPEN_ACTIVE_MINUTES must be > 0")
        return v

    MARKET_CLOSE_ACTIVE_MINUTES: int = 30

    @field_validator("MARKET_CLOSE_ACTIVE_MINUTES")
    @classmethod
    def validate_market_close_active_minutes(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MARKET_CLOSE_ACTIVE_MINUTES must be > 0")
        return v

    ACTIVE_PERIOD_INTERVAL_SECONDS: int = 900  # 15 minutes

    # Entry signal monitor interval (seconds) – how often to scan for entry signals.
    # For medium/long-term, 15 minutes is sufficient.
    ENTRY_SIGNAL_CHECK_INTERVAL_SECONDS: int = 900

    # Cooldown between entry-signal forced LLM evaluations (seconds)
    ENTRY_SIGNAL_COOLDOWN_SECONDS: int = 30

    @field_validator("ENTRY_SIGNAL_COOLDOWN_SECONDS")
    @classmethod
    def validate_entry_signal_cooldown_seconds(cls, v: int) -> int:
        if v < 0:
            raise ValueError("ENTRY_SIGNAL_COOLDOWN_SECONDS must be >= 0")
        return v

    # OHLCV timeframes for multi-timeframe analysis
    # Default order is longest to shortest to ensure larger timeframes are fetched first
    OHLCV_TIMEFRAMES: Annotated[list[str], NoDecode] = ["5Y", "3Y", "1Y", "6M", "3M", "1M", "1w", "1d", "1h"]

    # Market data download interval (seconds)
    MARKET_DATA_REFRESH_SECONDS: int = 900

    # Background quote refresh interval (seconds) – how often to fetch quotes
    # for all tradable assets to keep Redis and the database warm.
    # 900 seconds (15 minutes) is suitable for medium/long-term trading.
    QUOTE_REFRESH_INTERVAL_SECONDS: int = 900

    # Maximum age (seconds) for a quote to be considered fresh enough for trading.
    # The actual threshold is scaled by the symbol's timeframe (longer timeframes
    # allow staler quotes). Set to 0 to disable the staleness guard.
    QUOTE_MAX_STALENESS_SECONDS: float = 3600.0  # 1 hour

    @field_validator("QUOTE_MAX_STALENESS_SECONDS")
    @classmethod
    def validate_quote_max_staleness(cls, v: float) -> float:
        if v < 0:
            raise ValueError("QUOTE_MAX_STALENESS_SECONDS must be >= 0")
        return v

    # Full asset OHLCV download interval (seconds) – how often to backfill
    # OHLCV data for ALL tradable assets (stocks, ETFs, BTPs), not just the
    # currently selected symbols.
    FULL_ASSET_OHLCV_DOWNLOAD_INTERVAL_SECONDS: int = 21600  # 6 hours

    # Full asset news download interval (seconds) – how often to pre‑fetch news
    # for ALL tradable assets (stocks, ETFs, BTPs), not just the currently
    # selected symbols.
    FULL_ASSET_NEWS_DOWNLOAD_INTERVAL_SECONDS: int = 10800  # 3 hours

    # OHLCV download staggering (delay between symbols)
    OHLCV_DOWNLOAD_SYMBOL_DELAY_SECONDS: float = 2.0

    @field_validator("OHLCV_DOWNLOAD_SYMBOL_DELAY_SECONDS")
    @classmethod
    def validate_ohlcv_download_symbol_delay(cls, v: float) -> float:
        if v < 0:
            raise ValueError("OHLCV_DOWNLOAD_SYMBOL_DELAY_SECONDS must be >= 0")
        return v

    # Maximum number of OHLCV candles to insert in a single backfill call.
    # Prevents memory exhaustion and timeouts when backfilling large ranges.
    BACKFILL_MAX_CANDLES_PER_CALL: int = 5000

    @field_validator("BACKFILL_MAX_CANDLES_PER_CALL")
    @classmethod
    def validate_backfill_max_candles(cls, v: int) -> int:
        if v < 100:
            raise ValueError("BACKFILL_MAX_CANDLES_PER_CALL must be >= 100")
        return v

    # Maximum number of OHLCV gaps to fill per cycle to avoid rate limits
    MAX_GAPS_PER_CYCLE: int = 20

    # Maximum number of backtest variants per cycle
    MAX_BACKTEST_VARIANTS: int = 10

    @field_validator("MAX_BACKTEST_VARIANTS")
    @classmethod
    def validate_max_backtest_variants(cls, v: int) -> int:
        if v < 1 or v > 50:
            raise ValueError("MAX_BACKTEST_VARIANTS must be between 1 and 50")
        return v

    # Maximum number of concurrent backtest variants to run in parallel
    MAX_CONCURRENT_BACKTESTS: int = 8

    @field_validator("MAX_CONCURRENT_BACKTESTS")
    @classmethod
    def validate_max_concurrent_backtests(cls, v: int) -> int:
        if v < 1 or v > 32:
            raise ValueError("MAX_CONCURRENT_BACKTESTS must be between 1 and 32")
        return v

    # Thread pool sizes for dedicated executors
    DB_EXECUTOR_WORKERS: int = 10
    DOWNLOAD_EXECUTOR_WORKERS: int = 10
    QUOTE_EXECUTOR_WORKERS: int = 20  # increased from 12 to handle yfinance hangs

    @field_validator("DB_EXECUTOR_WORKERS", "DOWNLOAD_EXECUTOR_WORKERS", "QUOTE_EXECUTOR_WORKERS")
    @classmethod
    def validate_executor_workers(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Executor worker count must be at least 1")
        return v

    # Async semaphore concurrency limits
    EXCHANGE_SEMAPHORE_LIMIT: int = 10
    NEWS_SEMAPHORE_LIMIT: int = 5
    INDICATOR_SEMAPHORE_LIMIT: int = 4
    DOWNLOAD_SEMAPHORE_LIMIT: int = 5
    SYMBOL_PROCESSING_SEMAPHORE_LIMIT: int = 3
    FORCE_DOWNLOAD_ALL_CONCURRENCY: int = 2
    FORCE_DOWNLOAD_TRACKED_CONCURRENCY: int = 10
    FULL_DOWNLOAD_CONCURRENCY: int = 2

    @field_validator(
        "EXCHANGE_SEMAPHORE_LIMIT", "NEWS_SEMAPHORE_LIMIT", "INDICATOR_SEMAPHORE_LIMIT",
        "DOWNLOAD_SEMAPHORE_LIMIT", "SYMBOL_PROCESSING_SEMAPHORE_LIMIT",
        "FORCE_DOWNLOAD_ALL_CONCURRENCY", "FORCE_DOWNLOAD_TRACKED_CONCURRENCY",
        "FULL_DOWNLOAD_CONCURRENCY"
    )
    @classmethod
    def validate_semaphore_limits(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Semaphore limits must be >= 1")
        return v

    @field_validator("OHLCV_TIMEFRAMES")
    @classmethod
    def validate_ohlcv_timeframes(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list) or not all(isinstance(tf, str) for tf in v):
            raise ValueError("OHLCV_TIMEFRAMES must be a list of strings")
        allowed = {"1h", "1d", "1w", "1M", "3M", "6M", "1Y", "3Y", "5Y"}
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(f"OHLCV_TIMEFRAMES contains unsupported timeframes: {invalid}. Allowed medium/long-term timeframes: {allowed}")
        return v

    # Number of days of OHLCV data to retain and use for backtest / LLM analysis.
    # For long-term trading (1d, 1w, 1M, 3M, 6M, 1Y), we need at least 5-10 years of data.
    OHLCV_RETENTION_DAYS: int = 3650

    @field_validator("OHLCV_RETENTION_DAYS")
    @classmethod
    def validate_ohlcv_retention_days(cls, v: int) -> int:
        if v < 30:
            raise ValueError("OHLCV_RETENTION_DAYS must be at least 30 for medium/long-term analysis")
        return v

    # Yahoo Finance fallback for missing bid/ask quotes
    YAHOO_FINANCE_ENABLED: bool = True
    YAHOO_FINANCE_CACHE_SECONDS: int = 60

    # yfinance rate limiting and proxy settings
    YF_RATE_LIMIT_ENABLED: bool = True
    YF_RATE_LIMIT_MAX_REQUESTS: int = 30
    YF_RATE_LIMIT_WINDOW_SECONDS: int = 60
    HTTP_PROXY_ENABLED: bool = False
    HTTP_PROXIES: Annotated[list[str], NoDecode] = []

    # Alpha Vantage
    ALPHAVANTAGE_ENABLED: bool = False
    ALPHAVANTAGE_API_KEY: str = ""
    ALPHAVANTAGE_RATE_LIMIT_PER_MIN: int = 5  # free tier: 5 req/min

    # IEX Cloud
    IEX_ENABLED: bool = False
    IEX_API_KEY: str = ""




    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        if v not in ("ollama", "openai", "g4f"):
            raise ValueError("LLM_PROVIDER must be 'ollama', 'openai', or 'g4f'")
        return v

    @field_validator(
        "ALWAYS_INCLUDE_ETFS", "ADDITIONAL_TICKERS", "EXCLUDED_SYMBOLS",
        "HTTP_PROXIES", "RSS_FEEDS", "LLM_PROMPT_CACHING_PROVIDERS",
        "LLM_PROMPT_CACHING_CONTROL_PROVIDERS", "OHLCV_TIMEFRAMES",
        "OPENAI_FALLBACK_MODEL", "OPENAI_MIND_FALLBACK_MODEL",
        "OPENAI_ACTUATOR_FALLBACK_MODEL", "OPENAI_WEAK_FALLBACK_MODEL",
        "OLLAMA_FALLBACK_MODEL", "OLLAMA_MIND_FALLBACK_MODEL",
        "OLLAMA_ACTUATOR_FALLBACK_MODEL", "OLLAMA_WEAK_FALLBACK_MODEL",
        "OPENAI_MODEL", "OPENAI_MIND_MODEL", "OPENAI_ACTUATOR_MODEL", "OPENAI_WEAK_MODEL",
        "OLLAMA_MODEL", "OLLAMA_MIND_MODEL", "OLLAMA_ACTUATOR_MODEL", "OLLAMA_WEAK_MODEL",
        "AOL_LLM_MODEL",
        mode="before"
    )
    @classmethod
    def _parse_json_list(cls, v):
        if isinstance(v, str):
            if not v.strip():
                return []
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return []
        return v

    @model_validator(mode="after")
    def set_database_backend(self):
        # Determine if PostgreSQL should be used
        if all([self.DB_HOST, self.DB_PORT, self.DB_NAME, self.DB_USER, self.DB_PASSWORD]):
            self.DATABASE_BACKEND = "postgresql"
        else:
            self.DATABASE_BACKEND = "sqlite"
        return self

    @model_validator(mode="after")
    def set_database_path(self):
        if "DATABASE_PATH" not in self.model_fields_set:
            if self.DATABASE_BACKEND == "sqlite":
                if self.TRADING_MODE == "paper":
                    self.DATABASE_PATH = "data/paper.db"
                else:
                    self.DATABASE_PATH = "data/notify.db"
        return self

    # Ollama
    OLLAMA_BASE_URL: Optional[str] = None
    OLLAMA_MODEL: Annotated[list[str], NoDecode] = []
    OLLAMA_API_KEY: Optional[str] = None

    # LLM Provider selection
    LLM_PROVIDER: str = "ollama"   # "ollama" or "openai"

    # OpenAI-compatible API
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_MODEL: Annotated[list[str], NoDecode] = []

    # G4F (gpt4free)
    G4F_BASE_URL: Optional[str] = None
    G4F_API_KEY: Optional[str] = None
    # Note: g4f dynamically manages providers and models from code, so model names are not configured here.
    # Max input tokens for main models (defaults: 128K for mind/actuator, 64K for weak)
    G4F_MIND_MAX_INPUT_TOKENS: int = 128_000
    G4F_ACTUATOR_MAX_INPUT_TOKENS: int = 128_000
    G4F_WEAK_MAX_INPUT_TOKENS: int = 64_000

    # Max input tokens for fallback models (defaults: 16K for mind/actuator, 8K for weak)
    G4F_MIND_FALLBACK_MAX_INPUT_TOKENS: int = 16_384
    G4F_ACTUATOR_FALLBACK_MAX_INPUT_TOKENS: int = 16_384
    G4F_WEAK_FALLBACK_MAX_INPUT_TOKENS: int = 8_192

    # Mind model (complex reasoning: symbol selection, strategy generation)
    OLLAMA_MIND_MODEL: Annotated[list[str], NoDecode] = []
    OPENAI_MIND_MODEL: Annotated[list[str], NoDecode] = []

    # Actuator model (fast, time‑critical decisions: stop‑loss/take‑profit reviews, corrections)
    OLLAMA_ACTUATOR_MODEL: Annotated[list[str], NoDecode] = []
    OPENAI_ACTUATOR_MODEL: Annotated[list[str], NoDecode] = []

    # Per‑role provider overrides (empty = use global LLM_PROVIDER)
    LLM_MIND_PROVIDER: str = ""
    LLM_ACTUATOR_PROVIDER: str = ""

    # Enable automatic fallback to the other LLM provider if the primary fails.
    # Default: True to ensure the bot remains operational if the primary LLM provider fails.
    LLM_FALLBACK_ENABLED: bool = True
    # Prompt caching for DeepSeek (and other providers that support it)
    LLM_PROMPT_CACHING_ENABLED: bool = True
    LLM_PROMPT_CACHING_PROVIDERS: Annotated[list[str], NoDecode] = ["deepseek", "ollama", "openai"]
    # Providers that support the cache_control field (e.g., DeepSeek).
    # Only these providers will receive the cache_control header.
    LLM_PROMPT_CACHING_CONTROL_PROVIDERS: Annotated[list[str], NoDecode] = []

    # Cache version key to invalidate LLM cache on settings reload.
    # Automatically generated on instantiation; changes when settings.reload() is called.
    LLM_CACHE_VERSION: str = str(uuid.uuid4())

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

    # Weak model (minor tasks: summarization, etc.) - used to save tokens on mind/actuator models
    LLM_WEAK_PROVIDER: str = ""  # empty = use global LLM_PROVIDER

    # Per-role OpenAI settings for weak model (empty or None = use global OPENAI_*)
    OPENAI_WEAK_MODEL: Annotated[list[str], NoDecode] = []
    OPENAI_WEAK_API_KEY: Optional[str] = None
    OPENAI_WEAK_BASE_URL: Optional[str] = None

    # Per-role Ollama settings for weak model (empty or None = use global OLLAMA_*)
    OLLAMA_WEAK_MODEL: Annotated[list[str], NoDecode] = []
    OLLAMA_WEAK_BASE_URL: Optional[str] = None
    OLLAMA_WEAK_API_KEY: Optional[str] = None

    # Per-role temperature override for weak model
    LLM_WEAK_TEMPERATURE: Optional[str] = None

    @field_validator("LLM_WEAK_TEMPERATURE")
    @classmethod
    def validate_weak_temperature(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v.strip() == "":
            return None
        Settings.parse_temperature_range(v)  # raises ValueError if invalid
        return v

    # Always-Online Locally hosted weak model (last resort fallback)
    AOL_LLM_PROVIDER: str = ""
    AOL_LLM_MODEL: Annotated[list[str], NoDecode] = []
    AOL_LLM_API_KEY: Optional[str] = None
    AOL_BASE_URL: Optional[str] = None
    AOL_MAX_INPUT_TOKENS: int = 16_384

    # Dedicated timeout (seconds) for AOL (Always-Online) last-resort fallback calls.
    AOL_TIMEOUT: float = 120.0

    @field_validator("AOL_TIMEOUT")
    @classmethod
    def validate_aol_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("AOL_TIMEOUT must be positive")
        return v

    # Thinking mode (reasoning) control per model type.
    # When False, sends reasoning_effort="low" to the API to minimize deep thinking.
    # Mind: KEEP enabled — deep financial analysis, critical trading decisions.
    # Actuator: DISABLE — fast, time-critical decisions (SL/TP reviews, pause/resume).
    # Weak: DISABLE — summarization and simple text tasks.
    LLM_MIND_THINKING_ENABLED: bool = True
    LLM_ACTUATOR_THINKING_ENABLED: bool = False
    LLM_WEAK_THINKING_ENABLED: bool = False

    # Max input tokens for main models (defaults: 1M for mind/actuator, 128K for weak)
    OPENAI_MIND_MAX_INPUT_TOKENS: int = 128_000
    OPENAI_ACTUATOR_MAX_INPUT_TOKENS: int = 128_000
    OPENAI_WEAK_MAX_INPUT_TOKENS: int = 64_000
    OLLAMA_MIND_MAX_INPUT_TOKENS: int = 128_000
    OLLAMA_ACTUATOR_MAX_INPUT_TOKENS: int = 128_000
    OLLAMA_WEAK_MAX_INPUT_TOKENS: int = 64_000

    # Max input tokens for fallback models (defaults: 16K for mind/actuator, 8K for weak)
    OPENAI_MIND_FALLBACK_MAX_INPUT_TOKENS: int = 16_384
    OPENAI_ACTUATOR_FALLBACK_MAX_INPUT_TOKENS: int = 16_384
    OPENAI_WEAK_FALLBACK_MAX_INPUT_TOKENS: int = 16384
    OLLAMA_MIND_FALLBACK_MAX_INPUT_TOKENS: int = 16_384
    OLLAMA_ACTUATOR_FALLBACK_MAX_INPUT_TOKENS: int = 16_384
    OLLAMA_WEAK_FALLBACK_MAX_INPUT_TOKENS: int = 8_192

    # Fallback provider settings (empty = use global LLM_FALLBACK_PROVIDER)
    LLM_FALLBACK_PROVIDER: str = ""
    LLM_MIND_FALLBACK_PROVIDER: str = ""
    LLM_ACTUATOR_FALLBACK_PROVIDER: str = ""
    LLM_WEAK_FALLBACK_PROVIDER: str = ""

    # OpenAI fallback settings (empty or None = use global OPENAI_FALLBACK_*)
    OPENAI_FALLBACK_MODEL: Annotated[list[str], NoDecode] = []
    OPENAI_FALLBACK_BASE_URL: Optional[str] = None
    OPENAI_FALLBACK_API_KEY: Optional[str] = None

    OPENAI_MIND_FALLBACK_MODEL: Annotated[list[str], NoDecode] = []
    OPENAI_MIND_FALLBACK_BASE_URL: Optional[str] = None
    OPENAI_MIND_FALLBACK_API_KEY: Optional[str] = None

    OPENAI_ACTUATOR_FALLBACK_MODEL: Annotated[list[str], NoDecode] = []
    OPENAI_ACTUATOR_FALLBACK_BASE_URL: Optional[str] = None
    OPENAI_ACTUATOR_FALLBACK_API_KEY: Optional[str] = None

    OPENAI_WEAK_FALLBACK_MODEL: Annotated[list[str], NoDecode] = []
    OPENAI_WEAK_FALLBACK_BASE_URL: Optional[str] = None
    OPENAI_WEAK_FALLBACK_API_KEY: Optional[str] = None

    # Ollama fallback settings (empty or None = use global OLLAMA_FALLBACK_*)
    OLLAMA_FALLBACK_MODEL: Annotated[list[str], NoDecode] = []
    OLLAMA_FALLBACK_BASE_URL: Optional[str] = None
    OLLAMA_FALLBACK_API_KEY: Optional[str] = None

    OLLAMA_MIND_FALLBACK_MODEL: Annotated[list[str], NoDecode] = []
    OLLAMA_MIND_FALLBACK_BASE_URL: Optional[str] = None
    OLLAMA_MIND_FALLBACK_API_KEY: Optional[str] = None

    OLLAMA_ACTUATOR_FALLBACK_MODEL: Annotated[list[str], NoDecode] = []
    OLLAMA_ACTUATOR_FALLBACK_BASE_URL: Optional[str] = None
    OLLAMA_ACTUATOR_FALLBACK_API_KEY: Optional[str] = None

    OLLAMA_WEAK_FALLBACK_MODEL: Annotated[list[str], NoDecode] = []
    OLLAMA_WEAK_FALLBACK_BASE_URL: Optional[str] = None
    OLLAMA_WEAK_FALLBACK_API_KEY: Optional[str] = None

    # Threshold for choosing the "mind" model tier over "actuator".
    # Represents the minimum normalized weighted complexity score (0.0 to 1.0)
    # required to trigger the "mind" model. Lower values = more frequent use
    # of the "mind" model (higher cost/quality); higher values = more frequent
    # use of the "actuator" model (lower cost/faster).
    LLM_MIND_MODEL_THRESHOLD: float = 0.75

    @field_validator("LLM_MIND_MODEL_THRESHOLD")
    @classmethod
    def validate_llm_mind_model_threshold(cls, v: float) -> float:
        if v < 0.05 or v > 0.95:
            raise ValueError("LLM_MIND_MODEL_THRESHOLD must be between 0.05 and 0.95")
        return v

    @field_validator("LLM_PROMPT_CACHING_ENABLED")
    @classmethod
    def validate_llm_prompt_caching_enabled(cls, v: bool) -> bool:
        return v

    @field_validator("LLM_PROMPT_CACHING_PROVIDERS")
    @classmethod
    def validate_llm_prompt_caching_providers(cls, v: list[str]) -> list[str]:
        return v

    @field_validator("LLM_PROMPT_CACHING_CONTROL_PROVIDERS")
    @classmethod
    def validate_llm_prompt_caching_control_providers(cls, v: list[str]) -> list[str]:
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
    LLM_TIMEOUT: float = 300.0

    @field_validator("LLM_TIMEOUT")
    @classmethod
    def validate_llm_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("LLM_TIMEOUT must be positive")
        return v

    # LLM timeout (seconds) for time-critical actuator calls (stop-loss/take-profit reviews, pause/resume).
    # This should be shorter than LLM_TIMEOUT to prevent the market from moving too much during the wait.
    LLM_ACTUATOR_TIMEOUT: float = 60.0

    @field_validator("LLM_ACTUATOR_TIMEOUT")
    @classmethod
    def validate_llm_actuator_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("LLM_ACTUATOR_TIMEOUT must be positive")
        return v

    # LLM timeout (seconds) for fallback model calls.
    # Fallback models are often weaker/slower, so a longer timeout is recommended.
    LLM_FALLBACK_TIMEOUT: float = 600.0

    @field_validator("LLM_FALLBACK_TIMEOUT")
    @classmethod
    def validate_llm_fallback_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("LLM_FALLBACK_TIMEOUT must be positive")
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

    # Minimum viable trade amount (in base currency) to ensure round-trip fees
    # are a reasonable percentage of trade value. The LLM can override this
    # dynamically via the "min_viable_trade_amount" field in its JSON response.
    # Set to 0 to disable the check.
    MIN_VIABLE_TRADE_AMOUNT: float = 500.0

    @field_validator("MIN_VIABLE_TRADE_AMOUNT")
    @classmethod
    def validate_min_viable_trade_amount(cls, v: float) -> float:
        if v < 0:
            raise ValueError("MIN_VIABLE_TRADE_AMOUNT must be >= 0")
        return v

    # Maximum time (seconds) a queued limit order is allowed to stay open.
    # After this timeout the engine will cancel the order and free the capital.
    QUEUED_ORDER_TIMEOUT_SECONDS: float = 900.0   # 15 minutes (medium/long-term)

    @field_validator("QUEUED_ORDER_TIMEOUT_SECONDS")
    @classmethod
    def validate_queued_order_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("QUEUED_ORDER_TIMEOUT_SECONDS must be positive")
        return v

    LLM_CACHE_TTL: int = 1800

    @field_validator("LLM_CACHE_TTL")
    @classmethod
    def validate_llm_cache_ttl(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("LLM_CACHE_TTL must be positive")
        return v

    LLM_MIND_CACHE_TTL: int = 3600

    @field_validator("LLM_MIND_CACHE_TTL")
    @classmethod
    def validate_llm_mind_cache_ttl(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("LLM_MIND_CACHE_TTL must be positive")
        return v

    ORPHANED_ORDER_TIMEOUT_SECONDS: float = 600.0

    @field_validator("ORPHANED_ORDER_TIMEOUT_SECONDS")
    @classmethod
    def validate_orphaned_order_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("ORPHANED_ORDER_TIMEOUT_SECONDS must be positive")
        return v

    SYMBOL_EVALUATION_DELAY_SECONDS: float = 1.0

    @field_validator("SYMBOL_EVALUATION_DELAY_SECONDS")
    @classmethod
    def validate_symbol_evaluation_delay(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("SYMBOL_EVALUATION_DELAY_SECONDS must be positive")
        return v

    BACKTEST_MIN_CANDLES: int = 5

    @field_validator("BACKTEST_MIN_CANDLES")
    @classmethod
    def validate_backtest_min_candles(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("BACKTEST_MIN_CANDLES must be positive")
        return v

    # Backtest settings
    BACKTEST_SLIPPAGE_VOL_PERIOD: int = 20
    BACKTEST_SLIPPAGE_ATR_WEIGHT: float = 0.05
    BACKTEST_SLIPPAGE_MAX_VOL_RATIO: float = 3.0
    BACKTEST_DEFAULT_STOP_LOSS_PCT: float = 0.02
    BACKTEST_DEFAULT_TAKE_PROFIT_PCT: float = 0.05
    BACKTEST_DEFAULT_TRADE_VALUE: float = 10000.0
    BACKTEST_DEFAULT_POSITION_SIZE_FRACTION: float = 0.1
    BACKTEST_MAX_TRADES: int = 200
    BACKTEST_GAP_TOLERANCE_MULT: float = 1.5
    BACKTEST_SLIPPAGE_BASE_PCT: float = 0.001
    BACKTEST_SLIPPAGE_MAX_PCT: float = 0.01

    # Risk manager settings
    ATR_STALENESS_CHECK_SECONDS: int = 300
    TRAILING_STOP_LONG_TF_FETCH_INTERVAL_SECONDS: int = 3600
    TRAILING_STOP_FETCH_INTERVAL_MIN_SECONDS: int = 300
    TRAILING_STOP_FETCH_INTERVAL_MAX_SECONDS: int = 3600
    TRAILING_STOP_FETCH_INTERVAL_FRACTION: float = 0.1
    TRAILING_STOP_MIN_IMPROVEMENT_PCT: float = 0.001
    TICK_SIZE_LARGE: float = 0.01
    TICK_SIZE_SMALL: float = 0.0001
    NATIVE_STOP_TICK_THRESHOLD: float = 0.5
    BREAKEVEN_FALLBACK_BUFFER_PCT: float = 0.005

    STALENESS_NOTIFY_THRESHOLD_SECONDS: int = 3600

    @field_validator("STALENESS_NOTIFY_THRESHOLD_SECONDS")
    @classmethod
    def validate_staleness_notify_threshold(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("STALENESS_NOTIFY_THRESHOLD_SECONDS must be positive")
        return v

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 2
    REDIS_TLS: bool = False

    # Data directory for logs, database, etc.
    DATA_DIR: str = "data"

    # Database
    DATABASE_PATH: str = "data/trading_bot.db"
    DATABASE_BACKEND: str = "sqlite"  # "sqlite" or "postgresql"

    # PostgreSQL (optional – if all are set, PostgreSQL is used instead of SQLite)
    DB_HOST: str = ""
    DB_PORT: int = 5432
    DB_NAME: str = ""
    DB_USER: str = ""
    DB_PASSWORD: str = ""

    # News
    NEWS_ENABLED: bool = False
    NEWS_UPDATE_INTERVAL_MINUTES: int = 60

    # Fast news refresh for currently tracked symbols (minutes)
    NEWS_FAST_UPDATE_INTERVAL_MINUTES: int = 15

    NEWS_API_KEY: Optional[str] = None       # for NewsAPI.org
    TWITTER_BEARER_TOKEN: Optional[str] = None
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: str = "trade-ledger/1.0"
    NEWS_MAX_ARTICLES_PER_SYMBOL: int = 5
    NEWS_CACHE_TTL_SECONDS: int = 1800       # 30 minutes
    NEWS_HTTP_TIMEOUT_SECONDS: float = 30.0   # timeout for each news source HTTP request
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
    RSS_FEEDS: Annotated[list[str], NoDecode] = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
        "https://www.investing.com/rss/news.rss",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.bloomberg.com/feeds/podcasts/etf_report.xml",
        "https://www.borsaitaliana.it/borsa/notizie/rss.html",
        "https://www.milanofinanza.it/rss",
        "https://www.ilsole24ore.com/rss/finanza.xml",
        "https://it.investing.com/webmaster-tools/rss",
        "https://news.teleborsa.it/NewsFeed.ashx",
        "https://news.teleborsa.it/NewsFeed.ashx?channel=energia",
        "https://news.teleborsa.it/NewsFeed.ashx?channel=banche",
        "https://www.ilsole24ore.com/rss",
    ]

    # YouTube Data API v3
    YOUTUBE_API_KEY: Optional[str] = None
    YOUTUBE_MAX_RESULTS: int = 5

    # Google News RSS (free, no API key)
    GOOGLE_NEWS_MAX_ARTICLES: int = 5

    # StockTwits API
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
    WEB_USERNAME: Optional[str] = None
    WEB_PASSWORD: Optional[str] = None

    # Web rate limiting
    WEB_RATE_LIMIT_REQUESTS: int = 300  # max requests per window
    WEB_RATE_LIMIT_WINDOW: int = 60    # window size in seconds

    # Entry signal MACD magnitude threshold as a multiple of ATR
    ENTRY_SIGNAL_MACD_ATR_MULT: float = 0.05

    # News sentiment exit timeframe threshold (seconds) - long-term timeframes ignore short-term sentiment
    NEWS_SENTIMENT_EXIT_TF_SECONDS: int = 604_800  # 1 week

    # Medium-term threshold (seconds) - timeframes >= this value ignore sentiment entirely.
    # Timeframes between NEWS_SENTIMENT_EXIT_TF_SECONDS and this value use a stricter threshold.
    NEWS_SENTIMENT_EXIT_TF_SECONDS_MEDIUM: int = 2_592_000  # 30 days

    # Multiplier applied to the sentiment exit threshold for medium-term timeframes.
    # Makes the threshold more negative to require stronger negative sentiment for an exit.
    NEWS_SENTIMENT_EXIT_MEDIUM_THRESHOLD_MULTIPLIER: float = 1.5

    # Walk-forward backtest candle threshold
    WALK_FORWARD_CANDLE_THRESHOLD: int = 100

    # Minimum number of candles required for a statistically significant backtest
    MIN_STATISTICALLY_SIGNIFICANT_CANDLES: int = 10

    # Minimum number of candles required to run a backtest (lower than
    # MIN_STATISTICALLY_SIGNIFICANT_CANDLES to allow backtesting on long
    # timeframes like 3M/6M/1Y where few candles are available).
    MIN_BACKTEST_CANDLES: int = 10

    # Maximum number of symbols to include in correlation matrix computation
    MAX_CORR_SYMBOLS: int = 50

    # Sentiment shift threshold to trigger immediate re-evaluation
    SENTIMENT_SHIFT_THRESHOLD: float = 0.3

    # Maximum concurrent LLM chunk evaluations during symbol re-evaluation
    LLM_CHUNK_CONCURRENCY_LIMIT: int = 5

    # Partial fill volume cap (fraction of last minute's volume)
    PARTIAL_FILL_VOLUME_CAP_PCT: float = 0.1

    # Model tier scoring thresholds
    MODEL_TIER_RSI_EXTREME: float = 30.0
    MODEL_TIER_ADX_STRONG: float = 25.0
    MODEL_TIER_DRAWDOWN_PCT: float = 10.0
    MODEL_TIER_ATR_PERCENTILE_HIGH: float = 80.0
    MODEL_TIER_ATR_PERCENTILE_LOW: float = 20.0
    MODEL_TIER_BB_WIDTH_SQUEEZE: float = 0.02
    MODEL_TIER_BB_WIDTH_EXPANSION: float = 0.08
    MODEL_TIER_PORTFOLIO_EXPOSURE_HIGH: float = 70.0
    MODEL_TIER_PORTFOLIO_STOP_RISK_HIGH: float = 8.0
    MODEL_TIER_SENTIMENT_TREND_MAG: float = 0.2
    MODEL_TIER_VOLUME_TREND_HIGH: float = 3.0
    MODEL_TIER_CONSECUTIVE_LOSSES: int = 3
    MODEL_TIER_PE_HIGH: float = 50.0
    MODEL_TIER_STOCH_EXTREME: float = 20.0
    MODEL_TIER_MFI_EXTREME: float = 20.0
    MODEL_TIER_CCI_EXTREME: float = 100.0
    MODEL_TIER_WILLIAMS_R_EXTREME: float = 20.0
    MODEL_TIER_MARKET_BREADTH_EXTREME: float = 80.0
    MODEL_TIER_MACD_HIST_CHANGE: float = 0.0001

    # Logging
    LOG_LEVEL: str = "INFO"

    # Borsa Italiana BTP list URL
    BTP_URL: str = "https://www.borsaitaliana.it/borsa/obbligazioni/mot/btp/lista.html"

    # Borsa Italiana market code (e.g., XMIL for Borsa di Milano)
    MARKET_CODE: str = "XMIL"

    # Market hours (Rome time)
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MINUTE: int = 0
    MARKET_CLOSE_HOUR: int = 17
    MARKET_CLOSE_MINUTE: int = 30
    MARKET_TIMEZONE: str = "Europe/Rome"

    # BTP Bond Fees (Intesa Sanpaolo Investo)
    BTP_FEE_PERC: float = 0.0024

    @field_validator("BTP_FEE_PERC")
    @classmethod
    def validate_btp_fee_perc(cls, v: float) -> float:
        if v < 0:
            raise ValueError("BTP_FEE_PERC must be >= 0")
        return v

    BTP_MIN_FEE: float = 3.50

    @field_validator("BTP_MIN_FEE")
    @classmethod
    def validate_btp_min_fee(cls, v: float) -> float:
        if v < 0:
            raise ValueError("BTP_MIN_FEE must be >= 0")
        return v

    BTP_IS_PRIMARY_ISSUANCE: bool = False

    # Standard stock/ETF fee parameters (Intesa Sanpaolo Investo defaults)
    STOCK_FEE_PERC: float = 0.0024
    STOCK_FEE_MIN: float = 3.50
    STOCK_FEE_FIXED: float = 2.50
    TOBIN_TAX_RATE: float = 0.0012

    @field_validator("STOCK_FEE_PERC")
    @classmethod
    def validate_stock_fee_perc(cls, v: float) -> float:
        if v < 0:
            raise ValueError("STOCK_FEE_PERC must be >= 0")
        return v

    @field_validator("STOCK_FEE_MIN")
    @classmethod
    def validate_stock_fee_min(cls, v: float) -> float:
        if v < 0:
            raise ValueError("STOCK_FEE_MIN must be >= 0")
        return v

    @field_validator("STOCK_FEE_FIXED")
    @classmethod
    def validate_stock_fee_fixed(cls, v: float) -> float:
        if v < 0:
            raise ValueError("STOCK_FEE_FIXED must be >= 0")
        return v

    @field_validator("TOBIN_TAX_RATE")
    @classmethod
    def validate_tobin_tax_rate(cls, v: float) -> float:
        if v < 0:
            raise ValueError("TOBIN_TAX_RATE must be >= 0")
        return v

    # Banca d'Italia BCE comunicati scraping for BTP news
    BANCA_D_ITALIA_BTP_NEWS_ENABLED: bool = False

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


    def validate_llm_settings(self):
        """Validate that LLM provider settings are properly configured."""
        if self.LLM_PROVIDER == "ollama":
            if not (self.OLLAMA_MODEL or (self.OLLAMA_MIND_MODEL and self.OLLAMA_ACTUATOR_MODEL)):
                raise ValueError("Ollama LLM provider is selected but no model is configured (OLLAMA_MODEL or OLLAMA_MIND_MODEL/OLLAMA_ACTUATOR_MODEL).")
        elif self.LLM_PROVIDER == "openai":
            if not self.OPENAI_API_KEY:
                raise ValueError("OpenAI LLM provider is selected but OPENAI_API_KEY is not set.")
            if not (self.OPENAI_MODEL or (self.OPENAI_MIND_MODEL and self.OPENAI_ACTUATOR_MODEL)):
                raise ValueError("OpenAI LLM provider is selected but no model is configured (OPENAI_MODEL or OPENAI_MIND_MODEL/OPENAI_ACTUATOR_MODEL).")
        elif self.LLM_PROVIDER == "g4f":
            # g4f dynamically manages models, so no specific model or API key validation is needed here.
            pass

    def reload(self):
        """Reload settings from .env file and environment variables.

        Updates all fields on this singleton instance in-place so that all
        modules that imported ``settings`` see the new values immediately.

        **Safe to reload at runtime:**
        - LLM provider/model/temperature/timeout settings (including LLM_ACTUATOR_TIMEOUT)
        - News settings (NEWS_ENABLED, NEWS_API_KEY, RSS_FEEDS, etc.)
        - Trading mode (TRADING_MODE), MAX_SYMBOLS
        - Risk/engine loop intervals
        - OHLCV timeframes and retention
        - Yahoo Finance and yfinance rate limit settings
        - Proxy settings (HTTP_PROXY_ENABLED, HTTP_PROXIES)
        - Notification settings (NOTIFICATION_VERBOSITY, etc.)
        - BTP settings (BTP_URL, BTP_FEE_PERC, etc.)
        - PAPER_INITIAL_BALANCE — triggers a soft-restart of paper trading state

        **Requires restart (NOT safe to reload):**
        - DATABASE_BACKEND / DATABASE_PATH — connections already open
        - REDIS_HOST / REDIS_PORT / REDIS_DB — Redis client already connected
        - WEB_HOST / WEB_PORT — Uvicorn already listening
        """
        from dotenv import load_dotenv
        load_dotenv(override=True)

        try:
            new_settings = self.__class__()
            new_settings.validate_llm_settings()
        except Exception as e:
            logging.getLogger(__name__).exception("Failed to reload settings: %s", e)
            return

        old_paper_balance = self.PAPER_INITIAL_BALANCE

        # Fields that are NOT safe to reload at runtime
        unsafe_fields = {
            "DATABASE_BACKEND",
            "DATABASE_PATH",
            "REDIS_HOST",
            "REDIS_PORT",
            "REDIS_DB",
            "REDIS_TLS",
            "WEB_HOST",
            "WEB_PORT",
        }

        # LLM-related fields that should trigger cache invalidation if changed.
        # Automatically identified by prefix to avoid manual maintenance.
        # Temperature changes do not affect cache validity for the same prompt.
        temperature_fields = {
            "LLM_TEMPERATURE",
            "LLM_MIND_TEMPERATURE",
            "LLM_ACTUATOR_TEMPERATURE",
            "LLM_WEAK_TEMPERATURE",
        }
        llm_fields = {
            f for f in self.model_fields
            if f.startswith(("LLM_", "OLLAMA_", "OPENAI_", "G4F_")) and f not in temperature_fields
        }
        # Include fee and system-prompt-related fields that affect LLM cache validity.
        # When these change, LLM_CACHE_VERSION is regenerated to invalidate all cached responses.
        prompt_fields = {
            "STOCK_FEE_PERC", "STOCK_FEE_MIN", "STOCK_FEE_FIXED",
            "TOBIN_TAX_RATE", "BTP_FEE_PERC", "BTP_MIN_FEE",
            "BTP_IS_PRIMARY_ISSUANCE", "MAX_BACKTEST_VARIANTS",
            "ENTRY_CONDITION_MIN_TIMEOUT_MULT",
        }

        llm_changed = any(getattr(self, f) != getattr(new_settings, f) for f in llm_fields | prompt_fields)

        for field_name in self.model_fields:
            if field_name in unsafe_fields or field_name == "LLM_CACHE_VERSION":
                continue
            setattr(self, field_name, getattr(new_settings, field_name))

        if llm_changed:
            self.LLM_CACHE_VERSION = str(uuid.uuid4())

        if self.PAPER_INITIAL_BALANCE != old_paper_balance:
            self.PAPER_BALANCE_CHANGED = True

        # Notify registered callbacks (e.g., running asyncio tasks) that settings have been reloaded
        for cb in self._reload_callbacks:
            try:
                cb()
            except Exception:
                pass

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

from dotenv import load_dotenv
load_dotenv(override=True)
settings = Settings()
