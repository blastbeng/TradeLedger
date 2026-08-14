import asyncio
import hashlib
import json
import logging
import math
import random
import pandas_market_calendars as mcal
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from zoneinfo import ZoneInfo

from src.config.settings import settings
from src.utils.health_metrics import health_metrics
from src.exchanges.market_data import get_tradable_assets, get_quotes, get_quotes_cached, get_multi_timeframe_bars, get_bars_range, discover_btp_bonds, discover_italian_ucits_etfs, _get_yf_session, _check_yf_circuit
from src.exchanges.yahoo_finance import get_yahoo_quote, get_yahoo_fundamentals, get_yahoo_dividends
from src.exchanges.yf_session import _invalidate_yf_session
from src.trading.paper_trader import PaperTrader
from src.llm.cache import get_cached_llm_response, compute_market_hash, _should_use_primary_model
from src.llm.prompts import (
    build_system_prompt,
    build_stock_selection_prompt,
    build_final_selection_prompt,
    build_analysis_prompt,
    build_backtest_variants_prompt,
    build_final_decision_prompt,
    _format_news_for_prompt,
    compact_prompt,
)

def _get_compacted_system_prompt() -> str:
    """Build the compacted system prompt dynamically to pick up settings.reload()."""
    return compact_prompt(build_system_prompt())

from src.indicators import (
    compute_ema,
    compute_all_indicators,
    compute_vwap,
    compute_pivot_points,
    compute_atr_series,
    compute_adx_series,
    compute_rsi_series,
    compute_macd_series,
)
try:
    from src.news.fetcher import discover_trending_stocks, detect_upcoming_events, discover_tickers_from_news
except ImportError:
    discover_trending_stocks = None
    detect_upcoming_events = None
    discover_tickers_from_news = None
from src.strategies.base import Signal
from src.strategies.llm_parser import create_strategy_from_llm, LLMStrategy
from src.strategies.validator import validate_signal
from src.strategies.backtester import backtest_strategy, format_backtest_summary, walk_forward_backtest, format_walk_forward_summary
from src.utils.redis_client import get_redis_client, check_redis_connection, is_redis_available
from src.utils.symbol_utils import is_btp_isin
from src.utils.task_supervisor import TaskSupervisor
from src.utils.event_bus import EventBus
from src.database import load_trading_state, save_trading_state, insert_trade, get_performance, store_news_articles, get_aggregate_sentiment_from_db, get_aggregate_sentiment_for_symbols, get_news_for_symbol, get_ohlcv, get_latest_ohlcv_timestamp, get_latest_ohlcv_timestamps_batch, insert_ohlcv_batch, save_paper_balances, load_paper_balances, save_indicators, get_indicators, get_indicators_for_symbols, get_ohlcv_summary_for_symbols, get_all_trades, get_latest_close_prices, insert_position_pnl_snapshot, cleanup_old_position_pnl, save_backtest_result, get_recent_backtest_result, get_backtest_results_for_symbol, cleanup_old_backtest_results, reset_paper_trading_data, insert_dividend, cleanup_old_dividends, get_pending_llm_decisions, update_llm_decision_outcome, get_llm_decision_quality_metrics, cleanup_old_llm_decisions, get_pending_dividends_for_symbol, mark_dividend_reinvested
from src.trading.components.order_executor import OrderExecutor
from src.trading.components.buy_executor import BuyExecutor
from src.trading.components.exit_order_manager import ExitOrderManager
from src.trading.components.manual_trade_logger import ManualTradeLogger
from src.trading.components.risk_manager import RiskManager
from src.trading.components.state_persistence import StatePersistence
from src.trading.components.position_manager import PositionManager
from src.trading.components.signal_processor import SignalProcessor
from src.trading.components.backtest_manager import BacktestManager
from src.trading.components.market_data_manager import MarketDataManager, ClockInfo
from src.trading.components.symbol_reevaluator import SymbolReevaluator
from src.trading.components.shared_state import SharedState
from src.trading.components.engine_orchestrator import EngineOrchestrator
from src.trading.components.background_task_manager import BackgroundTaskManager
from src.trading.components.state_initializer import StateInitializer
from src.config.config_service import UnifiedConfigService
from src.trading.engine_utils import (
    timeframe_to_ms,
    timeframe_to_seconds,
    format_symbol_display,
    is_excluded,
    normalize_llm_symbol,
    get_effective_refresh_interval,
)

logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self):
        self.trader = None
        self.shared_state = SharedState()

        self.base_currency = settings.BASE_CURRENCY
        self.max_symbols = settings.MAX_SYMBOLS
        self.effective_max_symbols = self.max_symbols
        self._symbol_reevaluation_interval = settings.SYMBOL_REEVALUATION_INTERVAL
        self.redis = get_redis_client()
        self._clear_time_sensitive_redis_keys()
        self.event_bus = EventBus()
        self.config_service = UnifiedConfigService(self.redis)
        self._exchange_semaphore = asyncio.Semaphore(settings.EXCHANGE_SEMAPHORE_LIMIT)  # max 10 concurrent API calls
        self._news_semaphore = asyncio.Semaphore(settings.NEWS_SEMAPHORE_LIMIT)  # max 5 concurrent news fetches
        self._indicator_semaphore = asyncio.Semaphore(settings.INDICATOR_SEMAPHORE_LIMIT)  # limit concurrent indicator computations
        self._backtest_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_BACKTESTS)  # limit concurrent backtest variants
        self._download_semaphore = asyncio.Semaphore(settings.DOWNLOAD_SEMAPHORE_LIMIT)  # max 5 concurrent background OHLCV backfills
        self._symbol_processing_semaphore = asyncio.Semaphore(settings.SYMBOL_PROCESSING_SEMAPHORE_LIMIT)  # limit concurrent symbol evaluations

        # Dedicated thread pool for database writes – prevents write contention
        # from starving the default asyncio thread pool used by the web server,
        # Telegram bot, and all other to_thread calls.
        self._db_executor = ThreadPoolExecutor(max_workers=settings.DB_EXECUTOR_WORKERS, thread_name_prefix="dbwriter")
        # Dedicated thread pool for download/indicator operations – prevents
        # download tasks from exhausting the default asyncio thread pool used
        # by the web server, Telegram bot, and engine loop.
        self._download_executor = ThreadPoolExecutor(max_workers=settings.DOWNLOAD_EXECUTOR_WORKERS, thread_name_prefix="downloader")
        # Dedicated thread pool for quote fetching – prevents zombie get_quotes
        # threads (from asyncio.wait_for timeouts) from exhausting the default
        # asyncio thread pool used by the web server and Telegram bot.
        self._quote_executor = ThreadPoolExecutor(max_workers=settings.QUOTE_EXECUTOR_WORKERS, thread_name_prefix="quotes")

        self.initial_balance: float = 0.0
        self.notifier = None

        # _load_state() and _ensure_cost_basis() are now called in _initialize_clients()
        # after the trading client is available.

        # Clear stale pause keys immediately (Redis is already available)
        from src.utils.pause_utils import clear_trading_pause_keys
        clear_trading_pause_keys(self.redis)

        # --- Extracted components ---
        self.event_bus.subscribe("remove_symbol_if_paused", self._remove_symbol_if_paused)
        self._state_persistence = StatePersistence(self, self.event_bus)
        self._exit_order_manager = ExitOrderManager(self, self.event_bus)
        self._order_executor = OrderExecutor(self, self.event_bus)
        self._order_executor._exit_order_manager = self._exit_order_manager
        self._buy_executor = BuyExecutor(self, self.event_bus)
        self._buy_executor._exit_order_manager = self._exit_order_manager
        self._order_executor._buy_executor = self._buy_executor
        self._manual_trade_logger = ManualTradeLogger(self, self.event_bus)
        self._risk_manager = RiskManager(self, self.event_bus)
        self._symbol_reevaluator = SymbolReevaluator(self, self.event_bus)
        self._signal_processor = SignalProcessor(self, self.event_bus)
        self._position_manager = PositionManager(self, self.event_bus)
        self._backtest_manager = BacktestManager(self, self.event_bus)
        self._market_data_manager = MarketDataManager(self, self.event_bus)
        self._orchestrator = EngineOrchestrator(self)
        self._background_task_manager = BackgroundTaskManager(self, self.event_bus)
        self._state_initializer = StateInitializer(self, self.event_bus)
        self._symbol_reeval_lock = asyncio.Lock()
        self._tradable_assets_lock = asyncio.Lock()
        self._balance_cache_lock = asyncio.Lock()
        self._reeval_trigger = asyncio.Event()
        self._reeval_pending_force: bool = False
        self._force_reeval: bool = False
        self._user_forced_reeval: bool = False
        self._pre_market_reeval: bool = False
        self._rebalance_reeval: bool = False
        self._running = True
        self._settings_reload_event = asyncio.Event()
        def _reload_callback():
            self._settings_reload_event.set()
            asyncio.create_task(self._on_settings_reload())
        settings.register_reload_callback(_reload_callback)
        self._last_state_save = 0

        # Re-entrancy guards for periodic tasks
        self._reconcile_running = False
        self._reevaluate_running = False
        self._pause_check_running = False
        self._last_risk_check: Dict[str, float] = {}
        self._last_risk_check_lock = asyncio.Lock()
        self._news_cache_running = False
        self._news_fast_running = False
        self._market_data_running = False
        self._full_breadth_running = False
        self._full_download_running = False
        self._quotes_fetch_running = False
        self._supervisors: list = []
        self._background_tasks: list = []

        # Market-closed periodic notification tracking
        self._last_market_closed_notify_time: float = 0.0
        self._market_opening_soon_notified: bool = False

        # Performance metrics cache – avoids recomputing on every symbol evaluation
        self._perf_cache: Optional[Dict[str, Any]] = None
        self._perf_cache_trade_count: int = -1
        self._perf_cache_time: float = 0.0

        # Trade pattern analysis cache – recomputed only when new trades are added
        self._trade_pattern_cache: Optional[Dict[str, Any]] = None
        self._trade_pattern_cache_trade_count: int = -1

        # Cache for tradable assets list (refreshed every 5 minutes)
        self._tradable_assets_cache: List[str] = []
        self._tradable_assets_cache_time: float = 0.0

        # Cache for BTP bonds (refreshed every 30 minutes)
        self._btp_bonds_cache: List[Dict[str, Any]] = []
        self._btp_bonds_cache_time: float = 0.0
        # Cache for ETF symbols (refreshed every 1 hour)
        self._etf_symbols_cache: List[str] = []
        self._etf_symbols_cache_time: float = 0.0

        # Cache for asset info (min_order_size, name, etc.) – refreshed every 1 hour
        self._asset_cache: Dict[str, Any] = {}
        self._asset_cache_time: Dict[str, float] = {}

        # Balance cache – avoids redundant API calls within an evaluation cycle
        # Log the complete event subscription registry after all components are initialized
        self.event_bus.log_subscription_summary()

    # --- Wrapper methods for engine_utils functions (called by components) ---

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        return timeframe_to_seconds(timeframe)

    def _timeframe_to_ms(self, timeframe: str) -> int:
        return timeframe_to_ms(timeframe)

    def _is_excluded(self, symbol: str, timeframe: str) -> bool:
        return is_excluded(symbol, timeframe)

    def _normalize_llm_symbol(self, sym: str, sample_pairs: list) -> Optional[str]:
        return normalize_llm_symbol(sym, sample_pairs, self.base_currency)

    def _get_effective_refresh_interval(self, base_interval: int, loop_type: str = "data") -> int:
        return get_effective_refresh_interval(base_interval, self.shared_state.current_symbols, loop_type)

    @property
    def current_symbols(self):
        """Expose shared_state.current_symbols for backward compatibility."""
        return self.shared_state.current_symbols

    @property
    def positions(self):
        """Expose shared_state.positions for backward compatibility."""
        return self.shared_state.positions

    @property
    def queued_orders(self):
        """Expose shared_state.queued_orders for backward compatibility."""
        return self.shared_state.queued_orders

    @property
    def trade_history(self):
        """Expose shared_state.trade_history for backward compatibility."""
        return self.shared_state.trade_history

    def _clear_time_sensitive_redis_keys(self):
        """Clear time-sensitive Redis keys on startup to prevent stale data."""
        keys_to_clear = [
            "trading:last_triggered_reeval",
            "market:breadth:full",
            "reeval:incremental_offset",
        ]
        try:
            for key in keys_to_clear:
                self.redis.delete(key)
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Failed to clear time-sensitive Redis keys: {e}")

    async def _initialize_clients(self):
        await self._state_initializer._initialize_clients()

    async def reset_paper_trading_state(self):
        await self._state_initializer.reset_paper_trading_state()

    def set_notifier(self, notifier):
        """Attach a notification service (e.g., TelegramBot)."""
        self.notifier = notifier
        from src.exchanges import market_data
        market_data.set_notifier(notifier)
        if hasattr(self, '_market_data_manager') and self._market_data_manager:
            self._market_data_manager.notifier = notifier

    def trigger_symbol_reevaluation(self, force: bool = False):
        """Signal the periodic reevaluate loop to run immediately."""
        logger.info(f"Re-evaluation triggered (force={force})")
        if force:
            self._force_reeval = True
            self._user_forced_reeval = True
            if self._reevaluate_running:
                self._reeval_pending_force = True
                logger.info("Re-evaluation already running; queued forced re-evaluation for after current cycle completes.")
            # Invalidate correlation matrix cache on forced re-evaluation
            # so the LLM sees fresh correlations after significant market changes
            try:
                self.redis.delete("reeval:correlation_matrix")
            except (ConnectionError, TimeoutError, OSError):
                pass
        elif self._reevaluate_running:
            logger.info("Re-evaluation already running; queued re-evaluation for after current cycle completes.")
        self._reeval_trigger.set()

    def trigger_portfolio_rebalance(self):
        """Trigger a portfolio rebalance re-evaluation."""
        logger.info("Portfolio rebalance triggered")
        self._force_reeval = True
        self._rebalance_reeval = True
        self._reeval_trigger.set()

    async def force_download_all_assets(self):
        await self._market_data_manager.force_download_all_assets()

    async def force_download_tracked_symbols(self):
        await self._market_data_manager.force_download_tracked_symbols()

    async def _get_cached_balance(self, ttl: float = 30.0) -> Dict[str, float]:
        """Return cached balance, refreshing if older than ttl seconds."""
        return await self._market_data_manager._get_cached_balance(ttl)

    async def _get_cached_position_tickers(self, ttl: float = 30.0) -> Dict[str, Dict[str, Any]]:
        """Return cached position tickers, refreshing if older than ttl seconds."""
        return await self._market_data_manager._get_cached_position_tickers(ttl)

    async def _get_cached_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return aggregate news sentiment, cached for 60 seconds to reduce DB load."""
        return await self._market_data_manager._get_cached_sentiment(symbol)

    async def stop(self):
        """Gracefully stop the engine and all background tasks."""
        logger.info("Stopping trading engine...")
        self._running = False
        self._state_persistence.stop_periodic_save()
        for task in self.shared_state._delayed_entry_tasks:
            task.cancel()
        self.shared_state._delayed_entry_tasks.clear()
        logger.info("Cancelled delayed entry tasks.")

        # Cancel supervised background tasks
        for sup in self._supervisors:
            sup.cancel()
        for task in self._background_tasks:
            task.cancel()
        # Wait for them to actually finish cancelling
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        logger.info("Cancelled background tasks.")
        self._db_executor.shutdown(wait=True)
        self._download_executor.shutdown(wait=True)
        logger.info("Database write executor shut down.")
        logger.info("Download executor shut down.")
        self._quote_executor.shutdown(wait=False)
        logger.info("Quote executor shut down.")
        # Close the PostgreSQL connection pool if it was used
        from src.database import close_pool
        close_pool()
        logger.info("Trading engine stopped.")

    async def _interruptible_sleep(self, delay: float):
        """Sleep for delay seconds, but wake up immediately if settings are reloaded."""
        self._settings_reload_event.clear()
        sleep_task = asyncio.create_task(asyncio.sleep(delay))
        reload_task = asyncio.create_task(self._settings_reload_event.wait())
        await asyncio.wait([sleep_task, reload_task], return_when=asyncio.FIRST_COMPLETED)
        for task in (sleep_task, reload_task):
            if not task.done():
                try:
                    task.cancel()
                except asyncio.CancelledError:
                    pass

    async def _on_settings_reload(self):
        """Update cached settings values when settings are reloaded."""
        self.base_currency = settings.BASE_CURRENCY
        self.max_symbols = settings.MAX_SYMBOLS
        self.effective_max_symbols = self.max_symbols
        self._symbol_reevaluation_interval = settings.SYMBOL_REEVALUATION_INTERVAL
        # Invalidate yfinance session so it's recreated with new proxy settings
        _invalidate_yf_session()
        # Update backtest concurrency semaphore to pick up MAX_CONCURRENT_BACKTESTS changes
        self._backtest_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_BACKTESTS)
        # Update async semaphores to pick up concurrency limit changes
        self._exchange_semaphore = asyncio.Semaphore(settings.EXCHANGE_SEMAPHORE_LIMIT)
        self._news_semaphore = asyncio.Semaphore(settings.NEWS_SEMAPHORE_LIMIT)
        self._indicator_semaphore = asyncio.Semaphore(settings.INDICATOR_SEMAPHORE_LIMIT)
        self._download_semaphore = asyncio.Semaphore(settings.DOWNLOAD_SEMAPHORE_LIMIT)
        self._symbol_processing_semaphore = asyncio.Semaphore(settings.SYMBOL_PROCESSING_SEMAPHORE_LIMIT)
        # Update paper trader's base currency if it exists
        if self.trader is not None:
            self.trader.base_currency = settings.BASE_CURRENCY
        # Invalidate clock cache so market hours/timezone changes take effect immediately
        await self.event_bus.request("invalidate_clock_cache")

    async def _periodic_reconcile(self):
        await self._background_task_manager._periodic_reconcile()

    async def _periodic_reevaluate(self):
        await self._background_task_manager._periodic_reevaluate()

    async def _clear_pause_and_resume(self, reason: str, notification_msg: str, notification_summary: dict) -> None:
        await self._background_task_manager._clear_pause_and_resume(reason, notification_msg, notification_summary)

    async def _handle_missing_pause_duration(self, pause_start_raw: Optional[bytes]) -> Tuple[bool, bool]:
        return await self._background_task_manager._handle_missing_pause_duration(pause_start_raw)

    async def _handle_pause_duration_elapsed(self, pause_start_raw: bytes, pause_duration_raw: bytes) -> None:
        await self._background_task_manager._handle_pause_duration_elapsed(pause_start_raw, pause_duration_raw)

    async def _periodic_pause_check(self):
        await self._background_task_manager._periodic_pause_check()

    async def _periodic_full_market_breadth(self):
        await self._background_task_manager._periodic_full_market_breadth()

    async def _periodic_market_condition_check(self):
        await self._background_task_manager._periodic_market_condition_check()

    async def _periodic_portfolio_rebalance(self):
        await self._background_task_manager._periodic_portfolio_rebalance()

    async def _is_quote_too_stale(self, ticker: Dict[str, Any], timeframe: str) -> bool:
        """Check if the quote is too stale for trading based on the configured threshold."""
        return await self._market_data_manager._is_quote_too_stale(ticker, timeframe)

    async def _fetch_vix(self) -> Optional[float]:
        """Fetch a volatility proxy. Uses US VIX (^VIX) as a global market proxy,
        falling back to an internal proxy based on tracked symbols if unavailable."""
        return await self._market_data_manager._fetch_vix()


    async def _fetch_and_store_news_for_symbol(self, symbol: str):
        """Fetch news for a single symbol and store it in the database."""
        return await self._market_data_manager._fetch_and_store_news_for_symbol(symbol)

    async def _risk_management_loop(self):
        await self._background_task_manager._risk_management_loop()

    async def _refresh_current_symbols_news_fast(self):
        await self._background_task_manager._refresh_current_symbols_news_fast()

    async def _refresh_news_cache(self):
        await self._background_task_manager._refresh_news_cache()

    async def _download_market_data_loop(self):
        await self._background_task_manager._download_market_data_loop()

    async def _download_all_assets_data_loop(self):
        await self._background_task_manager._download_all_assets_data_loop()

    async def _download_all_news_loop(self):
        await self._background_task_manager._download_all_news_loop()

    async def _refresh_all_quotes_loop(self):
        await self._background_task_manager._refresh_all_quotes_loop()

    async def _refresh_ticker_discovery_loop(self):
        await self._background_task_manager._refresh_ticker_discovery_loop()

    async def _fetch_dividends_loop(self):
        await self._background_task_manager._fetch_dividends_loop()

    async def _reinvest_dividends_loop(self):
        await self._background_task_manager._reinvest_dividends_loop()

    def _daily_realized_pnl(self) -> float:
        """Return the sum of realized P&L for trades closed today (market timezone)."""
        tz = ZoneInfo(settings.MARKET_TIMEZONE)
        today = datetime.now(tz).date().isoformat()
        return self.shared_state._daily_realized_pnl.get(today, 0.0)

    def _daily_buy_fees(self) -> float:
        """Return the sum of buy-side fees for trades opened today (market timezone).

        These fees are not yet reflected in realized_pnl (which only includes
        fees from closed positions), so they must be accounted for separately
        in the daily loss limit check.
        """
        tz = ZoneInfo(settings.MARKET_TIMEZONE)
        today = datetime.now(tz).date().isoformat()
        return self.shared_state._daily_buy_fees.get(today, 0.0)

    def _log_task_exception(self, task: asyncio.Task) -> None:
        """Log exceptions from background tasks to prevent silent failures."""
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Background task {task.get_name()} failed: {type(e).__name__}: {e}", exc_info=True)

    async def _evaluate_llm_decisions_loop(self):
        await self._background_task_manager._evaluate_llm_decisions_loop()

    async def _record_unexpected_exception(self, context: str, exc: Exception) -> None:
        """Record metrics for unexpected exceptions in Redis and alert on persistent failures."""
        try:
            exc_type = type(exc).__name__
            key = f"metrics:unexpected_exception:{context}:{exc_type}"
            count = await asyncio.to_thread(self.redis.incr, key)
            await asyncio.to_thread(self.redis.expire, key, 86400)
            
            # Alert on persistent failures (e.g., 3 occurrences)
            if count == 3 and self.notifier:
                await self.notifier.send_notification(
                    f"⚠️ Persistent error: {exc_type} in {context} occurred {count} times.",
                    summary={"action": "ALERT", "reason": f"Persistent {exc_type} in {context}"}
                )
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError):
            pass

    async def run(self):
        """Main event-driven loop using WebSocket ticker updates."""
        self._state_persistence.start_periodic_save(interval=60)
        await self._orchestrator.run()


    async def _periodic_pause_resume_check(self):
        await self._background_task_manager._periodic_pause_resume_check()

    async def _redis_health_check_loop(self):
        await self._background_task_manager._redis_health_check_loop()

    async def _health_check_loop(self):
        await self._background_task_manager._health_check_loop()

    @staticmethod
    def _deduplicate_variants(variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove backtest variants whose key risk parameters are identical or nearly identical.

        Compares the parameters that actually change backtest outcomes:
        stop_loss_pct, take_profit_pct, stop_loss_atr_multiple, take_profit_atr_multiple,
        trailing_stop, trailing_stop_distance_pct, trailing_stop_atr_multiple,
        trailing_stop_activation_pct, max_hold_time_seconds, breakeven_activation_pct,
        partial_take_profit_levels, trailing_take_profit, trailing_take_profit_distance_pct,
        max_unrealized_loss_pct, position_size_fraction, backtest_entry_config.
        Variants whose values for ALL these keys match (within a small tolerance for floats)
        are considered duplicates; only the first is kept.
        """
        if not variants:
            return []

        KEY_PARAMS = [
            "stop_loss_pct", "take_profit_pct",
            "stop_loss_atr_multiple", "take_profit_atr_multiple",
            "trailing_stop", "trailing_stop_distance_pct",
            "trailing_stop_atr_multiple", "trailing_stop_activation_pct",
            "max_hold_time_seconds", "breakeven_activation_pct",
            "partial_take_profit_levels", "trailing_take_profit",
            "trailing_take_profit_distance_pct",
            "max_unrealized_loss_pct", "position_size_fraction",
            "backtest_entry_config",
        ]
        FLOAT_KEYS = {
            "stop_loss_pct", "take_profit_pct",
            "stop_loss_atr_multiple", "take_profit_atr_multiple",
            "trailing_stop_distance_pct", "trailing_stop_atr_multiple",
            "trailing_stop_activation_pct", "max_hold_time_seconds",
            "breakeven_activation_pct", "trailing_take_profit_distance_pct",
            "max_unrealized_loss_pct", "position_size_fraction",
        }

        def _signature(v: Dict[str, Any]) -> tuple:
            sig = []
            for key in KEY_PARAMS:
                val = v.get(key)
                if key in FLOAT_KEYS and val is not None:
                    try:
                        val = round(float(val), 8)
                    except (TypeError, ValueError):
                        pass
                # For list/dict params, use a JSON string for stable comparison
                if isinstance(val, (list, dict)):
                    val = json.dumps(val, sort_keys=True, default=str)
                sig.append((key, val))
            return tuple(sig)

        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for v in variants:
            if not isinstance(v, dict):
                continue
            sig = _signature(v)
            if sig not in seen:
                seen.add(sig)
                unique.append(v)

        if len(unique) < len(variants):
            logger.info(
                f"Deduplicated backtest variants: {len(variants)} -> {len(unique)} "
                f"(removed {len(variants) - len(unique)} duplicate(s))"
            )
        return unique

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Return performance summary grouped by symbol and timeframe from trade_history table."""
        return await asyncio.to_thread(get_performance)

    async def get_pause_status(self) -> Dict[str, Any]:
        """Return the current trading pause status, reason, remaining duration, and a formatted countdown."""
        return await self._state_persistence.get_pause_status()

    async def sell_all_positions(self):
        """Sell all open positions at market price."""
        await self.event_bus.request("sell_all_positions")

    async def sell_position(self, symbol: str):
        """Sell a specific open position at market price."""
        await self.event_bus.request("sell_position", symbol)

    async def log_manual_trade(self, ticker: str, side: str, quantity: float, money_spent: float, fee: float) -> dict:
        """Log a manually executed trade in notify mode. Persists to DB and updates positions."""
        return await self.event_bus.request("log_manual_trade", ticker, side, quantity, money_spent, fee)

    async def _monitor_entry_signals_loop(self):
        await self._background_task_manager._monitor_entry_signals_loop()

    async def _check_pending_entries(self):
        await self._background_task_manager._check_pending_entries()

    async def _execute_delayed_entry(self, symbol: str, signal, timeframe: str, delay_seconds: float):
        """Execute a delayed entry after waiting for the specified duration."""
        logger.info(f"Delayed entry: waiting {delay_seconds}s for {symbol}")
        await asyncio.sleep(delay_seconds)
        if not self._running:
            return
        # Check if market is still open before executing
        if not await self._is_market_open():
            logger.info(f"Skipping delayed BUY for {symbol}: market closed after delay elapsed.")
            if self.notifier:
                await self.notifier.send_notification(
                    f"⏸️ Delayed BUY for {symbol} skipped – market closed after delay.",
                    summary={"symbol": symbol, "action": "SKIP", "reason": "Market closed after delay"}
                )
            return
        # Check if the symbol already has a position (may have been bought by another path)
        if symbol in self.shared_state.positions:
            logger.info(f"Skipping delayed BUY for {symbol}: position already exists.")
            return
        # Check trading pause
        paused = await asyncio.to_thread(self.redis.get, "trading:paused")
        if paused:
            logger.info(f"Ignoring delayed BUY {symbol}: trading is now paused.")
            if self.notifier:
                await self.notifier.send_notification(
                    f"⏸️ Delayed BUY for {symbol} skipped – trading paused.",
                    summary={"symbol": symbol, "action": "SKIP", "reason": "Trading paused"}
                )
            return
        logger.info(f"Delay elapsed for {symbol}, executing BUY")
        await self.event_bus.request(
            "execute_signal",
            symbol=symbol,
            signal=signal,
            timeframe=timeframe,
            atr=None,
        )

    async def _get_global_risk_multiplier(self) -> Optional[float]:
        """Return the global risk multiplier, falling back to persisted value if Redis key expired."""
        raw = await asyncio.to_thread(self.redis.get, "trading:global_risk_multiplier")
        if raw:
            try:
                val = float(raw)
                if 0.0 <= val <= 1.0:
                    self.shared_state._global_risk_multiplier = val
                    return val
            except (ValueError, TypeError):
                pass
        # Redis key expired or invalid — fall back to last known persisted value
        if self.shared_state._global_risk_multiplier is not None:
            logger.warning(
                "Global risk multiplier Redis key expired — using persisted fallback "
                f"value {self.shared_state._global_risk_multiplier}. The LLM should set a new value."
            )
            return self.shared_state._global_risk_multiplier
        return None

    async def _set_global_risk_multiplier(self, value: float):
        """Set the global risk multiplier in Redis (with TTL) and persist it to the database."""
        await asyncio.to_thread(self.redis.setex, "trading:global_risk_multiplier", 3600, str(value))
        self.shared_state._global_risk_multiplier = value
        await asyncio.to_thread(save_trading_state, "global_risk_multiplier", value)

    async def _process_queued_orders(self):
        await self._background_task_manager._process_queued_orders()

    async def _cleanup_orphaned_orders(self):
        await self._background_task_manager._cleanup_orphaned_orders()

    async def _is_market_open(self) -> bool:
        """Return True if the Italian market (Borsa Italiana) is currently open."""
        clock = await self.event_bus.request("get_clock")
        if clock is None:
            # Fallback: if clock unavailable, assume closed to be safe
            return False
        return clock.is_open

    async def _remove_symbol_if_paused(self, symbol: str):
        """Clear pending entries for a symbol. Symbols are kept in current_symbols even when paused
        so the bot continues to generate and notify signals."""
        # Always clear any pending entry for this symbol
        self.shared_state._pending_entries.pop(symbol, None)
        self.shared_state._state_dirty = True

    async def simulate_backtest(self, symbol: str) -> Dict[str, Any]:
        """Simulate Step 1a (analysis), Step 1b (variants), and run backtest without executing trades."""
        return await self._signal_processor.simulation_manager.simulate_backtest(symbol)

    async def simulate_decision(self, symbol: str) -> Dict[str, Any]:
        """Simulate Step 1a (analysis), Step 1b (variants), and Step 2 (final decision) without executing trades."""
        return await self._signal_processor.simulation_manager.simulate_decision(symbol)
