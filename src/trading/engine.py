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
        """Initialize clients and load persisted state (non‑blocking)."""
        # Check if PAPER_INITIAL_BALANCE changed since last run
        state = await asyncio.to_thread(load_trading_state)
        persisted_balance = state.get("paper_initial_balance")

        # If paper_initial_balance was never persisted, infer it from paper_balances
        if persisted_balance is None:
            paper_balances = state.get("paper_balances")
            if paper_balances and isinstance(paper_balances, dict):
                persisted_balance = paper_balances.get(self.base_currency)
                if persisted_balance is not None:
                    logger.info(
                        f"paper_initial_balance not found in DB. "
                        f"Inferred {persisted_balance} from paper_balances. "
                        f"Current setting: {settings.PAPER_INITIAL_BALANCE}."
                    )

        if persisted_balance is not None and persisted_balance != settings.PAPER_INITIAL_BALANCE:
            logger.info(
                f"PAPER_INITIAL_BALANCE changed from {persisted_balance} to {settings.PAPER_INITIAL_BALANCE}. "
                "Resetting paper trading state."
            )
            await self.reset_paper_trading_state()
        else:
            self.trader = PaperTrader()
            logger.info(f"PaperTrader initialized for {settings.TRADING_MODE} trading mode.")
            try:
                self._state_persistence.load_state()
            except ValueError as e:
                logger.critical(f"State corruption detected during load: {e}. Resetting paper trading state.")
                await self.reset_paper_trading_state()
            self._position_manager.ensure_cost_basis()
            # Initialize _cycle_spent from any queued buy orders loaded from persisted
            # state so capital is reserved immediately at startup, before the first
            # re-evaluation cycle runs (which would otherwise leave _cycle_spent at 0.0
            # and allow over-allocation of capital already reserved by stale orders).
            queued_buy_total = sum(
                q.get('amount', 0.0) for q in self.shared_state.queued_orders
                if q.get('side') == 'buy'
            )
            async with self.shared_state._cycle_spent_lock:
                self.shared_state._cycle_spent = queued_buy_total
            if queued_buy_total > 0:
                logger.info(f"Initialized _cycle_spent={queued_buy_total:.2f} from {sum(1 for q in self.shared_state.queued_orders if q.get('side') == 'buy')} queued buy orders.")

        # Persist the current PAPER_INITIAL_BALANCE so we can detect changes on next startup
        await asyncio.to_thread(save_trading_state, "paper_initial_balance", settings.PAPER_INITIAL_BALANCE)

    async def reset_paper_trading_state(self):
        """Reset paper trading state."""
        logger.info("Resetting paper trading state...")

        # Clear in-memory state
        self.shared_state.positions.clear()
        self.shared_state.queued_orders.clear()
        self.shared_state.current_symbols.clear()
        self.shared_state._pending_entries.clear()
        async with self.shared_state._eval_state_lock:
            self.shared_state._last_strategy_eval.clear()
            self.shared_state._strategy_intervals.clear()
            self.shared_state._force_eval.clear()
            self.shared_state._force_eval_time.clear()
        self.shared_state._entry_signal_state.clear()
        self.shared_state._last_decisions.clear()
        self.shared_state._last_eval_snapshot.clear()
        async with self.shared_state._cycle_spent_lock:
            self.shared_state._cycle_spent = 0.0
        self.shared_state._balance_cache = None
        self.shared_state._balance_cache_time = 0.0
        self.shared_state._position_tickers_cache = None
        self.shared_state._position_tickers_cache_time = 0.0
        self._perf_cache = None
        self._perf_cache_time = 0.0
        self._perf_cache_trade_count = -1
        self._trade_pattern_cache = None
        self._trade_pattern_cache_trade_count = -1
        self.shared_state._trade_history_version = 0
        self.shared_state._realized_pnl_offset = 0.0
        self.shared_state.trade_history.clear()
        self.shared_state.recent_signals.clear()
        self.shared_state.last_loss_time.clear()
        self.shared_state.cooldown_durations.clear()
        self.shared_state._global_risk_multiplier = None
        self.shared_state._symbol_first_seen.clear()
        self.shared_state._sentiment_cache.clear()
        self.shared_state._market_breadth = None
        self.shared_state._daily_realized_pnl.clear()
        self.shared_state._daily_buy_fees.clear()
        self.initial_balance = settings.PAPER_INITIAL_BALANCE

        # Clear the persisted peak total equity so drawdown starts fresh
        await asyncio.to_thread(self.redis.delete, "trading:peak_total_equity")

        # Reset DB data (unconditionally clear all trade data for both modes)
        await asyncio.to_thread(reset_paper_trading_data, keep_trade_history=False)

        # Re-initialize paper trader with new balance
        self.trader = PaperTrader()

        # Save the fresh state
        self.shared_state._state_dirty = True
        await self._state_persistence.save_state(force=True)

        # Persist the new PAPER_INITIAL_BALANCE so we don't reset again on next restart
        await asyncio.to_thread(save_trading_state, "paper_initial_balance", settings.PAPER_INITIAL_BALANCE)

        # Persist the new initial_balance so profit calculations are correct after restart
        await asyncio.to_thread(save_trading_state, "initial_balance", settings.PAPER_INITIAL_BALANCE)

        if self.notifier:
            await self.notifier.send_notification(
                "♻️ Paper trading state has been reset.",
                summary={"action": "RESET", "reason": "State reset"}
            )
        logger.info("Paper trading state reset complete.")

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
        """Immediately download OHLCV data for all tradable assets (stocks, ETFs, BTPs)."""
        self._full_download_running = True
        logger.info("Force download: starting immediate OHLCV download for all assets...")
        try:
            plain_assets = await self.event_bus.request("get_tradable_assets")
            stock_pairs = [f"{sym}/{self.base_currency}" for sym in plain_assets]
            etf_symbols = await self.event_bus.request("get_etf_symbols")
            etf_pairs = [f"{sym}/{self.base_currency}" for sym in etf_symbols]

            btp_bonds = await self.event_bus.request("get_btp_bonds")
            btp_pairs = [f"{b['isin']}/{self.base_currency}" for b in btp_bonds]

            all_pairs = stock_pairs + etf_pairs + btp_pairs
            if not all_pairs:
                logger.warning("Force download: no tradable assets found.")
                return

            now_ms = int(time.time() * 1000)
            start_ms = now_ms - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000

            random.shuffle(all_pairs)

            async def _force_download_symbol(pair: str):
                for tf in settings.OHLCV_TIMEFRAMES:
                    await self.event_bus.request("download_symbol_ohlcv", pair, tf, start_ms, now_ms, quiet=True, force=True)

            # Limit concurrent symbol downloads to 2 to avoid exhausting the
            # _download_executor thread pool, leaving threads available for
            # tracked tickers.
            download_concurrency = asyncio.Semaphore(settings.FORCE_DOWNLOAD_ALL_CONCURRENCY)
            async def _limited_force_download(pair: str):
                async with download_concurrency:
                    await _force_download_symbol(pair)
            download_tasks = [_limited_force_download(pair) for pair in all_pairs]
            await asyncio.gather(*download_tasks)

            logger.info("Force download: complete.")
        except asyncio.CancelledError:
            raise
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Force download network/IO error: {type(e).__name__}: {e}")
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
            logger.error(f"Force download data/logic error: {type(e).__name__}: {e}", exc_info=True)
            await self._record_unexpected_exception("force_download_all_assets", e)
        finally:
            self._full_download_running = False

    async def force_download_tracked_symbols(self):
        """Immediately download OHLCV data for currently tracked symbols only."""
        logger.info("Force download: starting immediate OHLCV download for tracked symbols...")
        try:
            tracked_pairs = [entry["symbol"] for entry in self.shared_state.current_symbols]
            if not tracked_pairs:
                logger.warning("Force download: no tracked symbols found.")
                return

            now_ms = int(time.time() * 1000)
            start_ms = now_ms - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000

            async def _force_download_symbol(pair: str):
                for tf in settings.OHLCV_TIMEFRAMES:
                    await self.event_bus.request("download_symbol_ohlcv", pair, tf, start_ms, now_ms, quiet=True, force=True)

            download_concurrency = asyncio.Semaphore(settings.FORCE_DOWNLOAD_TRACKED_CONCURRENCY)
            async def _limited_force_download(pair: str):
                async with download_concurrency:
                    await _force_download_symbol(pair)
            download_tasks = [_limited_force_download(pair) for pair in tracked_pairs]
            await asyncio.gather(*download_tasks)

            logger.info("Force download: complete for tracked symbols.")
        except asyncio.CancelledError:
            raise
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Force download tracked symbols network/IO error: {type(e).__name__}: {e}")
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
            logger.error(f"Force download tracked symbols data/logic error: {type(e).__name__}: {e}", exc_info=True)
            await self._record_unexpected_exception("force_download_tracked_symbols", e)

    async def _get_cached_balance(self, ttl: float = 30.0) -> Dict[str, float]:
        """Return cached balance, refreshing if older than ttl seconds."""
        now = time.time()
        if self.shared_state._balance_cache is not None and (now - self.shared_state._balance_cache_time) < ttl:
            return self.shared_state._balance_cache
        
        async with self._balance_cache_lock:
            # Double-check after acquiring lock
            now = time.time()
            if self.shared_state._balance_cache is not None and (now - self.shared_state._balance_cache_time) < ttl:
                return self.shared_state._balance_cache
            
            balance = await asyncio.to_thread(self.trader.fetch_balance)
            self.shared_state._balance_cache = balance
            self.shared_state._balance_cache_time = now
            return balance

    async def _get_cached_position_tickers(self, ttl: float = 30.0) -> Dict[str, Dict[str, Any]]:
        """Return cached position tickers, refreshing if older than ttl seconds."""
        now = time.time()
        if (
            self.shared_state._position_tickers_cache is not None
            and (now - self.shared_state._position_tickers_cache_time) < ttl
        ):
            return self.shared_state._position_tickers_cache
        tickers = await self.event_bus.request("get_all_position_tickers")
        self.shared_state._position_tickers_cache = tickers
        self.shared_state._position_tickers_cache_time = now
        return tickers

    async def _get_cached_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return aggregate news sentiment, cached for 60 seconds to reduce DB load."""
        base = symbol.split("/")[0] if "/" in symbol else symbol
        now = time.time()
        cached = self.shared_state._sentiment_cache.get(base)
        if cached and (now - cached[0]) < 60:
            return cached[1]
        try:
            agg = await asyncio.to_thread(
                get_aggregate_sentiment_from_db, base, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS
            )
            self.shared_state._sentiment_cache[base] = (now, agg)
            return agg
        except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Failed to fetch sentiment for {base}: {type(e).__name__}: {e}")
            return None

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
        """Check if the quote is too stale for trading based on the configured threshold.

        The staleness threshold is scaled by the symbol's timeframe: longer timeframes
        tolerate staler quotes. Returns False if staleness cannot be determined or
        if the guard is disabled.

        When the market is closed, quotes cannot be refreshed, so stale quotes are
        not considered too stale. This prevents false positives over weekends and
        holidays where the most recent quote is necessarily from the last trading
        session.
        """
        if settings.QUOTE_MAX_STALENESS_SECONDS <= 0:
            return False
        last_update = ticker.get("last_update")
        if last_update is None:
            return False
        age_seconds = (time.time() * 1000 - last_update) / 1000
        # Scale the threshold by the timeframe: longer timeframes allow staler quotes.
        # Use at least the configured max staleness (1 hour), or 10% of the timeframe,
        # whichever is greater (capped at 6 hours for very long timeframes to avoid
        # trading on excessively stale prices).
        tf_seconds = timeframe_to_seconds(timeframe)
        scaled_threshold = max(settings.QUOTE_MAX_STALENESS_SECONDS, min(tf_seconds * 0.1, 21600))

        if age_seconds <= scaled_threshold:
            return False

        # The quote exceeds the staleness threshold, but if the market is currently
        # closed, we cannot obtain a fresher quote. In this case, don't flag it as
        # stale — the most recent available quote is the best we can get.
        try:
            is_open = await self._is_market_open()
            if not is_open:
                return False
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError):
            pass  # If market status can't be determined, fall back to the age-based check

        return True

    async def _fetch_vix(self) -> Optional[float]:
        """Fetch a volatility proxy. Uses US VIX (^VIX) as a global market proxy,
        falling back to an internal proxy based on tracked symbols if unavailable."""
        try:
            # Try fetching US VIX as a global volatility proxy
            quotes = await asyncio.to_thread(get_quotes_cached, ["^VIX"])
            vix_quote = quotes.get("^VIX", {})
            vix_price = vix_quote.get("last")
            if vix_price and vix_price > 0:
                return float(vix_price)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError):
            pass

        # Fallback: compute a simple internal volatility proxy from tracked symbols
        try:
            if not self.shared_state.current_symbols:
                return None
            
            tracked_symbols = [entry["symbol"].split("/")[0] for entry in self.shared_state.current_symbols]
            quotes = await asyncio.to_thread(get_quotes_cached, tracked_symbols)
            
            changes = []
            for sym, quote in quotes.items():
                pct_change = quote.get("percentage")
                if pct_change is not None:
                    changes.append(abs(pct_change))
            
            if changes:
                # Scale the average absolute percentage change to a VIX-like scale
                avg_abs_change = sum(changes) / len(changes)
                return round(avg_abs_change * 10, 2)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError):
            pass

        return None


    async def _fetch_and_store_news_for_symbol(self, symbol: str):
        """Fetch news for a single symbol and store it in the database."""
        if not settings.NEWS_ENABLED:
            return
        
        base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
        
        # Check if we recently fetched news and found 0 articles
        no_news_cache_key = f"news:no_articles:{base_symbol}"
        try:
            cached_no_news = await asyncio.to_thread(self.redis.get, no_news_cache_key)
            if cached_no_news:
                logger.debug(f"Skipping news fetch for {symbol}: recently found 0 articles.")
                return
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        try:
            from src.news.fetcher import fetch_news_for_symbol
            stock_name = await self.event_bus.request("get_stock_name", symbol)
            loop = asyncio.get_running_loop()
            articles = await fetch_news_for_symbol(symbol, stock_name)
            if articles:
                await loop.run_in_executor(self._db_executor, store_news_articles, base_symbol, articles)
            else:
                # Cache the fact that we found 0 articles to avoid re-fetching too soon
                try:
                    await asyncio.to_thread(
                        self.redis.setex, no_news_cache_key, 300, "1"
                    )
                except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
                    pass
        except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"News fetch/store failed for {symbol}: {type(e).__name__}: {e}")

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
        """Periodically evaluate past LLM decisions against actual market outcomes."""
        await asyncio.sleep(300)  # initial delay 5 minutes
        while self._running:
            try:
                # Evaluate decisions older than 1 hour (3600 seconds)
                pending_decisions = await asyncio.to_thread(get_pending_llm_decisions, 3600)
                if not pending_decisions:
                    await self._interruptible_sleep(600)
                    continue

                # Fetch latest close prices for all symbols in pending decisions
                symbols = [d["symbol"] for d in pending_decisions]
                # get_latest_close_prices expects base symbols without /currency
                base_symbols = [s.split("/")[0] if "/" in s else s for s in symbols]
                prices = await asyncio.to_thread(get_latest_close_prices, base_symbols)

                for decision in pending_decisions:
                    symbol = decision["symbol"]
                    base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
                    price_data = prices.get(base_symbol)
                    if not price_data or price_data.get("last") is None:
                        continue

                    # Skip decisions that haven't reached their timeframe-scaled
                    # evaluation window yet. This prevents judging a 1Y HOLD
                    # after just 1 hour.
                    tf_seconds = timeframe_to_seconds(decision.get("timeframe") or "1d")
                    eval_window = max(3600, min(tf_seconds * 0.1, 604800))
                    decision_age = (time.time() * 1000 - decision["timestamp"]) / 1000
                    if decision_age < eval_window:
                        continue

                    outcome_price = price_data["last"]
                    entry_price = decision["entry_price"]
                    action = decision["action"]

                    if entry_price is None or entry_price <= 0:
                        continue

                    # Determine if the decision was profitable using
                    # timeframe-aware thresholds to filter out noise.
                    tf_seconds = timeframe_to_seconds(decision.get("timeframe") or "1d")
                    # Minimum meaningful price movement (1% or 0.5% for long TFs)
                    min_move_pct = 0.005 if tf_seconds >= 2592000 else 0.01

                    price_change_pct = ((outcome_price - entry_price) / entry_price) if entry_price else 0.0

                    if action == "BUY":
                        # BUY is profitable if price went up by at least min_move_pct
                        profitable = price_change_pct >= min_move_pct
                    elif action == "SELL":
                        # SELL is profitable if price went down by at least min_move_pct
                        profitable = price_change_pct <= -min_move_pct
                    elif action == "HOLD":
                        # HOLD is profitable if price did NOT rise significantly
                        # (avoided a bad buy) OR if it fell (avoided a loss)
                        profitable = price_change_pct < min_move_pct
                    else:
                        profitable = False

                    # Generate a heuristic analysis of why the decision was right or wrong
                    price_change_display = abs(price_change_pct * 100)
                    analysis_parts = []
                    if profitable:
                        analysis_parts.append(f"Price moved {price_change_display:.2f}% in favor of the {action} decision (threshold: {min_move_pct*100:.1f}%).")
                    else:
                        analysis_parts.append(f"Price moved {price_change_display:.2f}% against the {action} decision (threshold: {min_move_pct*100:.1f}%).")
                    
                    ctx = decision.get("market_context")
                    if isinstance(ctx, dict):
                        if ctx.get("market_regime"):
                            analysis_parts.append(f"Market regime was '{ctx['market_regime']}'.")
                        if ctx.get("sentiment") is not None:
                            analysis_parts.append(f"News sentiment was {ctx['sentiment']}.")
                        if ctx.get("atr") is not None:
                            analysis_parts.append(f"ATR was {ctx['atr']:.4f}.")
                    
                    outcome_analysis = " ".join(analysis_parts)

                    await asyncio.to_thread(
                        update_llm_decision_outcome,
                        decision["id"],
                        outcome_price,
                        profitable,
                        outcome_analysis
                    )
                
                logger.info(f"Evaluated {len(pending_decisions)} LLM decisions for quality tracking.")

                # --- Check for accuracy degradation and alert ---
                metrics = await asyncio.to_thread(get_llm_decision_quality_metrics, 7)
                if metrics["total_evaluated"] >= 10 and metrics["accuracy"] < 40.0:
                    # Check cooldown to avoid spamming
                    alert_key = "llm:accuracy_alert_cooldown"
                    cooldown = await asyncio.to_thread(self.redis.get, alert_key)
                    if not cooldown:
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⚠️ LLM Decision Quality Alert: Accuracy has dropped to {metrics['accuracy']:.1f}% "
                                f"over the last 7 days ({metrics['total_evaluated']} evaluated decisions).",
                                summary={
                                    "action": "ALERT",
                                    "reason": "LLM accuracy degradation",
                                    "accuracy": metrics["accuracy"],
                                    "total_evaluated": metrics["total_evaluated"]
                                }
                            )
                        # Set cooldown for 24 hours
                        await asyncio.to_thread(self.redis.setex, alert_key, 86400, "1")

                # Clean up old evaluated decisions to prevent database bloat
                await asyncio.to_thread(cleanup_old_llm_decisions, 30)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"LLM decision evaluation loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"LLM decision evaluation loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("llm_decision_eval_loop", e)
            
            await self._interruptible_sleep(600)  # check every 10 minutes

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
        """Periodically ask the LLM whether to resume trading when paused."""
        await asyncio.sleep(60)  # initial delay
        while self._running:
            try:
                # Skip LLM pause/resume check if auto-resume cooldown is active
                cooldown_active = await asyncio.to_thread(self.redis.get, "trading:auto_resume_cooldown")
                if cooldown_active:
                    logger.debug("Skipping LLM pause/resume check – auto-resume cooldown is active.")
                    await asyncio.sleep(1800)
                    continue

                # Skip LLM pause/resume check if duration-based auto-resume is imminent
                pause_duration_raw = await asyncio.to_thread(self.redis.get, "trading:pause_duration")
                pause_start_raw = await asyncio.to_thread(self.redis.get, "trading:pause_start")
                if pause_duration_raw and pause_start_raw:
                    try:
                        pause_start = float(pause_start_raw)
                        pause_duration = int(pause_duration_raw)
                        remaining = (pause_start + pause_duration) - time.time()
                        if remaining < 120:  # less than 2 minutes until auto-resume
                            logger.debug("Skipping LLM pause/resume check – duration-based auto-resume imminent.")
                            await asyncio.sleep(1800)
                            continue
                    except (ValueError, TypeError):
                        pass

                if await self._is_market_open():
                    await self.event_bus.request("check_pause_resume_decision")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Pause/resume check network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Pause/resume check data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("pause_resume_check", e)
            await asyncio.sleep(1800)  # every 30 minutes

    async def _redis_health_check_loop(self):
        """Periodically check Redis connection and alert on state changes."""
        await asyncio.sleep(30)
        last_degraded_notify_time = 0.0
        while self._running:
            try:
                was_available = is_redis_available()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._db_executor, check_redis_connection)
                is_available = is_redis_available()
                if was_available and not is_available:
                    logger.critical("Redis connection lost. Degrading to no-cache mode.")
                    if self.notifier:
                        await self.notifier.send_notification(
                            "⚠️ Redis connection lost. The bot is now running in degraded mode (caching disabled).",
                            summary={"action": "ERROR", "reason": "Redis connection lost"}
                        )
                    last_degraded_notify_time = time.time()
                elif not was_available and is_available:
                    logger.info("Redis connection restored. Caching resumed.")
                    if self.notifier:
                        await self.notifier.send_notification(
                            "✅ Redis connection restored. Caching resumed.",
                            summary={"action": "INFO", "reason": "Redis connection restored"}
                        )
                elif not is_available:
                    # Periodic reminder every 6 hours if still in degraded mode
                    if time.time() - last_degraded_notify_time >= 21600:
                        if self.notifier:
                            await self.notifier.send_notification(
                                "⚠️ Redis is still unavailable. The bot remains in degraded mode (caching disabled).",
                                summary={"action": "ERROR", "reason": "Redis still unavailable"}
                            )
                        last_degraded_notify_time = time.time()
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Redis health check loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Redis health check loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("redis_health_check_loop", e)
            await asyncio.sleep(60)

    async def _health_check_loop(self):
        """Periodically check the health of supervised background tasks and alert on failures."""
        await asyncio.sleep(300)  # initial delay 5 minutes
        alerted_supervisors = set()
        while self._running:
            try:
                for sup in self._supervisors:
                    health = sup.get_health()
                    if not health["is_healthy"] or not health["running"]:
                        if sup.name not in alerted_supervisors:
                            alerted_supervisors.add(sup.name)
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"🚨 Critical background task '{health['name']}' is not healthy or has stopped running. "
                                    f"Last error: {health['last_exception']}",
                                    summary={"action": "CRITICAL", "reason": f"Task {health['name']} unhealthy"}
                                )
                    else:
                        # If the task somehow becomes healthy/running again, clear the alert flag
                        alerted_supervisors.discard(sup.name)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Health check loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Health check loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("health_check_loop", e)
            await asyncio.sleep(300)  # check every 5 minutes

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
        """Periodically check tracked symbols for favourable entry conditions.
        When a condition is met, force an immediate LLM evaluation."""
        await asyncio.sleep(10)  # initial delay
        while self._running:
            try:
                for entry in self.shared_state.current_symbols:
                    symbol = entry["symbol"]
                    tf = entry["timeframe"]
                    # Skip entry signal monitoring for very long timeframes (>= 1 month)
                    # where short-term crossovers are irrelevant.
                    tf_seconds = timeframe_to_seconds(tf)
                    # Avoid re‑triggering too often – enforce a cooldown of at least
                    # the normal strategy interval.
                    # Use a short, dedicated cooldown so the bot reacts quickly to new signals
                    cooldown = settings.ENTRY_SIGNAL_COOLDOWN_SECONDS
                    async with self.shared_state._eval_state_lock:
                        last_forced = self.shared_state._force_eval_time.get(symbol, 0)
                    if time.time() - last_forced < cooldown:
                        continue

                    if await self.event_bus.request("detect_entry_signal", symbol, tf):
                        logger.info(f"Entry signal detected for {symbol}, forcing LLM evaluation.")
                        async with self.shared_state._eval_state_lock:
                            self.shared_state._force_eval[symbol] = True
                            self.shared_state._force_eval_time[symbol] = time.time()
                            # Clear last evaluation timestamp so the main loop picks it up immediately
                            self.shared_state._last_strategy_eval.pop(symbol, None)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Entry signal monitor network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Entry signal monitor data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("entry_signal_monitor", e)
            await self._interruptible_sleep(settings.ENTRY_SIGNAL_CHECK_INTERVAL_SECONDS)

    async def _check_pending_entries(self):
        """Periodically check pending entry conditions and execute if met."""
        await asyncio.sleep(10)  # short initial delay
        while self._running:
            try:
                now = time.time()
                for symbol in list(self.shared_state._pending_entries.keys()):
                    await self.event_bus.request("process_pending_entry", symbol, now)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Pending entries check network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Error checking pending entries data/logic: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("check_pending_entries", e)
            await asyncio.sleep(60)  # check every 60 seconds (medium/long-term)

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
        """Periodically check queued limit orders in the simulator and process fills,
        including partial fills.

        Instead of re-executing the signal (which would place a duplicate order
        while the original is still open), we poll the actual simulator order
        status. When the order fills (fully or partially), we record the trade
        and update positions exactly as a normal fill would. We never place a
        new order here.
        """
        await asyncio.sleep(10)
        # --- Notify mode: no queued order processing ---
        if settings.TRADING_MODE == "notify":
            logger.info("Notify mode: skipping queued order processing.")
            return
        while self._running:
            try:
                for queued in list(self.shared_state.queued_orders):
                    await self.event_bus.request("process_single_queued_order", queued)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Queued orders processing network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Error processing queued orders data/logic: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("process_queued_orders", e)
            await asyncio.sleep(15)  # check every 15 seconds for faster fill detection

    async def _cleanup_orphaned_orders(self):
        """Periodically cancel any open orders that are older than 10 minutes,
        but never cancel orders that are still being tracked as queued."""
        await asyncio.sleep(120)  # initial delay
        # --- Notify mode: no orphaned order cleanup ---
        if settings.TRADING_MODE == "notify":
            return
        while self._running:
            try:
                await self.event_bus.request("cancel_orphaned_orders")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Orphaned order cleanup network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Orphaned order cleanup data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("cleanup_orphaned_orders", e)
            await asyncio.sleep(900)  # every 15 minutes

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
        data = await self._signal_processor.simulation_manager.prepare_simulation_data(symbol)
        if "error" in data:
            return data

        model_type = data.get("model_type", "mind")
        temperature = data.get("temperature", 0.2)
        market_hash = data.get("market_hash")

        _analysis, step1b_response, preliminary_signal, error = await self._signal_processor.simulation_manager.run_simulation_step1(symbol, data)
        if error is not None:
            return error

        if preliminary_signal.action in ("BUY", "HOLD"):
            backtest_results, combined_bt_summary = await self._backtest_manager.run_simulation_backtests(
                symbol=symbol,
                data=data,
                preliminary_signal=preliminary_signal,
            )

            return {
                "step1_response": step1b_response,
                "action": preliminary_signal.action,
                "backtest_summary": combined_bt_summary,
                "backtest_results": backtest_results,
            }
        else:
            return {
                "step1_response": step1b_response,
                "action": preliminary_signal.action,
                "backtest_summary": "No backtest performed (action is SELL)",
            }

    async def simulate_decision(self, symbol: str) -> Dict[str, Any]:
        """Simulate Step 1a (analysis), Step 1b (variants), and Step 2 (final decision) without executing trades."""
        data = await self._signal_processor.simulation_manager.prepare_simulation_data(symbol)
        if "error" in data:
            return data

        model_type = data.get("model_type", "mind")
        temperature = data.get("temperature", 0.2)
        market_hash = data.get("market_hash")

        _analysis, step1b_response, preliminary_signal, error = await self._signal_processor.simulation_manager.run_simulation_step1(symbol, data)
        if error is not None:
            return error

        if preliminary_signal.action == "SELL":
            return {
                "step1_response": step1b_response,
                "step2_response": "N/A (Step 1 action is SELL)",
                "action": preliminary_signal.action,
                "backtest_summary": "No backtest performed (action is SELL)",
            }

        backtest_results, combined_bt_summary = await self._backtest_manager.run_simulation_backtests(
            symbol=symbol,
            data=data,
            preliminary_signal=preliminary_signal,
        )

        data["step1b_response"] = step1b_response
        step2_response, _error, final_signal, error_dict = await self._backtest_manager.run_simulation_step2(
            symbol=symbol,
            data=data,
            preliminary_signal=preliminary_signal,
            backtest_results=backtest_results,
            combined_bt_summary=combined_bt_summary,
        )
        if error_dict is not None:
            return error_dict

        return {
            "step1_response": step1b_response,
            "step2_response": step2_response,
            "action": final_signal.action,
            "backtest_summary": combined_bt_summary,
            "backtest_results": backtest_results,
        }
