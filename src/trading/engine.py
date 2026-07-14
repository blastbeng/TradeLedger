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
from src.exchanges.market_data import get_tradable_assets, get_quotes, get_quotes_cached, get_multi_timeframe_bars, get_bars_range, discover_btp_bonds, discover_italian_ucits_etfs, _get_yf_session, _check_yf_circuit
from src.exchanges.yahoo_finance import get_yahoo_quote, get_yahoo_fundamentals, get_yahoo_dividends
from src.exchanges.yf_session import _invalidate_yf_session
from src.trading.paper_trader import PaperTrader
from src.llm.cache import get_cached_llm_response, compute_market_hash
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
from src.database import load_trading_state, save_trading_state, insert_trade, get_performance, store_news_articles, get_aggregate_sentiment_from_db, get_aggregate_sentiment_for_symbols, get_news_for_symbol, get_ohlcv, get_latest_ohlcv_timestamp, get_latest_ohlcv_timestamps_batch, insert_ohlcv_batch, save_paper_balances, load_paper_balances, cleanup_old_ohlcv, save_indicators, get_indicators, get_indicators_for_symbols, get_ohlcv_summary_for_symbols, get_all_trades, get_latest_close_prices, insert_position_pnl_snapshot, cleanup_old_position_pnl, save_backtest_result, get_recent_backtest_result, get_backtest_results_for_symbol, cleanup_old_backtest_results, reset_paper_trading_data, insert_dividend, cleanup_old_dividends, get_pending_llm_decisions, update_llm_decision_outcome, get_llm_decision_quality_metrics, cleanup_old_llm_decisions
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
from src.config.config_service import UnifiedConfigService

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
        self._exchange_semaphore = asyncio.Semaphore(10)  # max 10 concurrent API calls
        self._news_semaphore = asyncio.Semaphore(5)  # max 5 concurrent news fetches
        self._indicator_semaphore = asyncio.Semaphore(4)  # limit concurrent indicator computations
        self._backtest_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_BACKTESTS)  # limit concurrent backtest variants
        self._download_semaphore = asyncio.Semaphore(5)  # max 5 concurrent background OHLCV backfills
        self._symbol_processing_semaphore = asyncio.Semaphore(3)  # limit concurrent symbol evaluations

        # Dedicated thread pool for database writes – prevents write contention
        # from starving the default asyncio thread pool used by the web server,
        # Telegram bot, and all other to_thread calls.
        self._db_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="dbwriter")
        # Dedicated thread pool for download/indicator operations – prevents
        # download tasks from exhausting the default asyncio thread pool used
        # by the web server, Telegram bot, and engine loop.
        self._download_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="downloader")
        # Dedicated thread pool for quote fetching – prevents zombie get_quotes
        # threads (from asyncio.wait_for timeouts) from exhausting the default
        # asyncio thread pool used by the web server and Telegram bot.
        self._quote_executor = ThreadPoolExecutor(max_workers=12, thread_name_prefix="quotes")

        self.initial_balance: float = 0.0
        self.notifier = None

        # _load_state() and _ensure_cost_basis() are now called in _initialize_clients()
        # after the trading client is available.

        # Clear stale pause keys immediately (Redis is already available)
        from src.utils.pause_utils import clear_trading_pause_keys
        clear_trading_pause_keys(self.redis)

        # Track quote currency spent in the current cycle to avoid over-allocating
        self._cycle_spent_lock = self.shared_state._cycle_spent_lock
        self._positions_lock = self.shared_state._positions_lock
        self._pending_entries_lock = self.shared_state._pending_entries_lock
        self._queued_orders_lock = self.shared_state._queued_orders_lock
        self._state_lock = self.shared_state._state_lock
        self._trade_history_lock = self.shared_state._trade_history_lock
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
        self._symbol_reeval_lock = asyncio.Lock()
        self._tradable_assets_lock = asyncio.Lock()
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
            self._on_settings_reload()
        settings.register_reload_callback(_reload_callback)
        self._last_state_save = 0

        # Re-entrancy guards for periodic tasks
        self._reconcile_running = False
        self._reevaluate_running = False
        self._pause_check_running = False
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
        # Lock to protect _force_eval, _force_eval_time, _last_strategy_eval, and _strategy_intervals
        self._eval_state_lock = self.shared_state._eval_state_lock

        # Log the complete event subscription registry after all components are initialized
        self.event_bus.log_subscription_summary()

    def _clear_time_sensitive_redis_keys(self):
        """Clear time-sensitive Redis keys on startup to prevent stale data."""
        keys_to_clear = [
            "trading:last_triggered_reeval",
            "market:breadth:full",
        ]
        try:
            for key in keys_to_clear:
                self.redis.delete(key)
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Failed to clear time-sensitive Redis keys: {e}")

    # --- Scalar state properties (proxy to SharedState) ---
    @property
    def current_symbols(self) -> List[Dict[str, str]]:
        return self.shared_state.current_symbols

    @current_symbols.setter
    def current_symbols(self, value: List[Dict[str, str]]) -> None:
        self.shared_state.current_symbols = value

    @property
    def positions(self) -> Dict[str, Dict[str, Any]]:
        return self.shared_state.positions

    @positions.setter
    def positions(self, value: Dict[str, Dict[str, Any]]) -> None:
        self.shared_state.positions = value

    @property
    def trade_history(self) -> List[Dict[str, Any]]:
        return self.shared_state.trade_history

    @trade_history.setter
    def trade_history(self, value: List[Dict[str, Any]]) -> None:
        self.shared_state.trade_history = value

    @property
    def recent_signals(self) -> List[Dict[str, Any]]:
        return self.shared_state.recent_signals

    @recent_signals.setter
    def recent_signals(self, value: List[Dict[str, Any]]) -> None:
        self.shared_state.recent_signals = value

    @property
    def last_loss_time(self) -> Dict[str, float]:
        return self.shared_state.last_loss_time

    @last_loss_time.setter
    def last_loss_time(self, value: Dict[str, float]) -> None:
        self.shared_state.last_loss_time = value

    @property
    def cooldown_durations(self) -> Dict[str, float]:
        return self.shared_state.cooldown_durations

    @cooldown_durations.setter
    def cooldown_durations(self, value: Dict[str, float]) -> None:
        self.shared_state.cooldown_durations = value

    @property
    def _last_strategy_eval(self) -> Dict[str, float]:
        return self.shared_state._last_strategy_eval

    @_last_strategy_eval.setter
    def _last_strategy_eval(self, value: Dict[str, float]) -> None:
        self.shared_state._last_strategy_eval = value

    @property
    def _strategy_intervals(self) -> Dict[str, float]:
        return self.shared_state._strategy_intervals

    @_strategy_intervals.setter
    def _strategy_intervals(self, value: Dict[str, float]) -> None:
        self.shared_state._strategy_intervals = value

    @property
    def _symbol_first_seen(self) -> Dict[str, float]:
        return self.shared_state._symbol_first_seen

    @_symbol_first_seen.setter
    def _symbol_first_seen(self, value: Dict[str, float]) -> None:
        self.shared_state._symbol_first_seen = value

    @property
    def queued_orders(self) -> List[Dict[str, Any]]:
        return self.shared_state.queued_orders

    @queued_orders.setter
    def queued_orders(self, value: List[Dict[str, Any]]) -> None:
        self.shared_state.queued_orders = value

    @property
    def _force_eval(self) -> Dict[str, bool]:
        return self.shared_state._force_eval

    @_force_eval.setter
    def _force_eval(self, value: Dict[str, bool]) -> None:
        self.shared_state._force_eval = value

    @property
    def _force_eval_time(self) -> Dict[str, float]:
        return self.shared_state._force_eval_time

    @_force_eval_time.setter
    def _force_eval_time(self, value: Dict[str, float]) -> None:
        self.shared_state._force_eval_time = value

    @property
    def _entry_signal_state(self) -> Dict[str, Dict[str, Any]]:
        return self.shared_state._entry_signal_state

    @_entry_signal_state.setter
    def _entry_signal_state(self, value: Dict[str, Dict[str, Any]]) -> None:
        self.shared_state._entry_signal_state = value

    @property
    def _last_decisions(self) -> Dict[str, Dict[str, Any]]:
        return self.shared_state._last_decisions

    @_last_decisions.setter
    def _last_decisions(self, value: Dict[str, Dict[str, Any]]) -> None:
        self.shared_state._last_decisions = value

    @property
    def _last_eval_snapshot(self) -> Dict[str, Dict[str, float]]:
        return self.shared_state._last_eval_snapshot

    @_last_eval_snapshot.setter
    def _last_eval_snapshot(self, value: Dict[str, Dict[str, float]]) -> None:
        self.shared_state._last_eval_snapshot = value

    @property
    def _pending_entries(self) -> Dict[str, Dict[str, Any]]:
        return self.shared_state._pending_entries

    @_pending_entries.setter
    def _pending_entries(self, value: Dict[str, Dict[str, Any]]) -> None:
        self.shared_state._pending_entries = value

    @property
    def _sentiment_cache(self) -> Dict[str, tuple]:
        return self.shared_state._sentiment_cache

    @_sentiment_cache.setter
    def _sentiment_cache(self, value: Dict[str, tuple]) -> None:
        self.shared_state._sentiment_cache = value

    @property
    def _cycle_spent(self) -> float:
        return self.shared_state._cycle_spent

    @_cycle_spent.setter
    def _cycle_spent(self, value: float) -> None:
        self.shared_state._cycle_spent = value

    @property
    def _state_save_pending(self) -> bool:
        return self.shared_state._state_save_pending

    @_state_save_pending.setter
    def _state_save_pending(self, value: bool) -> None:
        self.shared_state._state_save_pending = value

    @property
    def _state_dirty(self) -> bool:
        return self.shared_state._state_dirty

    @_state_dirty.setter
    def _state_dirty(self, value: bool) -> None:
        self.shared_state._state_dirty = value

    @property
    def _global_risk_multiplier(self) -> Optional[float]:
        return self.shared_state._global_risk_multiplier

    @_global_risk_multiplier.setter
    def _global_risk_multiplier(self, value: Optional[float]) -> None:
        self.shared_state._global_risk_multiplier = value

    @property
    def _balance_cache(self) -> Optional[Dict[str, float]]:
        return self.shared_state._balance_cache

    @_balance_cache.setter
    def _balance_cache(self, value: Optional[Dict[str, float]]) -> None:
        self.shared_state._balance_cache = value

    @property
    def _balance_cache_time(self) -> float:
        return self.shared_state._balance_cache_time

    @_balance_cache_time.setter
    def _balance_cache_time(self, value: float) -> None:
        self.shared_state._balance_cache_time = value

    @property
    def _position_tickers_cache(self) -> Optional[Dict[str, Dict[str, Any]]]:
        return self.shared_state._position_tickers_cache

    @_position_tickers_cache.setter
    def _position_tickers_cache(self, value: Optional[Dict[str, Dict[str, Any]]]) -> None:
        self.shared_state._position_tickers_cache = value

    @property
    def _position_tickers_cache_time(self) -> float:
        return self.shared_state._position_tickers_cache_time

    @_position_tickers_cache_time.setter
    def _position_tickers_cache_time(self, value: float) -> None:
        self.shared_state._position_tickers_cache_time = value

    @property
    def _market_breadth(self) -> Optional[Dict[str, Any]]:
        return self.shared_state._market_breadth

    @_market_breadth.setter
    def _market_breadth(self, value: Optional[Dict[str, Any]]) -> None:
        self.shared_state._market_breadth = value

    @property
    def _trade_history_version(self) -> int:
        return self.shared_state._trade_history_version

    @_trade_history_version.setter
    def _trade_history_version(self, value: int) -> None:
        self.shared_state._trade_history_version = value

    @property
    def _realized_pnl_offset(self) -> float:
        return self.shared_state._realized_pnl_offset

    @_realized_pnl_offset.setter
    def _realized_pnl_offset(self, value: float) -> None:
        self.shared_state._realized_pnl_offset = value

    @property
    def _portfolio_exposure_cache(self) -> Optional[Dict[str, float]]:
        return self.shared_state._portfolio_exposure_cache

    @_portfolio_exposure_cache.setter
    def _portfolio_exposure_cache(self, value: Optional[Dict[str, float]]) -> None:
        self.shared_state._portfolio_exposure_cache = value

    @property
    def _delayed_entry_tasks(self) -> set:
        return self.shared_state._delayed_entry_tasks

    @_delayed_entry_tasks.setter
    def _delayed_entry_tasks(self, value: set) -> None:
        self.shared_state._delayed_entry_tasks = value

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
                q.get('amount', 0.0) for q in self.queued_orders
                if q.get('side') == 'buy'
            )
            async with self._cycle_spent_lock:
                self._cycle_spent = queued_buy_total
            if queued_buy_total > 0:
                logger.info(f"Initialized _cycle_spent={queued_buy_total:.2f} from {sum(1 for q in self.queued_orders if q.get('side') == 'buy')} queued buy orders.")

        # Persist the current PAPER_INITIAL_BALANCE so we can detect changes on next startup
        await asyncio.to_thread(save_trading_state, "paper_initial_balance", settings.PAPER_INITIAL_BALANCE)

    async def reset_paper_trading_state(self):
        """Reset paper trading state."""
        logger.info("Resetting paper trading state...")

        # Clear in-memory state
        self.positions.clear()
        self.queued_orders.clear()
        self.current_symbols.clear()
        self._pending_entries.clear()
        async with self._eval_state_lock:
            self._last_strategy_eval.clear()
            self._strategy_intervals.clear()
            self._force_eval.clear()
            self._force_eval_time.clear()
        self._entry_signal_state.clear()
        self._last_decisions.clear()
        self._last_eval_snapshot.clear()
        async with self._cycle_spent_lock:
            self._cycle_spent = 0.0
        self._balance_cache = None
        self._balance_cache_time = 0.0
        self._position_tickers_cache = None
        self._position_tickers_cache_time = 0.0
        self._perf_cache = None
        self._perf_cache_time = 0.0
        self._perf_cache_trade_count = -1
        self._trade_pattern_cache = None
        self._trade_pattern_cache_trade_count = -1
        self._trade_history_version = 0
        self._realized_pnl_offset = 0.0
        self.trade_history.clear()
        self.recent_signals.clear()
        self.last_loss_time.clear()
        self.cooldown_durations.clear()
        self._global_risk_multiplier = None
        self._symbol_first_seen.clear()
        self._sentiment_cache.clear()
        self._market_breadth = None
        self.initial_balance = settings.PAPER_INITIAL_BALANCE

        # Reset DB data (unconditionally clear all trade data for both modes)
        await asyncio.to_thread(reset_paper_trading_data, keep_trade_history=False)

        # Re-initialize paper trader with new balance
        self.trader = PaperTrader()

        # Save the fresh state
        self._state_dirty = True
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
            plain_assets = await self._market_data_manager.get_tradable_assets()
            stock_pairs = [f"{sym}/{self.base_currency}" for sym in plain_assets]

            btp_bonds = await self._market_data_manager.get_btp_bonds()
            btp_pairs = [f"{b['isin']}/{self.base_currency}" for b in btp_bonds]

            all_pairs = stock_pairs + btp_pairs
            if not all_pairs:
                logger.warning("Force download: no tradable assets found.")
                return

            now_ms = int(time.time() * 1000)
            start_ms = now_ms - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000

            random.shuffle(all_pairs)

            async def _force_download_symbol(pair: str):
                for tf in settings.OHLCV_TIMEFRAMES:
                    await self._market_data_manager._download_symbol_ohlcv(pair, tf, start_ms, now_ms, quiet=True, force=True)

            # Limit concurrent symbol downloads to 2 to avoid exhausting the
            # _download_executor thread pool, leaving threads available for
            # tracked tickers.
            download_concurrency = asyncio.Semaphore(2)
            async def _limited_force_download(pair: str):
                async with download_concurrency:
                    await _force_download_symbol(pair)
            download_tasks = [_limited_force_download(pair) for pair in all_pairs]
            await asyncio.gather(*download_tasks)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._db_executor, cleanup_old_ohlcv, settings.OHLCV_RETENTION_DAYS)
            logger.info("Force download: complete.")
        except asyncio.CancelledError:
            raise
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Force download network/IO error: {type(e).__name__}: {e}")
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
            logger.error(f"Force download data/logic error: {type(e).__name__}: {e}", exc_info=True)
            await self._record_unexpected_exception("force_download_all_assets", e)
        except Exception as e:
            logger.error(f"Force download error: {type(e).__name__}: {e}", exc_info=True)
            await self._record_unexpected_exception("force_download_all_assets", e)
        finally:
            self._full_download_running = False

    async def force_download_tracked_symbols(self):
        """Immediately download OHLCV data for currently tracked symbols only."""
        logger.info("Force download: starting immediate OHLCV download for tracked symbols...")
        try:
            tracked_pairs = [entry["symbol"] for entry in self.current_symbols]
            if not tracked_pairs:
                logger.warning("Force download: no tracked symbols found.")
                return

            now_ms = int(time.time() * 1000)
            start_ms = now_ms - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000

            async def _force_download_symbol(pair: str):
                for tf in settings.OHLCV_TIMEFRAMES:
                    await self._market_data_manager._download_symbol_ohlcv(pair, tf, start_ms, now_ms, quiet=True, force=True)

            download_concurrency = asyncio.Semaphore(10)
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
        except Exception as e:
            logger.error(f"Force download tracked symbols error: {type(e).__name__}: {e}", exc_info=True)
            await self._record_unexpected_exception("force_download_tracked_symbols", e)

    async def _get_cached_balance(self, ttl: float = 30.0) -> Dict[str, float]:
        """Return cached balance, refreshing if older than ttl seconds."""
        now = time.time()
        if self._balance_cache is not None and (now - self._balance_cache_time) < ttl:
            return self._balance_cache
        balance = await asyncio.to_thread(self.trader.fetch_balance)
        self._balance_cache = balance
        self._balance_cache_time = now
        return balance

    async def _get_cached_position_tickers(self, ttl: float = 30.0) -> Dict[str, Dict[str, Any]]:
        """Return cached position tickers, refreshing if older than ttl seconds."""
        now = time.time()
        if (
            self._position_tickers_cache is not None
            and (now - self._position_tickers_cache_time) < ttl
        ):
            return self._position_tickers_cache
        tickers = await self._market_data_manager._get_all_position_tickers()
        self._position_tickers_cache = tickers
        self._position_tickers_cache_time = now
        return tickers

    async def _get_cached_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return aggregate news sentiment, cached for 60 seconds to reduce DB load."""
        base = symbol.split("/")[0] if "/" in symbol else symbol
        now = time.time()
        cached = self._sentiment_cache.get(base)
        if cached and (now - cached[0]) < 60:
            return cached[1]
        try:
            agg = await asyncio.to_thread(
                get_aggregate_sentiment_from_db, base, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS
            )
            self._sentiment_cache[base] = (now, agg)
            return agg
        except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Failed to fetch sentiment for {base}: {type(e).__name__}: {e}")
            return None

    async def stop(self):
        """Gracefully stop the engine and all background tasks."""
        logger.info("Stopping trading engine...")
        self._running = False
        for task in self._delayed_entry_tasks:
            task.cancel()
        self._delayed_entry_tasks.clear()
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

    def _on_settings_reload(self):
        """Update cached settings values when settings are reloaded."""
        self.base_currency = settings.BASE_CURRENCY
        self.max_symbols = settings.MAX_SYMBOLS
        self.effective_max_symbols = self.max_symbols
        self._symbol_reevaluation_interval = settings.SYMBOL_REEVALUATION_INTERVAL
        # Invalidate yfinance session so it's recreated with new proxy settings
        _invalidate_yf_session()
        # Update backtest concurrency semaphore to pick up MAX_CONCURRENT_BACKTESTS changes
        self._backtest_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_BACKTESTS)
        # Update paper trader's base currency if it exists
        if self.trader is not None:
            self.trader.base_currency = settings.BASE_CURRENCY
        # Invalidate clock cache so market hours/timezone changes take effect immediately
        self._market_data_manager.invalidate_clock_cache()

    async def _periodic_reconcile(self):
        """Run position reconciliation every 5 minutes (medium/long-term)."""
        while self._running:
            if self._reconcile_running:
                logger.warning("Reconcile still running; skipping this cycle.")
                await asyncio.sleep(60)
                continue
            self._reconcile_running = True
            try:
                await self.event_bus.request("reconcile_positions")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Reconcile network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Reconcile data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("reconcile", e)
            except Exception as e:
                logger.error(f"Reconcile error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("reconcile", e)
            finally:
                self._reconcile_running = False
            await asyncio.sleep(300)

    async def _periodic_reevaluate(self):
        """Re-evaluate stock selection periodically."""
        # Initial delay to allow WebSocket and Telegram bot to initialize
        logger.info(
            f"Waiting {settings.INITIAL_EVALUATION_DELAY_SECONDS}s before initial symbol evaluation..."
        )
        await asyncio.sleep(settings.INITIAL_EVALUATION_DELAY_SECONDS)
        while self._running:
            if self._reevaluate_running:
                # Wait briefly for the current re-evaluation to finish.
                # Use a short sleep so queued triggers are picked up quickly.
                await asyncio.sleep(settings.SYMBOL_EVALUATION_DELAY_SECONDS)
                continue
            self._reevaluate_running = True
            try:
                # Always run re-evaluation, even if paused, to keep generating signals
                reeval_start_time = time.time()
                logger.info("Starting symbol re-evaluation...")
                is_forced = self._force_reeval or self._reeval_pending_force
                self._force_reeval = False
                self._reeval_pending_force = False
                async with self._symbol_reeval_lock:
                    await self.event_bus.request("reevaluate_symbols_impl", force=is_forced)
                elapsed = time.time() - reeval_start_time
                logger.info(f"Symbol re-evaluation complete (took {elapsed:.1f}s).")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Stock re-evaluation network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Stock re-evaluation data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("reevaluate", e)
                if self.notifier:
                    await self.notifier.send_notification(
                        f"❌ Stock re-evaluation failed: {str(e)[:200]}",
                        summary={
                            "action": "ERROR",
                            "reason": f"Re-evaluation error: {str(e)[:200]}",
                        }
                    )
            except Exception as e:
                logger.error(f"Stock re-evaluation error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("reevaluate", e)
                if self.notifier:
                    await self.notifier.send_notification(
                        f"❌ Stock re-evaluation failed: {str(e)[:200]}",
                        summary={
                            "action": "ERROR",
                            "reason": f"Re-evaluation error: {str(e)[:200]}",
                        }
                    )
            finally:
                self._reevaluate_running = False
            self._settings_reload_event.clear()
            wait_task = asyncio.create_task(self._reeval_trigger.wait())
            reload_task = asyncio.create_task(self._settings_reload_event.wait())
            await asyncio.wait(
                [wait_task, reload_task],
                timeout=settings.SYMBOL_REEVALUATION_INTERVAL,
                return_when=asyncio.FIRST_COMPLETED
            )
            for task in (wait_task, reload_task):
                if not task.done():
                    try:
                        task.cancel()
                    except asyncio.CancelledError:
                        pass
            self._reeval_trigger.clear()

    async def _clear_pause_and_resume(self, reason: str, notification_msg: str, notification_summary: dict) -> None:
        """Helper to clear pause keys, set resume cooldown, and notify."""
        from src.utils.pause_utils import clear_trading_pause_keys
        await asyncio.to_thread(clear_trading_pause_keys, self.redis)
        self._reeval_trigger.set()
        await asyncio.to_thread(self.redis.set, "trading:last_auto_resume", str(time.time()))
        await asyncio.to_thread(self.redis.setex, "trading:auto_resume_cooldown", 600, "1")
        if self.notifier:
            await self.notifier.send_notification(notification_msg, summary=notification_summary)

    async def _handle_missing_pause_duration(self, pause_start_raw: Optional[bytes]) -> Tuple[bool, bool]:
        """Handle fallback when no pause_duration was set.
        
        Returns a tuple (skip_normal_logic, resumed):
        - skip_normal_logic: True if the caller should skip normal duration logic.
        - resumed: True if trading was actually resumed.
        """
        default_max_pause = settings.MIN_LLM_PAUSE_DURATION
        try:
            raw = await self.config_service.get_config("min_llm_pause_duration")
            if raw:
                default_max_pause = int(raw)
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError):
            pass

        if pause_start_raw is None:
            logger.warning("Pause has no duration and no start time; forcing auto-resume immediately.")
            await self._clear_pause_and_resume(
                "Fallback: no pause start time",
                "⏰ Trading auto-resumed (pause had no duration and no start time).",
                {"action": "RESUME", "reason": "Fallback: no pause start time"}
            )
            return True, True

        try:
            elapsed = time.time() - float(pause_start_raw)
            if elapsed >= default_max_pause:
                logger.warning(f"Pause has no duration; forcing auto-resume after default fallback ({default_max_pause // 60} minutes).")
                await self._clear_pause_and_resume(
                    "Fallback pause timeout",
                    "⏰ Trading auto‑resumed after maximum pause duration (no LLM‑set duration).",
                    {"action": "RESUME", "reason": "Fallback pause timeout"}
                )
                return True, True
        except (ValueError, TypeError):
            pass
        
        # Did not resume, but we still need to skip normal duration logic
        return True, False

    async def _handle_pause_duration_elapsed(self, pause_start_raw: bytes, pause_duration_raw: bytes) -> None:
        """Check if the pause duration has elapsed and resume if so."""
        try:
            pause_start = float(pause_start_raw)
            pause_duration = int(pause_duration_raw)
            if time.time() - pause_start >= pause_duration:
                logger.info("Pause duration elapsed – auto-resuming trading.")
                stored_reason_raw = await asyncio.to_thread(self.redis.get, "trading:pause_reason")
                stored_reason = stored_reason_raw.decode() if isinstance(stored_reason_raw, bytes) else (stored_reason_raw or "")
                reason_text = f" (was paused: {stored_reason})" if stored_reason else ""
                await self._clear_pause_and_resume(
                    f"Pause duration elapsed{reason_text}",
                    f"▶️ Trading auto-resumed after pause duration elapsed.{reason_text}",
                    {"action": "RESUME", "reason": f"Pause duration elapsed{reason_text}"}
                )
        except (ValueError, TypeError):
            pass  # ignore malformed values

    async def _periodic_pause_check(self):
        """Check and handle auto-resume from pause (only for LLM-initiated pauses)."""
        while self._running:
            if self._pause_check_running:
                logger.warning("Pause check still running; skipping this cycle.")
                await asyncio.sleep(30)
                continue
            self._pause_check_running = True
            try:
                paused = await asyncio.to_thread(self.redis.get, "trading:paused")
                if paused:
                    # Only auto-resume if the pause was initiated by the LLM
                    source = await asyncio.to_thread(self.redis.get, "trading:pause_source")
                    if source and (source.decode() if isinstance(source, bytes) else source) == "llm":
                        pause_start_raw = await asyncio.to_thread(self.redis.get, "trading:pause_start")
                        pause_duration_raw = await asyncio.to_thread(self.redis.get, "trading:pause_duration")

                        # --- Fallback if no pause_duration was set ---
                        if not pause_duration_raw:
                            skip_normal, _resumed = await self._handle_missing_pause_duration(pause_start_raw)
                            if skip_normal:
                                await asyncio.sleep(30)
                                continue   # skip the original duration logic, proceed to next loop iteration

                        if pause_start_raw and pause_duration_raw:
                            await self._handle_pause_duration_elapsed(pause_start_raw, pause_duration_raw)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Pause check network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Pause check data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("pause_check", e)
            except Exception as e:
                logger.error(f"Pause check error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("pause_check", e)
            finally:
                self._pause_check_running = False
            await asyncio.sleep(30)

    async def _periodic_full_market_breadth(self):
        """Periodically compute market breadth over all available pairs.

        Uses cached quotes (Redis/DB only, no network calls) to avoid
        thread pool exhaustion. Falls back to DB close prices for symbols
        without cached quotes. Uses a random sample of up to 500 symbols
        when the universe is larger, ensuring a representative sample.
        """
        await asyncio.sleep(60)  # initial delay
        while self._running:
            if self._full_breadth_running:
                logger.warning("Full market breadth computation still running; skipping this cycle.")
                await asyncio.sleep(300)
                continue
            self._full_breadth_running = True
            try:
                # Fetch all asset types for stratified sampling
                stock_assets = await self._market_data_manager.get_tradable_assets()
                stock_pairs = [f"{sym}/{self.base_currency}" for sym in stock_assets]
                etf_symbols = await self._market_data_manager.get_etf_symbols()
                etf_pairs = [f"{sym}/{self.base_currency}" for sym in etf_symbols]
                btp_bonds = await self._market_data_manager.get_btp_bonds()
                btp_pairs = [f"{b['isin']}/{self.base_currency}" for b in btp_bonds]

                # Build strata: (pairs, label) for each asset type
                strata = [
                    (stock_pairs, "stocks"),
                    (etf_pairs, "etfs"),
                    (btp_pairs, "btps"),
                ]
                # Filter out empty strata
                strata = [(pairs, label) for pairs, label in strata if pairs]

                available_pairs = stock_pairs + etf_pairs + btp_pairs
                if available_pairs:
                    MAX_BREADTH_SAMPLE = 200
                    if len(available_pairs) <= MAX_BREADTH_SAMPLE:
                        # Universe is small enough — use everything
                        breadth_pairs = available_pairs
                    else:
                        # Proportional stratified sampling across asset types
                        total_universe = len(available_pairs)
                        breadth_pairs = []
                        for pairs, label in strata:
                            # Proportional allocation: stratum_size / total * MAX_SAMPLE
                            stratum_sample_size = max(1, round(len(pairs) / total_universe * MAX_BREADTH_SAMPLE))
                            # Cap at the stratum's actual size
                            stratum_sample_size = min(stratum_sample_size, len(pairs))
                            sampled = random.sample(pairs, stratum_sample_size)
                            breadth_pairs.extend(sampled)
                            logger.debug(
                                f"Breadth stratum '{label}': {len(pairs)} total, "
                                f"sampled {len(sampled)}"
                            )
                        # If rounding caused us to exceed the cap, trim randomly
                        if len(breadth_pairs) > MAX_BREADTH_SAMPLE:
                            breadth_pairs = random.sample(breadth_pairs, MAX_BREADTH_SAMPLE)
                    plain_breadth = [s.split("/")[0] for s in breadth_pairs]

                    # Use cached quotes (Redis/DB only, no network calls)
                    loop = asyncio.get_running_loop()
                    raw_breadth = await loop.run_in_executor(self._quote_executor, get_quotes_cached, plain_breadth)
                    breadth_tickers = {pair: raw_breadth.get(pair.split("/")[0], {}) for pair in breadth_pairs}

                    # Fall back to DB close prices for symbols without cached quotes
                    missing_breadth = [
                        s.split("/")[0] for s in breadth_pairs
                        if breadth_tickers.get(s, {}).get('percentage') is None
                    ]
                    if missing_breadth:
                        try:
                            db_candles = await loop.run_in_executor(self._db_executor, get_latest_close_prices, missing_breadth)
                            for pair in breadth_pairs:
                                base = pair.split("/")[0]
                                if base in db_candles and db_candles[base].get("last", 0) > 0:
                                    last = db_candles[base]["last"]
                                    prev_close = db_candles[base].get("prev_close")
                                    if prev_close and prev_close > 0:
                                        pct = ((last - prev_close) / prev_close) * 100
                                        breadth_tickers[pair] = {
                                            "last": last,
                                            "percentage": round(pct, 4),
                                        }
                        except Exception as e:
                            logger.warning(f"DB close price fallback for breadth failed: {e}")

                    positive_count = sum(
                        1 for sym in breadth_pairs
                        if (breadth_tickers.get(sym, {}).get('percentage') or 0) > 0
                    )
                    total_count = len(breadth_pairs)
                    full_market_breadth = {
                        "positive_pct": round(positive_count / total_count * 100, 1) if total_count > 0 else 0.0,
                        "positive_count": positive_count,
                        "total_count": total_count,
                        "universe_size": len(available_pairs),
                    }
                    await asyncio.to_thread(
                        self.redis.setex, "market:breadth:full", 600, json.dumps(full_market_breadth)
                    )
                    logger.info(f"Full market breadth updated: {full_market_breadth} (sampled from {len(available_pairs)} symbols)")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Full market breadth computation network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Full market breadth computation data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("full_market_breadth", e)
            except Exception as e:
                logger.error(f"Full market breadth computation error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("full_market_breadth", e)
            finally:
                self._full_breadth_running = False
            await asyncio.sleep(1800)  # every 30 minutes (medium/long-term)

    async def _periodic_market_condition_check(self):
        """Check for market conditions that warrant more frequent symbol re-evaluation.

        Triggers re-evaluation when:
        - Significant news sentiment shifts are detected on tracked symbols
        - Unusually active market (many stocks with large daily price movements)
        - Extreme indicator values or Bollinger Band squeeze breakouts on tracked symbols
        """
        await asyncio.sleep(120)  # initial delay
        while self._running:
            try:
                await self.event_bus.request("check_market_conditions")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Market condition check network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Market condition check data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("market_condition_check", e)
            except Exception as e:
                logger.error(f"Market condition check error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("market_condition_check", e)
            await asyncio.sleep(1800)  # check every 30 minutes (medium/long-term)

    async def _periodic_portfolio_rebalance(self):
        """Periodically trigger portfolio rebalance for long-term trading."""
        if not settings.PORTFOLIO_REBALANCE_ENABLED:
            logger.info("Portfolio rebalance is disabled (PORTFOLIO_REBALANCE_ENABLED=False). Task sleeping.")
            while self._running:
                await self._interruptible_sleep(3600)
            return
        await asyncio.sleep(3600)  # initial delay
        while self._running:
            try:
                logger.info("Periodic portfolio rebalance triggered.")
                self.trigger_portfolio_rebalance()
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Periodic portfolio rebalance network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Periodic portfolio rebalance data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("periodic_portfolio_rebalance", e)
            except Exception as e:
                logger.error(f"Periodic portfolio rebalance error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("periodic_portfolio_rebalance", e)
            await self._interruptible_sleep(settings.PORTFOLIO_REBALANCE_INTERVAL_SECONDS)

    async def _is_quote_too_stale(self, ticker: Dict[str, Any], timeframe: str) -> bool:
        """Check if the quote is too stale for trading based on the configured threshold.

        The staleness threshold is scaled by the symbol's timeframe: longer timeframes
        tolerate staler quotes. Returns False if staleness cannot be determined or
        if the guard is disabled.
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
        tf_seconds = self._timeframe_to_seconds(timeframe)
        scaled_threshold = max(settings.QUOTE_MAX_STALENESS_SECONDS, min(tf_seconds * 0.1, 21600))
        return age_seconds > scaled_threshold

    async def _fetch_vix(self) -> Optional[float]:
        """VIX is not available for the Italian market via yfinance. Returns None."""
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
            stock_name = await self._market_data_manager.get_stock_name(symbol)
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
        """Check stop-loss, take-profit, and other risk rules on every ticker update."""
        await asyncio.sleep(5)  # initial delay
        last_risk_check: Dict[str, float] = {}

        while self._running:
            try:
                now = time.time()
                symbols_to_check = []
                min_interval = settings.RISK_CHECK_INTERVAL_SECONDS
                for symbol, pos in self.positions.items():
                    pos_tf = pos.get("timeframe")
                    if not pos_tf:
                        pos_tf_secs = settings.RISK_CHECK_INTERVAL_SECONDS
                    else:
                        pos_tf_secs = self._timeframe_to_seconds(pos_tf)

                    if pos_tf_secs >= 31_536_000:  # >= 1 year
                        pos_interval = settings.RISK_CHECK_INTERVAL_VERY_LONG_TF_SECONDS
                    elif pos_tf_secs >= settings.LONG_TERM_TF_SECONDS:  # >= 1 month
                        pos_interval = max(3600, min(3600, int(pos_tf_secs * 0.01)))
                    else:
                        pos_interval = settings.RISK_CHECK_INTERVAL_SECONDS

                    if pos_interval < min_interval:
                        min_interval = pos_interval

                    last_check = last_risk_check.get(symbol, 0)
                    if now - last_check >= pos_interval:
                        symbols_to_check.append(symbol)
                        last_risk_check[symbol] = now

                # Clean up last_risk_check for closed positions
                closed_symbols = [s for s in last_risk_check if s not in self.positions]
                for s in closed_symbols:
                    del last_risk_check[s]

                if symbols_to_check:
                    await self.event_bus.request("check_risk_management", symbols_to_check)
                    await self._state_persistence.save_state()
                    self._state_dirty = True

                # Dynamically compute sleep interval based on the shortest timeframe
                # among current positions. This ensures the interval is updated immediately
                # when positions are closed and the shortest timeframe changes.
                await self._interruptible_sleep(min_interval)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Risk management loop network/IO error: {type(e).__name__}: {e}")
                await self._interruptible_sleep(settings.RISK_CHECK_INTERVAL_SECONDS)
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Risk management loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("risk_management_loop", e)
                await self._interruptible_sleep(settings.RISK_CHECK_INTERVAL_SECONDS)
            except Exception as e:
                logger.error(f"Risk management loop error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("risk_management_loop", e)
                await self._interruptible_sleep(settings.RISK_CHECK_INTERVAL_SECONDS)

    async def _refresh_current_symbols_news_fast(self):
        """Fast news refresh loop – only for the symbols currently tracked by the engine."""
        if not settings.NEWS_ENABLED:
            logger.info("News is disabled (NEWS_ENABLED=False). Fast news refresh task sleeping.")
            while self._running:
                await self._interruptible_sleep(3600)
            return
        # Fetch immediately on startup, then periodically
        while self._running:
            if self._news_fast_running:
                logger.warning("Fast news refresh still running; skipping this cycle.")
                await self._interruptible_sleep(settings.NEWS_FAST_UPDATE_INTERVAL_MINUTES * 60)
                continue
            self._news_fast_running = True
            try:
                symbols = [entry["symbol"] for entry in self.current_symbols]
                if symbols:
                    logger.info(f"Fast news refresh for {len(symbols)} current symbols")
                    async def _fetch_news_with_limit(sym):
                        async with self._news_semaphore:
                            await self._fetch_and_store_news_for_symbol(sym)
                    await asyncio.gather(
                        *[_fetch_news_with_limit(sym) for sym in symbols]
                    )
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Fast news refresh network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Fast news refresh data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("fast_news_refresh", e)
            except Exception as e:
                logger.error(f"Fast news refresh error: {type(e).__name__}: {e}")
                await self._record_unexpected_exception("fast_news_refresh", e)
            finally:
                self._news_fast_running = False
            await self._interruptible_sleep(settings.NEWS_FAST_UPDATE_INTERVAL_MINUTES * 60)

    async def _refresh_news_cache(self):
        """Periodically fetch news for tracked stocks/ETFs and top-volume stocks to keep cache warm."""
        if not settings.NEWS_ENABLED:
            logger.info("News is disabled (NEWS_ENABLED=False). News cache refresh task sleeping.")
            while self._running:
                await self._interruptible_sleep(3600)
            return
        try:
            from src.news.fetcher import fetch_news_for_symbol
        except ImportError:
            logger.warning("News module not available; skipping background news refresh.")
            return

        while self._running:
            if self._news_cache_running:
                logger.warning("News cache refresh still running; skipping this cycle.")
                await self._interruptible_sleep(settings.NEWS_UPDATE_INTERVAL_MINUTES * 60)
                continue
            self._news_cache_running = True
            try:
                cycle_start = time.time()
                # Slow refresh: all available pairs EXCEPT the stocks already handled by the fast loop
                current_symbols = {entry["symbol"] for entry in self.current_symbols}
                symbols_to_refresh = set()
                try:
                    plain_assets = await self._market_data_manager.get_tradable_assets()
                    available_pairs = [f"{sym}/{self.base_currency}" for sym in plain_assets]
                    # Fetch tickers for a subset to determine top volume symbols
                    # (limit to 200 to avoid excessive API calls)
                    sample_for_vol = available_pairs[:200]
                    plain_sample = [s.split("/")[0] for s in sample_for_vol]
                    raw_quotes = await self._market_data_manager._get_quotes_batched(plain_sample, timeout_per_chunk=45.0)
                    tickers = {pair: raw_quotes.get(pair.split("/")[0], {}) for pair in sample_for_vol}
                    def _vol(sym):
                        t = tickers.get(sym, {})
                        return t.get('quoteVolume', 0) or 0
                    symbols_to_refresh = set(sample_for_vol) - current_symbols
                except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
                    logger.warning(f"Could not get available pairs for news refresh: {e}")

                for sym in symbols_to_refresh:
                    try:
                        async with self._news_semaphore:
                            stock_name = await self._market_data_manager.get_stock_name(sym)
                            articles = await fetch_news_for_symbol(sym, stock_name)
                            if articles:
                                base_symbol = sym.split("/")[0] if "/" in sym else sym
                                await asyncio.to_thread(store_news_articles, base_symbol, articles)
                    except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
                        logger.info(f"News refresh failed for {sym}: {e}")
                    await asyncio.sleep(0.2)

                logger.info(f"News cache refreshed for {len(symbols_to_refresh)} symbols in {time.time() - cycle_start:.2f}s")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Background news refresh network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Background news refresh data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("news_cache_refresh", e)
            except Exception as e:
                logger.error(f"Background news refresh error: {type(e).__name__}: {e}")
                await self._record_unexpected_exception("news_cache_refresh", e)
            finally:
                self._news_cache_running = False

            # Clean up old news articles
            try:
                from src.database import cleanup_old_news
                await asyncio.to_thread(cleanup_old_news, settings.NEWS_RETENTION_SECONDS)
            except (ValueError, TypeError, ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"News cleanup failed: {e}")

            await self._interruptible_sleep(settings.NEWS_UPDATE_INTERVAL_MINUTES * 60)

    @staticmethod
    def _timeframe_to_ms(timeframe: str) -> int:
        """Convert a timeframe string (e.g., '1m', '5m', '1h') to milliseconds."""
        units = {
            'm': 60_000,
            'h': 3_600_000,
            'd': 86_400_000,
            'w': 604_800_000,
            'M': 2_592_000_000,  # approximate (30 days)
            'Y': 31_536_000_000, # approximate (365 days)
        }
        match = re.match(r'^(\d+)([mhdwMY])$', timeframe)
        if not match:
            return 3_600_000  # default to 1h
        amount = int(match.group(1))
        unit = match.group(2)
        return amount * units.get(unit, 3_600_000)

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert a timeframe string (e.g., '5m', '1h') to seconds."""
        return self._timeframe_to_ms(timeframe) // 1000

    def _get_effective_refresh_interval(self, base_interval: int, loop_type: str = "data") -> int:
        """Scale refresh interval based on the longest tracked timeframe.

        For long-term timeframes (1Y+), use much longer refresh cycles to
        avoid wasting bandwidth and API calls on data that barely changes.

        loop_type: "quotes" for quote refresh, "data" for OHLCV downloads,
                   "news" for news downloads.
        """
        if not self.current_symbols:
            return base_interval

        max_tf_seconds = 0
        for entry in self.current_symbols:
            tf = entry.get("timeframe", "1d")
            tf_secs = self._timeframe_to_seconds(tf)
            if tf_secs > max_tf_seconds:
                max_tf_seconds = tf_secs

        if loop_type == "quotes":
            # Quotes: even for long timeframes, prices still move intraday
            if max_tf_seconds >= 31_536_000:  # 1Y+
                return max(base_interval, 3600)  # 1 hour
            elif max_tf_seconds >= 2_592_000:  # 1M+
                return max(base_interval, 1800)  # 30 minutes
            return base_interval
        elif loop_type == "news":
            # News: daily is sufficient for long-term trading
            if max_tf_seconds >= 31_536_000:  # 1Y+
                return max(base_interval, 86400)  # daily
            elif max_tf_seconds >= 2_592_000:  # 1M+
                return max(base_interval, 43200)  # 12 hours
            return base_interval
        else:  # "data" – OHLCV downloads
            if max_tf_seconds >= 31_536_000:  # 1Y+
                return max(base_interval, 86400)  # daily
            elif max_tf_seconds >= 15_552_000:  # 6M+
                return max(base_interval, 43200)  # 12 hours
            elif max_tf_seconds >= 7_776_000:  # 3M+
                return max(base_interval, 21600)  # 6 hours
            elif max_tf_seconds >= 2_592_000:  # 1M+
                return max(base_interval, 10800)  # 3 hours
            elif max_tf_seconds >= 604_800:  # 1w+
                return max(base_interval, 3600)  # 1 hour
            return base_interval

    @staticmethod
    def _format_symbol_display(symbol: str, stock_name: str, timeframe: Optional[str] = None) -> str:
        """Return a display string like 'AAPL[Apple Inc.]' or 'AAPL[Apple Inc.] (15m)'."""
        base = symbol.split("/")[0] if "/" in symbol else symbol
        if stock_name and stock_name != base:
            display = f"{base}[{stock_name}]"
        else:
            display = base
        if timeframe:
            display += f" ({timeframe})"
        return display

    async def _download_market_data_loop(self):
        """Periodically download and store OHLCV data for tracked stocks, with gap detection."""
        # Initial delay to let the engine settle
        await asyncio.sleep(30)
        while self._running:
            if self._market_data_running:
                logger.warning("Market data download still running; skipping this cycle.")
                await self._interruptible_sleep(self._get_effective_refresh_interval(settings.MARKET_DATA_REFRESH_SECONDS, "data"))
                continue
            self._market_data_running = True
            try:
                if not self.current_symbols:
                    logger.info("No symbols tracked; skipping market data download.")
                else:
                    logger.info("Starting market data download cycle...")
                    now_ms = int(time.time() * 1000)
                    start_ms = now_ms - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000

                    async def _download_symbol_data(symbol_entry):
                        symbol = symbol_entry["symbol"]
                        tf = symbol_entry["timeframe"]
                        logger.debug(f"Downloading market data for {symbol} ({tf})")
                        await self._market_data_manager._download_symbol_ohlcv(symbol, tf, start_ms, now_ms)

                    shuffled_symbols = list(self.current_symbols)
                    random.shuffle(shuffled_symbols)
                    download_tasks = [_download_symbol_data(entry) for entry in shuffled_symbols]
                    await asyncio.gather(*download_tasks)
                    logger.info("Market data download cycle complete.")
                    # Clean up old OHLCV data (older than retention period)
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(self._db_executor, cleanup_old_ohlcv, settings.OHLCV_RETENTION_DAYS)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Market data download loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Market data download loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("market_data_download_loop", e)
            except Exception as e:
                logger.error(f"Market data download loop error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("market_data_download_loop", e)
            finally:
                self._market_data_running = False

            await self._interruptible_sleep(self._get_effective_refresh_interval(settings.MARKET_DATA_REFRESH_SECONDS, "data"))

    async def _download_all_assets_data_loop(self):
        """Periodically download OHLCV for ALL tradable assets (stocks, ETFs, BTPs)."""
        await asyncio.sleep(120)  # initial delay to let the engine settle
        while self._running:
            if self._full_download_running:
                logger.info("Full download already running (likely force download); skipping this cycle.")
                await self._interruptible_sleep(self._get_effective_refresh_interval(settings.FULL_ASSET_OHLCV_DOWNLOAD_INTERVAL_SECONDS, "data"))
                continue
            self._full_download_running = True
            try:
                logger.info("Starting full asset OHLCV download cycle...")
                # 1. Get all stock + ETF symbols
                plain_assets = await self._market_data_manager.get_tradable_assets()
                stock_pairs = [f"{sym}/{self.base_currency}" for sym in plain_assets]

                # 2. Get all BTP symbols
                btp_bonds = await self._market_data_manager.get_btp_bonds()
                btp_pairs = [f"{b['isin']}/{self.base_currency}" for b in btp_bonds]

                all_pairs = stock_pairs + btp_pairs
                if not all_pairs:
                    logger.info("No tradable assets found; skipping full download.")
                    await self._interruptible_sleep(self._get_effective_refresh_interval(settings.FULL_ASSET_OHLCV_DOWNLOAD_INTERVAL_SECONDS, "data"))
                    continue

                now_ms = int(time.time() * 1000)
                start_ms = now_ms - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000

                # Prioritize symbols with missing or stale data for configured timeframes
                loop = asyncio.get_running_loop()
                latest_timestamps = await loop.run_in_executor(
                    self._db_executor,
                    get_latest_ohlcv_timestamps_batch,
                    all_pairs,
                    settings.OHLCV_TIMEFRAMES
                )

                pairs_with_stale_data = []
                pairs_complete = []
                now_ms = int(time.time() * 1000)

                for pair in all_pairs:
                    stale_tfs = []
                    for tf in settings.OHLCV_TIMEFRAMES:
                        latest_ts = latest_timestamps.get(pair, {}).get(tf)
                        if latest_ts is None:
                            stale_tfs.append(tf)
                        else:
                            interval_ms = self._timeframe_to_ms(tf)
                            if latest_ts < now_ms - interval_ms:
                                stale_tfs.append(tf)
                    if stale_tfs:
                        pairs_with_stale_data.append((pair, stale_tfs))
                    else:
                        pairs_complete.append(pair)

                random.shuffle(pairs_with_stale_data)
                if pairs_with_stale_data:
                    logger.info(f"Prioritizing {len(pairs_with_stale_data)} symbols with stale/missing OHLCV data out of {len(all_pairs)} total.")

                async def _download_symbol_data(pair: str, tfs: List[str]):
                    for tf in tfs:
                        await self._market_data_manager._download_symbol_ohlcv(pair, tf, start_ms, now_ms, quiet=True)

                # Limit concurrent symbol downloads to 2 to avoid exhausting the
                # _download_executor thread pool, leaving threads available for
                # tracked tickers.
                download_concurrency = asyncio.Semaphore(2)
                async def _limited_download(pair: str, tfs: List[str]):
                    async with download_concurrency:
                        await _download_symbol_data(pair, tfs)
                download_tasks = [_limited_download(pair, tfs) for pair, tfs in pairs_with_stale_data]
                await asyncio.gather(*download_tasks)

                # Clean up old data
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._db_executor, cleanup_old_ohlcv, settings.OHLCV_RETENTION_DAYS)
                await loop.run_in_executor(self._db_executor, cleanup_old_position_pnl, 90)
                await loop.run_in_executor(self._db_executor, cleanup_old_backtest_results, 90)
                logger.info("Full asset OHLCV download cycle complete.")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Full asset download loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Full asset download loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("full_asset_download_loop", e)
            except Exception as e:
                logger.error(f"Full asset download loop error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("full_asset_download_loop", e)
            finally:
                self._full_download_running = False

            # Wait before next full download
            await self._interruptible_sleep(self._get_effective_refresh_interval(settings.FULL_ASSET_OHLCV_DOWNLOAD_INTERVAL_SECONDS, "data"))

    async def _download_all_news_loop(self):
        """Periodically pre‑fetch news for ALL tradable assets (stocks, ETFs, BTPs)."""
        if not settings.NEWS_ENABLED:
            logger.info("News is disabled (NEWS_ENABLED=False). Full news download task sleeping.")
            while self._running:
                await self._interruptible_sleep(3600)
            return
        await asyncio.sleep(180)  # initial delay to let the engine settle
        while self._running:
            try:
                logger.info("Starting full asset news download cycle...")
                # 1. Get all stock + ETF symbols
                plain_assets = await self._market_data_manager.get_tradable_assets()
                stock_pairs = [f"{sym}/{self.base_currency}" for sym in plain_assets]

                # 2. Get all BTP symbols
                btp_bonds = await self._market_data_manager.get_btp_bonds()
                btp_pairs = [f"{b['isin']}/{self.base_currency}" for b in btp_bonds]

                all_pairs = stock_pairs + btp_pairs
                if not all_pairs:
                    logger.info("No tradable assets found; skipping full news download.")
                    await self._interruptible_sleep(self._get_effective_refresh_interval(settings.FULL_ASSET_NEWS_DOWNLOAD_INTERVAL_SECONDS, "news"))
                    continue

                # Prioritize currently tracked symbols first, then the rest.
                current_symbol_set = {entry["symbol"] for entry in self.current_symbols}
                priority_pairs = [p for p in all_pairs if p in current_symbol_set]
                other_pairs = [p for p in all_pairs if p not in current_symbol_set]
                ordered_pairs = priority_pairs + other_pairs

                # Download concurrently, respecting rate limits via _news_semaphore
                async def _download_news_for_symbol(pair: str):
                    try:
                        async with self._news_semaphore:
                            await self._fetch_and_store_news_for_symbol(pair)
                    except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
                        logger.warning(f"Full news download failed for {pair}: {e}")

                news_tasks = [_download_news_for_symbol(pair) for pair in ordered_pairs]
                await asyncio.gather(*news_tasks)

                logger.info("Full asset news download cycle complete.")
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Full asset news download loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Full asset news download loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("full_asset_news_download_loop", e)
            except Exception as e:
                logger.error(f"Full asset news download loop error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("full_asset_news_download_loop", e)

            await self._interruptible_sleep(self._get_effective_refresh_interval(settings.FULL_ASSET_NEWS_DOWNLOAD_INTERVAL_SECONDS, "news"))

    async def _refresh_all_quotes_loop(self):
        """Periodically fetch quotes for all tradable assets and cache them in Redis."""
        await asyncio.sleep(60)  # initial delay
        while self._running:
            if self._quotes_fetch_running:
                logger.info("Quotes fetch already running (likely re-evaluation or breadth); skipping this cycle.")
                await self._interruptible_sleep(self._get_effective_refresh_interval(settings.QUOTE_REFRESH_INTERVAL_SECONDS, "quotes"))
                continue
            self._quotes_fetch_running = True
            try:
                # Do NOT skip when the circuit breaker is open — get_quotes
                # internally checks the circuit breaker and falls back to DB
                # close prices (from market_data candles).  Skipping here
                # prevents those fallback prices from being saved to the
                # quotes table, leaving it stale when yfinance is down.
                plain_assets = await self._market_data_manager.get_tradable_assets()
                etf_symbols = await self._market_data_manager.get_etf_symbols()
                btp_bonds = await self._market_data_manager.get_btp_bonds()
                btp_isins = [b["isin"] for b in btp_bonds if b.get("isin")]

                all_quote_symbols = plain_assets + etf_symbols + btp_isins
                if all_quote_symbols:
                    # Fetch quotes in batches to avoid yfinance timeouts on large symbol lists
                    await self._market_data_manager._get_quotes_batched(all_quote_symbols, timeout_per_chunk=180.0)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Background quote refresh network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Background quote refresh data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("quote_refresh_loop", e)
            except Exception as e:
                logger.error(f"Background quote refresh error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("quote_refresh_loop", e)
            finally:
                self._quotes_fetch_running = False
            await self._interruptible_sleep(self._get_effective_refresh_interval(settings.QUOTE_REFRESH_INTERVAL_SECONDS, "quotes"))

    async def _refresh_ticker_discovery_loop(self):
        """Periodically discover tickers from news RSS feeds and trending stocks.
        Caches results in Redis so re-evaluation never blocks on slow HTTP calls."""
        await asyncio.sleep(120)  # initial delay
        while self._running:
            try:
                plain_assets = await self._market_data_manager.get_tradable_assets()
                available_pairs = [f"{sym}/{self.base_currency}" for sym in plain_assets]

                if settings.NEWS_ENABLED and settings.NEWS_TICKER_DISCOVERY_ENABLED and discover_tickers_from_news is not None:
                    logger.info("Background: refreshing RSS ticker discovery...")
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        self._download_executor,
                        lambda: discover_tickers_from_news(
                            existing_pairs=available_pairs,
                            cache_only=False,
                        )
                    )

                if settings.NEWS_ENABLED and settings.NEWS_SYMBOL_DISCOVERY_ENABLED and discover_trending_stocks is not None:
                    logger.info("Background: refreshing trending stock discovery...")
                    await loop.run_in_executor(
                        self._download_executor,
                        lambda: discover_trending_stocks(
                            self.base_currency,
                            available_pairs,
                            max_symbols=settings.NEWS_SYMBOL_DISCOVERY_MAX_SYMBOLS,
                            min_sentiment=settings.NEWS_SYMBOL_DISCOVERY_MIN_SENTIMENT,
                            min_articles=settings.NEWS_SYMBOL_DISCOVERY_MIN_ARTICLES,
                            cache_only=False,
                        )
                    )
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Ticker discovery refresh network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Ticker discovery refresh data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("ticker_discovery_loop", e)
            except Exception as e:
                logger.error(f"Ticker discovery refresh error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("ticker_discovery_loop", e)
            await asyncio.sleep(3600)  # every 60 minutes (medium/long-term)

    async def _fetch_dividends_loop(self):
        """Periodically fetch and store dividends for tracked symbols."""
        await self._interruptible_sleep(300)  # initial delay 5 minutes
        while self._running:
            try:
                symbols = [entry["symbol"] for entry in self.current_symbols]
                if not symbols:
                    await self._interruptible_sleep(3600)
                    continue
                # Only fetch for non-BTP symbols (BTPs use coupons, not dividends)
                stock_symbols = [s for s in symbols if not is_btp_isin(s.split("/")[0])]
                if stock_symbols:
                    async def _fetch_dividends(symbol: str):
                        try:
                            divs = await asyncio.to_thread(get_yahoo_dividends, symbol)
                            for d in divs:
                                await asyncio.to_thread(insert_dividend, symbol, d["date"], d["amount"])
                        except (ValueError, TypeError, KeyError, ConnectionError, TimeoutError, OSError) as e:
                            logger.debug(f"Dividend fetch failed for {symbol}: {e}")
                    await asyncio.gather(*[_fetch_dividends(s) for s in stock_symbols])
                # Cleanup old dividends
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._db_executor, cleanup_old_dividends, 365)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Dividend fetch loop network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Dividend fetch loop data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("dividend_fetch_loop", e)
            except Exception as e:
                logger.error(f"Dividend fetch loop error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("dividend_fetch_loop", e)
            await self._interruptible_sleep(86400)  # daily

    def _daily_realized_pnl(self) -> float:
        """Return the sum of realized P&L for trades closed today (market timezone)."""
        from datetime import datetime, timezone
        tz = ZoneInfo(settings.MARKET_TIMEZONE)
        today = datetime.now(tz).date()
        total = 0.0
        with self._trade_history_lock:
            trades_snapshot = list(self.trade_history)
        for trade in trades_snapshot:
            if trade.get("side") != "sell":
                continue
            ts = trade.get("timestamp", 0)
            if ts:
                trade_date = datetime.fromtimestamp(ts / 1000.0, tz=tz).date()
                if trade_date == today:
                    total += trade.get("realized_pnl", 0.0)
        return total

    def _daily_buy_fees(self) -> float:
        """Return the sum of buy-side fees for trades opened today (market timezone).

        These fees are not yet reflected in realized_pnl (which only includes
        fees from closed positions), so they must be accounted for separately
        in the daily loss limit check.
        """
        tz = ZoneInfo(settings.MARKET_TIMEZONE)
        today = datetime.now(tz).date()
        total = 0.0
        with self._trade_history_lock:
            trades_snapshot = list(self.trade_history)
        for trade in trades_snapshot:
            if trade.get("side") != "buy":
                continue
            ts = trade.get("timestamp", 0)
            if ts:
                trade_date = datetime.fromtimestamp(ts / 1000.0, tz=tz).date()
                if trade_date == today:
                    fee = trade.get("fee", {})
                    total += float(fee.get("cost", 0.0) or 0.0)
        return total

    def _log_task_exception(self, task: asyncio.Task) -> None:
        """Log exceptions from background tasks to prevent silent failures."""
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
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

                    outcome_price = price_data["last"]
                    entry_price = decision["entry_price"]
                    action = decision["action"]

                    if entry_price is None or entry_price <= 0:
                        continue

                    # Determine if the decision was profitable
                    # BUY: profitable if price went up
                    # SELL: profitable if price went down
                    # HOLD: profitable if price didn't go up (opportunity cost avoided)
                    if action == "BUY":
                        profitable = outcome_price > entry_price
                    elif action == "SELL":
                        profitable = outcome_price < entry_price
                    elif action == "HOLD":
                        profitable = outcome_price <= entry_price
                    else:
                        profitable = False

                    await asyncio.to_thread(
                        update_llm_decision_outcome,
                        decision["id"],
                        outcome_price,
                        profitable
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
            except Exception as e:
                logger.error(f"LLM decision evaluation loop error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("llm_decision_eval_loop", e)
            
            await self._interruptible_sleep(600)  # check every 10 minutes

    async def _record_unexpected_exception(self, context: str, exc: Exception) -> None:
        """Record metrics for unexpected exceptions in Redis."""
        try:
            exc_type = type(exc).__name__
            key = f"metrics:unexpected_exception:{context}:{exc_type}"
            await asyncio.to_thread(self.redis.incr, key)
            await asyncio.to_thread(self.redis.expire, key, 86400)
        except Exception:
            pass

    async def run(self):
        """Main event-driven loop using WebSocket ticker updates."""
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
            except Exception as e:
                logger.error(f"Pause/resume check error: {type(e).__name__}: {e}", exc_info=True)
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
            except Exception as e:
                logger.error(f"Redis health check loop error: {type(e).__name__}: {e}", exc_info=True)
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
            except Exception as e:
                logger.error(f"Health check loop error: {type(e).__name__}: {e}", exc_info=True)
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
                for entry in self.current_symbols:
                    symbol = entry["symbol"]
                    tf = entry["timeframe"]
                    # Skip entry signal monitoring for very long timeframes (>= 1 month)
                    # where short-term crossovers are irrelevant.
                    tf_seconds = self._timeframe_to_seconds(tf)
                    # Avoid re‑triggering too often – enforce a cooldown of at least
                    # the normal strategy interval.
                    # Use a short, dedicated cooldown so the bot reacts quickly to new signals
                    cooldown = settings.ENTRY_SIGNAL_COOLDOWN_SECONDS
                    async with self._eval_state_lock:
                        last_forced = self._force_eval_time.get(symbol, 0)
                    if time.time() - last_forced < cooldown:
                        continue

                    if await self.event_bus.request("detect_entry_signal", symbol, tf):
                        logger.info(f"Entry signal detected for {symbol}, forcing LLM evaluation.")
                        async with self._eval_state_lock:
                            self._force_eval[symbol] = True
                            self._force_eval_time[symbol] = time.time()
                            # Clear last evaluation timestamp so the main loop picks it up immediately
                            self._last_strategy_eval.pop(symbol, None)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Entry signal monitor network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Entry signal monitor data/logic error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("entry_signal_monitor", e)
            except Exception as e:
                logger.error(f"Entry signal monitor error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("entry_signal_monitor", e)
            await self._interruptible_sleep(settings.ENTRY_SIGNAL_CHECK_INTERVAL_SECONDS)

    async def _check_pending_entries(self):
        """Periodically check pending entry conditions and execute if met."""
        await asyncio.sleep(10)  # short initial delay
        while self._running:
            try:
                now = time.time()
                for symbol in list(self._pending_entries.keys()):
                    await self.event_bus.request("process_pending_entry", symbol, now)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Pending entries check network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Error checking pending entries data/logic: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("check_pending_entries", e)
            except Exception as e:
                logger.error(f"Error checking pending entries: {type(e).__name__}: {e}", exc_info=True)
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
        if symbol in self.positions:
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
                    self._global_risk_multiplier = val
                    return val
            except (ValueError, TypeError):
                pass
        # Redis key expired or invalid — fall back to last known persisted value
        if self._global_risk_multiplier is not None:
            logger.warning(
                "Global risk multiplier Redis key expired — using persisted fallback "
                f"value {self._global_risk_multiplier}. The LLM should set a new value."
            )
            return self._global_risk_multiplier
        return None

    async def _set_global_risk_multiplier(self, value: float):
        """Set the global risk multiplier in Redis (with TTL) and persist it to the database."""
        await asyncio.to_thread(self.redis.setex, "trading:global_risk_multiplier", 3600, str(value))
        self._global_risk_multiplier = value
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
                for queued in list(self.queued_orders):
                    await self.event_bus.request("process_single_queued_order", queued)
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Queued orders processing network/IO error: {type(e).__name__}: {e}")
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Error processing queued orders data/logic: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("process_queued_orders", e)
            except Exception as e:
                logger.error(f"Error processing queued orders: {type(e).__name__}: {e}", exc_info=True)
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
            except Exception as e:
                logger.error(f"Orphaned order cleanup error: {type(e).__name__}: {e}", exc_info=True)
                await self._record_unexpected_exception("cleanup_orphaned_orders", e)
            await asyncio.sleep(900)  # every 15 minutes

    def _is_excluded(self, symbol: str, timeframe: str) -> bool:
        """Return True if (symbol, timeframe) is in the EXCLUDED_SYMBOLS list."""
        for entry in settings.EXCLUDED_SYMBOLS:
            parts = entry.split("/")
            if len(parts) == 2:
                # "BASE/QUOTE" → exclude all timeframes for this pair
                if parts[0] == symbol.split("/")[0] and parts[1] == symbol.split("/")[1]:
                    return True
            elif len(parts) == 3:
                # "BASE/QUOTE/TIMEFRAME" → exclude only that specific timeframe
                if (parts[0] == symbol.split("/")[0] and
                    parts[1] == symbol.split("/")[1] and
                    parts[2] == timeframe):
                    return True
        return False

    def _normalize_llm_symbol(self, sym: str, sample_pairs: list) -> Optional[str]:
        """Normalize an LLM-returned symbol to match the format in sample_pairs.

        The LLM may return symbols without the /EUR suffix (e.g., 'ENI.MI' instead
        of 'ENI.MI/EUR'), or with/without exchange-specific suffixes (e.g., 'ENI'
        vs 'ENI.MI'). This method tries multiple formats to find a match.
        Returns the matched pair string, or None if no match is found.
        """
        if sym in sample_pairs:
            return sym
        # Try adding /{base_currency} suffix
        with_suffix = f"{sym}/{self.base_currency}"
        if with_suffix in sample_pairs:
            return with_suffix
        # Try matching by base symbol (strip any suffix the LLM may have added)
        base = sym.split("/")[0]
        for pair in sample_pairs:
            if pair.split("/")[0] == base:
                return pair
        # Try matching by stripping exchange suffixes from both sides
        # e.g., LLM returns 'ENI' but sample has 'ENI.MI', or vice versa
        configured_suffix = getattr(settings, 'TICKER_SUFFIX', '')

        def _strip_suffix(symbol_base: str) -> str:
            # Strip the configured ticker suffix first
            if configured_suffix and symbol_base.endswith(configured_suffix):
                return symbol_base[:-len(configured_suffix)]
            # Strip common exchange suffixes (e.g., .MI, .PA, .L, .N, .SW)
            parts = symbol_base.rsplit('.', 1)
            if len(parts) == 2 and 1 <= len(parts[1]) <= 3 and parts[1].isalpha() and parts[1].isupper():
                return parts[0]
            return symbol_base

        stripped_base = _strip_suffix(base)
        for pair in sample_pairs:
            pair_base = pair.split("/")[0]
            if stripped_base == _strip_suffix(pair_base):
                return pair
        return None

    async def _is_market_open(self) -> bool:
        """Return True if the Italian market (Borsa Italiana) is currently open."""
        clock = await self._market_data_manager.get_clock()
        if clock is None:
            # Fallback: if clock unavailable, assume closed to be safe
            return False
        return clock.is_open

    async def _remove_symbol_if_paused(self, symbol: str):
        """Clear pending entries for a symbol. Symbols are kept in current_symbols even when paused
        so the bot continues to generate and notify signals."""
        # Always clear any pending entry for this symbol
        self._pending_entries.pop(symbol, None)
        self._state_dirty = True

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
