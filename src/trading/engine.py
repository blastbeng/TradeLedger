import asyncio
import hashlib
import json
import logging
import math
import random
import pandas_market_calendars as mcal
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from zoneinfo import ZoneInfo

from src.config.settings import settings
from src.exchanges.market_data import get_tradable_assets, get_quotes, get_quotes_cached, get_multi_timeframe_bars, get_bars_range, discover_btp_bonds, discover_italian_ucits_etfs, _get_yf_session, _check_yf_circuit
from src.exchanges.yahoo_finance import get_yahoo_quote, get_yahoo_fundamentals
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
    get_cached_news_summary,
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
from src.utils.redis_client import get_redis_client
from src.utils.symbol_utils import is_btp_isin
from src.database import load_trading_state, save_trading_state, insert_trade, get_performance, store_news_articles, get_aggregate_sentiment_from_db, get_aggregate_sentiment_for_symbols, get_news_for_symbol, get_ohlcv, get_latest_ohlcv_timestamp, insert_ohlcv_batch, save_paper_balances, load_paper_balances, cleanup_old_ohlcv, save_indicators, get_indicators, get_indicators_for_symbols, get_ohlcv_summary_for_symbols, get_all_trades, get_latest_close_prices, insert_position_pnl_snapshot, cleanup_old_position_pnl, save_backtest_result, get_recent_backtest_result, get_backtest_results_for_symbol, cleanup_old_backtest_results
from src.trading.components.order_executor import OrderExecutor
from src.trading.components.risk_manager import RiskManager
from src.trading.components.state_persistence import StatePersistence
from src.trading.components.position_manager import PositionManager
from src.trading.components.signal_processor import SignalProcessor
from src.trading.components.backtest_manager import BacktestManager
from src.trading.components.market_data_manager import MarketDataManager
from src.trading.components.symbol_reevaluator import SymbolReevaluator

logger = logging.getLogger(__name__)

@dataclass
class ClockInfo:
    """Market clock info for Euronext Milan (XMIL)."""
    is_open: bool
    timestamp: datetime
    next_open: datetime


@dataclass
class AssetInfo:
    """Asset info for yfinance-based trading (min order size, fractionability, etc.)."""
    name: str = ""
    min_order_size: Optional[float] = 0.0
    fractionable: bool = True


class TradingEngine:
    def __init__(self):
        self.trader = None

        self.base_currency = settings.BASE_CURRENCY
        self.max_symbols = settings.MAX_SYMBOLS
        self.effective_max_symbols = self.max_symbols
        self.redis = get_redis_client()
        self._exchange_semaphore = asyncio.Semaphore(10)  # max 10 concurrent API calls
        self._news_semaphore = asyncio.Semaphore(5)  # max 5 concurrent news fetches
        self._indicator_semaphore = asyncio.Semaphore(4)  # limit concurrent indicator computations
        self._backtest_semaphore = asyncio.Semaphore(4)  # limit concurrent backtest variants
        self._download_semaphore = asyncio.Semaphore(5)  # max 5 concurrent background OHLCV backfills

        # Dedicated thread pool for database writes – prevents write contention
        # from starving the default asyncio thread pool used by the web server,
        # Telegram bot, and all other to_thread calls.
        self._db_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dbwriter")
        # Dedicated thread pool for download/indicator operations – prevents
        # download tasks from exhausting the default asyncio thread pool used
        # by the web server, Telegram bot, and engine loop.
        self._download_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="downloader")
        # Dedicated thread pool for quote fetching – prevents zombie get_quotes
        # threads (from asyncio.wait_for timeouts) from exhausting the default
        # asyncio thread pool used by the web server and Telegram bot.
        self._quote_executor = ThreadPoolExecutor(max_workers=12, thread_name_prefix="quotes")

        self.current_symbols: List[Dict[str, str]] = []   # each dict: {"symbol": ..., "timeframe": ...}
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position info
        self.trade_history: List[Dict[str, Any]] = []
        self.initial_balance: float = 0.0
        self.last_loss_time: Dict[str, float] = {}  # symbol -> timestamp of last losing trade
        self.cooldown_durations: Dict[str, float] = {}  # symbol -> cooldown seconds set by LLM
        self._global_risk_multiplier: Optional[float] = None
        self._last_strategy_eval: Dict[str, float] = {}   # symbol -> timestamp of last strategy evaluation
        self._strategy_intervals: Dict[str, float] = {}    # symbol -> custom interval in seconds
        self._symbol_reevaluation_interval = settings.SYMBOL_REEVALUATION_INTERVAL
        self.notifier = None

        # _load_state() and _ensure_cost_basis() are now called in _initialize_clients()
        # after the trading client is available.

        # Clear stale pause keys immediately (Redis is already available)
        pause_keys = [
            "trading:paused",
            "trading:pause_source",
            "trading:pause_start",
            "trading:pause_duration",
            "trading:pause_reason",
            "trading:llm_pause_time",
        ]
        for key in pause_keys:
            self.redis.delete(key)

        # Track quote currency spent in the current cycle to avoid over-allocating
        self._cycle_spent = 0.0
        self._symbol_first_seen: Dict[str, float] = {}  # symbol -> timestamp when first added
        self._market_breadth: Optional[Dict[str, Any]] = None
        self._cycle_spent_lock = asyncio.Lock()
        self._positions_lock = asyncio.Lock()
        self._queued_orders_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._state_save_pending = False
        self._state_dirty: bool = False
        # --- Extracted components ---
        self._state_persistence = StatePersistence(self)
        self._order_executor = OrderExecutor(self)
        self._risk_manager = RiskManager(self)
        self._symbol_reevaluator = SymbolReevaluator(self)
        self._signal_processor = SignalProcessor(self)
        self._position_manager = PositionManager(self)
        self._backtest_manager = BacktestManager(self)
        self._market_data_manager = MarketDataManager(self)
        self._symbol_reeval_lock = asyncio.Lock()
        self._tradable_assets_lock = asyncio.Lock()
        self._reeval_trigger = asyncio.Event()
        self._reeval_pending_force: bool = False
        self._force_reeval: bool = False
        self._user_forced_reeval: bool = False
        self._pre_market_reeval: bool = False
        self._running = True
        self._last_state_save = 0
        self._last_eval_snapshot: Dict[str, Dict[str, float]] = {}  # symbol -> indicator snapshot
        self._last_decisions: Dict[str, Dict[str, Any]] = {}  # symbol -> last LLM decision
        self.recent_signals: List[Dict[str, Any]] = []
        self._pending_entries: Dict[str, Dict[str, Any]] = {}  # symbol -> pending entry condition info

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
        self._delayed_entry_tasks: set = set()

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
        self._trade_history_version: int = 0
        self._realized_pnl_offset: float = 0.0

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
        self._balance_cache: Optional[Dict[str, float]] = None
        self._balance_cache_time: float = 0.0
        self._portfolio_exposure_cache: Optional[Dict[str, float]] = None
        self._portfolio_exposure_cache_time: float = 0.0
        self._sentiment_cache: Dict[str, tuple] = {}  # symbol -> (timestamp, sentiment_dict)
        self.queued_orders: List[Dict[str, Any]] = []
        # Force immediate LLM evaluation when an entry signal is detected
        self._force_eval: Dict[str, bool] = {}
        # Track when we last forced an LLM evaluation per symbol (for entry signal cooldown)
        self._force_eval_time: Dict[str, float] = {}
        # Per‑symbol state for crossover detection (stores last known indicator values)
        self._entry_signal_state: Dict[str, Dict[str, Any]] = {}

        # Market clock cache
        self._clock_cache: Optional[Any] = None
        self._clock_cache_time: float = 0.0

    async def _initialize_clients(self):
        """Initialize clients and load persisted state (non‑blocking)."""
        self.trader = PaperTrader()
        logger.info(f"PaperTrader initialized for {settings.TRADING_MODE} trading mode.")
        self._load_state()
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

    def set_notifier(self, notifier):
        """Attach a notification service (e.g., TelegramBot)."""
        self.notifier = notifier

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
            except Exception:
                pass
        elif self._reevaluate_running:
            logger.info("Re-evaluation already running; queued re-evaluation for after current cycle completes.")
        self._reeval_trigger.set()

    async def force_download_all_assets(self):
        """Immediately download OHLCV data for all tradable assets (stocks, ETFs, BTPs)."""
        self._full_download_running = True
        logger.info("Force download: starting immediate OHLCV download for all assets...")
        try:
            plain_assets = await self._get_tradable_assets()
            stock_pairs = [f"{sym}/{self.base_currency}" for sym in plain_assets]

            btp_bonds = await self._get_btp_bonds()
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
                    await self._download_symbol_ohlcv(pair, tf, start_ms, now_ms, quiet=True, force=True)

            download_concurrency = asyncio.Semaphore(10)
            async def _limited_force_download(pair: str):
                async with download_concurrency:
                    await _force_download_symbol(pair)
            download_tasks = [_limited_force_download(pair) for pair in all_pairs]
            await asyncio.gather(*download_tasks)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._db_executor, cleanup_old_ohlcv, settings.OHLCV_RETENTION_DAYS)
            logger.info("Force download: complete.")
        except Exception as e:
            logger.error(f"Force download error: {e}", exc_info=True)
        finally:
            self._full_download_running = False

    async def _get_tradable_assets(self) -> List[str]:
        """Return tradable assets, cached for 5 minutes to reduce API calls."""
        now = time.time()
        if self._tradable_assets_cache and (now - self._tradable_assets_cache_time) < 300:
            return self._tradable_assets_cache
        async with self._tradable_assets_lock:
            # Double-check cache after acquiring lock (another task may have populated it)
            now = time.time()
            if self._tradable_assets_cache and (now - self._tradable_assets_cache_time) < 300:
                return self._tradable_assets_cache
            assets = await asyncio.to_thread(get_tradable_assets)
            self._tradable_assets_cache = assets
            self._tradable_assets_cache_time = now
            return assets

    async def _get_btp_bonds(self) -> List[Dict[str, Any]]:
        """Return BTP bonds, cached for 30 minutes to reduce scraping calls."""
        now = time.time()
        if self._btp_bonds_cache and (now - self._btp_bonds_cache_time) < 1800:
            return self._btp_bonds_cache
        async with self._tradable_assets_lock:
            # Double-check after lock
            now = time.time()
            if self._btp_bonds_cache and (now - self._btp_bonds_cache_time) < 1800:
                return self._btp_bonds_cache
            bonds = await asyncio.to_thread(discover_btp_bonds)
            # Merge with DB-saved BTPs so nothing is lost between runs
            try:
                from src.database import get_all_discovered_symbols
                db_symbols = get_all_discovered_symbols()
                existing_isins = {b["isin"] for b in bonds}
                for db_entry in db_symbols:
                    if db_entry.get("asset_type") == "btp" and db_entry["symbol"] not in existing_isins:
                        bonds.append({
                            "isin": db_entry["symbol"],
                            "name": db_entry.get("name") or db_entry["symbol"],
                            "last_price": None,
                            "change_pct": 0.0,
                            "coupon": db_entry.get("coupon"),
                            "maturity": db_entry.get("maturity"),
                        })
                        existing_isins.add(db_entry["symbol"])
            except Exception as e:
                logger.warning(f"Failed to merge BTPs from DB: {e}")
            self._btp_bonds_cache = bonds
            self._btp_bonds_cache_time = now
            return bonds

    async def _get_etf_symbols(self) -> List[str]:
        """Return Italian UCITS ETF symbols, cached for 1 hour."""
        now = time.time()
        if self._etf_symbols_cache and (now - self._etf_symbols_cache_time) < 3600:
            return self._etf_symbols_cache
        async with self._tradable_assets_lock:
            # Double-check after lock
            now = time.time()
            if self._etf_symbols_cache and (now - self._etf_symbols_cache_time) < 3600:
                return self._etf_symbols_cache
            symbols = await asyncio.to_thread(discover_italian_ucits_etfs)
            self._etf_symbols_cache = symbols
            self._etf_symbols_cache_time = now
            return symbols

    async def _get_asset_info(self, symbol: str) -> Any:
        """Return asset info (min order size, name, etc.), cached for 1 hour.

        Fetches from yfinance (subject to circuit breaker) with database
        fallback for the name. Returns permissive defaults only when no
        data is available.
        """
        base = symbol.split("/")[0] if "/" in symbol else symbol
        now = time.time()
        if base in self._asset_cache and (now - self._asset_cache_time.get(base, 0)) < 3600:
            return self._asset_cache[base]

        name = base
        min_order_size: Optional[float] = None
        fractionable = True

        # Try yfinance first (subject to circuit breaker)
        if not _check_yf_circuit():
            try:
                def _fetch_yf_info():
                    import yfinance as yf
                    ticker = yf.Ticker(base, session=_get_yf_session())
                    return ticker.info
                info = await asyncio.to_thread(_fetch_yf_info)
                if info:
                    name = info.get("longName") or info.get("shortName") or base
                    raw_min = info.get("minimumOrderSize")
                    if raw_min is not None:
                        try:
                            min_order_size = float(raw_min)
                        except (TypeError, ValueError):
                            pass
                    raw_frac = info.get("fractionalTrading")
                    if raw_frac is not None:
                        fractionable = bool(raw_frac)
            except Exception as e:
                logger.debug(f"yfinance asset info fetch failed for {base}: {e}")

        # Database fallback for name
        if name == base:
            try:
                from src.database import get_symbol_name_from_db
                db_name = await asyncio.to_thread(get_symbol_name_from_db, base)
                if db_name:
                    name = db_name
            except Exception:
                pass

        # Default to permissive 0.0 when no minimum was found
        if min_order_size is None:
            min_order_size = 0.0

        asset = AssetInfo(name=name, min_order_size=min_order_size, fractionable=fractionable)
        self._asset_cache[base] = asset
        self._asset_cache_time[base] = now
        return asset

    async def _get_all_position_tickers(self) -> Dict[str, Dict[str, Any]]:
        """Fetch tickers for all open positions, batching missing ones into a single API call."""
        self._portfolio_exposure_cache = None
        tickers: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        for sym in self.positions:
            missing.append(sym.split("/")[0])
        if missing:
            try:
                raw = await self._get_quotes_batched(missing, timeout_per_chunk=45.0)
                for sym in self.positions:
                    base = sym.split("/")[0]
                    if base in raw:
                        tickers[sym] = raw[base]
            except Exception as e:
                logger.warning(f"Batch quote fetch failed for positions: {e}")
        return tickers

    def _get_all_position_tickers_sync(self) -> Dict[str, Dict[str, Any]]:
        """Fetch tickers for all open positions synchronously, batching missing ones.

        Uses get_quotes_cached (Redis/DB only, no network calls) to avoid
        blocking the default asyncio thread pool with slow yfinance requests.
        """
        self._portfolio_exposure_cache = None
        tickers: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        for sym in self.positions:
            missing.append(sym.split("/")[0])
        if missing:
            try:
                raw = get_quotes_cached(missing)
                for sym in self.positions:
                    base = sym.split("/")[0]
                    if base in raw:
                        tickers[sym] = raw[base]
            except Exception as e:
                logger.warning(f"Sync batch quote fetch failed for positions: {e}")
        return tickers

    async def _get_cached_balance(self, ttl: float = 30.0) -> Dict[str, float]:
        """Return cached balance, refreshing if older than ttl seconds."""
        now = time.time()
        if self._balance_cache is not None and (now - self._balance_cache_time) < ttl:
            return self._balance_cache
        balance = await asyncio.to_thread(self.trader.fetch_balance)
        self._balance_cache = balance
        self._balance_cache_time = now
        return balance

    async def _get_quotes_async(self, symbols: List[str], timeout: float = 45.0) -> Dict[str, Dict[str, Any]]:
        """Fetch quotes using the dedicated quote thread pool with a timeout.
        This prevents slow yfinance calls from blocking the default asyncio thread pool."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self._quote_executor, get_quotes, symbols),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Quote fetch timed out for {len(symbols)} symbols")
            return {}
        except Exception as e:
            logger.warning(f"Quote fetch failed for {len(symbols)} symbols: {e}")
            return {}

    async def _get_quotes_batched(self, symbols: List[str], timeout_per_chunk: float = 45.0, chunk_size: int = 50) -> Dict[str, Dict[str, Any]]:
        """Fetch quotes for a large list of symbols in batches to avoid yfinance timeouts.

        Splits symbols into chunks of ``chunk_size`` and fetches each chunk
        sequentially.  ``get_quotes`` uses a global lock internally, so
        concurrent calls would queue behind the lock and potentially time out.
        """
        if not symbols:
            return {}

        if len(symbols) <= chunk_size:
            return await self._get_quotes_async(symbols, timeout=timeout_per_chunk)

        result: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            chunk_result = await self._get_quotes_async(chunk, timeout=timeout_per_chunk)
            result.update(chunk_result)
        return result

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
        except Exception as e:
            logger.warning(f"Failed to fetch sentiment for {base}: {e}")
            return None

    async def _get_clock(self, ttl: float = 30.0) -> Optional[ClockInfo]:
        """Return Euronext Milan market clock info, cached for `ttl` seconds.

        Uses pandas_market_calendars only to detect holidays/weekends.
        Open/close times are hardcoded to Borsa Italiana continuous trading:
        09:00–17:30 Rome time (Monday–Friday, excluding holidays).
        """
        now = time.time()
        if self._clock_cache is not None and (now - self._clock_cache_time) < ttl:
            return self._clock_cache

        rome_tz = ZoneInfo(settings.MARKET_TIMEZONE)
        now_rome = datetime.now(timezone.utc).astimezone(rome_tz)
        today = now_rome.date()

        # Configurable trading hours
        MARKET_OPEN_HOUR = settings.MARKET_OPEN_HOUR
        MARKET_OPEN_MINUTE = settings.MARKET_OPEN_MINUTE
        MARKET_CLOSE_HOUR = settings.MARKET_CLOSE_HOUR
        MARKET_CLOSE_MINUTE = settings.MARKET_CLOSE_MINUTE

        market_open_today = datetime(today.year, today.month, today.day,
                                     MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, tzinfo=rome_tz)
        market_close_today = datetime(today.year, today.month, today.day,
                                      MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE, tzinfo=rome_tz)

        is_open = False
        next_open = None

        try:
            # Run mcal calls in a thread to avoid blocking the event loop
            def _get_mcal_schedule():
                cal = mcal.get_calendar('XMIL')
                # Fetch schedule for a window around today to find next trading days
                return cal.schedule(start_date=today - timedelta(days=1),
                                    end_date=today + timedelta(days=10))
            schedule = await asyncio.to_thread(_get_mcal_schedule)

            # --- Fallback when the calendar has no data for the requested range ---
            if schedule.empty:
                # Simple weekday + hardcoded hours check
                if today.weekday() < 5 and market_open_today <= now_rome < market_close_today:
                    is_open = True
                else:
                    is_open = False

                if is_open:
                    # Next open is tomorrow (or next weekday) at 09:00
                    next_open = market_open_today + timedelta(days=1)
                    while next_open.weekday() >= 5:
                        next_open += timedelta(days=1)
                else:
                    if now_rome < market_open_today and today.weekday() < 5:
                        next_open = market_open_today
                    else:
                        next_open = market_open_today + timedelta(days=1)
                        while next_open.weekday() >= 5:
                            next_open += timedelta(days=1)

                clock = ClockInfo(is_open=is_open, timestamp=now_rome, next_open=next_open)
                self._clock_cache = clock
                self._clock_cache_time = now
                return clock
            # --- End fallback ---

            # Determine if today is a trading day (any session that covers today's date)
            today_is_trading_day = False
            next_trading_day = None

            if not schedule.empty:
                for idx in range(len(schedule)):
                    session_start = schedule.iloc[idx]['market_open'].tz_convert(rome_tz)
                    session_end = schedule.iloc[idx]['market_close'].tz_convert(rome_tz)
                    session_date = session_start.date()

                    if session_date == today:
                        today_is_trading_day = True
                    elif session_date > today and next_trading_day is None:
                        next_trading_day = session_start

            if today_is_trading_day:
                if market_open_today <= now_rome < market_close_today:
                    is_open = True
                    # Next open is tomorrow's session (if exists) else next weekday 09:00
                    if next_trading_day is not None:
                        next_open = next_trading_day.replace(hour=MARKET_OPEN_HOUR,
                                                             minute=MARKET_OPEN_MINUTE,
                                                             second=0, microsecond=0)
                    else:
                        # Fallback: next weekday at 09:00
                        next_open = market_open_today + timedelta(days=1)
                        while next_open.weekday() >= 5:
                            next_open += timedelta(days=1)
                elif now_rome < market_open_today:
                    # Before open today
                    next_open = market_open_today
                else:
                    # After close today
                    if next_trading_day is not None:
                        next_open = next_trading_day.replace(hour=MARKET_OPEN_HOUR,
                                                             minute=MARKET_OPEN_MINUTE,
                                                             second=0, microsecond=0)
                    else:
                        next_open = market_open_today + timedelta(days=1)
                        while next_open.weekday() >= 5:
                            next_open += timedelta(days=1)
            else:
                # Today is not a trading day (holiday/weekend)
                if next_trading_day is not None:
                    next_open = next_trading_day.replace(hour=MARKET_OPEN_HOUR,
                                                         minute=MARKET_OPEN_MINUTE,
                                                         second=0, microsecond=0)
                else:
                    # No trading days in schedule – fallback to next weekday 09:00
                    next_open = market_open_today + timedelta(days=1)
                    while next_open.weekday() >= 5:
                        next_open += timedelta(days=1)

        except Exception as e:
            logger.error(f"Failed to get market clock from pandas_market_calendars: {e}")
            # Fallback: simple weekday + time check, assume no holidays
            if today.weekday() < 5 and market_open_today <= now_rome < market_close_today:
                is_open = True
            next_open = market_open_today + timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)

        if next_open is None:
            next_open = now_rome + timedelta(days=1)

        clock = ClockInfo(is_open=is_open, timestamp=now_rome, next_open=next_open)
        self._clock_cache = clock
        self._clock_cache_time = now
        return clock

    async def stop(self):
        """Gracefully stop the engine and all background tasks."""
        logger.info("Stopping trading engine...")
        self._running = False
        for task in self._delayed_entry_tasks:
            task.cancel()
        self._delayed_entry_tasks.clear()
        logger.info("Cancelled delayed entry tasks.")
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

    async def _periodic_reconcile(self):
        """Run position reconciliation every 5 minutes (medium/long-term)."""
        while self._running:
            if self._reconcile_running:
                logger.warning("Reconcile still running; skipping this cycle.")
                await asyncio.sleep(60)
                continue
            self._reconcile_running = True
            try:
                await self._reconcile_positions()
            except Exception as e:
                logger.error(f"Reconcile error: {e}", exc_info=True)
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
                await asyncio.sleep(1)
                continue
            self._reevaluate_running = True
            try:
                # Always run re-evaluation, even if paused, to keep generating signals
                reeval_start_time = time.time()
                logger.info("Starting symbol re-evaluation...")
                is_forced = self._force_reeval or self._reeval_pending_force
                self._force_reeval = False
                self._reeval_pending_force = False
                await self._reevaluate_symbols(force=is_forced)
                elapsed = time.time() - reeval_start_time
                logger.info(f"Symbol re-evaluation complete (took {elapsed:.1f}s).")
            except Exception as e:
                logger.error(f"Stock re-evaluation error: {e}", exc_info=True)
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
            try:
                await asyncio.wait_for(self._reeval_trigger.wait(), timeout=self._symbol_reevaluation_interval)
            except asyncio.TimeoutError:
                pass
            self._reeval_trigger.clear()

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
                            # No LLM-set duration → resume after the LLM-decided minimum pause duration
                            default_max_pause = settings.MIN_LLM_PAUSE_DURATION
                            try:
                                raw = await asyncio.to_thread(self.redis.get, "trading:min_llm_pause_duration")
                                if raw:
                                    default_max_pause = int(raw)
                            except Exception:
                                pass
                            if pause_start_raw is None:
                                logger.warning(
                                    "Pause has no duration and no start time; "
                                    "forcing auto-resume immediately."
                                )
                                stored_reason_raw = await asyncio.to_thread(self.redis.get, "trading:pause_reason")
                                stored_reason = stored_reason_raw.decode() if isinstance(stored_reason_raw, bytes) else (stored_reason_raw or "")
                                pause_keys = [
                                    "trading:paused",
                                    "trading:pause_source",
                                    "trading:pause_start",
                                    "trading:pause_duration",
                                    "trading:pause_reason",
                                    "trading:llm_pause_time",
                                ]
                                for key in pause_keys:
                                    await asyncio.to_thread(self.redis.delete, key)
                                self._reeval_trigger.set()
                                await asyncio.to_thread(self.redis.set, "trading:last_auto_resume", str(time.time()))
                                await asyncio.to_thread(self.redis.setex, "trading:auto_resume_cooldown", 600, "1")
                                if self.notifier:
                                    await self.notifier.send_notification(
                                        "⏰ Trading auto-resumed (pause had no duration and no start time).",
                                        summary={"action": "RESUME", "reason": "Fallback: no pause start time"}
                                    )
                            else:
                                try:
                                    elapsed = time.time() - float(pause_start_raw)
                                    if elapsed >= default_max_pause:
                                        logger.warning(
                                            "Pause has no duration; forcing auto‑resume after default fallback (2 hours)."
                                        )
                                        stored_reason_raw = await asyncio.to_thread(self.redis.get, "trading:pause_reason")
                                        stored_reason = stored_reason_raw.decode() if isinstance(stored_reason_raw, bytes) else (stored_reason_raw or "")
                                        pause_keys = [
                                            "trading:paused",
                                            "trading:pause_source",
                                            "trading:pause_start",
                                            "trading:pause_duration",
                                            "trading:pause_reason",
                                            "trading:llm_pause_time",
                                        ]
                                        for key in pause_keys:
                                            await asyncio.to_thread(self.redis.delete, key)
                                        self._reeval_trigger.set()
                                        await asyncio.to_thread(self.redis.set, "trading:last_auto_resume", str(time.time()))
                                        await asyncio.to_thread(self.redis.setex, "trading:auto_resume_cooldown", 600, "1")
                                        if self.notifier:
                                            await self.notifier.send_notification(
                                                "⏰ Trading auto‑resumed after maximum pause duration (no LLM‑set duration).",
                                                summary={"action": "RESUME", "reason": "Fallback pause timeout"}
                                            )
                                    else:
                                        # still waiting, but we already know there is no duration, don't spam log
                                        pass
                                except (ValueError, TypeError):
                                    pass
                            await asyncio.sleep(30)
                            continue   # skip the original duration logic, proceed to next loop iteration
                        if pause_start_raw and pause_duration_raw:
                            try:
                                pause_start = float(pause_start_raw)
                                pause_duration = int(pause_duration_raw)
                                if time.time() - pause_start >= pause_duration:
                                    logger.info("Pause duration elapsed – auto-resuming trading.")
                                    stored_reason_raw = await asyncio.to_thread(self.redis.get, "trading:pause_reason")
                                    stored_reason = stored_reason_raw.decode() if isinstance(stored_reason_raw, bytes) else (stored_reason_raw or "")
                                    # Delete all pause keys
                                    pause_keys = [
                                        "trading:paused",
                                        "trading:pause_source",
                                        "trading:pause_start",
                                        "trading:pause_duration",
                                        "trading:pause_reason",
                                        "trading:llm_pause_time",
                                    ]
                                    for key in pause_keys:
                                        await asyncio.to_thread(self.redis.delete, key)
                                    self._reeval_trigger.set()
                                    await asyncio.to_thread(self.redis.set, "trading:last_auto_resume", str(time.time()))
                                    await asyncio.to_thread(self.redis.setex, "trading:auto_resume_cooldown", 600, "1")
                                    if self.notifier:
                                        reason_text = f" (was paused: {stored_reason})" if stored_reason else ""
                                        await self.notifier.send_notification(
                                            f"▶️ Trading auto-resumed after pause duration elapsed.{reason_text}",
                                            summary={"action": "RESUME", "reason": f"Pause duration elapsed{reason_text}"}
                                        )
                            except (ValueError, TypeError):
                                pass  # ignore malformed values
            except Exception as e:
                logger.error(f"Pause check error: {e}", exc_info=True)
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
                stock_assets = await self._get_tradable_assets()
                stock_pairs = [f"{sym}/{self.base_currency}" for sym in stock_assets]
                etf_symbols = await self._get_etf_symbols()
                etf_pairs = [f"{sym}/{self.base_currency}" for sym in etf_symbols]
                btp_bonds = await self._get_btp_bonds()
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
                    MAX_BREADTH_SAMPLE = 500
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
                    raw_breadth = await asyncio.to_thread(get_quotes_cached, plain_breadth)
                    breadth_tickers = {pair: raw_breadth.get(pair.split("/")[0], {}) for pair in breadth_pairs}

                    # Fall back to DB close prices for symbols without cached quotes
                    missing_breadth = [
                        s.split("/")[0] for s in breadth_pairs
                        if breadth_tickers.get(s, {}).get('percentage') is None
                    ]
                    if missing_breadth:
                        try:
                            db_candles = await asyncio.to_thread(get_latest_close_prices, missing_breadth)
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
            except Exception as e:
                logger.error(f"Full market breadth computation error: {e}", exc_info=True)
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
                await self._symbol_reevaluator.check_market_conditions()
            except Exception as e:
                logger.error(f"Market condition check error: {e}", exc_info=True)
            await asyncio.sleep(1800)  # check every 30 minutes (medium/long-term)

    async def _market_clock_monitor(self):
        """Periodically check market clock and pause/resume trading based on market open/close."""
        await asyncio.sleep(5)  # initial delay
        while self._running:
            try:
                clock = await self._get_clock()
                if clock is None:
                    await asyncio.sleep(30)
                    continue

                is_open = clock.is_open
                paused = await asyncio.to_thread(self.redis.get, "trading:paused")

                if not is_open:
                    # Market closed – always pause trading immediately
                    now_rome = clock.timestamp.astimezone(ZoneInfo(settings.MARKET_TIMEZONE))
                    next_open_rome = clock.next_open.astimezone(ZoneInfo(settings.MARKET_TIMEZONE))
                    remaining_seconds = (clock.next_open - clock.timestamp).total_seconds()
                    if remaining_seconds > 3600:
                        hours = int(remaining_seconds // 3600)
                        minutes = int((remaining_seconds % 3600) // 60)
                        countdown_str = f"{hours}h {minutes}m"
                    elif remaining_seconds > 60:
                        minutes = int(remaining_seconds // 60)
                        seconds = int(remaining_seconds % 60)
                        countdown_str = f"{minutes}m {seconds}s"
                    else:
                        countdown_str = f"{int(remaining_seconds)}s"
                    reason = "Market closed"
                    # Check if we already sent the market closed notification
                    source_raw = await asyncio.to_thread(self.redis.get, "trading:pause_source")
                    source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")
                    already_market_closed = (source == "market_closed")
                    # Delete all existing pause keys to ensure clean state
                    pause_keys = [
                        "trading:paused",
                        "trading:pause_source",
                        "trading:pause_start",
                        "trading:pause_duration",
                        "trading:pause_reason",
                        "trading:llm_pause_time",
                        "trading:market_next_open",
                    ]
                    for key in pause_keys:
                        await asyncio.to_thread(self.redis.delete, key)
                    # Set new pause keys unconditionally
                    await asyncio.to_thread(self.redis.set, "trading:paused", "1")
                    await asyncio.to_thread(self.redis.set, "trading:pause_source", "market_closed")
                    await asyncio.to_thread(self.redis.set, "trading:pause_reason", reason)
                    await asyncio.to_thread(self.redis.set, "trading:market_next_open", clock.next_open.isoformat())
                    logger.debug(f"Market closed, pausing trading. Reason: {reason}")
                    if self.notifier and not already_market_closed:
                        await self.notifier.send_notification(
                            f"⏸️ {reason}",
                            summary={"action": "PAUSE", "reason": reason}
                        )
                    # Invalidate clock cache so the next monitor cycle sees the updated state
                    self._clock_cache = None

                # --- Periodic countdown updates while market is closed ---
                # Only send updates if we are currently paused due to market_closed
                source_raw = await asyncio.to_thread(self.redis.get, "trading:pause_source")
                source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")
                if source == "market_closed" and not is_open:
                    now_ts = time.time()
                    # Recompute remaining seconds from the live clock (or fallback)
                    if clock is not None:
                        remaining_seconds = (clock.next_open - datetime.now(timezone.utc)).total_seconds()
                    else:
                        # Fallback to stored next_open
                        next_open_raw = await asyncio.to_thread(self.redis.get, "trading:market_next_open")
                        if next_open_raw:
                            next_open_str = next_open_raw.decode() if isinstance(next_open_raw, bytes) else next_open_raw
                            next_open_dt = datetime.fromisoformat(next_open_str)
                            remaining_seconds = (next_open_dt - datetime.now(timezone.utc)).total_seconds()
                        else:
                            remaining_seconds = 0

                    # Periodic update every 30 minutes (1800 seconds)
                    if now_ts - self._last_market_closed_notify_time >= 1800:
                        if remaining_seconds > 0:
                            hours = int(remaining_seconds // 3600)
                            minutes = int((remaining_seconds % 3600) // 60)
                            if hours > 0:
                                countdown_str = f"{hours}h {minutes}m"
                            elif minutes > 0:
                                seconds = int(remaining_seconds % 60)
                                countdown_str = f"{minutes}m {seconds}s"
                            else:
                                countdown_str = f"{int(remaining_seconds)}s"
                            next_open_rome = clock.next_open.astimezone(ZoneInfo(settings.MARKET_TIMEZONE)) if clock else None
                            next_open_str = next_open_rome.strftime('%H:%M %d/%m/%Y') if next_open_rome else "?"
                            update_msg = (
                                f"⏸️ Market still closed. Reopens in {countdown_str} "
                                f"(at {next_open_str})"
                            )
                            if self.notifier:
                                await self.notifier.send_notification(
                                    update_msg,
                                    summary={"action": "PAUSE", "reason": update_msg}
                                )
                        self._last_market_closed_notify_time = now_ts

                    # "Opening soon" alert when less than 5 minutes remain
                    if 0 < remaining_seconds <= 900 and not self._market_opening_soon_notified:
                        minutes_left = int(remaining_seconds // 60)
                        soon_msg = f"⏰ Market opens in ~{minutes_left} minute(s) – trading will resume automatically."
                        if self.notifier:
                            await self.notifier.send_notification(
                                soon_msg,
                                summary={"action": "INFO", "reason": "Market opening soon"}
                            )
                        self._market_opening_soon_notified = True
                        # Invalidate correlation matrix cache before pre-market re-evaluation
                        await asyncio.to_thread(self.redis.delete, "reeval:correlation_matrix")
                        # Trigger pre-market re-evaluation so we're prepared with fresh signals
                        self._force_reeval = True
                        self._pre_market_reeval = True
                        self._reeval_trigger.set()
                else:
                    # Market open – resume trading only if paused due to market closure.
                    # Respect LLM-initiated and manual pauses while the market is open.
                    paused = await asyncio.to_thread(self.redis.get, "trading:paused")
                    if paused:
                        source_raw = await asyncio.to_thread(self.redis.get, "trading:pause_source")
                        source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")
                        if source == "market_closed":
                            # Only clear market-closed pauses; respect LLM and manual pauses
                            pause_keys = [
                                "trading:paused",
                                "trading:pause_source",
                                "trading:pause_start",
                                "trading:pause_duration",
                                "trading:pause_reason",
                                "trading:llm_pause_time",
                                "trading:market_next_open",
                            ]
                            for key in pause_keys:
                                await asyncio.to_thread(self.redis.delete, key)
                            logger.info("Market opened, clearing market-closed pause (trading resumed).")
                            if self.notifier:
                                await self.notifier.send_notification(
                                    "▶️ Market opened, trading resumed.",
                                    summary={"action": "RESUME", "reason": "Market opened"}
                                )
                            # Invalidate correlation matrix cache on market open
                            await asyncio.to_thread(self.redis.delete, "reeval:correlation_matrix")
                            # Only trigger re-evaluation when we actually resumed from a pause
                            self._reeval_trigger.set()
                            # Invalidate clock cache so subsequent calls get fresh data
                            self._clock_cache = None
                        else:
                            logger.debug("Market open, but trading paused by '%s' – not clearing.", source)
                    else:
                        logger.debug("Market open, trading already active.")
                        # Do NOT trigger re-evaluation when already active — let the normal
                        # periodic interval handle it to avoid spamming re-evaluations every 60s.
                    # Reset the "opening soon" notification flag
                    self._market_opening_soon_notified = False
                    # Reset the periodic countdown timer so the first update
                    # after the next market close is not skipped due to a stale timestamp.
                    self._last_market_closed_notify_time = 0.0
            except Exception as e:
                logger.error(f"Market clock monitor error: {e}", exc_info=True)
            await asyncio.sleep(30)  # check every 30 seconds

    async def _get_sentiment_str(self, symbol: str) -> str:
        """Get a short news sentiment string for notifications, including an LLM summary."""
        if not settings.NEWS_ENABLED:
            return ""
        try:
            base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
            agg_sent = await asyncio.to_thread(get_aggregate_sentiment_from_db, base_symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
            if not agg_sent:
                return ""

            compound = agg_sent["avg_compound"]
            sentiment_label = "positive" if compound > 0.05 else "negative" if compound < -0.05 else "neutral"
            total = agg_sent["total_articles"]

            # Try to get an LLM-generated summary of the news
            summary = ""
            try:
                summary_raw = await asyncio.to_thread(get_cached_news_summary, symbol)
                if isinstance(summary_raw, dict):
                    summary = summary_raw.get("summary", "")
                else:
                    summary = summary_raw
                if summary in ("No recent news.", "Could not generate summary."):
                    summary = ""
            except Exception:
                pass  # fallback to no summary

            base = f"📰 (sentiment: {compound:+.2f}[{sentiment_label}], {total} articles)"
            if summary:
                return f"{base} – {summary}"
            return base
        except Exception:
            pass
        return ""


    @staticmethod
    def _get_quote_staleness_warning(ticker: Dict[str, Any]) -> str:
        """Return a warning string if the quote data is stale, or empty string if fresh."""
        last_update = ticker.get("last_update")
        source = ticker.get("source", "unknown")

        if last_update is None:
            # No timestamp available — can't determine staleness
            return ""

        age_seconds = (time.time() * 1000 - last_update) / 1000

        # Only warn for potentially stale sources
        if source == "db_close" and age_seconds > 900:  # 15 minutes
            age_minutes = int(age_seconds / 60)
            return (
                f"\n⚠️ **STALE QUOTE WARNING:** The current price ({ticker.get('last')}) "
                f"is from database close prices and is {age_minutes} minutes old. "
                f"It may not reflect real-time market conditions. "
                f"Exercise extra caution and consider waiting for fresher data.\n"
            )
        elif source == "db_quotes" and age_seconds > 900:
            age_minutes = int(age_seconds / 60)
            return (
                f"\n⚠️ **STALE QUOTE WARNING:** The current price ({ticker.get('last')}) "
                f"is from cached database quotes and is {age_minutes} minutes old. "
                f"It may not reflect real-time market conditions.\n"
            )
        elif source == "yfinance" and age_seconds > 900:
            age_minutes = int(age_seconds / 60)
            return (
                f"\n⚠️ **STALE QUOTE WARNING:** The current price ({ticker.get('last')}) "
                f"is from Yahoo Finance and is {age_minutes} minutes old. "
                f"It may not reflect real-time market conditions. "
                f"Exercise extra caution and consider waiting for fresher data.\n"
            )

        return ""

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
        # Use at least the configured max staleness, or 10% of the timeframe,
        # whichever is greater (capped at 1 day for very long timeframes).
        tf_seconds = self._timeframe_to_seconds(timeframe)
        scaled_threshold = max(settings.QUOTE_MAX_STALENESS_SECONDS, min(tf_seconds * 0.1, 86400))
        return age_seconds > scaled_threshold

    async def _fetch_vix(self) -> Optional[float]:
        """VIX is not available for the Italian market via yfinance. Returns None."""
        return None



    async def _compute_volume_trend(self, symbol: str, current_volume: float, timeframe: Optional[str] = None) -> Optional[float]:
        """Compute volume trend as ratio of current 24h volume to EMA of past volumes.

        Returns the ratio (e.g., 2.0 means current volume is 2× the average).
        Returns None if volume data is unavailable.

        For long-term timeframes (>= 1 day), the ratio is cached in Redis
        to avoid recomputing on every evaluation cycle.
        """
        if current_volume <= 0:
            return None

        # Determine cache TTL based on timeframe
        cache_ttl = 0  # no caching by default
        if timeframe is not None:
            tf_seconds = self._timeframe_to_seconds(timeframe)
            if tf_seconds >= 2_592_000:  # >= 1 month
                cache_ttl = 3600       # 1 hour
            elif tf_seconds >= 604_800:  # >= 1 week
                cache_ttl = 1800       # 30 minutes
            elif tf_seconds >= 86_400:  # >= 1 day
                cache_ttl = 900        # 15 minutes

        # Check ratio cache first (long timeframes only)
        ratio_cache_key = f"volume_trend:ratio:{symbol}"
        if cache_ttl > 0:
            try:
                cached_ratio = await asyncio.to_thread(self.redis.get, ratio_cache_key)
                if cached_ratio is not None:
                    return round(float(cached_ratio), 3)
            except Exception:
                pass

        redis_key = f"volume_trend:ema:{symbol}"
        alpha = 0.3  # EMA smoothing factor

        try:
            stored = await asyncio.to_thread(self.redis.get, redis_key)
            if stored is not None:
                old_avg = float(stored)
                new_avg = alpha * current_volume + (1 - alpha) * old_avg
                ratio = current_volume / old_avg if old_avg > 0 else 1.0
                # Store the updated average with 7-day TTL
                await asyncio.to_thread(self.redis.setex, redis_key, 7 * 24 * 3600, str(new_avg))
                # Cache the ratio for long-term timeframes
                if cache_ttl > 0:
                    await asyncio.to_thread(self.redis.setex, ratio_cache_key, cache_ttl, str(ratio))
                return round(ratio, 3)
            else:
                # First observation: initialize with current volume, ratio = 1.0
                await asyncio.to_thread(self.redis.setex, redis_key, 7 * 24 * 3600, str(current_volume))
                if cache_ttl > 0:
                    await asyncio.to_thread(self.redis.setex, ratio_cache_key, cache_ttl, "1.0")
                return 1.0
        except Exception as e:
            logger.info(f"Volume trend computation failed for {symbol}: {e}")
            return None

    async def _compute_and_store_indicators(self, symbol: str, timeframe: str, candles: List[List]):
        """Compute indicators for a symbol/timeframe using TA-Lib and store in DB."""
        await self._market_data_manager.compute_and_store_indicators(symbol, timeframe, candles)

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
        except Exception:
            pass

        try:
            from src.news.fetcher import fetch_news_for_symbol
            stock_name = await self._get_stock_name(symbol)
            articles = await asyncio.to_thread(fetch_news_for_symbol, symbol, stock_name)
            if articles:
                await asyncio.to_thread(store_news_articles, base_symbol, articles)
            else:
                # Cache the fact that we found 0 articles to avoid re-fetching too soon
                try:
                    await asyncio.to_thread(
                        self.redis.setex, no_news_cache_key, settings.NEWS_CACHE_TTL_SECONDS, "1"
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.info(f"News fetch/store failed for {symbol}: {e}")

    async def _risk_management_loop(self):
        """Check stop-loss, take-profit, and other risk rules on every ticker update."""
        await asyncio.sleep(5)  # initial delay
        while self._running:
            try:
                # Scale risk check interval based on the shortest timeframe among
                # open positions.  For very long timeframes (>= 1 month), checking
                # every 2 minutes is wasteful — use ~1% of the timeframe instead,
                # capped at 1 hour minimum and 1 day maximum.
                min_tf_seconds = None
                for pos in self.positions.values():
                    pos_tf = pos.get("timeframe")
                    if pos_tf:
                        pos_tf_secs = self._timeframe_to_seconds(pos_tf)
                        if min_tf_seconds is None or pos_tf_secs < min_tf_seconds:
                            min_tf_seconds = pos_tf_secs
                if min_tf_seconds is not None and min_tf_seconds >= settings.LONG_TERM_TF_SECONDS:  # >= 1 month
                    risk_interval = max(3600, min(86400, int(min_tf_seconds * 0.01)))
                else:
                    risk_interval = settings.RISK_CHECK_INTERVAL_SECONDS
                await asyncio.sleep(risk_interval)
                await self._check_risk_management()
                await self._save_state()
                self._state_dirty = True
            except Exception as e:
                logger.error(f"Risk management loop error: {e}", exc_info=True)

    async def _refresh_current_symbols_news_fast(self):
        """Fast news refresh loop – only for the symbols currently tracked by the engine."""
        if not settings.NEWS_ENABLED:
            return
        # Fetch immediately on startup, then periodically
        while self._running:
            if self._news_fast_running:
                logger.warning("Fast news refresh still running; skipping this cycle.")
                await asyncio.sleep(settings.NEWS_FAST_UPDATE_INTERVAL_MINUTES * 60)
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
            except Exception as e:
                logger.error(f"Fast news refresh error: {e}")
            finally:
                self._news_fast_running = False
            await asyncio.sleep(settings.NEWS_FAST_UPDATE_INTERVAL_MINUTES * 60)

    async def _refresh_news_cache(self):
        """Periodically fetch news for tracked stocks/ETFs and top-volume stocks to keep cache warm."""
        if not settings.NEWS_ENABLED:
            return
        try:
            from src.news.fetcher import fetch_news_for_symbol
        except ImportError:
            logger.warning("News module not available; skipping background news refresh.")
            return

        while self._running:
            if self._news_cache_running:
                logger.warning("News cache refresh still running; skipping this cycle.")
                await asyncio.sleep(settings.NEWS_UPDATE_INTERVAL_MINUTES * 60)
                continue
            self._news_cache_running = True
            try:
                cycle_start = time.time()
                # Slow refresh: all available pairs EXCEPT the stocks already handled by the fast loop
                current_symbols = {entry["symbol"] for entry in self.current_symbols}
                symbols_to_refresh = set()
                try:
                    plain_assets = await self._get_tradable_assets()
                    available_pairs = [f"{sym}/{self.base_currency}" for sym in plain_assets]
                    # Fetch tickers for a subset to determine top volume symbols
                    # (limit to 200 to avoid excessive API calls)
                    sample_for_vol = available_pairs[:200]
                    plain_sample = [s.split("/")[0] for s in sample_for_vol]
                    raw_quotes = await self._get_quotes_batched(plain_sample, timeout_per_chunk=45.0)
                    tickers = {pair: raw_quotes.get(pair.split("/")[0], {}) for pair in sample_for_vol}
                    def _vol(sym):
                        t = tickers.get(sym, {})
                        return t.get('quoteVolume', 0) or 0
                    symbols_to_refresh = set(sample_for_vol) - current_symbols
                except Exception as e:
                    logger.warning(f"Could not get available pairs for news refresh: {e}")

                for sym in symbols_to_refresh:
                    try:
                        async with self._news_semaphore:
                            stock_name = await self._get_stock_name(sym)
                            articles = await asyncio.to_thread(fetch_news_for_symbol, sym, stock_name)
                            if articles:
                                base_symbol = sym.split("/")[0] if "/" in sym else sym
                                await asyncio.to_thread(store_news_articles, base_symbol, articles)
                    except Exception as e:
                        logger.info(f"News refresh failed for {sym}: {e}")
                    await asyncio.sleep(0.2)

                logger.info(f"News cache refreshed for {len(symbols_to_refresh)} symbols in {time.time() - cycle_start:.2f}s")
            except Exception as e:
                logger.error(f"Background news refresh error: {e}")
            finally:
                self._news_cache_running = False

            # Clean up old news articles
            try:
                from src.database import cleanup_old_news
                await asyncio.to_thread(cleanup_old_news, settings.NEWS_RETENTION_SECONDS)
            except Exception as e:
                logger.warning(f"News cleanup failed: {e}")

            await asyncio.sleep(settings.NEWS_UPDATE_INTERVAL_MINUTES * 60)

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

    async def _get_stock_name(self, symbol: str) -> str:
        """Return the human-readable company name for a symbol, cached in Redis."""
        return await self._market_data_manager.get_stock_name(symbol)

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

    async def _backfill_ohlcv(self, symbol: str, timeframe: str, start_ms: int, end_ms: int, max_candles: int = None, ignore_existing: bool = False, force: bool = False, quiet: bool = False) -> int:
        """Fetch and store all missing OHLCV candles between start_ms and end_ms.
        Returns the number of candles inserted."""
        logger.debug(f"Backfill started for {symbol} {timeframe}: {start_ms} → {end_ms}")
        if ignore_existing:
            since = start_ms
        else:
            latest_ts = await asyncio.to_thread(get_latest_ohlcv_timestamp, symbol, timeframe)
            if latest_ts is None:
                since = start_ms
            else:
                # Skip the API call entirely if the latest candle is recent enough
                # (within one candle interval of now). No new data to fetch.
                if not force:
                    interval_ms = self._timeframe_to_ms(timeframe)
                    now_ms = int(time.time() * 1000)
                    if latest_ts >= now_ms - interval_ms:
                        logger.debug(f"Skipping backfill for {symbol} {timeframe}: data is up to date (latest_ts={latest_ts})")
                        return 0
                since = max(start_ms, latest_ts + 1)

        total_inserted = 0
        if max_candles is None:
            max_candles = 10000  # Fetch all available history in one go
        while since < end_ms:
            try:
                async with self._download_semaphore:
                    loop = asyncio.get_running_loop()
                    candles = await loop.run_in_executor(
                        self._download_executor,
                        get_bars_range,
                        symbol.split("/")[0], timeframe, since, 10000
                    )
            except Exception as e:
                logger.warning(f"get_bars_range failed for {symbol} {timeframe} at {since}: {e}")
                break

            if not candles:
                break

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._db_executor, insert_ohlcv_batch, symbol, timeframe, candles)
            batch_count = len(candles)
            total_inserted += batch_count
            logger.debug(f"Backfill batch: {symbol} {timeframe} fetched {batch_count} candles from {since}")

            if total_inserted >= max_candles:
                logger.debug(
                    f"Backfill limit reached for {symbol} {timeframe}: {total_inserted} candles inserted "
                    f"(max {max_candles}). Remaining range will be filled in next cycle."
                )
                break

            last_ts = candles[-1][0]
            if last_ts <= since:
                # Avoid infinite loop if exchange returns same candle
                break
            since = last_ts + 1
            # Small delay to avoid rate limits
            await asyncio.sleep(0.05)

        if total_inserted >= max_candles:
            logger.debug(f"Backfill partial for {symbol} {timeframe}: {total_inserted} candles inserted (limit reached)")
        elif quiet:
            logger.debug(f"Backfill complete for {symbol} {timeframe}: {total_inserted} candles inserted")
        else:
            logger.info(f"Backfill complete for {symbol} {timeframe}: {total_inserted} candles inserted")
        return total_inserted

    async def _fill_gaps(self, symbol: str, timeframe: str):
        """Detect and fill gaps in stored OHLCV data for a symbol/timeframe."""
        interval_ms = self._timeframe_to_ms(timeframe)
        if interval_ms <= 0:
            return

        # Get all stored timestamps
        loop = asyncio.get_running_loop()
        candles = await loop.run_in_executor(self._download_executor, get_ohlcv, symbol, timeframe, 50000)
        if len(candles) < 2:
            logger.debug(f"Not enough data to check gaps for {symbol} {timeframe}")
            return

        timestamps = sorted(c["timestamp"] for c in candles)

        # Find all gaps larger than 1.5x the expected interval
        gaps: List[Tuple[int, int, int]] = []  # (gap_size, gap_start, gap_end)
        for i in range(len(timestamps) - 1):
            gap = timestamps[i + 1] - timestamps[i]
            if gap > interval_ms * 1.5:
                gap_start = timestamps[i] + interval_ms
                gap_end = timestamps[i + 1] - interval_ms
                if gap_end > gap_start:
                    gaps.append((gap, gap_start, gap_end))

        gaps_found = len(gaps)
        gaps_filled = 0
        max_gaps_per_cycle = 5  # Limit gap fills per cycle to avoid rate limits

        # Sort by gap size descending so the largest (most impactful) gaps are filled first
        gaps.sort(key=lambda g: g[0], reverse=True)

        for gap_size, gap_start, gap_end in gaps[:max_gaps_per_cycle]:
            logger.debug(f"Gap detected for {symbol} {timeframe}: {gap_start} → {gap_end} (size {gap_size}ms)")
            await self._backfill_ohlcv(symbol, timeframe, gap_start, gap_end, ignore_existing=True)
            gaps_filled += 1

        if gaps_found == 0:
            logger.debug(f"No gaps found for {symbol} {timeframe}")
        else:
            logger.debug(f"Gap check for {symbol} {timeframe}: {gaps_found} gaps found, {gaps_filled} filled")

    async def _backfill_new_symbol(self, symbol: str, timeframe: str):
        """Immediately backfill 30 days of OHLCV data for a newly selected symbol (assigned timeframe only)."""
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000
        logger.debug(f"Starting immediate backfill for newly selected symbol {symbol} ({timeframe})")
        await self._download_symbol_ohlcv(symbol, timeframe, start_ms, now_ms)
        logger.debug(f"Immediate backfill complete for {symbol} ({timeframe})")

    async def _download_symbol_ohlcv(self, symbol: str, timeframe: str, start_ms: int, end_ms: int, quiet: bool = False, force: bool = False) -> None:
        """Download OHLCV, fill gaps, and compute/store indicators for a single symbol/timeframe."""
        loop = asyncio.get_running_loop()
        try:
            inserted = await self._backfill_ohlcv(symbol, timeframe, start_ms, end_ms, quiet=quiet, force=force)
            if inserted > 0 or force:
                await self._fill_gaps(symbol, timeframe)
                db_candles = await loop.run_in_executor(self._download_executor, get_ohlcv, symbol, timeframe, 200)
                if db_candles:
                    raw_candles = [[c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in db_candles]
                    await self._compute_and_store_indicators(symbol, timeframe, raw_candles)
        except Exception as e:
            logger.warning(f"Download failed for {symbol} {timeframe}: {e}")

    async def _download_market_data_loop(self):
        """Periodically download and store OHLCV data for tracked stocks, with gap detection."""
        # Initial delay to let the engine settle
        await asyncio.sleep(30)
        while self._running:
            if self._market_data_running:
                logger.warning("Market data download still running; skipping this cycle.")
                await asyncio.sleep(self._get_effective_refresh_interval(settings.MARKET_DATA_REFRESH_SECONDS, "data"))
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
                        await self._download_symbol_ohlcv(symbol, tf, start_ms, now_ms)

                    shuffled_symbols = list(self.current_symbols)
                    random.shuffle(shuffled_symbols)
                    download_tasks = [_download_symbol_data(entry) for entry in shuffled_symbols]
                    await asyncio.gather(*download_tasks)
                    logger.info("Market data download cycle complete.")
                    # Clean up old OHLCV data (older than retention period)
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(self._db_executor, cleanup_old_ohlcv, settings.OHLCV_RETENTION_DAYS)
            except Exception as e:
                logger.error(f"Market data download loop error: {e}", exc_info=True)
            finally:
                self._market_data_running = False

            await asyncio.sleep(self._get_effective_refresh_interval(settings.MARKET_DATA_REFRESH_SECONDS, "data"))

    async def _download_all_assets_data_loop(self):
        """Periodically download OHLCV for ALL tradable assets (stocks, ETFs, BTPs)."""
        await asyncio.sleep(120)  # initial delay to let the engine settle
        while self._running:
            if self._full_download_running:
                logger.info("Full download already running (likely force download); skipping this cycle.")
                await asyncio.sleep(self._get_effective_refresh_interval(settings.FULL_ASSET_OHLCV_DOWNLOAD_INTERVAL_SECONDS, "data"))
                continue
            self._full_download_running = True
            try:
                logger.info("Starting full asset OHLCV download cycle...")
                # 1. Get all stock + ETF symbols
                plain_assets = await self._get_tradable_assets()
                stock_pairs = [f"{sym}/{self.base_currency}" for sym in plain_assets]

                # 2. Get all BTP symbols
                btp_bonds = await self._get_btp_bonds()
                btp_pairs = [f"{b['isin']}/{self.base_currency}" for b in btp_bonds]

                all_pairs = stock_pairs + btp_pairs
                if not all_pairs:
                    logger.info("No tradable assets found; skipping full download.")
                    await asyncio.sleep(self._get_effective_refresh_interval(settings.FULL_ASSET_OHLCV_DOWNLOAD_INTERVAL_SECONDS, "data"))
                    continue

                now_ms = int(time.time() * 1000)
                start_ms = now_ms - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000

                # Prioritize symbols with missing data for configured timeframes
                async def _has_missing_data(pair: str) -> bool:
                    """Return True if pair is missing OHLCV data for any configured timeframe."""
                    for tf in settings.OHLCV_TIMEFRAMES:
                        latest_ts = await asyncio.to_thread(get_latest_ohlcv_timestamp, pair, tf)
                        if latest_ts is None:
                            return True
                    return False

                missing_checks = await asyncio.gather(*[_has_missing_data(pair) for pair in all_pairs])
                pairs_missing = [pair for pair, missing in zip(all_pairs, missing_checks) if missing]
                pairs_complete = [pair for pair, missing in zip(all_pairs, missing_checks) if not missing]
                random.shuffle(pairs_missing)
                random.shuffle(pairs_complete)
                if pairs_missing:
                    logger.info(f"Prioritizing {len(pairs_missing)} symbols with missing OHLCV data out of {len(all_pairs)} total.")
                all_pairs = pairs_missing + pairs_complete

                async def _download_symbol_data(pair: str):
                    for tf in settings.OHLCV_TIMEFRAMES:
                        await self._download_symbol_ohlcv(pair, tf, start_ms, now_ms, quiet=True)

                # Limit concurrent symbol downloads to avoid thread pool exhaustion
                download_concurrency = asyncio.Semaphore(10)
                async def _limited_download(pair: str):
                    async with download_concurrency:
                        await _download_symbol_data(pair)
                download_tasks = [_limited_download(pair) for pair in all_pairs]
                await asyncio.gather(*download_tasks)

                # Clean up old data
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._db_executor, cleanup_old_ohlcv, settings.OHLCV_RETENTION_DAYS)
                await loop.run_in_executor(self._db_executor, cleanup_old_position_pnl, 90)
                await loop.run_in_executor(self._db_executor, cleanup_old_backtest_results, 90)
                logger.info("Full asset OHLCV download cycle complete.")
            except Exception as e:
                logger.error(f"Full asset download loop error: {e}", exc_info=True)
            finally:
                self._full_download_running = False

            # Wait before next full download
            await asyncio.sleep(self._get_effective_refresh_interval(settings.FULL_ASSET_OHLCV_DOWNLOAD_INTERVAL_SECONDS, "data"))

    async def _cleanup_yf_cache_loop(self):
        """No-op: yfinance manages its own cache. External deletion caused SQLite errors."""
        pass

    async def _download_all_news_loop(self):
        """Periodically pre‑fetch news for ALL tradable assets (stocks, ETFs, BTPs)."""
        if not settings.NEWS_ENABLED:
            return
        await asyncio.sleep(180)  # initial delay to let the engine settle
        while self._running:
            try:
                logger.info("Starting full asset news download cycle...")
                # 1. Get all stock + ETF symbols
                plain_assets = await self._get_tradable_assets()
                stock_pairs = [f"{sym}/{self.base_currency}" for sym in plain_assets]

                # 2. Get all BTP symbols
                btp_bonds = await self._get_btp_bonds()
                btp_pairs = [f"{b['isin']}/{self.base_currency}" for b in btp_bonds]

                all_pairs = stock_pairs + btp_pairs
                if not all_pairs:
                    logger.info("No tradable assets found; skipping full news download.")
                    await asyncio.sleep(self._get_effective_refresh_interval(settings.FULL_ASSET_NEWS_DOWNLOAD_INTERVAL_SECONDS, "news"))
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
                    except Exception as e:
                        logger.warning(f"Full news download failed for {pair}: {e}")

                news_tasks = [_download_news_for_symbol(pair) for pair in ordered_pairs]
                await asyncio.gather(*news_tasks)

                logger.info("Full asset news download cycle complete.")
            except Exception as e:
                logger.error(f"Full asset news download loop error: {e}", exc_info=True)

            await asyncio.sleep(self._get_effective_refresh_interval(settings.FULL_ASSET_NEWS_DOWNLOAD_INTERVAL_SECONDS, "news"))

    async def _refresh_all_quotes_loop(self):
        """Periodically fetch quotes for all tradable assets and cache them in Redis."""
        await asyncio.sleep(60)  # initial delay
        while self._running:
            if self._quotes_fetch_running:
                logger.info("Quotes fetch already running (likely re-evaluation or breadth); skipping this cycle.")
                await asyncio.sleep(self._get_effective_refresh_interval(settings.QUOTE_REFRESH_INTERVAL_SECONDS, "quotes"))
                continue
            self._quotes_fetch_running = True
            try:
                # Do NOT skip when the circuit breaker is open — get_quotes
                # internally checks the circuit breaker and falls back to DB
                # close prices (from market_data candles).  Skipping here
                # prevents those fallback prices from being saved to the
                # quotes table, leaving it stale when yfinance is down.
                plain_assets = await self._get_tradable_assets()
                if plain_assets:
                    # Fetch quotes in batches to avoid yfinance timeouts on large symbol lists
                    await self._get_quotes_batched(plain_assets, timeout_per_chunk=90.0)
                    self._portfolio_exposure_cache = None
            except Exception as e:
                logger.error(f"Background quote refresh error: {e}", exc_info=True)
            finally:
                self._quotes_fetch_running = False
            await asyncio.sleep(self._get_effective_refresh_interval(settings.QUOTE_REFRESH_INTERVAL_SECONDS, "quotes"))

    async def _refresh_ticker_discovery_loop(self):
        """Periodically discover tickers from news RSS feeds and trending stocks.
        Caches results in Redis so re-evaluation never blocks on slow HTTP calls."""
        await asyncio.sleep(120)  # initial delay
        while self._running:
            try:
                plain_assets = await self._get_tradable_assets()
                available_pairs = [f"{sym}/{self.base_currency}" for sym in plain_assets]

                if settings.NEWS_ENABLED and settings.NEWS_TICKER_DISCOVERY_ENABLED and discover_tickers_from_news is not None:
                    logger.info("Background: refreshing RSS ticker discovery...")
                    await asyncio.to_thread(
                        discover_tickers_from_news,
                        existing_pairs=available_pairs,
                        cache_only=False,
                    )

                if settings.NEWS_ENABLED and settings.NEWS_SYMBOL_DISCOVERY_ENABLED and discover_trending_stocks is not None:
                    logger.info("Background: refreshing trending stock discovery...")
                    await asyncio.to_thread(
                        discover_trending_stocks,
                        self.base_currency,
                        available_pairs,
                        max_symbols=settings.NEWS_SYMBOL_DISCOVERY_MAX_SYMBOLS,
                        min_sentiment=settings.NEWS_SYMBOL_DISCOVERY_MIN_SENTIMENT,
                        min_articles=settings.NEWS_SYMBOL_DISCOVERY_MIN_ARTICLES,
                        cache_only=False,
                    )
            except Exception as e:
                logger.error(f"Ticker discovery refresh error: {e}", exc_info=True)
            await asyncio.sleep(3600)  # every 60 minutes (medium/long-term)

    def _daily_realized_pnl(self) -> float:
        """Return the sum of realized P&L for trades closed today (UTC)."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date()
        total = 0.0
        for trade in self.trade_history:
            if trade.get("side") != "sell":
                continue
            ts = trade.get("timestamp", 0)
            if ts:
                trade_date = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).date()
                if trade_date == today:
                    total += trade.get("realized_pnl", 0.0)
        return total

    def _compute_performance_metrics(self) -> Dict[str, Any]:
        """Analyze trade history to produce per-symbol and per-strategy performance summaries."""
        return self._position_manager.compute_performance_metrics()

    def _compute_trade_pattern_analysis(self) -> Dict[str, Any]:
        """Analyze closed trades to identify which conditions, timeframes, and parameters
        have historically led to wins vs losses. Cached and only recomputed when new trades arrive."""
        return self._position_manager.compute_trade_pattern_analysis()

    async def _classify_market_regime(
        self,
        adx: Optional[float],
        plus_di: Optional[float],
        minus_di: Optional[float],
        ema_9: Optional[float],
        ema_21: Optional[float],
        bb_upper: Optional[float],
        bb_lower: Optional[float],
        bb_middle: Optional[float],
        atr: Optional[float],
        atr_percentile: Optional[float],
        current_price: float,
    ) -> str:
        """Classify market regime using multiple indicators."""
        return await self._signal_processor.classify_market_regime(
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            ema_9=ema_9,
            ema_21=ema_21,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            bb_middle=bb_middle,
            atr=atr,
            atr_percentile=atr_percentile,
            current_price=current_price,
        )

    async def _reconcile_positions(self):
        """Detect and handle external changes: delisted symbols, externally sold positions."""
        await self._position_manager.reconcile_positions()

    def _append_trade(self, trade: Dict[str, Any]):
        """Append a trade to history and prune old entries to bound memory usage."""
        self._trade_history_version += 1
        self.trade_history.append(trade)
        if len(self.trade_history) > settings.MAX_TRADES_IN_MEMORY:
            # Accumulate realized P&L of pruned trades so the equity curve
            # in _compute_performance_metrics remains accurate.
            pruned = self.trade_history[:-settings.MAX_TRADES_IN_MEMORY]
            for t in pruned:
                if t.get("side") == "sell":
                    self._realized_pnl_offset += t.get("realized_pnl", 0.0)
            # Keep only the most recent trades
            self.trade_history = self.trade_history[-settings.MAX_TRADES_IN_MEMORY:]

    def _load_state(self):
        """Load current symbols, positions, trade history, and initial balance from SQLite."""
        self._state_persistence.load_state()

    async def _save_state(self, force: bool = False):
        """Persist current symbols, positions, and trade history to SQLite."""
        await self._state_persistence.save_state(force=force)

    async def run(self):
        """Main event‑driven loop using WebSocket ticker updates."""
        logger.info("Trading engine initializing...")
        await self._initialize_clients()
        logger.info("Trading engine started.")
        # Start background tasks
        self._background_tasks: list = []
        self._background_tasks.append(asyncio.create_task(self._refresh_news_cache()))
        self._background_tasks.append(asyncio.create_task(self._refresh_current_symbols_news_fast()))
        self._background_tasks.append(asyncio.create_task(self._download_market_data_loop()))
        self._background_tasks.append(asyncio.create_task(self._download_all_assets_data_loop()))
        # yfinance cache cleanup removed – it caused OperationalError('unable to open database file')
        # when deleting the cache directory while yfinance was actively using it from other tasks.
        # yfinance manages its own cache internally and does not need external cleanup.
        self._background_tasks.append(asyncio.create_task(self._download_all_news_loop()))
        self._background_tasks.append(asyncio.create_task(self._risk_management_loop()))
        self._background_tasks.append(asyncio.create_task(self._periodic_reconcile()))
        self._background_tasks.append(asyncio.create_task(self._periodic_reevaluate()))
        self._background_tasks.append(asyncio.create_task(self._periodic_pause_check()))
        self._background_tasks.append(asyncio.create_task(self._periodic_pause_resume_check()))
        self._background_tasks.append(asyncio.create_task(self._periodic_full_market_breadth()))
        self._background_tasks.append(asyncio.create_task(self._periodic_market_condition_check()))
        self._background_tasks.append(asyncio.create_task(self._check_pending_entries()))
        self._background_tasks.append(asyncio.create_task(self._cleanup_orphaned_orders()))
        self._background_tasks.append(asyncio.create_task(self._process_queued_orders()))
        self._background_tasks.append(asyncio.create_task(self._monitor_entry_signals_loop()))
        self._background_tasks.append(asyncio.create_task(self._market_clock_monitor()))
        self._background_tasks.append(asyncio.create_task(self._refresh_all_quotes_loop()))
        self._background_tasks.append(asyncio.create_task(self._refresh_ticker_discovery_loop()))

        while self._running:
            try:
                await asyncio.sleep(settings.ENGINE_LOOP_INTERVAL_SECONDS)

                # Process any symbol whose evaluation interval has elapsed
                now = time.time()

                # Compute active period status once per loop iteration
                clock = await self._get_clock()
                is_active_period = False
                if clock and clock.is_open:
                    now_rome = clock.timestamp
                    market_open_dt = now_rome.replace(hour=settings.MARKET_OPEN_HOUR, minute=settings.MARKET_OPEN_MINUTE, second=0, microsecond=0)
                    minutes_since_open = (now_rome - market_open_dt).total_seconds() / 60
                    if 0 <= minutes_since_open < settings.MARKET_OPEN_ACTIVE_MINUTES:
                        is_active_period = True
                    if not is_active_period:
                        market_close_dt = now_rome.replace(hour=settings.MARKET_CLOSE_HOUR, minute=settings.MARKET_CLOSE_MINUTE, second=0, microsecond=0)
                        minutes_to_close = (market_close_dt - now_rome).total_seconds() / 60
                        if 0 < minutes_to_close < settings.MARKET_CLOSE_ACTIVE_MINUTES:
                            is_active_period = True

                # --- Dynamic evaluation interval based on timeframe and market conditions ---

                # Fetch full market breadth once per loop iteration
                full_market_breadth = None
                try:
                    full_breadth_raw = await asyncio.to_thread(self.redis.get, "market:breadth:full")
                    if full_breadth_raw:
                        full_market_breadth = json.loads(full_breadth_raw)
                except Exception:
                    pass

                is_highly_active = False
                if full_market_breadth:
                    pos_pct = full_market_breadth.get("positive_pct", 50)
                    if pos_pct > 80 or pos_pct < 20:
                        is_highly_active = True

                # Check for significant news sentiment shifts for all tracked symbols
                # We store this in a dict so we can use it per-symbol
                symbol_has_significant_news: Dict[str, bool] = {}
                if settings.NEWS_ENABLED:
                    for entry in self.current_symbols:
                        symbol = entry["symbol"]
                        try:
                            agg = await self._get_cached_sentiment(symbol)
                            if agg:
                                base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
                                prev_key = f"sentiment:reeval_baseline:{base_symbol}"
                                prev_raw = await asyncio.to_thread(self.redis.get, prev_key)
                                current_compound = agg.get("avg_compound", 0)
                                if prev_raw:
                                    prev_compound = float(prev_raw)
                                    if abs(current_compound - prev_compound) > 0.3:
                                        symbol_has_significant_news[symbol] = True
                        except Exception:
                            continue

                for symbol_entry in self.current_symbols:
                    symbol = symbol_entry["symbol"]
                    tf = symbol_entry.get("timeframe", "1d")

                    # Base interval proportional to timeframe
                    if tf in ("1h",):
                        tf_base_interval = 900  # 15 minutes
                    elif tf in ("1d",):
                        tf_base_interval = 1800 # 30 minutes
                    elif tf in ("1w",):
                        tf_base_interval = 3600 # 1 hour
                    elif tf in ("1M",):
                        tf_base_interval = 86400 # 1 day
                    elif tf in ("3M",):
                        tf_base_interval = 172800 # 2 days
                    elif tf in ("6M", "1Y"):
                        tf_base_interval = 604800 # 1 week
                    else:
                        tf_base_interval = 3600 # 1 hour default

                    # Adjust based on market conditions
                    if is_active_period or is_highly_active:
                        # Active market: evaluate more frequently (halve the interval, min 15m)
                        tf_base_interval = max(900, tf_base_interval // 2)

                    if symbol_has_significant_news.get(symbol, False):
                        # Significant news for this specific ticker: evaluate quickly (min 15m)
                        tf_base_interval = max(900, min(tf_base_interval, 1800))

                    if full_market_breadth and 40 <= full_market_breadth.get("positive_pct", 50) <= 60 and not is_active_period:
                        # Quiet market: evaluate less frequently (double the interval, max 8h)
                        # Use max() so the cap never reduces the interval below the base for long timeframes
                        tf_base_interval = max(tf_base_interval, min(tf_base_interval * 2, 28800))

                    # Use the dynamically computed tf_base_interval, but allow LLM to override per-symbol
                    default_interval = tf_base_interval
                    interval = self._strategy_intervals.get(symbol, default_interval)
                    last_eval = self._last_strategy_eval.get(symbol, 0)
                    if now - last_eval >= interval:
                        # Check if trading is paused (skip BUY signals)
                        paused = await asyncio.to_thread(self.redis.get, "trading:paused")
                        trading_paused = paused is not None and paused == "1"
                        try:
                            await asyncio.wait_for(
                                self._process_symbol(symbol_entry, trading_paused=trading_paused),
                                timeout=settings.LLM_TIMEOUT + 10  # slightly longer than the LLM timeout
                            )
                            self._last_strategy_eval[symbol] = now
                        except asyncio.TimeoutError:
                            logger.error(f"Timeout processing symbol {symbol_entry['symbol']} – skipping. Will retry on next loop iteration.")
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"⏱️ Processing timeout for {symbol_entry['symbol']} – skipping this cycle.",
                                    summary={"symbol": symbol_entry["symbol"], "action": "SKIP", "reason": "Processing timeout"}
                                )
                        await asyncio.sleep(0.2)   # small delay to reduce contention

                # Save state periodically (every 5 minutes) when dirty
                if now - self._last_state_save > 300 and self._state_dirty:
                    await self._save_state()
                    self._last_state_save = now
                    self._state_dirty = False

            except Exception as e:
                logger.error(f"Engine loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _reevaluate_symbols(self, force: bool = False):
        """Use LLM to select which symbols to trade."""
        async with self._symbol_reeval_lock:
            return await self._reevaluate_symbols_impl(force=force)

    async def _reevaluate_symbols_impl(self, force: bool = False):
        _cooldown_result = await self._symbol_reevaluator.check_cooldown_and_reset(force)
        if _cooldown_result is None:
            return
        is_user_forced, is_market_condition_trigger, now = _cooldown_result
        _assets_result = await self._symbol_reevaluator.fetch_and_filter_candidate_assets(now)
        if _assets_result is None:
            return
        available_pairs, btp_pairs, etf_pairs, old_symbols, last_key = _assets_result
        _quotes_result = await self._symbol_reevaluator.fetch_quotes_and_sort(
            available_pairs, btp_pairs, etf_pairs, now, last_key
        )
        if _quotes_result is None:
            return
        balance, base_balance, per_symbol_budget, tickers, sample_pairs, stock_pairs = _quotes_result
        logger.info("Re-evaluation step 6/12: Batch-fetching news sentiment for %d symbols...", len(sample_pairs))
        news_sentiment, sentiment_trend, market_trend = await self._symbol_reevaluator.fetch_news_sentiment_and_trends(
            sample_pairs, tickers
        )


        # Fetch OHLCV from database only for ALL candidate pairs.
        # Background tasks (_download_all_assets_data_loop) keep the DB populated.
        # This avoids blocking reevaluation on slow API calls.
        sorted_by_vol = sample_pairs
        logger.info("Re-evaluation step 7/12: Fetching OHLCV from DB for %d symbols...", len(sorted_by_vol))
        ohlcv_data, available_timeframes_by_symbol = await self._symbol_reevaluator.fetch_ohlcv_from_db(sorted_by_vol)

        logger.info("Re-evaluation step 8/12: Batch-fetching indicators for %d symbols...", len(sorted_by_vol))
        symbol_indicators, symbol_trend_scores = await self._symbol_reevaluator.fetch_indicators_and_trend_scores(
            sorted_by_vol, sample_pairs
        )

        # Use asset info for minimum order size constraints
        market_limits = await self._symbol_reevaluator.compute_market_limits(sample_pairs, tickers)

        # effective_max_symbols is set by the LLM's max_stocks field.
        # Do NOT zero it out based on per-symbol budget calculations.
        # The LLM decides how many symbols to trade and how to allocate capital dynamically.
        self.effective_max_symbols = self.max_symbols

        # Recompute per-symbol budget with the effective max
        per_symbol_budget = base_balance / self.effective_max_symbols

        # No hardcoded minimum viable trade amount gate.
        # The LLM decides position sizes dynamically based on all available parameters.
        # The only hard limits are exchange minimums (min_order_size, min_order_cost),
        # which are checked at order execution time.
        min_viable_amount = 0.0

        logger.info("Re-evaluation step 10/12: Computing correlation matrix and performance metrics...")
        correlation_matrix = await self._symbol_reevaluator.get_or_compute_correlation_matrix(
            ohlcv_data, sorted_by_vol
        )

        perf = await asyncio.to_thread(self._compute_performance_metrics)
        trade_pattern_analysis = await asyncio.to_thread(self._compute_trade_pattern_analysis)

        # --- Composite opportunity score and shortlist building ---
        composite_scores, shortlist = self._symbol_reevaluator.compute_composite_scores_and_shortlist(
            sample_pairs, symbol_trend_scores, news_sentiment, trade_pattern_analysis, etf_pairs, btp_pairs
        )
        sorted_by_composite = sorted(sample_pairs, key=lambda s: composite_scores.get(s, 0), reverse=True)
        sample_pairs = shortlist
        logger.info(f"LLM candidate list: {len(sample_pairs)} symbols (will be evaluated in chunks)")

        symbol_events, session_info, market_breadth, full_market_breadth, vix = await self._symbol_reevaluator.fetch_shortlist_context(
            sample_pairs, tickers, market_trend
        )

        trading_paused_bool, symbol_tenure, symbol_max_tenure, auto_resume_note, ohlcv_summary, effective_temp = await self._symbol_reevaluator.prepare_reeval_prompt_context(
            now=now,
            sample_pairs=sample_pairs,
            ohlcv_data=ohlcv_data,
            sentiment_trend=sentiment_trend,
            market_breadth=market_breadth,
        )

        # --- Chunked LLM evaluation ---
        chunk_results = await self._symbol_reevaluator.evaluate_llm_chunks(
            sample_pairs=sample_pairs,
            tickers=tickers,
            ohlcv_summary=ohlcv_summary,
            symbol_indicators=symbol_indicators,
            market_limits=market_limits,
            symbol_events=symbol_events,
            symbol_trend_scores=symbol_trend_scores,
            sentiment_trend=sentiment_trend,
            correlation_matrix=correlation_matrix,
            ohlcv_data=ohlcv_data,
            perf=perf,
            market_trend=market_trend,
            session_info=session_info,
            market_breadth=market_breadth,
            trading_paused_bool=trading_paused_bool,
            symbol_tenure=symbol_tenure,
            symbol_max_tenure=symbol_max_tenure,
            vix=vix,
            trade_pattern_analysis=trade_pattern_analysis,
            min_viable_amount=min_viable_amount,
            base_balance=base_balance,
            per_symbol_budget=per_symbol_budget,
            auto_resume_note=auto_resume_note,
            effective_temp=effective_temp,
        )

        # --- Final selection call ---
        response, llm_provider, llm_model = await self._symbol_reevaluator.run_final_selection_llm_call(
            chunk_results=chunk_results,
            sample_pairs=sample_pairs,
            base_balance=base_balance,
            per_symbol_budget=per_symbol_budget,
            perf=perf,
            market_trend=market_trend,
            session_info=session_info,
            market_breadth=market_breadth,
            full_market_breadth=full_market_breadth,
            trading_paused_bool=trading_paused_bool,
            symbol_tenure=symbol_tenure,
            symbol_max_tenure=symbol_max_tenure,
            trade_pattern_analysis=trade_pattern_analysis,
            vix=vix,
            min_viable_amount=min_viable_amount,
            market_limits=market_limits,
            available_timeframes_by_symbol=available_timeframes_by_symbol,
            auto_resume_note=auto_resume_note,
            effective_temp=effective_temp,
        )

        logger.info("Re-evaluation: LLM response received (%d chars), parsing...", len(response) if response else 0)
        if response:
            # Truncate long responses to avoid flooding logs with HTML error pages
            if len(response) > 500:
                logger.info("LLM stock selection raw response (truncated): %.500s...", response)
            else:
                logger.info("LLM stock selection raw response: %s", response)
            # Warn if the response looks like HTML (common when the LLM endpoint returns an error page)
            if response.lstrip().startswith('<'):
                logger.warning(
                    "LLM stock selection response appears to be HTML (length %d). "
                    "The LLM endpoint may be returning an error page.",
                    len(response)
                )
        else:
            logger.info("LLM stock selection returned empty response")
            if self.notifier:
                await self.notifier.send_notification(
                    "⚠️ LLM symbol selection failed after all retries. " +
                    ("Keeping previously tracked symbols." if old_symbols else "Will attempt fallback selection."),
                    summary={
                        "action": "ERROR",
                        "reason": "LLM symbol selection failed after all retries",
                        "model_type": "mind",
                    }
                )

        # Initialize variables that may be used later even if LLM fails
        parsed = {}
        pause_trading = None
        pause_reason = ""
        pause_duration = None
        new_symbols: List[Dict[str, str]] = []

        # Retry JSON parsing if the first attempt fails
        if response is not None:
            try:
                json.loads(response)  # validate
            except json.JSONDecodeError:
                response, llm_provider, llm_model = await self._symbol_reevaluator.retry_json_parsing(
                    response=response,
                    effective_temp=effective_temp,
                )

        if response is not None:
            try:
                parsed = json.loads(response)
                llm_max_stocks = parsed.get("max_stocks") if isinstance(parsed, dict) else None
                deduped = self._symbol_reevaluator.parse_and_validate_symbols(
                    response=response,
                    sample_pairs=sample_pairs,
                    ohlcv_data=ohlcv_data,
                )
                if deduped is None:
                    deduped = []

                # --- Extract pause_trading early so MIN_SYMBOLS enforcement can respect it ---
                pause_trading = parsed.get("pause_trading")
                if isinstance(pause_trading, str):
                    low = pause_trading.strip().lower()
                    if low in ("true", "1"):
                        pause_trading = True
                    elif low in ("false", "0"):
                        pause_trading = False
                    else:
                        pause_trading = None

                # Use the LLM's chosen number of symbols to update effective_max_symbols
                if llm_max_stocks is not None and isinstance(llm_max_stocks, int) and 0 <= llm_max_stocks <= self.max_symbols:
                    self.effective_max_symbols = llm_max_stocks
                else:
                    # Fallback: use the length of the deduped list, capped at the engine's max
                    self.effective_max_symbols = min(len(deduped), self.effective_max_symbols)

                self._symbol_reevaluator.enforce_min_symbols(
                    deduped=deduped,
                    pause_trading=pause_trading,
                    sorted_by_composite=sorted_by_composite,
                    market_limits=market_limits,
                    base_balance=base_balance,
                )

                # --- Store LLM-decided parameters to Redis ---
                await self._symbol_reevaluator.store_llm_decided_parameters(parsed)

                pause_trading, pause_reason, pause_duration = await self._symbol_reevaluator.handle_pause_resume_and_risk_multiplier(
                    parsed=parsed,
                    pause_trading=pause_trading,
                    trading_paused_bool=trading_paused_bool,
                )

                self._symbol_reevaluator.update_current_symbols(
                    deduped=deduped,
                    old_symbols=old_symbols,
                )

            except json.JSONDecodeError:
                logger.error("Failed to parse symbol selection response.")

        await self._symbol_reevaluator.apply_fallback_selection(
            sample_pairs=sample_pairs,
            composite_scores=composite_scores,
            tickers=tickers,
            market_limits=market_limits,
            base_balance=base_balance,
            old_symbols=old_symbols,
            pause_trading=pause_trading,
        )

        # Ensure all open positions remain in current_symbols so they continue to be managed by the LLM strategy
        for symbol, pos in self.positions.items():
            if not any(entry["symbol"] == symbol for entry in self.current_symbols):
                tf = pos.get("timeframe") or (settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h")
                self.current_symbols.append({"symbol": symbol, "timeframe": tf})
                logger.info(f"Keeping {symbol} in current_symbols due to open position (timeframe={tf})")

        # If trading is paused, we still keep all symbols so the LLM can generate signals
        # (which will be notified but not executed in paper mode).
        # The LLM may have just set pause_trading = true, so re-read Redis.
        paused_now = await asyncio.to_thread(self.redis.get, "trading:paused")
        if paused_now and paused_now == "1" and not force:
            logger.info("Trading is paused. Keeping all symbols for signal generation.")

        # Update symbol tenure tracking
        now_ts = time.time()
        new_symbol_set = {entry["symbol"] for entry in self.current_symbols}
        for sym in new_symbol_set:
            if sym not in self._symbol_first_seen:
                self._symbol_first_seen[sym] = now_ts
        for sym in list(self._symbol_first_seen.keys()):
            if sym not in new_symbol_set:
                del self._symbol_first_seen[sym]

        # Trigger immediate backfill for newly selected symbols
        old_symbol_set = {entry["symbol"] for entry in old_symbols}
        for entry in self.current_symbols:
            if entry["symbol"] not in old_symbol_set:
                sym = entry["symbol"]
                tf = entry["timeframe"]
                logger.info(f"Triggering immediate backfill for newly selected symbol {sym} ({tf})")
                asyncio.create_task(self._backfill_new_symbol(sym, tf))

        # Also trigger immediate news fetch for newly selected symbols
        if settings.NEWS_ENABLED:
            for entry in deduped:
                sym = entry["symbol"]
                logger.info(f"Triggering immediate news fetch for newly selected symbol {sym}")
                asyncio.create_task(self._fetch_and_store_news_for_symbol(sym))

        await self._symbol_reevaluator.build_and_send_reeval_notification(
            base_balance=base_balance,
            per_symbol_budget=per_symbol_budget,
            pause_trading=pause_trading,
            pause_reason=pause_reason,
            pause_duration=pause_duration,
            trading_paused_bool=trading_paused_bool,
            force=force,
            is_user_forced=is_user_forced,
            parsed=parsed,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

        # If no symbols were selected, shorten the re‑evaluation interval to retry sooner.
        if not self.current_symbols:
            self._symbol_reevaluation_interval = max(self._symbol_reevaluation_interval, settings.MIN_SYMBOL_REEVALUATION_INTERVAL)
            logger.info(f"No symbols selected – next re‑evaluation in {self._symbol_reevaluation_interval}s")
        # else: keep the current interval (may have been set by LLM via
        # stock_revaluation_interval_seconds, or the default SYMBOL_REEVALUATION_INTERVAL)

        # Set the triggered cooldown key AFTER a successful market-condition-triggered
        # re-evaluation to prevent the market condition monitor from firing again too soon.
        # This must be set at the END, not at the trigger point, otherwise the re-evaluation
        # itself would see the cooldown as active and skip itself.
        if is_market_condition_trigger:
            await asyncio.to_thread(self.redis.set, "trading:last_triggered_reeval", str(time.time()))
            await asyncio.to_thread(self.redis.expire, "trading:last_triggered_reeval", 7200)

        # --- Cleanup stale entries from engine state dicts and caches ---
        self._symbol_reevaluator.cleanup_stale_state_entries()

        self._state_dirty = True
        logger.info("Re-evaluation complete: %d symbols selected.", len(self.current_symbols))
        await asyncio.to_thread(self.redis.set, last_key, now)

    async def _periodic_pause_resume_check(self):
        """Periodically ask the LLM whether to resume trading when paused."""
        await asyncio.sleep(60)  # initial delay
        while self._running:
            try:
                if await self._is_market_open():
                    await self._check_pause_resume_decision()
            except Exception as e:
                logger.error(f"Pause/resume check error: {e}", exc_info=True)
            await asyncio.sleep(1800)  # every 30 minutes

    async def _check_pause_resume_decision(self):
        """When trading is paused, ask the LLM whether to resume (lightweight)."""
        await self._signal_processor.check_pause_resume_decision()

    async def _compute_multi_tf_indicators(
        self, symbol: str, ohlcv_data: Dict[str, List[List]], assigned_tf: str
    ) -> Dict[str, Any]:
        """Batch-fetch indicators from DB and extract assigned-timeframe values."""
        return await self._signal_processor.compute_multi_tf_indicators(
            symbol, ohlcv_data, assigned_tf
        )

    async def _gather_prompt_context(
        self,
        symbol: str,
        assigned_tf: str,
        tf_seconds: int,
        ticker: Dict[str, Any],
        base_balance: float,
        ohlcv_data: Dict[str, List[List]],
        multi_tf_indicators: Dict[str, Dict[str, Any]],
        multi_tf_raw_candles: Dict[str, List[List]],
        atr: Optional[float],
        rsi: Optional[float],
        macd: Optional[float],
        macd_signal: Optional[float],
        macd_hist: Optional[float],
        bb_upper: Optional[float],
        bb_middle: Optional[float],
        bb_lower: Optional[float],
        ema_9: Optional[float],
        ema_21: Optional[float],
        adx: Optional[float],
        plus_di: Optional[float],
        minus_di: Optional[float],
    ) -> Dict[str, Any]:
        """Gather all additional market context needed for the strategy prompt."""
        return await self._signal_processor.gather_prompt_context(
            symbol=symbol,
            assigned_tf=assigned_tf,
            tf_seconds=tf_seconds,
            ticker=ticker,
            base_balance=base_balance,
            ohlcv_data=ohlcv_data,
            multi_tf_indicators=multi_tf_indicators,
            multi_tf_raw_candles=multi_tf_raw_candles,
            atr=atr,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            macd_hist=macd_hist,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            ema_9=ema_9,
            ema_21=ema_21,
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
        )

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

    async def _run_backtest_and_final_decision(
        self,
        symbol: str,
        assigned_tf: str,
        tf_seconds: int,
        current_price: float,
        atr: Optional[float],
        historical_ohlcv: Optional[List[List]],
        raw_candles: Optional[List[List]],
        base_balance: float,
        is_btp: bool,
        trading_paused: bool,
        strategy_model_type: str,
        effective_temp: float,
        preliminary_signal: Signal,
        display_symbol: str,
        ticker: Dict[str, Any],
    ) -> Tuple[Signal, str, Optional[str], Optional[str]]:
        """Run backtests and the Step 2 LLM call to produce the final signal."""
        return await self._backtest_manager.run_backtest_and_final_decision(
            symbol=symbol,
            assigned_tf=assigned_tf,
            tf_seconds=tf_seconds,
            current_price=current_price,
            atr=atr,
            historical_ohlcv=historical_ohlcv,
            raw_candles=raw_candles,
            base_balance=base_balance,
            is_btp=is_btp,
            trading_paused=trading_paused,
            strategy_model_type=strategy_model_type,
            effective_temp=effective_temp,
            preliminary_signal=preliminary_signal,
            display_symbol=display_symbol,
            ticker=ticker,
        )

    async def _fetch_symbol_market_data(self, symbol: str, assigned_tf: str) -> Optional[Dict[str, Any]]:
        """Fetch all raw market data for a symbol: ticker, fundamentals, balance, OHLCV, and multi-TF indicators."""
        return await self._signal_processor.fetch_symbol_market_data(symbol, assigned_tf)

    async def _process_symbol(self, symbol_entry: Dict[str, str], trading_paused: bool = False):
        """Fetch market data, get LLM strategy, validate, and execute."""
        await self._signal_processor.process_symbol(symbol_entry, trading_paused)

    async def get_profit_summary(self) -> Dict[str, Any]:
        """Return profit/loss summary including queued orders."""
        return await self._position_manager.get_profit_summary()

    def get_recent_signals(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent LLM signals for the web dashboard."""
        return self.recent_signals[-limit:]

    async def get_open_trades(self) -> List[Dict[str, Any]]:
        """Return current open positions as trade-like dicts with unrealized P&L."""
        open_trades = []
        pos_tickers = await asyncio.to_thread(self._get_all_position_tickers_sync)
        for symbol, pos in self.positions.items():
            # Skip invalid positions (zero amount or zero price)
            if pos.get("amount", 0) <= 0 or pos.get("price", 0) <= 0:
                continue
            try:
                t = pos_tickers.get(symbol)
                current_price = t['last'] if t and t.get('last') else pos['price']
            except Exception:
                current_price = pos['price']  # fallback to entry price

            entry_price = pos['price']
            amount = pos['amount']
            cost_basis = pos.get('cost_basis', amount * entry_price)
            unrealized_pnl = (current_price - entry_price) * amount
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

            # Try to get fee from the most recent buy trade for this symbol
            fee = {}
            for t in reversed(self.trade_history):
                if t['symbol'] == symbol and t['side'] == 'buy':
                    fee = t.get('fee', {})
                    break

            open_trades.append({
                'symbol': symbol,
                'timeframe': pos.get('timeframe'),
                'side': 'buy',
                'amount': amount,
                'price': entry_price,
                'timestamp': pos.get('timestamp', 0),
                'fee': fee,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'cost_basis': cost_basis,
            })
        return open_trades

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Return performance summary grouped by symbol and timeframe from trade_history table."""
        return await asyncio.to_thread(get_performance)

    async def get_pause_status(self) -> Dict[str, Any]:
        """Return the current trading pause status, reason, remaining duration, and a formatted countdown."""
        paused_raw = await asyncio.to_thread(self.redis.get, "trading:paused")
        is_paused = paused_raw is not None and paused_raw == "1"

        reason_raw = await asyncio.to_thread(self.redis.get, "trading:pause_reason")
        reason = reason_raw.decode() if isinstance(reason_raw, bytes) else (reason_raw or "")

        source_raw = await asyncio.to_thread(self.redis.get, "trading:pause_source")
        source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")

        remaining_seconds = None
        countdown_str = None

        if is_paused:
            market_time_str = None
            if source == "market_closed":
                # Fetch the current clock to compute a live countdown and current market time
                clock = await self._get_clock()

                market_time_str = None
                if clock is not None:
                    market_time_str = clock.timestamp.astimezone(ZoneInfo(settings.MARKET_TIMEZONE)).strftime('%H:%M %d/%m/%Y')
                    if not clock.is_open:
                        now_utc = datetime.now(timezone.utc)
                        next_open = clock.next_open
                        remaining = (next_open - now_utc).total_seconds()
                        if remaining > 0:
                            remaining_seconds = int(remaining)
                            if remaining_seconds > 3600:
                                hours = remaining_seconds // 3600
                                minutes = (remaining_seconds % 3600) // 60
                                countdown_str = f"{hours}h {minutes}m"
                            elif remaining_seconds > 60:
                                minutes = remaining_seconds // 60
                                seconds = remaining_seconds % 60
                                countdown_str = f"{minutes}m {seconds}s"
                            else:
                                countdown_str = f"{remaining_seconds}s"
                else:
                    # Fallback to the stored next_open if the clock is unavailable
                    next_open_raw = await asyncio.to_thread(self.redis.get, "trading:market_next_open")
                    if next_open_raw:
                        try:
                            next_open_str = next_open_raw.decode() if isinstance(next_open_raw, bytes) else next_open_raw
                            next_open_dt = datetime.fromisoformat(next_open_str)
                            now_utc = datetime.now(timezone.utc)
                            remaining = (next_open_dt - now_utc).total_seconds()
                            if remaining > 0:
                                remaining_seconds = int(remaining)
                                if remaining_seconds > 3600:
                                    hours = remaining_seconds // 3600
                                    minutes = (remaining_seconds % 3600) // 60
                                    countdown_str = f"{hours}h {minutes}m"
                                elif remaining_seconds > 60:
                                    minutes = remaining_seconds // 60
                                    seconds = remaining_seconds % 60
                                    countdown_str = f"{minutes}m {seconds}s"
                                else:
                                    countdown_str = f"{remaining_seconds}s"
                                reason = "Market closed"
                        except Exception:
                            pass
            else:
                # LLM or manual pause with duration
                pause_start_raw = await asyncio.to_thread(self.redis.get, "trading:pause_start")
                pause_duration_raw = await asyncio.to_thread(self.redis.get, "trading:pause_duration")
                if pause_start_raw and pause_duration_raw:
                    try:
                        pause_start = float(pause_start_raw)
                        pause_duration = int(pause_duration_raw)
                        elapsed = time.time() - pause_start
                        remaining = pause_duration - elapsed
                        if remaining > 0:
                            remaining_seconds = int(remaining)
                            if remaining_seconds > 3600:
                                hours = remaining_seconds // 3600
                                minutes = (remaining_seconds % 3600) // 60
                                countdown_str = f"{hours}h {minutes}m"
                            elif remaining_seconds > 60:
                                minutes = remaining_seconds // 60
                                seconds = remaining_seconds % 60
                                countdown_str = f"{minutes}m {seconds}s"
                            else:
                                countdown_str = f"{remaining_seconds}s"
                    except (ValueError, TypeError):
                        pass

        return {
            "is_paused": is_paused,
            "reason": reason,
            "remaining_seconds": remaining_seconds,
            "countdown_str": countdown_str,
            "source": source,
            "market_time_str": market_time_str if is_paused and source == "market_closed" else None,
        }

    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Return current risk/exposure metrics."""
        return await self._position_manager.get_risk_metrics()

    async def sell_all_positions(self):
        """Sell all open positions at market price."""
        if not await self._is_market_open():
            logger.warning("Sell all positions skipped: market is closed.")
            if self.notifier:
                await self.notifier.send_notification(
                    "⏸️ Sell all skipped: market is currently closed.",
                    summary={"action": "SKIP", "reason": "Market closed"}
                )
            return
        for symbol in list(self.positions.keys()):
            await self._execute_signal(
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Manual sell all"),
                exit_reason="manual_sell_all"
            )

    async def sell_position(self, symbol: str):
        """Sell a specific open position at market price."""
        if not await self._is_market_open():
            logger.warning(f"Sell position {symbol} skipped: market is closed.")
            if self.notifier:
                await self.notifier.send_notification(
                    f"⏸️ Sell {symbol} skipped: market is currently closed.",
                    summary={"symbol": symbol, "action": "SKIP", "reason": "Market closed"}
                )
            return
        if symbol in self.positions:
            await self._execute_signal(
                symbol,
                Signal(action="SELL", confidence=1.0, reasoning="Manual sell"),
                exit_reason="manual_sell"
            )
        else:
            logger.warning(f"No open position for {symbol}")

    async def log_manual_trade(self, ticker: str, side: str, quantity: float, money_spent: float, fee: float) -> dict:
        """Log a manually executed trade in notify mode. Persists to DB and updates positions."""
        symbol = f"{ticker}/{self.base_currency}"
        base = ticker
        quote = self.base_currency
        price = money_spent / quantity if quantity > 0 else 0.0
        cost = money_spent
        timestamp = int(time.time() * 1000)

        # If fee is not provided (0.0), calculate it using the Intesa Sanpaolo Investo logic
        if fee == 0.0:
            from src.exchanges.fees import calculate_transaction_costs
            costs = calculate_transaction_costs(side.upper(), price, quantity, symbol=ticker)
            fee = costs["total_costs"]

        trade = {
            "id": f"manual_{timestamp}",
            "symbol": symbol,
            "side": side,
            "amount": quantity,
            "price": price,
            "cost": cost,
            "fee": {"cost": fee, "currency": quote},
            "timestamp": timestamp,
            "note": "manual",
            "status": "closed",
            "strategy_type": "manual",
        }

        if side == "buy":
            cost_basis = cost + fee
            net_base = quantity
            if symbol in self.positions:
                old_pos = self.positions[symbol]
                old_cost_basis = old_pos.get("cost_basis", old_pos["amount"] * old_pos["price"])
                old_net_base = old_pos.get("net_base", old_pos["amount"])
                new_cost_basis = old_cost_basis + cost_basis
                new_net_base = old_net_base + net_base
                new_price = new_cost_basis / new_net_base if new_net_base > 0 else 0.0
                self.positions[symbol]["amount"] = new_net_base
                self.positions[symbol]["price"] = new_price
                self.positions[symbol]["cost_basis"] = new_cost_basis
                self.positions[symbol]["net_base"] = new_net_base
            else:
                entry_price = cost_basis / net_base if net_base > 0 else price
                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": "buy",
                    "amount": net_base,
                    "price": entry_price,
                    "timestamp": timestamp,
                    "stop_loss": None,
                    "take_profit": None,
                    "cost_basis": cost_basis,
                    "net_base": net_base,
                    "timeframe": None,
                    "entry_order_type": "manual",
                    "buy_confidence": 1.0,
                    "buy_reasoning": "Manual trade",
                }
            self._balance_cache = None

            # Update virtual cash balance
            self.trader._balances[quote] = self.trader._balances.get(quote, 0.0) - cost_basis
            self.trader._balances[base] = self.trader._balances.get(base, 0.0) + net_base
            self.trader._balances_dirty = True
            await asyncio.to_thread(self.trader._save_balances)
        elif side == "sell":
            pos = self.positions.get(symbol)
            if pos:
                cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                net_quote = cost - fee
                realized_pnl = net_quote - cost_basis
                trade["realized_pnl"] = realized_pnl
                trade["cost_basis"] = cost_basis
                trade["exit_reason"] = "manual_sell"
                if "timestamp" in pos:
                    trade["hold_time_seconds"] = (timestamp - pos["timestamp"]) / 1000.0
                self.positions.pop(symbol, None)
                self._balance_cache = None

                # Update virtual cash balance
                self.trader._balances[base] = self.trader._balances.get(base, 0.0) - quantity
                self.trader._balances[quote] = self.trader._balances.get(quote, 0.0) + net_quote
                self.trader._balances_dirty = True
                await asyncio.to_thread(self.trader._save_balances)
            else:
                # Check if the user actually holds enough of the base asset
                current_base_balance = self.trader._balances.get(base, 0.0)
                if current_base_balance < quantity:
                    logger.warning(
                        f"Manual sell rejected for {symbol}: insufficient {base} balance "
                        f"(have {current_base_balance}, need {quantity})"
                    )
                    return {
                        "status": "error",
                        "error": f"Insufficient {base} balance: have {current_base_balance}, need {quantity}",
                    }

                trade["realized_pnl"] = 0.0
                trade["cost_basis"] = 0.0
                trade["exit_reason"] = "manual_sell"

                # Update virtual cash balance even if position wasn't tracked
                self.trader._balances[base] = current_base_balance - quantity
                self.trader._balances[quote] = self.trader._balances.get(quote, 0.0) + (cost - fee)
                self.trader._balances_dirty = True
                await asyncio.to_thread(self.trader._save_balances)

        self._append_trade(trade)
        await asyncio.to_thread(insert_trade, trade)
        await self._save_state(force=True)
        self._portfolio_exposure_cache = None
        logger.info(f"Manual trade logged: {side} {quantity} {symbol} @ {price:.4f}")
        return {"status": "ok", "trade": trade}

    async def _record_position_pnl_snapshots(self):
        """Record P&L snapshots for all open positions to the database."""
        await self._risk_manager.record_position_pnl_snapshots()

    async def _process_native_exit_fill(
        self,
        symbol: str,
        order_id: str,
        order_obj: Any,
        pos: Dict[str, Any],
        exit_reason: str,
    ):
        """Process a filled native exit order (stop-loss or take-profit) inline."""
        await self._order_executor.process_native_exit_fill(symbol, order_id, order_obj, pos, exit_reason)

    async def _check_risk_management(self):
        """Check open positions and close if stop-loss, take-profit, or trailing stop is hit."""
        await self._risk_manager.check_risk_management()

    async def _execute_signal(self, symbol: str, signal, timeframe: str = None, exit_reason: str = None, atr: Optional[float] = None):
        """Execute a BUY or SELL signal."""
        # --- Format symbol for notifications ---
        stock_name = await self._get_stock_name(symbol)
        tf = timeframe or (self.positions.get(symbol, {}).get("timeframe") if symbol in self.positions else None)
        display_symbol = self._format_symbol_display(symbol, stock_name, tf)

        # --- Notify mode: do not execute any orders, only send notifications ---
        if settings.TRADING_MODE == "notify":
            logger.info(f"Notify mode: skipping order execution for {signal.action} {symbol}.")
            return

        # --- Paper mode + Paused: do not execute automated BUY orders, only send notifications ---
        # Manual overrides (exit_reason starts with "manual") are still allowed.
        # Automated SELL orders are allowed if the market is open (to manage open positions).
        paused = await asyncio.to_thread(self.redis.get, "trading:paused")
        if settings.TRADING_MODE == "paper" and paused and not (exit_reason and exit_reason.startswith("manual")):
            is_market_open = await self._is_market_open()
            if signal.action == "SELL" and is_market_open:
                logger.info(f"Paper mode + Paused: allowing automated SELL for risk management {symbol}.")
                # Fall through to execute the SELL order
            else:
                logger.info(f"Paper mode + Paused: skipping automated order execution for {signal.action} {symbol}.")
                return

        # Prevent executing new signals if an order is already queued for this symbol
        # (unless it's a manual override)
        async with self._queued_orders_lock:
            has_queued = any(q['symbol'] == symbol for q in self.queued_orders)
        if has_queued and not (exit_reason and exit_reason.startswith("manual")):
            logger.info(f"Skipping {signal.action} for {symbol}: order already queued.")
            return

        # If this is a manual sell, cancel any queued SELL order for this symbol to avoid duplicate sells
        if exit_reason and exit_reason.startswith("manual") and signal.action == "SELL":
            async with self._queued_orders_lock:
                self.queued_orders = [q for q in self.queued_orders if not (q['symbol'] == symbol and q['side'] == 'sell')]

        # In live mode, only execute during regular market hours (manual overrides are allowed anytime)
        if not await self._is_market_open() and not (exit_reason and exit_reason.startswith("manual")):
            logger.info(f"Skipping {signal.action} for {symbol}: market closed (live mode).")
            if self.notifier:
                await self.notifier.send_notification(
                    f"⏸️ Skipping {signal.action} for {display_symbol}: market closed.",
                    summary={"symbol": symbol, "action": "SKIP", "reason": "Market closed"}
                )
            return

        parts = symbol.split("/")
        if len(parts) != 2:
            logger.error(f"Invalid symbol format: {symbol}")
            return
        base, quote = parts
        _exec_base = symbol.split("/")[0]
        _exec_is_btp = is_btp_isin(symbol)
        balance = await self._get_cached_balance()

        if signal.action == "BUY":
            await self._order_executor.execute_buy(
                symbol=symbol,
                display_symbol=display_symbol,
                signal=signal,
                timeframe=timeframe,
                exit_reason=exit_reason,
                atr=atr,
                balance=balance,
            )
        elif signal.action == "SELL":
            await self._order_executor.execute_sell(
                symbol=symbol,
                display_symbol=display_symbol,
                signal=signal,
                timeframe=timeframe,
                exit_reason=exit_reason,
                atr=atr,
                balance=balance,
            )

    async def _should_skip_llm_eval(
        self,
        symbol: str,
        current_price: float,
        atr: Optional[float],
        rsi: Optional[float],
        macd_hist: Optional[float],
        atr_percentile: Optional[float],
        market_regime: str,
        sentiment_trend_val: Optional[float],
        timeframe_seconds: float,
        has_position: bool,
        is_critical: bool,
    ) -> bool:
        """Return True if it's safe to skip the LLM call and just HOLD."""
        return await self._signal_processor.should_skip_llm_eval(
            symbol=symbol,
            current_price=current_price,
            atr=atr,
            rsi=rsi,
            macd_hist=macd_hist,
            atr_percentile=atr_percentile,
            market_regime=market_regime,
            sentiment_trend_val=sentiment_trend_val,
            timeframe_seconds=timeframe_seconds,
            has_position=has_position,
            is_critical=is_critical,
        )

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
                    last_forced = self._force_eval_time.get(symbol, 0)
                    if time.time() - last_forced < cooldown:
                        continue

                    if await self._detect_entry_signal(symbol, tf):
                        logger.info(f"Entry signal detected for {symbol}, forcing LLM evaluation.")
                        self._force_eval[symbol] = True
                        self._force_eval_time[symbol] = time.time()
                        # Clear last evaluation timestamp so the main loop picks it up immediately
                        self._last_strategy_eval.pop(symbol, None)
            except Exception as e:
                logger.error(f"Entry signal monitor error: {e}", exc_info=True)
            await asyncio.sleep(settings.ENTRY_SIGNAL_CHECK_INTERVAL_SECONDS)

    async def _detect_entry_signal(self, symbol: str, timeframe: str) -> bool:
        """Return True if a favourable entry condition is detected for the symbol.
        Uses recent OHLCV data from the database and compares with previous state."""
        return await self._signal_processor.detect_entry_signal(symbol, timeframe)

    async def _check_pending_entries(self):
        """Periodically check pending entry conditions and execute if met."""
        await asyncio.sleep(10)  # short initial delay
        while self._running:
            try:
                now = time.time()
                for symbol in list(self._pending_entries.keys()):
                    await self._signal_processor.process_pending_entry(symbol, now)
            except Exception as e:
                logger.error(f"Error checking pending entries: {e}", exc_info=True)
            await asyncio.sleep(60)  # check every 60 seconds (medium/long-term)

    async def _check_entry_condition_once(
        self, symbol: str, condition: Dict[str, Any], timeframe: str
    ) -> bool:
        """Check a single entry condition immediately. Return True if met."""
        return await self._signal_processor.check_entry_condition_once(symbol, condition, timeframe)

    async def _execute_delayed_entry(self, symbol: str, signal, timeframe: str, delay_seconds: float):
        """Execute a delayed entry after waiting for the specified duration."""
        logger.info(f"Delayed entry: waiting {delay_seconds}s for {symbol}")
        await asyncio.sleep(delay_seconds)
        if not self._running:
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
        await self._execute_signal(
            symbol,
            signal,
            timeframe=timeframe,
            atr=None,
        )

    def _choose_model_tier(
        self,
        atr: Optional[float] = None,
        atr_percentile: Optional[float] = None,
        rsi: Optional[float] = None,
        macd: Optional[float] = None,
        macd_signal: Optional[float] = None,
        macd_hist: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_middle: Optional[float] = None,
        bb_lower: Optional[float] = None,
        ema_9: Optional[float] = None,
        ema_21: Optional[float] = None,
        stochastic_k: Optional[float] = None,
        adx: Optional[float] = None,
        plus_di: Optional[float] = None,
        minus_di: Optional[float] = None,
        mfi: Optional[float] = None,
        cci: Optional[float] = None,
        williams_r: Optional[float] = None,
        ichimoku: Optional[Dict[str, Any]] = None,
        market_regime: str = "",
        market_breadth: Optional[Dict[str, Any]] = None,
        full_market_breadth: Optional[Dict[str, Any]] = None,
        sentiment_trend_val: Optional[float] = None,
        volume_trend: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        drawdown_pct: Optional[float] = None,
        portfolio_exposure_pct: Optional[float] = None,
        portfolio_stop_risk_pct: Optional[float] = None,
        is_critical: bool = False,
        trading_paused: bool = False,
        symbol_event: Optional[Dict[str, Any]] = None,
        fundamentals: Optional[Dict[str, Any]] = None,
        consecutive_losses: int = 0,
        current_price: Optional[float] = None,
    ) -> str:
        """Return "mind" or "actuator" based on market complexity."""
        return self._signal_processor.choose_model_tier(
            atr=atr, atr_percentile=atr_percentile, rsi=rsi, macd=macd,
            macd_signal=macd_signal, macd_hist=macd_hist,
            bb_upper=bb_upper, bb_middle=bb_middle, bb_lower=bb_lower,
            ema_9=ema_9, ema_21=ema_21, stochastic_k=stochastic_k,
            adx=adx, plus_di=plus_di, minus_di=minus_di,
            mfi=mfi, cci=cci, williams_r=williams_r, ichimoku=ichimoku,
            market_regime=market_regime, market_breadth=market_breadth,
            full_market_breadth=full_market_breadth,
            sentiment_trend_val=sentiment_trend_val, volume_trend=volume_trend,
            unrealized_pnl=unrealized_pnl, drawdown_pct=drawdown_pct,
            portfolio_exposure_pct=portfolio_exposure_pct,
            portfolio_stop_risk_pct=portfolio_stop_risk_pct,
            is_critical=is_critical, trading_paused=trading_paused,
            symbol_event=symbol_event, fundamentals=fundamentals,
            consecutive_losses=consecutive_losses, current_price=current_price,
        )

    def _compute_prompt_complexity(
        self,
        num_candidates: int = 0,
        volatility_percentile: Optional[float] = None,
        rsi: Optional[float] = None,
        macd: Optional[float] = None,
        macd_signal: Optional[float] = None,
        macd_hist: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_middle: Optional[float] = None,
        bb_lower: Optional[float] = None,
        ema_9: Optional[float] = None,
        ema_21: Optional[float] = None,
        stochastic_k: Optional[float] = None,
        adx: Optional[float] = None,
        plus_di: Optional[float] = None,
        minus_di: Optional[float] = None,
        mfi: Optional[float] = None,
        cci: Optional[float] = None,
        williams_r: Optional[float] = None,
        ichimoku: Optional[Dict[str, Any]] = None,
        market_breadth: Optional[Dict[str, Any]] = None,
        full_market_breadth: Optional[Dict[str, Any]] = None,
        sentiment_trend_magnitude: Optional[float] = None,
        volume_trend: Optional[float] = None,
        market_regime: str = "",
        unrealized_pnl: Optional[float] = None,
        drawdown_pct: Optional[float] = None,
        portfolio_exposure_pct: Optional[float] = None,
        portfolio_stop_risk_pct: Optional[float] = None,
        is_critical: bool = False,
        trading_paused: bool = False,
        symbol_event: Optional[Dict[str, Any]] = None,
        fundamentals: Optional[Dict[str, Any]] = None,
        consecutive_losses: int = 0,
        current_price: Optional[float] = None,
        fear_greed: Optional[Dict[str, Any]] = None,
        conflicting_signals: bool = False,
    ) -> float:
        """Return a complexity score between 0.0 (simple) and 1.0 (very complex)."""
        return self._signal_processor.compute_prompt_complexity(
            num_candidates=num_candidates,
            volatility_percentile=volatility_percentile,
            rsi=rsi, macd=macd, macd_signal=macd_signal, macd_hist=macd_hist,
            bb_upper=bb_upper, bb_middle=bb_middle, bb_lower=bb_lower,
            ema_9=ema_9, ema_21=ema_21, stochastic_k=stochastic_k,
            adx=adx, plus_di=plus_di, minus_di=minus_di,
            mfi=mfi, cci=cci, williams_r=williams_r, ichimoku=ichimoku,
            market_breadth=market_breadth, full_market_breadth=full_market_breadth,
            sentiment_trend_magnitude=sentiment_trend_magnitude,
            volume_trend=volume_trend, market_regime=market_regime,
            unrealized_pnl=unrealized_pnl, drawdown_pct=drawdown_pct,
            portfolio_exposure_pct=portfolio_exposure_pct,
            portfolio_stop_risk_pct=portfolio_stop_risk_pct,
            is_critical=is_critical, trading_paused=trading_paused,
            symbol_event=symbol_event, fundamentals=fundamentals,
            consecutive_losses=consecutive_losses, current_price=current_price,
            fear_greed=fear_greed, conflicting_signals=conflicting_signals,
        )

    def _get_effective_temperature(self, model_type: str, complexity: float) -> float:
        """Return the temperature to use for a given model_type and complexity score (0-1)."""
        from src.config.settings import Settings
        raw = settings.LLM_MIND_TEMPERATURE if model_type == "mind" else settings.LLM_ACTUATOR_TEMPERATURE
        parsed = Settings.parse_temperature_range(raw)
        if parsed is None:
            # Fall back to global LLM_TEMPERATURE
            return settings.LLM_TEMPERATURE
        lo, hi = parsed
        if lo == hi:
            return lo
        # Map complexity 0→lo, 1→hi
        return lo + (hi - lo) * complexity

    def _update_last_eval_snapshot(self, symbol: str, price: float, rsi: Optional[float], macd_hist: Optional[float]):
        self._last_eval_snapshot[symbol] = {
            "timestamp": time.time(),
            "price": price,
            "rsi": rsi,
            "macd_hist": macd_hist,
        }

    def _create_fallback_hold_signal(
        self, symbol: str, reason: str, strategy_model_type: str = "actuator"
    ) -> Signal:
        """Create a default HOLD signal when LLM calls fail after all retries.

        This ensures the bot continues to function even if the LLM is temporarily
        unavailable or producing invalid output.
        """
        return Signal(
            action="HOLD",
            confidence=0.0,
            reasoning=f"Fallback: {reason}",
            strategy_type="fallback",
            strategy_params={},
            model_type=strategy_model_type,
            llm_provider="fallback",
            llm_model="default_hold",
        )

    def _parse_analysis_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the Step 1a analysis LLM response into a dict.

        Expected fields: action, confidence, reasoning, strategy_direction.
        Returns None if parsing fails.
        """
        try:
            parsed = json.loads(response)
            if not isinstance(parsed, dict):
                return None
            action = parsed.get("action", "").upper()
            if action not in ("BUY", "SELL", "HOLD"):
                return None
            return {
                "action": action,
                "confidence": float(parsed.get("confidence", 0.0)),
                "reasoning": parsed.get("reasoning", ""),
                "strategy_direction": parsed.get("strategy_direction", ""),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

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

    def _compute_exit_order_prices(
        self,
        entry_price: float,
        signal: Signal,
        atr: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        return self._order_executor.compute_exit_order_prices(entry_price, signal, atr)

    async def _cancel_exit_orders(self, symbol: str):
        """Cancel any native stop-loss and take-profit orders for a symbol."""
        await self._order_executor.cancel_exit_orders(symbol)

    async def _place_exit_orders(
        self,
        symbol: str,
        signal: Signal,
        exit_prices: Dict[str, Optional[float]],
        timeframe: Optional[str] = None,
    ):
        """Place native stop-loss and take-profit orders for a position."""
        await self._order_executor.place_exit_orders(symbol, signal, exit_prices, timeframe)

    async def _replace_native_stop_order(
        self,
        symbol: str,
        pos: Dict[str, Any],
        old_stop_price: float,
        new_stop_price: float,
    ):
        """Cancel the existing native stop order and place a new one with the updated stop price."""
        await self._order_executor.replace_native_stop_order(symbol, pos, old_stop_price, new_stop_price)

    async def _execute_partial_sell(
        self,
        symbol: str,
        sell_amount: float,
        level_label: str,
        exit_reason: str,
        ticker: Optional[Dict[str, Any]] = None,
        atr: Optional[float] = None,
        current_price: float = 0.0,
        cleanup_callback=None,
        extra_summary: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Execute a partial sell (used by partial take-profit single and multi-level)."""
        return await self._order_executor.execute_partial_sell(
            symbol, sell_amount, level_label, exit_reason, ticker, atr, current_price, cleanup_callback, extra_summary
        )

    async def _execute_partial_tp_single(
        self, symbol: str, current_price: float, atr: Optional[float], ticker: Dict[str, Any]
    ) -> None:
        """Execute a single partial take-profit sell for a position."""
        await self._order_executor.execute_partial_tp_single(symbol, current_price, atr, ticker)

    async def _execute_partial_tp_level(
        self, symbol: str, level_index: int, current_price: float, atr: Optional[float], ticker: Dict[str, Any]
    ) -> None:
        """Execute a partial take-profit sell for a specific level."""
        await self._order_executor.execute_partial_tp_level(symbol, level_index, current_price, atr, ticker)

    async def _sweep_dust(self, symbol: str):
        """Sell any remaining dust balance of a symbol after a partial sell."""
        await self._order_executor.sweep_dust(symbol)

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
                    await self._order_executor.process_single_queued_order(queued)
            except Exception as e:
                logger.error(f"Error processing queued orders: {e}", exc_info=True)
            await asyncio.sleep(15)  # check every 15 seconds for faster fill detection

    async def _handle_queued_buy_fill(self, trade_dict: Dict[str, Any], queued: Dict[str, Any]):
        """Process a queued BUY limit order that has filled in the simulator."""
        await self._order_executor.handle_queued_buy_fill(trade_dict, queued)

    async def _handle_queued_sell_fill(self, trade_dict: Dict[str, Any], queued: Dict[str, Any], partial: bool = False):
        """Process a queued SELL limit order that has filled in the simulator."""
        await self._order_executor.handle_queued_sell_fill(trade_dict, queued, partial)

    async def _cleanup_orphaned_orders(self):
        """Periodically cancel any open orders that are older than 10 minutes,
        but never cancel orders that are still being tracked as queued."""
        await asyncio.sleep(120)  # initial delay
        # --- Notify mode: no orphaned order cleanup ---
        if settings.TRADING_MODE == "notify":
            return
        while self._running:
            try:
                await self._order_executor.cleanup_orphaned_orders()
            except Exception as e:
                logger.error(f"Orphaned order cleanup error: {e}", exc_info=True)
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
        clock = await self._get_clock()
        if clock is None:
            # Fallback: if clock unavailable, assume closed to be safe
            return False
        return clock.is_open

    def _is_regular_hours(self) -> bool:
        """Return True if the market is currently open."""
        if self._clock_cache is None:
            return False
        return self._clock_cache.is_open

    def _get_session_info(self) -> dict:
        """Return current Italian market session info using Europe/Rome timezone."""
        now_rome = datetime.now(timezone.utc).astimezone(ZoneInfo(settings.MARKET_TIMEZONE))
        weekday = now_rome.weekday()
        hour = now_rome.hour + now_rome.minute / 60.0
        open_hour = settings.MARKET_OPEN_HOUR + settings.MARKET_OPEN_MINUTE / 60.0
        close_hour = settings.MARKET_CLOSE_HOUR + settings.MARKET_CLOSE_MINUTE / 60.0
        if weekday >= 5:
            session = "Closed (weekend)"
        elif hour < open_hour:
            session = "Closed (pre-open)"
        elif hour < close_hour:
            session = "Regular"
        else:
            session = "Closed (after hours)"
        return {"utc_hour": datetime.now(timezone.utc).hour, "session": session}

    def _default_limit_price(
        self, symbol: str, action: str, ticker: Dict[str, Any], atr: Optional[float] = None
    ) -> Optional[float]:
        """Compute a default aggressive limit price for extended‑hours trading.

        The buffer is scaled by ATR when available: buffer_pct = atr / price,
        clamped to [0.001, 0.02] (0.1%–2%). Falls back to 0.2% when ATR is
        unavailable.
        """
        last = ticker.get('last')
        if not last or last <= 0:
            return None

        # Compute buffer percentage from ATR, clamped to [0.1%, 2%]
        if atr is not None and atr > 0:
            buffer_pct = max(0.001, min(atr / last, 0.02))
        else:
            buffer_pct = 0.002  # fallback 0.2%

        if action == "BUY":
            limit = last * (1 + buffer_pct)
        elif action == "SELL":
            limit = last * (1 - buffer_pct)
        else:
            return None

        if last >= 1.0:
            limit = round(limit, 2)
        else:
            limit = round(limit, 4)
        return limit

    async def _remove_symbol_if_paused(self, symbol: str):
        """Clear pending entries for a symbol. Symbols are kept in current_symbols even when paused
        so the bot continues to generate and notify signals."""
        # Always clear any pending entry for this symbol
        self._pending_entries.pop(symbol, None)
        self._state_dirty = True

    def _get_tickers_for_symbols_sync(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch latest quotes for a list of symbols synchronously, batching missing ones.

        Uses get_quotes_cached (Redis/DB only, no network calls) to avoid
        blocking the default asyncio thread pool with slow yfinance requests.
        """
        tickers: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        for sym in symbols:
            missing.append(sym.split("/")[0])
        if missing:
            try:
                raw = get_quotes_cached(missing)
                for sym in symbols:
                    base = sym.split("/")[0]
                    if base in raw:
                        tickers[sym] = raw[base]
            except Exception as e:
                logger.warning(f"Sync batch quote fetch failed: {e}")
        return tickers

    async def _run_backtest_variant(
        self,
        symbol: str,
        variant_params: Dict[str, Any],
        preliminary_signal: Signal,
        atr: Optional[float],
        current_price: float,
        tf_secs: int,
        assigned_tf: str,
        historical_ohlcv: Optional[List[List]],
        raw_candles: Optional[List[List]],
        base_balance: float,
        is_btp: bool,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Run a single backtest variant with database persistence and concurrency limiting."""
        return await self._backtest_manager._run_backtest_variant(
            symbol=symbol,
            variant_params=variant_params,
            preliminary_signal=preliminary_signal,
            atr=atr,
            current_price=current_price,
            tf_secs=tf_secs,
            assigned_tf=assigned_tf,
            historical_ohlcv=historical_ohlcv,
            raw_candles=raw_candles,
            base_balance=base_balance,
            is_btp=is_btp,
        )

    async def _run_backtest_from_signal(
        self,
        symbol: str,
        signal: Signal,
        atr: Optional[float],
        current_price: float,
        tf_secs: int,
        assigned_tf: str,
        historical_ohlcv: Optional[List[List]],
        raw_candles: Optional[List[List]],
        base_balance: float,
        is_btp: bool,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Run a backtest using the parameters from a signal. Returns (stats, summary)."""
        return await self._backtest_manager._run_backtest_from_signal(
            symbol=symbol,
            signal=signal,
            atr=atr,
            current_price=current_price,
            tf_secs=tf_secs,
            assigned_tf=assigned_tf,
            historical_ohlcv=historical_ohlcv,
            raw_candles=raw_candles,
            base_balance=base_balance,
            is_btp=is_btp,
        )

    async def _prepare_simulation_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch all necessary data and build the strategy prompt for simulation."""
        return await self._signal_processor.prepare_simulation_data(symbol)

    async def simulate_backtest(self, symbol: str) -> Dict[str, Any]:
        """Simulate Step 1a (analysis), Step 1b (variants), and run backtest without executing trades."""
        data = await self._prepare_simulation_data(symbol)
        if "error" in data:
            return data

        model_type = data.get("model_type", "mind")
        temperature = data.get("temperature", 0.2)
        market_hash = data.get("market_hash")

        _analysis, step1b_response, preliminary_signal, error = await self._signal_processor.run_simulation_step1(symbol, data)
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
        data = await self._prepare_simulation_data(symbol)
        if "error" in data:
            return data

        model_type = data.get("model_type", "mind")
        temperature = data.get("temperature", 0.2)
        market_hash = data.get("market_hash")

        _analysis, step1b_response, preliminary_signal, error = await self._signal_processor.run_simulation_step1(symbol, data)
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
