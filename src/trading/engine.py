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
    SYSTEM_PROMPT,
    build_stock_selection_prompt,
    build_final_selection_prompt,
    build_analysis_prompt,
    build_backtest_variants_prompt,
    build_final_decision_prompt,
    _format_news_for_prompt,
    compact_prompt,
    get_cached_news_summary,
)

COMPACTED_SYSTEM_PROMPT = compact_prompt(SYSTEM_PROMPT)
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
from src.database import load_trading_state, save_trading_state, insert_trade, get_performance, store_news_articles, get_aggregate_sentiment_from_db, get_aggregate_sentiment_for_symbols, get_news_for_symbol, get_ohlcv, get_latest_ohlcv_timestamp, insert_ohlcv_batch, save_paper_balances, load_paper_balances, cleanup_old_ohlcv, save_indicators, get_indicators, get_indicators_for_symbols, get_ohlcv_summary_for_symbols, get_all_trades, get_latest_close_prices, insert_position_pnl_snapshot, cleanup_old_position_pnl, save_backtest_result, get_recent_backtest_result, get_backtest_results_for_symbol, cleanup_old_backtest_results

logger = logging.getLogger(__name__)

SYMBOL_REEVALUATION_INTERVAL = 14400  # seconds (4 hours) – medium/long-term
DEFAULT_STRATEGY_INTERVAL = 3600   # fallback when no timeframe or no symbols (1 hour)
MIN_SYMBOL_REEVALUATION_INTERVAL = 3600  # 1 hour – prevents rapid toggling
MAX_TRADES_IN_MEMORY = 1000  # cap in-memory trade history to prevent unbounded growth


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
        self._symbol_reevaluation_interval = SYMBOL_REEVALUATION_INTERVAL
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
        self._ensure_cost_basis()

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
                loop = asyncio.get_running_loop()
                # Download timeframes in the exact order defined in OHLCV_TIMEFRAMES (longest to shortest)
                for tf in settings.OHLCV_TIMEFRAMES:
                    try:
                        inserted = await self._backfill_ohlcv(pair, tf, start_ms, now_ms, force=True, quiet=True)
                        if inserted > 0:
                            await self._fill_gaps(pair, tf)
                        # Always compute indicators for force download
                        db_candles = await loop.run_in_executor(self._download_executor, get_ohlcv, pair, tf, 200)
                        if db_candles:
                            raw_candles = [[c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in db_candles]
                            await self._compute_and_store_indicators(pair, tf, raw_candles)
                    except Exception as e:
                        logger.warning(f"Force download failed for {pair} {tf}: {e}")

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
        """Return asset info (min order size, name, etc.), cached for 1 hour."""
        base = symbol.split("/")[0] if "/" in symbol else symbol
        now = time.time()
        if base in self._asset_cache and (now - self._asset_cache_time.get(base, 0)) < 3600:
            return self._asset_cache[base]

        # Return a dummy AssetInfo with permissive defaults
        asset = AssetInfo(name=base, min_order_size=0.0, fractionable=True)
        self._asset_cache[base] = asset
        self._asset_cache_time[base] = now
        return asset

    async def _get_all_position_tickers(self) -> Dict[str, Dict[str, Any]]:
        """Fetch tickers for all open positions, batching missing ones into a single API call."""
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

    async def _get_clock(self, ttl: float = 300.0) -> Optional[ClockInfo]:
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
                            return   # skip the original duration logic
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
                # Respect a cooldown so we don't re-evaluate too frequently
                last_triggered_key = "trading:last_triggered_reeval"
                last_triggered = await asyncio.to_thread(self.redis.get, last_triggered_key)
                if last_triggered:
                    elapsed = time.time() - float(last_triggered)
                    if elapsed < settings.TRIGGERED_REEVALUATION_COOLDOWN:
                        await asyncio.sleep(300)
                        continue

                should_trigger = False

                # 1. Significant news sentiment shift on tracked symbols
                if settings.NEWS_ENABLED and self.current_symbols:
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
                                        logger.info(f"Significant sentiment shift for {symbol}, triggering re-evaluation")
                                        should_trigger = True
                                        break
                                # Update the re-evaluation baseline regardless of whether we triggered
                                if current_compound is not None:
                                    await asyncio.to_thread(
                                        self.redis.setex, prev_key, 3600, str(current_compound)
                                    )
                        except Exception:
                            continue

                # 2. Unusually active market (many stocks with >5% daily change)
                if not should_trigger:
                    try:
                        plain_assets = await self._get_tradable_assets()
                        sample_pairs = [f"{sym}/{self.base_currency}" for sym in plain_assets[:50]]
                        plain_sample = [s.split("/")[0] for s in sample_pairs]
                        quotes = await self._get_quotes_batched(plain_sample, timeout_per_chunk=45.0)
                        large_movers = sum(
                            1 for q in quotes.values()
                            if abs(q.get("percentage") or 0) > 5.0
                        )
                        if large_movers >= 5:
                            logger.info(f"Unusually active market: {large_movers} stocks with >5% daily change, triggering re-evaluation")
                            should_trigger = True
                    except Exception:
                        pass

                # 3. Extreme indicator values or BB squeeze breakout on tracked symbols
                if not should_trigger:
                    for entry in self.current_symbols:
                        symbol = entry["symbol"]
                        tf = entry["timeframe"]
                        try:
                            # Fetch pre-computed indicators from DB
                            ind = await asyncio.to_thread(get_indicators, symbol, tf)
                            if not ind:
                                continue

                            # Extreme RSI — use LLM-configured thresholds (fallback to 20/80)
                            rsi_oversold = 20.0
                            rsi_overbought = 80.0
                            try:
                                raw = await asyncio.to_thread(self.redis.get, "trading:skip_eval_rsi_oversold")
                                if raw:
                                    rsi_oversold = float(raw)
                                raw = await asyncio.to_thread(self.redis.get, "trading:skip_eval_rsi_overbought")
                                if raw:
                                    rsi_overbought = float(raw)
                            except Exception:
                                pass
                            rsi = ind.get("rsi")
                            if rsi is not None and (rsi < rsi_oversold or rsi > rsi_overbought):
                                logger.info(f"Extreme RSI ({rsi:.1f}) for {symbol}, triggering re-evaluation")
                                should_trigger = True
                                break

                            # Bollinger Band squeeze breakout
                            bb_upper = ind.get("bb_upper")
                            bb_lower = ind.get("bb_lower")
                            bb_middle = ind.get("bb_middle")
                            if bb_upper and bb_lower and bb_middle and bb_middle > 0:
                                bb_width = (bb_upper - bb_lower) / bb_middle
                                bb_squeeze_width = 0.02
                                try:
                                    raw = await asyncio.to_thread(self.redis.get, "trading:regime_bb_squeeze_width")
                                    if raw:
                                        bb_squeeze_width = float(raw)
                                except Exception:
                                    pass
                                if bb_width < bb_squeeze_width:
                                    db_candles = await asyncio.to_thread(get_ohlcv, symbol, tf, limit=1)
                                    if db_candles:
                                        current_close = db_candles[-1]["close"]
                                        if current_close > bb_upper or current_close < bb_lower:
                                            logger.info(f"Bollinger Band squeeze breakout for {symbol}, triggering re-evaluation")
                                            should_trigger = True
                                            break
                        except Exception:
                            continue

                if should_trigger:
                    logger.info("Market condition trigger fired – forcing symbol re-evaluation")
                    # Invalidate correlation matrix cache due to significant market changes
                    await asyncio.to_thread(self.redis.delete, "reeval:correlation_matrix")
                    if self.notifier:
                        await self.notifier.send_notification(
                            "🔄 Market conditions changed – triggering immediate symbol re-evaluation.",
                            summary={"action": "INFO", "reason": "Market condition triggered re-evaluation"}
                        )
                    self._force_reeval = True
                    self._reeval_trigger.set()
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
            await asyncio.sleep(300)  # check every 5 minutes (medium/long-term)

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
        if not candles or len(candles) < 2:
            return
        try:
            async with self._indicator_semaphore:
                loop = asyncio.get_running_loop()
                ind = await loop.run_in_executor(self._download_executor, compute_all_indicators, candles)
            if ind:
                latest_ts = candles[-1][0]
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._db_executor, save_indicators, symbol, timeframe, latest_ts, ind)
                logger.debug(f"Indicators computed and stored for {symbol} {timeframe}")
        except Exception as e:
            logger.warning(f"Failed to compute/store indicators for {symbol} {timeframe}: {e}")

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
        """Return the human-readable company name for a symbol, cached in Redis.

        Uses yfinance to fetch the name.
        """
        base = symbol.split("/")[0] if "/" in symbol else symbol

        if re.match(r'^IT[A-Z0-9]{10}$', base):
            # It's a BTP bond, try to get the name from the BTP cache (includes DB-merged BTPs)
            try:
                btp_bonds = await self._get_btp_bonds()
                for b in btp_bonds:
                    if b["isin"] == base:
                        name = b.get("name") or base
                        if name and name != b["isin"]:
                            return name
            except Exception:
                pass
            # Fallback: try DB directly
            try:
                from src.database import get_symbol_name_from_db
                db_name = await asyncio.to_thread(get_symbol_name_from_db, base)
                if db_name:
                    return db_name
            except Exception:
                pass

            # If we got a name from the BTP cache, save it to DB for future lookups
            try:
                btp_bonds = await self._get_btp_bonds()
                for b in btp_bonds:
                    if b["isin"] == base:
                        name = b.get("name") or base
                        if name and name != b["isin"]:
                            from src.database import save_discovered_symbol
                            await asyncio.to_thread(
                                save_discovered_symbol, base, base, "btp", name,
                                country="italy"
                            )
                            return name
            except Exception:
                pass
            return base

        # Check Redis cache first
        cache_key = f"stock_name:{base}"
        try:
            cached = await asyncio.to_thread(self.redis.get, cache_key)
            if cached:
                return cached.decode() if isinstance(cached, bytes) else cached
        except Exception:
            pass

        # Check discovered_symbols table (works even when yf circuit is open)
        try:
            from src.database import get_symbol_name_from_db
            db_name = await asyncio.to_thread(get_symbol_name_from_db, base)
            if db_name:
                try:
                    await asyncio.to_thread(self.redis.setex, cache_key, 7 * 24 * 3600, db_name)
                except Exception:
                    pass
                return db_name
        except Exception:
            pass

        if _check_yf_circuit():
            return base

        try:
            def _fetch_yf_name():
                import yfinance as yf
                ticker = yf.Ticker(base, session=_get_yf_session())
                info = ticker.info
                return info.get("longName") or info.get("shortName") or base
            name = await asyncio.to_thread(_fetch_yf_name)
        except Exception:
            name = base

        # If yfinance returned a name, save it to the DB for future use
        if name and name != base:
            try:
                from src.database import save_discovered_symbol
                db_base = base
                suffix = settings.TICKER_SUFFIX
                if suffix and db_base.endswith(suffix):
                    db_base = db_base[:-len(suffix)]
                save_discovered_symbol(db_base, None, None, name, country=None)
            except Exception:
                pass

        # Cache for 7 days (names rarely change)
        try:
            await asyncio.to_thread(self.redis.setex, cache_key, 7 * 24 * 3600, name)
        except Exception:
            pass
        return name

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

        # Find and fill gaps larger than 1.5x the expected interval
        gaps_found = 0
        gaps_filled = 0
        max_gaps_per_cycle = 5  # Limit gap fills per cycle to avoid rate limits
        for i in range(len(timestamps) - 1):
            if gaps_filled >= max_gaps_per_cycle:
                break
            gap = timestamps[i + 1] - timestamps[i]
            if gap > interval_ms * 1.5:
                gaps_found += 1
                gap_start = timestamps[i] + interval_ms
                gap_end = timestamps[i + 1] - interval_ms
                if gap_end > gap_start:
                    logger.debug(f"Gap detected for {symbol} {timeframe}: {gap_start} → {gap_end} (size {gap}ms)")
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
        try:
            inserted = await self._backfill_ohlcv(symbol, timeframe, start_ms, now_ms)
            await self._fill_gaps(symbol, timeframe)
            # Compute and store indicators after backfill
            loop = asyncio.get_running_loop()
            db_candles = await loop.run_in_executor(self._download_executor, get_ohlcv, symbol, timeframe, 200)
            if db_candles:
                raw_candles = [[c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in db_candles]
                await self._compute_and_store_indicators(symbol, timeframe, raw_candles)
        except Exception as e:
            logger.error(f"Initial backfill failed for {symbol} {timeframe}: {e}")
        logger.debug(f"Immediate backfill complete for {symbol} ({timeframe})")

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
                        loop = asyncio.get_running_loop()
                        try:
                            inserted = await self._backfill_ohlcv(symbol, tf, start_ms, now_ms)
                            if inserted > 0:
                                await self._fill_gaps(symbol, tf)
                                # Compute and store indicators after candles are downloaded
                                db_candles = await loop.run_in_executor(self._download_executor, get_ohlcv, symbol, tf, 200)
                                if db_candles:
                                    raw_candles = [[c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in db_candles]
                                    await self._compute_and_store_indicators(symbol, tf, raw_candles)
                        except Exception as e:
                            logger.warning(f"Market data download failed for {symbol} {tf}: {e}")

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

                # Download concurrently, respecting rate limits via _exchange_semaphore
                random.shuffle(all_pairs)

                async def _download_symbol_data(pair: str):
                    loop = asyncio.get_running_loop()
                    for tf in settings.OHLCV_TIMEFRAMES:
                        try:
                            inserted = await self._backfill_ohlcv(pair, tf, start_ms, now_ms, quiet=True)
                            if inserted > 0:
                                await self._fill_gaps(pair, tf)
                                # Compute and store indicators after candles are downloaded
                                db_candles = await loop.run_in_executor(self._download_executor, get_ohlcv, pair, tf, 200)
                                if db_candles:
                                    raw_candles = [[c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in db_candles]
                                    await self._compute_and_store_indicators(pair, tf, raw_candles)
                        except Exception as e:
                            logger.warning(f"Full download failed for {pair} {tf}: {e}")

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

    def _ensure_cost_basis(self):
        """If positions lack cost_basis, compute it from amount and price (backward compat)."""
        for sym, pos in self.positions.items():
            if 'cost_basis' not in pos or 'net_base' not in pos:
                # Assume no fees for old positions; cost_basis = amount * price
                pos['cost_basis'] = pos['amount'] * pos['price']
                pos['net_base'] = pos['amount']

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
        # Cache check: if no new trades have been added since the last computation,
        # return the cached result to avoid expensive iteration over trade_history.
        now = time.time()
        if (
            self._trade_history_version == self._perf_cache_trade_count
            and self._perf_cache is not None
            and (now - self._perf_cache_time) < 60  # 60-second TTL for unrealized P&L freshness
        ):
            return self._perf_cache

        # Snapshot trade_history to avoid concurrent modification during iteration
        trades_snapshot = list(self.trade_history)

        from collections import defaultdict

        now = time.time()
        symbol_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0.0, "last_trade_ts": 0})
        strategy_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0.0})
        symbol_stop_losses = defaultdict(int)
        symbol_hold_times = defaultdict(list)

        for trade in trades_snapshot:
            if trade.get("side") != "sell":
                continue
            symbol = trade["symbol"]
            pnl = trade.get("realized_pnl", 0.0)
            strategy = trade.get("strategy_type", "unknown")
            exit_reason = trade.get("exit_reason", "")
            if exit_reason == "stop_loss":
                symbol_stop_losses[symbol] += 1
            hold_time = trade.get("hold_time_seconds")
            if hold_time is not None:
                symbol_hold_times[symbol].append(hold_time)

            symbol_stats[symbol]["trades"] += 1
            symbol_stats[symbol]["total_pnl"] += pnl
            if pnl > 0:
                symbol_stats[symbol]["wins"] += 1
            symbol_stats[symbol]["last_trade_ts"] = max(symbol_stats[symbol]["last_trade_ts"], trade.get("timestamp", 0) / 1000.0)

            strategy_stats[strategy]["trades"] += 1
            strategy_stats[strategy]["total_pnl"] += pnl
            if pnl > 0:
                strategy_stats[strategy]["wins"] += 1

        # Convert to dicts with win rates
        symbol_perf = {}
        for sym, s in symbol_stats.items():
            win_rate = s["wins"] / s["trades"] if s["trades"] > 0 else 0.0
            avg_pnl = s["total_pnl"] / s["trades"] if s["trades"] > 0 else 0.0
            symbol_perf[sym] = {
                "trades": s["trades"],
                "win_rate": round(win_rate, 3),
                "avg_pnl": round(avg_pnl, 4),
                "total_pnl": round(s["total_pnl"], 4),
                "last_trade_seconds_ago": round(now - s["last_trade_ts"]) if s["last_trade_ts"] else None,
                "stop_loss_hits": symbol_stop_losses.get(sym, 0),
                "avg_hold_time_seconds": round(sum(symbol_hold_times[sym]) / len(symbol_hold_times[sym]), 1) if symbol_hold_times.get(sym) else None,
            }

        strategy_perf = {}
        for st, s in strategy_stats.items():
            win_rate = s["wins"] / s["trades"] if s["trades"] > 0 else 0.0
            avg_pnl = s["total_pnl"] / s["trades"] if s["trades"] > 0 else 0.0
            strategy_perf[st] = {
                "trades": s["trades"],
                "win_rate": round(win_rate, 3),
                "avg_pnl": round(avg_pnl, 4),
                "total_pnl": round(s["total_pnl"], 4),
            }

        # Overall equity curve summary: last 10 trades P&L trend
        recent_sells = [t for t in trades_snapshot if t.get("side") == "sell"][-10:]
        recent_pnl = [t.get("realized_pnl", 0.0) for t in recent_sells]
        total_recent_pnl = sum(recent_pnl)
        trend = "up" if total_recent_pnl > 0 else "down" if total_recent_pnl < 0 else "flat"

        # Compute drawdown based on total equity (initial balance + cumulative realized P&L)
        equity_series = []
        running_equity = self.initial_balance + self._realized_pnl_offset
        for trade in trades_snapshot:
            if trade.get("side") == "sell":
                running_equity += trade.get("realized_pnl", 0.0)
            equity_series.append(running_equity)
        peak = max(equity_series) if equity_series else self.initial_balance

        # Current equity includes unrealized P&L from open positions
        current_realized_equity = equity_series[-1] if equity_series else self.initial_balance
        unrealized_pnl = 0.0
        try:
            pos_tickers = self._get_all_position_tickers_sync()
            for sym, pos in self.positions.items():
                t = pos_tickers.get(sym)
                if t and t.get('last'):
                    unrealized_pnl += (t['last'] - pos['price']) * pos['amount']
        except Exception:
            pass
        current_equity = current_realized_equity + unrealized_pnl
        # If current equity exceeds peak, update peak (new all-time high including unrealized)
        if current_equity > peak:
            peak = current_equity
        drawdown_pct = ((peak - current_equity) / peak * 100) if peak > 0 else 0.0

        daily_pnl = self._daily_realized_pnl()

        # Count consecutive losing trades (most recent first)
        consecutive_losses = 0
        for trade in reversed(trades_snapshot):
            if trade.get("side") == "sell":
                pnl = trade.get("realized_pnl", 0.0)
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    break

        result = {
            "stock_performance": symbol_perf,
            "strategy_performance": strategy_perf,
            "equity_curve": {
                "total_pnl": round(self._realized_pnl_offset + sum(t.get("realized_pnl", 0.0) for t in trades_snapshot if t.get("side") == "sell"), 4),
                "recent_10_trades_pnl": round(total_recent_pnl, 4),
                "trend": trend,
                "drawdown_pct": round(drawdown_pct, 2),
                "daily_pnl": round(daily_pnl, 4),
                "consecutive_losses": consecutive_losses,
            },
        }

        # Update the cache so subsequent calls with the same trade count are fast
        self._perf_cache = result
        self._perf_cache_trade_count = self._trade_history_version
        self._perf_cache_time = now

        return result

    def _compute_trade_pattern_analysis(self) -> Dict[str, Any]:
        """Analyze closed trades to identify which conditions, timeframes, and parameters
        have historically led to wins vs losses. Cached and only recomputed when new trades arrive."""
        if self._trade_history_version == self._trade_pattern_cache_trade_count and self._trade_pattern_cache is not None:
            return self._trade_pattern_cache

        # Snapshot trade_history to avoid concurrent modification during iteration
        trades_snapshot = list(self.trade_history)

        from collections import defaultdict

        sells = [t for t in trades_snapshot if t.get("side") == "sell" and "realized_pnl" in t]
        if not sells:
            result: Dict[str, Any] = {}
            self._trade_pattern_cache = result
            self._trade_pattern_cache_trade_count = self._trade_history_version
            return result

        def _win_rate_stats(trades: list) -> Optional[Dict[str, Any]]:
            if not trades:
                return None
            wins = [t for t in trades if t["realized_pnl"] > 0]
            total_pnl = sum(t["realized_pnl"] for t in trades)
            return {
                "win_rate": round(len(wins) / len(trades), 3),
                "trades": len(trades),
                "avg_pnl": round(total_pnl / len(trades), 6),
            }

        # --- Entry conditions (strategy type + confidence range as proxies) ---
        condition_groups: Dict[str, list] = defaultdict(list)
        for t in sells:
            strategy = t.get("strategy_type", "unknown")
            condition_groups[f"strategy={strategy}"].append(t)

        best_entry_conditions = []
        for cond, trades in condition_groups.items():
            stats = _win_rate_stats(trades)
            if stats and stats["trades"] >= 3:
                best_entry_conditions.append({"condition": cond, **stats})
        best_entry_conditions.sort(key=lambda x: x["win_rate"], reverse=True)
        best_entry_conditions = best_entry_conditions[:5]

        # --- Timeframes ---
        tf_groups: Dict[str, list] = defaultdict(list)
        for t in sells:
            tf = t.get("timeframe", "unknown")
            tf_groups[tf].append(t)
        best_timeframes = []
        for tf, trades in tf_groups.items():
            stats = _win_rate_stats(trades)
            if stats and stats["trades"] >= 3:
                best_timeframes.append({"timeframe": tf, **stats})
        best_timeframes.sort(key=lambda x: x["win_rate"], reverse=True)

        # --- Exit reasons ---
        exit_groups: Dict[str, list] = defaultdict(list)
        for t in sells:
            reason = t.get("exit_reason", "unknown")
            exit_groups[reason].append(t)
        best_exit_reasons = []
        for reason, trades in exit_groups.items():
            stats = _win_rate_stats(trades)
            if stats and stats["trades"] >= 2:
                best_exit_reasons.append({"exit_reason": reason, **stats})
        best_exit_reasons.sort(key=lambda x: x["win_rate"], reverse=True)

        # --- Confidence ranges ---
        conf_groups: Dict[str, list] = defaultdict(list)
        for t in sells:
            conf = t.get("buy_confidence", 0.5)
            if conf >= 0.8:
                conf_groups["0.8-1.0"].append(t)
            elif conf >= 0.5:
                conf_groups["0.5-0.8"].append(t)
            elif conf >= 0.3:
                conf_groups["0.3-0.5"].append(t)
            else:
                conf_groups["0.0-0.3"].append(t)
        best_confidence_ranges = []
        for rng, trades in conf_groups.items():
            stats = _win_rate_stats(trades)
            if stats and stats["trades"] >= 3:
                best_confidence_ranges.append({"range": rng, **stats})
        best_confidence_ranges.sort(key=lambda x: x["win_rate"], reverse=True)

        # --- Per-symbol performance ---
        symbol_groups: Dict[str, list] = defaultdict(list)
        for t in sells:
            symbol_groups[t["symbol"]].append(t)
        best_symbols = []
        worst_symbols = []
        for sym, trades in symbol_groups.items():
            stats = _win_rate_stats(trades)
            if stats and stats["trades"] >= 3:
                if stats["win_rate"] >= 0.5:
                    best_symbols.append({"symbol": sym, **stats})
                else:
                    worst_symbols.append({"symbol": sym, **stats})
        best_symbols.sort(key=lambda x: x["avg_pnl"], reverse=True)
        best_symbols = best_symbols[:5]
        worst_symbols.sort(key=lambda x: x["avg_pnl"])
        worst_symbols = worst_symbols[:5]

        # --- Hold time analysis ---
        winning_holds = [t.get("hold_time_seconds") for t in sells if t["realized_pnl"] > 0 and t.get("hold_time_seconds")]
        losing_holds = [t.get("hold_time_seconds") for t in sells if t["realized_pnl"] < 0 and t.get("hold_time_seconds")]
        avg_hold_winning = round(sum(winning_holds) / len(winning_holds)) if winning_holds else None
        avg_hold_losing = round(sum(losing_holds) / len(losing_holds)) if losing_holds else None

        result = {
            "best_entry_conditions": best_entry_conditions,
            "best_timeframes": best_timeframes,
            "best_exit_reasons": best_exit_reasons,
            "best_confidence_ranges": best_confidence_ranges,
            "best_symbols": best_symbols,
            "worst_symbols": worst_symbols,
            "avg_hold_time_winning": avg_hold_winning,
            "avg_hold_time_losing": avg_hold_losing,
        }

        self._trade_pattern_cache = result
        self._trade_pattern_cache_trade_count = self._trade_history_version
        return result

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
        if current_price is None or current_price <= 0:
            return "unknown"

        # Read LLM-decided regime thresholds from Redis (set during stock selection).
        # If any threshold is missing, we cannot classify the regime reliably –
        # return "unknown" instead of falling back to hardcoded defaults.
        adx_strong = None
        adx_moderate = None
        vol_high_pct = None
        vol_low_pct = None
        bb_squeeze_width = None
        bb_expansion_width = None
        try:
            raw = await asyncio.to_thread(self.redis.get, "trading:regime_adx_strong")
            if raw:
                adx_strong = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:regime_adx_moderate")
            if raw:
                adx_moderate = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:regime_volatility_high_pct")
            if raw:
                vol_high_pct = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:regime_volatility_low_pct")
            if raw:
                vol_low_pct = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:regime_bb_squeeze_width")
            if raw:
                bb_squeeze_width = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:regime_bb_expansion_width")
            if raw:
                bb_expansion_width = float(raw)
        except Exception:
            pass  # leave as None if Redis is unavailable

        # If the LLM has not provided all required thresholds, we cannot
        # classify the regime – return "unknown" rather than using defaults.
        if (
            adx_strong is None
            or adx_moderate is None
            or vol_high_pct is None
            or vol_low_pct is None
            or bb_squeeze_width is None
            or bb_expansion_width is None
        ):
            return "unknown"

        # --- Trend direction and strength ---
        trend_dir = "neutral"
        trend_strength = "weak"
        if adx is not None and plus_di is not None and minus_di is not None:
            if adx > adx_strong:
                trend_strength = "strong"
            elif adx > adx_moderate:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"

            if plus_di > minus_di:
                trend_dir = "uptrend"
            elif minus_di > plus_di:
                trend_dir = "downtrend"
            else:
                trend_dir = "neutral"

        # --- Moving average alignment ---
        ma_alignment = "neutral"
        if ema_9 is not None and ema_21 is not None:
            if ema_9 > ema_21:
                ma_alignment = "bullish"
            else:
                ma_alignment = "bearish"

        # --- Volatility state ---
        volatility = "normal"
        if atr is not None and current_price > 0:
            atr_pct = (atr / current_price) * 100
            if atr_percentile is not None:
                if atr_percentile > vol_high_pct:
                    volatility = "high"
                elif atr_percentile < vol_low_pct:
                    volatility = "low"
                else:
                    volatility = "normal"
            else:
                # Fallback to simple thresholds
                if atr_pct > (bb_expansion_width * 100):
                    volatility = "high"
                elif atr_pct < (bb_squeeze_width * 100):
                    volatility = "low"

        # --- Bollinger Band squeeze/expansion ---
        bb_state = ""
        if bb_upper is not None and bb_lower is not None and bb_middle is not None and bb_middle > 0:
            bb_width = (bb_upper - bb_lower) / bb_middle
            if bb_width < bb_squeeze_width:   # very narrow bands
                bb_state = " squeeze"
            elif bb_width > bb_expansion_width: # wide bands
                bb_state = " expansion"

        # --- Compose final regime string ---
        if trend_strength in ("strong", "moderate") and trend_dir != "neutral":
            regime = f"{trend_strength} {trend_dir}"
        else:
            regime = "ranging"

        # Add volatility
        regime += f", {volatility} volatility"

        # Add Bollinger state if meaningful
        if bb_state:
            regime += bb_state

        # Add MA alignment if it conflicts with ADX trend (e.g., weak trend but bullish MA)
        if trend_strength == "weak" and ma_alignment != "neutral":
            regime += f" ({ma_alignment} MA bias)"

        return regime

    async def _reconcile_positions(self):
        """Detect and handle external changes: delisted symbols, externally sold positions."""
        # --- Delisted stocks ---
        plain_assets = await self._get_tradable_assets()
        available_pairs = [f"{sym}/{self.base_currency}" for sym in plain_assets]
        # Include BTP bonds and ETFs so they are not removed during reconciliation
        btp_bonds = await self._get_btp_bonds()
        available_pairs += [f"{b['isin']}/{self.base_currency}" for b in btp_bonds]
        etf_symbols = await self._get_etf_symbols()
        available_pairs += [f"{sym}/{self.base_currency}" for sym in etf_symbols]

        # Build BTP maturity map for maturity checking
        btp_maturity_map: Dict[str, Optional[str]] = {}
        for b in btp_bonds:
            isin = b.get("isin")
            maturity = b.get("maturity")
            if isin and maturity:
                btp_maturity_map[isin] = maturity

        # --- Matured BTP bonds: close at par value (100.0) ---
        now_dt = datetime.now(timezone.utc)
        for entry in list(self.current_symbols):
            symbol = entry["symbol"]
            base = symbol.split("/")[0]
            if not re.match(r'^IT[A-Z0-9]{10}$', base):
                continue
            maturity_str = btp_maturity_map.get(base)
            if maturity_str is None:
                continue
            try:
                maturity_str_clean = maturity_str.strip()
                if "T" in maturity_str_clean:
                    maturity_dt = datetime.fromisoformat(maturity_str_clean.replace("Z", "+00:00"))
                else:
                    maturity_dt = datetime.fromisoformat(maturity_str_clean)
                if maturity_dt.tzinfo is None:
                    maturity_dt = maturity_dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                try:
                    maturity_dt = datetime.strptime(maturity_str, "%d/%m/%Y").replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    logger.debug(f"Could not parse maturity date '{maturity_str}' for BTP {symbol}")
                    continue
            if now_dt < maturity_dt:
                continue
            # BTP has matured – close at par value
            logger.info(f"BTP {symbol} has matured (maturity: {maturity_str}). Closing at par value.")
            self.current_symbols.remove(entry)
            async with self._queued_orders_lock:
                self.queued_orders = [q for q in self.queued_orders if q['symbol'] != symbol]
            if symbol in self.positions:
                await self._cancel_exit_orders(symbol)
                async with self._positions_lock:
                    pos = self.positions.pop(symbol)
                par_value = 100.0
                cost = pos["amount"] * par_value
                from src.exchanges.fees import calculate_transaction_costs
                costs = calculate_transaction_costs("SELL", par_value, pos["amount"], symbol=symbol)
                fee_cost = costs["total_costs"]
                cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                net_quote = cost - fee_cost
                realized_pnl = net_quote - cost_basis
                trade = {
                    "symbol": symbol,
                    "side": "sell",
                    "amount": pos["amount"],
                    "price": par_value,
                    "cost": cost,
                    "fee": {"cost": fee_cost, "currency": self.base_currency},
                    "timestamp": time.time() * 1000,
                    "note": "btp_matured",
                    "exit_reason": "btp_matured",
                    "realized_pnl": realized_pnl,
                    "cost_basis": cost_basis,
                }
                self._append_trade(trade)
                await asyncio.to_thread(insert_trade, trade)
                logger.info(f"Matured BTP {symbol}: closed {pos['amount']} at par value {par_value}.")
                if self.notifier:
                    stock_name = await self._get_stock_name(symbol)
                    display_symbol = self._format_symbol_display(symbol, stock_name, pos.get("timeframe"))
                    await self.notifier.send_notification(
                        f"💰 BTP {display_symbol} matured – closed at par value {par_value}. P&L: {realized_pnl:+.4f}",
                        summary={
                            "symbol": symbol,
                            "action": "SELL",
                            "reason": "BTP matured",
                            "price": par_value,
                            "realized_pnl": realized_pnl,
                            "exit_reason": "btp_matured",
                        }
                    )
                await self._remove_symbol_if_paused(symbol)

        for entry in list(self.current_symbols):
            symbol = entry["symbol"]
            if symbol not in available_pairs:
                logger.warning(f"Stock {symbol} no longer available. Removing from tracking.")
                self.current_symbols.remove(entry)
                # Remove any queued orders for this delisted symbol
                async with self._queued_orders_lock:
                    self.queued_orders = [q for q in self.queued_orders if q['symbol'] != symbol]
                if symbol in self.positions:
                    await self._cancel_exit_orders(symbol)
                    async with self._positions_lock:
                        pos = self.positions.pop(symbol)
                    cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                    base = symbol.split("/")[0]
                    is_btp = re.match(r'^IT[A-Z0-9]{10}$', base) is not None
                    if is_btp:
                        close_price = 100.0  # par value for delisted BTPs
                        close_cost = pos["amount"] * close_price
                        from src.exchanges.fees import calculate_transaction_costs
                        costs = calculate_transaction_costs("SELL", close_price, pos["amount"], symbol=symbol)
                        fee_cost = costs["total_costs"]
                        net_quote = close_cost - fee_cost
                        realized_pnl = net_quote - cost_basis
                        note = "btp_delisted"
                        exit_reason = "btp_delisted"
                    else:
                        close_price = 0.0
                        close_cost = 0.0
                        fee_cost = 0.0
                        realized_pnl = -cost_basis
                        note = "delisted"
                        exit_reason = "delisted"
                    trade = {
                        "symbol": symbol,
                        "side": "sell",
                        "amount": pos["amount"],
                        "price": close_price,
                        "cost": close_cost,
                        "fee": {"cost": fee_cost, "currency": self.base_currency},
                        "timestamp": time.time() * 1000,
                        "note": note,
                        "exit_reason": exit_reason,
                        "realized_pnl": realized_pnl,
                        "cost_basis": cost_basis,
                    }
                    self._append_trade(trade)
                    await asyncio.to_thread(insert_trade, trade)
                    logger.warning(f"Delisted {symbol}: recorded forced sell of {pos['amount']} at {close_price}.")
                    await self._remove_symbol_if_paused(symbol)

        # --- Externally modified balances ---
        # Fetch all balances at once instead of per-position API calls
        try:
            all_balances = await asyncio.to_thread(self.trader.fetch_balance)
        except Exception as e:
            logger.error(f"Failed to fetch balances for reconciliation: {e}")
            all_balances = {}
        for symbol, pos in list(self.positions.items()):
            base = symbol.split('/')[0]
            try:
                actual_balance = all_balances.get(base, 0.0)
            except Exception as e:
                logger.error(f"Failed to get balance for {base}: {e}")
                continue

            recorded_amount = pos.get("amount", 0.0)
            if actual_balance < recorded_amount - 1e-8:
                # External sell detected
                sold_amount = recorded_amount - actual_balance
                try:
                    tickers_map = await self._get_quotes_async([symbol.split("/")[0]], timeout=45.0)
                    ticker = tickers_map.get(symbol.split("/")[0])
                    current_price = ticker['last'] if ticker else pos.get("price", 0.0)
                except Exception:
                    current_price = pos.get("price", 0.0)  # fallback to entry price
                cost = sold_amount * current_price
                from src.exchanges.fees import calculate_transaction_costs
                costs = calculate_transaction_costs("SELL", current_price, sold_amount, symbol=symbol)
                fee_cost = costs["total_costs"]
                trade = {
                    "symbol": symbol,
                    "side": "sell",
                    "amount": sold_amount,
                    "price": current_price,
                    "cost": cost,
                    "fee": {"cost": fee_cost, "currency": self.base_currency},
                    "timestamp": time.time() * 1000,
                    "note": "external_sell",
                    "exit_reason": "external_sell"
                }
                # Compute realized P&L for the externally sold portion
                cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                net_base = pos.get("net_base", pos["amount"])
                prorated_cost_basis = cost_basis * (sold_amount / net_base) if net_base > 0 else 0.0
                net_quote = cost - fee_cost
                trade["realized_pnl"] = net_quote - prorated_cost_basis
                trade["cost_basis"] = prorated_cost_basis
                self._append_trade(trade)
                await asyncio.to_thread(insert_trade, trade)
                logger.warning(
                    f"External sell detected for {symbol}: {sold_amount} sold at ~{current_price}. "
                    f"Updating position from {recorded_amount} to {actual_balance}."
                )
                if actual_balance == 0.0:
                    await self._cancel_exit_orders(symbol)
                    async with self._positions_lock:
                        del self.positions[symbol]
                    await self._remove_symbol_if_paused(symbol)
                else:
                    async with self._positions_lock:
                        self.positions[symbol]["amount"] = actual_balance
                        self.positions[symbol]["cost_basis"] = cost_basis - prorated_cost_basis
                        self.positions[symbol]["net_base"] = net_base - sold_amount
                        new_net_base = self.positions[symbol]["net_base"]
                        new_cost_basis = self.positions[symbol]["cost_basis"]
                        self.positions[symbol]["price"] = new_cost_basis / new_net_base if new_net_base > 0 else 0.0
            elif actual_balance > recorded_amount + 1e-8:
                # External deposit – sync to actual balance
                logger.warning(
                    f"Balance of {base} increased externally from {recorded_amount} to {actual_balance}. "
                    f"Updating position."
                )
                async with self._positions_lock:
                    self.positions[symbol]["amount"] = actual_balance
                    self.positions[symbol]["net_base"] = actual_balance
                    cost_basis = self.positions[symbol].get("cost_basis", 0.0)
                    self.positions[symbol]["price"] = cost_basis / actual_balance if actual_balance > 0 else 0.0

        # --- Handle positions that were loaded without LLM risk parameters ---
        for symbol, pos in list(self.positions.items()):
            if pos.get("_needs_risk_params"):
                # Check if risk parameters have been populated by a re-evaluation
                if pos.get("stop_loss") is not None and pos.get("take_profit") is not None:
                    logger.info(f"Risk parameters obtained for {symbol}; clearing _needs_risk_params flag.")
                    pos.pop("_needs_risk_params", None)
                    pos.pop("_needs_risk_params_attempts", None)
                    continue

                # Risk parameters still missing — increment attempt counter
                attempts = pos.get("_needs_risk_params_attempts", 0) + 1
                pos["_needs_risk_params_attempts"] = attempts

                # Force another re-evaluation so the LLM gets another chance
                self._force_eval[symbol] = True
                self._last_strategy_eval.pop(symbol, None)

                max_attempts = 3  # ~15 minutes across 3 reconcile cycles (5 min each)
                if attempts >= max_attempts:
                    logger.warning(
                        f"Force-closing {symbol}: missing LLM risk parameters after {attempts} "
                        f"re-evaluation attempts."
                    )
                    stock_name = await self._get_stock_name(symbol)
                    display_symbol = self._format_symbol_display(symbol, stock_name, pos.get("timeframe"))
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"🔻 Closing {display_symbol} – missing LLM risk parameters after {attempts} attempts.",
                            summary={
                                "symbol": symbol,
                                "action": "SELL",
                                "reason": f"Missing LLM risk parameters after {attempts} attempts",
                                "exit_reason": "force_close",
                            }
                        )
                    signal = Signal(action="SELL", confidence=1.0, reasoning="Missing LLM risk parameters after re-evaluation attempts")
                    await self._execute_signal(symbol, signal, exit_reason="force_close")
                else:
                    logger.info(
                        f"Position {symbol} still missing risk parameters "
                        f"(attempt {attempts}/{max_attempts}); forcing re-evaluation."
                    )

        # Persist any changes made during reconciliation
        await self._save_state(force=True)

    def _append_trade(self, trade: Dict[str, Any]):
        """Append a trade to history and prune old entries to bound memory usage."""
        self._trade_history_version += 1
        self.trade_history.append(trade)
        if len(self.trade_history) > MAX_TRADES_IN_MEMORY:
            # Accumulate realized P&L of pruned trades so the equity curve
            # in _compute_performance_metrics remains accurate.
            pruned = self.trade_history[:-MAX_TRADES_IN_MEMORY]
            for t in pruned:
                if t.get("side") == "sell":
                    self._realized_pnl_offset += t.get("realized_pnl", 0.0)
            # Keep only the most recent trades
            self.trade_history = self.trade_history[-MAX_TRADES_IN_MEMORY:]

    def _load_state(self):
        """Load current symbols, positions, trade history, and initial balance from SQLite."""
        state = load_trading_state()

        raw_symbols = state.get("current_symbols", [])
        # Convert old format (list of strings) to new format if needed
        if raw_symbols and isinstance(raw_symbols[0], str):
            default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
            self.current_symbols = [{"symbol": s, "timeframe": default_tf} for s in raw_symbols]
        else:
            self.current_symbols = raw_symbols
        self.positions = state.get("positions", {})
        # Remove any position that lacks LLM-defined risk parameters.
        # Such positions cannot be managed safely.
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            if "stop_loss" not in pos or "take_profit" not in pos:
                logger.warning(
                    f"Position for {symbol} is missing stop_loss/take_profit. "
                    f"Will attempt to re-evaluate to obtain LLM risk parameters before force-closing."
                )
                pos["_needs_risk_params"] = True
                pos["_needs_risk_params_attempts"] = 0
                # Force immediate re-evaluation so the LLM can provide risk parameters
                self._force_eval[symbol] = True
                self._last_strategy_eval.pop(symbol, None)

        # Discard positions with zero amount or zero price (corrupted state)
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            amount = pos.get("amount", 0)
            price = pos.get("price", 0)
            if amount <= 0 or price <= 0:
                logger.warning(
                    f"Position for {symbol} has invalid amount={amount} or price={price}. Removing it."
                )
                del self.positions[symbol]

        # Initialize trailing stop tracking fields for positions with trailing stops.
        # This ensures _highest_price is not set to a pre-entry price on the first
        # check after a restart (which would make the trailing stop too tight).
        for symbol, pos in self.positions.items():
            if pos.get("trailing_stop"):
                if "_highest_price" not in pos:
                    pos["_highest_price"] = pos.get("price", 0.0)
                if "_last_trailing_check_ts" not in pos:
                    pos["_last_trailing_check_ts"] = time.time()

        all_trades = get_all_trades()
        self.trade_history = all_trades[-MAX_TRADES_IN_MEMORY:]
        # Compute the realized P&L offset for trades that were pruned at load time
        self._realized_pnl_offset = sum(
            t.get("realized_pnl", 0.0)
            for t in all_trades[:-MAX_TRADES_IN_MEMORY]
            if t.get("side") == "sell"
        )
        self.queued_orders = state.get("queued_orders", [])
        for q in self.queued_orders:
            q['order_book'] = None
        self.recent_signals = state.get("recent_signals", [])
        self._symbol_first_seen = state.get("symbol_first_seen", {})
        self._entry_signal_state = state.get("entry_signal_state", {})
        self._last_eval_snapshot = state.get("last_eval_snapshot", {})
        self._force_eval = state.get("force_eval", {})
        self._force_eval_time = state.get("force_eval_time", {})
        self._strategy_intervals = state.get("strategy_intervals", {})
        self._last_decisions = state.get("last_decisions", {})
        self.last_loss_time = state.get("last_loss_time", {})
        self.cooldown_durations = state.get("cooldown_durations", {})
        self._global_risk_multiplier = state.get("global_risk_multiplier")

        # Restore pending entries (reconstruct Signal objects from dicts)
        raw_pending = state.get("pending_entries", {})
        self._pending_entries = {}
        for symbol, entry in raw_pending.items():
            try:
                signal = Signal(**entry["signal"])
                self._pending_entries[symbol] = {
                    "signal": signal,
                    "deadline": entry["deadline"],
                    "timeframe": entry["timeframe"],
                    "condition": entry["condition"],
                }
            except Exception as e:
                logger.warning(f"Failed to restore pending entry for {symbol}: {e}")

        # Prune any pending entries whose deadline has already passed
        now = time.time()
        expired = [sym for sym, e in self._pending_entries.items() if now >= e["deadline"]]
        for sym in expired:
            logger.info(f"Discarding expired pending entry for {sym} (deadline passed during downtime).")
            del self._pending_entries[sym]

        if "initial_balance" in state:
            self.initial_balance = float(state["initial_balance"])
        else:
            balance = self.trader.fetch_balance()
            self.initial_balance = balance.get(self.base_currency, 0.0)
            save_trading_state("initial_balance", self.initial_balance)

        logger.info(
            "Loaded trading state: %d symbols, %d positions, %d trades",
            len(self.current_symbols),
            len(self.positions),
            len(self.trade_history),
        )

    async def _save_state(self, force: bool = False):
        """Persist current symbols, positions, and trade history to SQLite.

        Uses a lock to serialize concurrent calls and a debounce flag to
        coalesce multiple save requests into fewer DB write batches.

        When *force* is True, the method waits for the lock instead of
        debouncing, guaranteeing the state is flushed even if another
        save is in progress.  Use force=True after critical state changes
        (trade execution, position closure, etc.) to avoid data loss on crash.
        """
        # If a save is already in progress:
        #   - force=True  → wait for the lock, then save (no debounce)
        #   - force=False → mark pending and return (debounce)
        if self._state_lock.locked():
            if not force:
                self._state_save_pending = True
                return
            # Fall through to acquire the lock (wait for the current save to finish)

        async with self._state_lock:
            await self._save_state_impl()
            # If another save was requested while we were saving, do one more
            while self._state_save_pending:
                self._state_save_pending = False
                await self._save_state_impl()

    async def _save_state_impl(self):
        """Actual state persistence (must be called under _state_lock)."""
        await asyncio.to_thread(save_trading_state, "current_symbols", self.current_symbols)
        async with self._positions_lock:
            positions_snapshot = dict(self.positions)
        await asyncio.to_thread(save_trading_state, "positions", positions_snapshot)
        await asyncio.to_thread(save_trading_state, "queued_orders", self.queued_orders)
        await asyncio.to_thread(save_trading_state, "recent_signals", self.recent_signals)
        # Serialize pending entries (convert Signal objects to dicts for JSON storage)
        pending_entries_serializable = {}
        for symbol, entry in self._pending_entries.items():
            pending_entries_serializable[symbol] = {
                "signal": asdict(entry["signal"]),
                "deadline": entry["deadline"],
                "timeframe": entry["timeframe"],
                "condition": entry["condition"],
            }
        await asyncio.to_thread(save_trading_state, "pending_entries", pending_entries_serializable)
        await asyncio.to_thread(save_trading_state, "symbol_first_seen", self._symbol_first_seen)
        await asyncio.to_thread(save_trading_state, "entry_signal_state", self._entry_signal_state)
        await asyncio.to_thread(save_trading_state, "last_eval_snapshot", self._last_eval_snapshot)
        await asyncio.to_thread(save_trading_state, "force_eval", self._force_eval)
        await asyncio.to_thread(save_trading_state, "force_eval_time", self._force_eval_time)
        await asyncio.to_thread(save_trading_state, "strategy_intervals", self._strategy_intervals)
        await asyncio.to_thread(save_trading_state, "last_decisions", self._last_decisions)
        await asyncio.to_thread(save_trading_state, "last_loss_time", self.last_loss_time)
        await asyncio.to_thread(save_trading_state, "cooldown_durations", self.cooldown_durations)
        await asyncio.to_thread(save_trading_state, "global_risk_multiplier", self._global_risk_multiplier)
        logger.debug("Saved trading state: %d symbols, %d positions, %d trades",
                     len(self.current_symbols), len(self.positions), len(self.trade_history))
        self._state_dirty = False

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
                    elif tf in ("3Y", "5Y"):
                        tf_base_interval = 1209600 # 2 weeks
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
        # Reset per-cycle spending tracker so new buys are not blocked by prior cycle spending
        async with self._cycle_spent_lock:
            self._cycle_spent = 0.0
        logger.info("Re-evaluation step 1/12: Checking cooldown and fetching asset lists...")

        # Respect triggered re-evaluation cooldown for market-condition triggers only.
        # Pre-market re-evaluations are always allowed (they are time-critical).
        # Forced re-evaluations (explicit user or critical condition requests) always bypass
        # the cooldown since they are intentionally requested.
        # Capture whether this is a market-condition trigger before clearing flags
        is_market_condition_trigger = force and not self._pre_market_reeval and not self._user_forced_reeval

        if is_market_condition_trigger:
            # Check if this was triggered by the market condition monitor (not a user action).
            # The market condition monitor sets _force_reeval directly without going through
            # trigger_symbol_reevaluation, so we check the triggered cooldown key.
            # User-initiated forced re-evaluations (from the web UI or Telegram) bypass this cooldown.
            last_triggered = await asyncio.to_thread(self.redis.get, "trading:last_triggered_reeval")
            if last_triggered:
                elapsed = time.time() - float(last_triggered)
                if elapsed < settings.TRIGGERED_REEVALUATION_COOLDOWN:
                    logger.info(f"Forced re-evaluation skipped: triggered cooldown active ({settings.TRIGGERED_REEVALUATION_COOLDOWN - elapsed:.0f}s remaining)")
                    return
        is_user_forced = self._user_forced_reeval
        # Clear the pre-market flag after reading it
        self._pre_market_reeval = False
        # Clear the user-forced flag after reading it
        self._user_forced_reeval = False

        # Only re-evaluate every SYMBOL_REVALUATION_INTERVAL
        last_key = "trading:last_symbol_eval"
        last_eval = await asyncio.to_thread(self.redis.get, last_key)
        now = time.time()
        if last_eval and (now - float(last_eval)) < self._symbol_reevaluation_interval and self.current_symbols and not force:
            logger.info("Skipping symbol re-evaluation: last eval was recent and symbols are already loaded.")
            return

        logger.info("Re-evaluation step 2/12: Fetching tradable assets, BTPs, and ETFs...")
        old_symbols = list(self.current_symbols)
        plain_assets = await self._get_tradable_assets()
        stock_pairs = [f"{sym}/{self.base_currency}" for sym in plain_assets]

        # Fetch BTP bonds
        btp_bonds = await self._get_btp_bonds()
        btp_pairs = [f"{b['isin']}/{self.base_currency}" for b in btp_bonds]

        # Fetch ETFs
        etf_symbols = await self._get_etf_symbols()
        etf_pairs = [f"{sym}/{self.base_currency}" for sym in etf_symbols]
        available_pairs = stock_pairs + btp_pairs

        # --- Filter: only include symbols that have a name in discovered_symbols ---
        from src.database import get_discovered_symbols_with_names
        symbols_with_names = await asyncio.to_thread(get_discovered_symbols_with_names)
        _suffix = settings.TICKER_SUFFIX

        def _has_name(pair: str) -> bool:
            base = pair.split("/")[0]
            db_base = base
            if _suffix and db_base.endswith(_suffix):
                db_base = db_base[:-len(_suffix)]
            return db_base in symbols_with_names or base in symbols_with_names

        available_pairs = [p for p in available_pairs if _has_name(p)]
        btp_pairs = [p for p in btp_pairs if p.split("/")[0] in symbols_with_names]
        etf_pairs = [p for p in etf_pairs if _has_name(p)]

        if not available_pairs and not btp_pairs:
            logger.warning("No symbols with names in discovered_symbols. Skipping re-evaluation.")
            await asyncio.to_thread(self.redis.set, last_key, now)
            return

        logger.info("Re-evaluation step 3/12: RSS and news-driven symbol discovery...")
        # --- RSS-based ticker discovery: scan news feeds for symbols with TICKER_SUFFIX ---
        if settings.NEWS_ENABLED and settings.NEWS_TICKER_DISCOVERY_ENABLED and discover_tickers_from_news is not None:
            try:
                rss_discovered = await asyncio.to_thread(
                    discover_tickers_from_news,
                    existing_pairs=available_pairs,
                    cache_only=True,
                )
                # Convert discovered base symbols to full pairs and add to the front
                for base in rss_discovered:
                    pair = f"{base}/{self.base_currency}"
                    if pair not in available_pairs:
                        available_pairs.insert(0, pair)
                if rss_discovered:
                    logger.info(f"RSS ticker discovery added {len(rss_discovered)} new symbols: {rss_discovered}")
            except Exception as e:
                logger.warning(f"RSS ticker discovery failed: {e}")

        if not available_pairs:
            logger.warning("No available pairs found.")
            return

        # --- News-driven symbol discovery: add trending symbols not in the top 50 ---
        if settings.NEWS_ENABLED and settings.NEWS_SYMBOL_DISCOVERY_ENABLED and discover_trending_stocks is not None:
            try:
                discovered = await asyncio.to_thread(
                    discover_trending_stocks,
                    self.base_currency,
                    available_pairs,
                    max_symbols=settings.NEWS_SYMBOL_DISCOVERY_MAX_SYMBOLS,
                    min_sentiment=settings.NEWS_SYMBOL_DISCOVERY_MIN_SENTIMENT,
                    min_articles=settings.NEWS_SYMBOL_DISCOVERY_MIN_ARTICLES,
                    cache_only=True,
                )
                # Add discovered symbols to the front of the list so they are included in the sample
                for pair in discovered:
                    if pair not in available_pairs:
                        available_pairs.insert(0, pair)
                if discovered:
                    logger.info(f"Added {len(discovered)} news-discovered symbols to candidate pool.")
            except Exception as e:
                logger.warning(f"News stock discovery failed: {e}")

        logger.info("Re-evaluation step 4/12: Fetching balance and quotes (from %d available pairs)...", len(available_pairs))
        # Fetch balance and compute per-symbol budget
        balance = await self._get_cached_balance()
        base_balance = balance.get(self.base_currency, 0.0)
        per_symbol_budget = base_balance / self.max_symbols if self.max_symbols > 0 else 0.0

        # Fetch tickers for a subset to keep prompt size manageable
        # Apply sentiment filter if configured
        if settings.SYMBOL_SELECTION_MIN_SENTIMENT > -1.0 and settings.NEWS_ENABLED:
            candidate_pairs = available_pairs
            async def _fetch_sentiment_filter(sym):
                try:
                    base_symbol = sym.split("/")[0] if "/" in sym else sym
                    agg = await asyncio.to_thread(get_aggregate_sentiment_from_db, base_symbol, max_age_seconds=settings.NEWS_CACHE_TTL_SECONDS)
                    if agg and agg["avg_compound"] >= settings.SYMBOL_SELECTION_MIN_SENTIMENT:
                        return sym
                    elif not agg:
                        return sym
                    return None
                except Exception:
                    return sym
            sentiment_filter_tasks = [_fetch_sentiment_filter(sym) for sym in candidate_pairs]
            sentiment_filter_results = await asyncio.gather(*sentiment_filter_tasks)
            sample_pairs = [sym for sym in sentiment_filter_results if sym is not None]
        else:
            sample_pairs = available_pairs

        # Ensure BTPs and ETFs are always included in the candidate pool so they flow
        # through volume sorting, OHLCV fetch, indicator computation, etc.
        for btp in btp_pairs:
            if btp not in sample_pairs:
                sample_pairs.append(btp)
        for etf in etf_pairs:
            if etf not in sample_pairs:
                sample_pairs.append(etf)

        # Remove fully excluded symbols from the candidate pool
        sample_pairs = [
            sym for sym in sample_pairs
            if not any(
                entry.split("/")[0] == sym.split("/")[0] and
                entry.split("/")[1] == sym.split("/")[1] and
                len(entry.split("/")) == 2
                for entry in settings.EXCLUDED_SYMBOLS
            )
        ]

        # --- Do NOT pre-rank or limit candidates by volume ---
        # We want the LLM to evaluate ALL symbols that have a quote in cache/DB.
        # The quote fetch uses get_quotes_cached which only reads from Redis/DB
        # (no network calls), so fetching quotes for hundreds of symbols is fast.

        logger.info(f"Step 4: Fetching quotes for {len(sample_pairs)} symbols from Redis/DB cache")

        # Fetch quotes from Redis/DB cache only — no network calls.
        # The background _refresh_all_quotes_loop keeps the cache warm.
        # This prevents the re-evaluation from hanging on slow yfinance
        # or Borsa Italiana API calls.
        # BTPs are included so their quotes are fetched from DB close prices
        # (same as stocks), not from the Borsa Italiana bond list.
        plain_sample = [s.split("/")[0] for s in sample_pairs]
        raw_quotes = await asyncio.to_thread(get_quotes_cached, plain_sample)
        tickers = {pair: raw_quotes.get(pair.split("/")[0], {}) for pair in sample_pairs}

        # Filter out symbols with no valid last price
        valid_sample_pairs = [
            sym for sym in sample_pairs
            if tickers.get(sym, {}).get('last') is not None and tickers[sym]['last'] > 0
        ]
        if not valid_sample_pairs:
            logger.warning("No symbols with valid price data. Idling until next evaluation.")
            await asyncio.to_thread(self.redis.set, last_key, now)
            # Cooldown: only send the notification once per hour to avoid spam
            no_price_key = "trading:no_price_data_notify"
            last_notify = await asyncio.to_thread(self.redis.get, no_price_key)
            should_notify = True
            if last_notify:
                try:
                    if (time.time() - float(last_notify)) < 3600:  # 1 hour cooldown
                        should_notify = False
                except (ValueError, TypeError):
                    pass
            if should_notify and self.notifier:
                await self.notifier.send_notification(
                    "⚠️ No symbols with valid price data. Bot will idle.",
                    summary={"action": "HOLD", "reason": "No valid price data"}
                )
                await asyncio.to_thread(self.redis.set, no_price_key, str(time.time()))
            return
        sample_pairs = valid_sample_pairs

        logger.info("Re-evaluation step 5/12: Yahoo Finance fallback for missing quotes...")
        # --- Yahoo Finance fallback for missing quotes (last, bid, ask) ---
        if settings.YAHOO_FINANCE_ENABLED:
            missing_quotes = [
                sym for sym in sample_pairs
                if tickers.get(sym, {}).get('last') is None or tickers.get(sym, {}).get('bid') is None or tickers.get(sym, {}).get('ask') is None
            ]
            # Limit to 20 symbols per cycle to stay under Yahoo's rate limits
            missing_quotes = missing_quotes[:20]
            async def _fetch_yahoo_quote(sym):
                base = sym.split("/")[0]
                yahoo = await asyncio.to_thread(get_yahoo_quote, base)
                if yahoo:
                    t = tickers.setdefault(sym, {})
                    if t.get('last') is None:
                        t['last'] = yahoo.get('last')
                    if t.get('bid') is None:
                        t['bid'] = yahoo.get('bid')
                    if t.get('ask') is None:
                        t['ask'] = yahoo.get('ask')
            await asyncio.gather(*[_fetch_yahoo_quote(sym) for sym in missing_quotes])

        # --- Sort candidate pool by 24h volume (preserve BTPs and ETFs) ---
        def _volume(sym):
            t = tickers.get(sym, {})
            return t.get('quoteVolume', 0) or 0
        stock_sample_sorted = sorted([s for s in sample_pairs if s in stock_pairs and s not in etf_pairs], key=_volume, reverse=True)
        etf_sample_sorted = [s for s in sample_pairs if s in etf_pairs]
        # Pass ALL discovered stocks, ETFs, and BTPs to the LLM
        sample_pairs = stock_sample_sorted + etf_sample_sorted + [s for s in sample_pairs if s in btp_pairs]
        logger.info("Re-evaluation step 6/12: Batch-fetching news sentiment for %d symbols...", len(sample_pairs))
        news_sentiment = {}
        if settings.NEWS_ENABLED:
            batch_sentiment = await asyncio.to_thread(
                get_aggregate_sentiment_for_symbols, sample_pairs, settings.NEWS_CACHE_TTL_SECONDS
            )
            for sym, agg in batch_sentiment.items():
                if agg:
                    base = sym.split("/")[0] if "/" in sym else sym
                    news_sentiment[base] = agg


        # Sentiment trend (delta from previous cycle)
        sentiment_trend: Dict[str, Optional[float]] = {}
        for sym in sample_pairs:
            base_symbol = sym.split("/")[0] if "/" in sym else sym
            current_compound = None
            if base_symbol in news_sentiment:
                current_compound = news_sentiment[base_symbol].get("avg_compound")
            prev_key = f"sentiment:prev:{base_symbol}"
            prev_raw = await asyncio.to_thread(self.redis.get, prev_key)
            prev_compound = float(prev_raw) if prev_raw else None
            if current_compound is not None:
                await asyncio.to_thread(self.redis.setex, prev_key, settings.NEWS_CACHE_TTL_SECONDS, str(current_compound))
            if current_compound is not None and prev_compound is not None:
                sentiment_trend[base_symbol] = round(current_compound - prev_compound, 4)
            else:
                sentiment_trend[base_symbol] = None

        # Overall market trend (use configured benchmark, e.g., FTSEMIB.MI)
        market_trend = None
        benchmark_symbol = settings.BENCHMARK_SYMBOL
        if benchmark_symbol in tickers:
            benchmark_ticker = tickers[benchmark_symbol]
            market_trend = {
                "symbol": benchmark_symbol,
                "change_24h": benchmark_ticker.get("percentage"),
                "last": benchmark_ticker.get("last"),
            }
        elif sample_pairs:
            first = sample_pairs[0]
            if first in tickers:
                t = tickers[first]
                market_trend = {
                    "symbol": first,
                    "change_24h": t.get("percentage"),
                    "last": t.get("last"),
                }


        # Fetch OHLCV from database only for ALL candidate pairs.
        # Background tasks (_download_all_assets_data_loop) keep the DB populated.
        # This avoids blocking reevaluation on slow API calls.
        sorted_by_vol = sample_pairs
        logger.info("Re-evaluation step 7/12: Fetching OHLCV from DB for %d symbols...", len(sorted_by_vol))

        ohlcv_data = {}
        if settings.OHLCV_TIMEFRAMES:
            async def fetch_ohlcv_from_db(sym):
                data = {}
                for tf in settings.OHLCV_TIMEFRAMES:
                    try:
                        db_candles = await asyncio.to_thread(
                            get_ohlcv, sym, tf, limit=50
                        )
                        if db_candles:
                            data[tf] = [
                                [c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
                                for c in db_candles
                            ]
                    except Exception as e:
                        logger.debug(f"DB OHLCV fetch failed for {sym} {tf}: {e}")
                return sym, data
            tasks = [fetch_ohlcv_from_db(sym) for sym in sorted_by_vol]
            results = await asyncio.gather(*tasks)
            ohlcv_data = dict(results)

        # Build available timeframes per symbol for validation and final selection prompt
        available_timeframes_by_symbol = {}
        for sym, tf_data in ohlcv_data.items():
            available_tfs = [tf for tf in settings.OHLCV_TIMEFRAMES if tf in tf_data and tf_data[tf]]
            if available_tfs:
                available_timeframes_by_symbol[sym] = available_tfs

        logger.info("Re-evaluation step 8/12: Batch-fetching indicators for %d symbols...", len(sorted_by_vol))
        primary_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"

        # Batch-fetch all indicators in a single DB query
        batch_indicators = await asyncio.to_thread(
            get_indicators_for_symbols, sorted_by_vol, settings.OHLCV_TIMEFRAMES
        )

        def _compute_trend_score(sym: str, sym_indicators: Dict[str, Dict[str, Any]]) -> float:
            trend_score = 0.0
            try:
                ind = sym_indicators.get(primary_tf, {})
                score = 0.0
                components = 0

                adx_val = ind.get('adx')
                if adx_val is not None:
                    score += min(1.0, adx_val / 50.0)
                    components += 1

                ema_9_val = ind.get('ema_9')
                ema_21_val = ind.get('ema_21')
                if ema_9_val is not None and ema_21_val is not None:
                    score += 1.0 if ema_9_val > ema_21_val else 0.0
                    components += 1

                rsi_val = ind.get('rsi')
                if rsi_val is not None:
                    if 40 <= rsi_val <= 70:
                        score += 1.0
                    elif 30 <= rsi_val <= 80:
                        score += 0.5
                    else:
                        score += 0.0
                    components += 1

                macd_hist_val = ind.get('macd_hist')
                if macd_hist_val is not None:
                    score += 1.0 if macd_hist_val > 0 else 0.0
                    components += 1

                plus_di_val = ind.get('plus_di')
                minus_di_val = ind.get('minus_di')
                if plus_di_val is not None and minus_di_val is not None:
                    score += 1.0 if plus_di_val > minus_di_val else 0.0
                    components += 1

                if components > 0:
                    trend_score = round(score / components, 3)
            except Exception:
                pass
            return trend_score

        symbol_indicators = {}
        symbol_trend_scores: Dict[str, float] = {}
        for sym in sorted_by_vol:
            sym_inds = batch_indicators.get(sym, {})
            symbol_indicators[sym] = sym_inds
            symbol_trend_scores[sym] = _compute_trend_score(sym, sym_inds)

        # Ensure all sample_pairs have a trend score even if OHLCV was missing
        for sym in sample_pairs:
            if sym not in symbol_trend_scores:
                symbol_trend_scores[sym] = 0.0

        # Use asset info for minimum order size constraints
        market_limits = {}
        for symbol in sample_pairs:
            base = symbol.split('/')[0]
            try:
                asset = await self._get_asset_info(symbol)
                min_amount = float(asset.min_order_size) if asset.min_order_size else None
            except Exception:
                min_amount = None
            ticker = tickers.get(symbol, {})
            last_price = ticker.get('last', 0)
            if min_amount is not None and last_price:
                numeric_min_cost = min_amount * last_price
            else:
                numeric_min_cost = 0.0
            market_limits[symbol] = {
                'min_cost': numeric_min_cost,
                'min_amount': min_amount,
            }

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

        # Compute pairwise correlation matrix from OHLCV close prices (run in thread to avoid blocking)
        def _compute_correlation_matrix():
            corr_matrix: Dict[str, Dict[str, float]] = {}
            if ohlcv_data and settings.OHLCV_TIMEFRAMES:
                # Try timeframes from longest to shortest. For long timeframes
                # (5Y, 3Y, 1Y) with very few candles, fall back to shorter
                # timeframes that have more candles. Require a minimum of 20
                # data points so the Pearson correlation is statistically
                # significant.
                MIN_CANDLES = 20
                MIN_RETURNS = 19

                returns_series: Dict[str, List[float]] = {}
                used_tf = None
                for tf in settings.OHLCV_TIMEFRAMES:
                    close_series: Dict[str, List[float]] = {}
                    for sym in sorted_by_vol:
                        if sym in ohlcv_data and tf in ohlcv_data[sym]:
                            candles = ohlcv_data[sym][tf]
                            if len(candles) >= MIN_CANDLES:
                                close_series[sym] = [c[4] for c in candles]
                    # Compute percentage returns
                    candidate_returns: Dict[str, List[float]] = {}
                    for sym, closes in close_series.items():
                        returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
                                   for i in range(1, len(closes)) if closes[i - 1] != 0]
                        if len(returns) >= MIN_RETURNS:
                            candidate_returns[sym] = returns
                    # Use this timeframe if at least 2 symbols have enough data
                    if len(candidate_returns) >= 2:
                        returns_series = candidate_returns
                        used_tf = tf
                        break

                if used_tf:
                    logger.debug(
                        f"Correlation matrix computed using {used_tf} timeframe "
                        f"({len(returns_series)} symbols)"
                    )
                # Pairwise Pearson correlation
                corr_symbols = list(returns_series.keys())
                for sym_a in corr_symbols:
                    corr_matrix[sym_a] = {}
                    for sym_b in corr_symbols:
                        if sym_a == sym_b:
                            corr_matrix[sym_a][sym_b] = 1.0
                        elif sym_b in corr_matrix and sym_a in corr_matrix[sym_b]:
                            corr_matrix[sym_a][sym_b] = corr_matrix[sym_b][sym_a]
                        else:
                            ret_a = returns_series[sym_a]
                            ret_b = returns_series[sym_b]
                            min_len = min(len(ret_a), len(ret_b))
                            if min_len < 2:
                                continue
                            a = ret_a[-min_len:]
                            b = ret_b[-min_len:]
                            mean_a = sum(a) / min_len
                            mean_b = sum(b) / min_len
                            cov = sum((a[k] - mean_a) * (b[k] - mean_b) for k in range(min_len)) / min_len
                            std_a = (sum((x - mean_a) ** 2 for x in a) / min_len) ** 0.5
                            std_b = (sum((x - mean_b) ** 2 for x in b) / min_len) ** 0.5
                            if std_a > 0 and std_b > 0:
                                corr_matrix[sym_a][sym_b] = round(cov / (std_a * std_b), 3)
            return corr_matrix

        logger.info("Re-evaluation step 10/12: Computing correlation matrix and performance metrics...")
        # Cache correlation matrix in Redis for 30 minutes (it changes slowly)
        corr_cache_key = "reeval:correlation_matrix"
        correlation_matrix = None
        try:
            cached_corr = await asyncio.to_thread(self.redis.get, corr_cache_key)
            if cached_corr:
                correlation_matrix = json.loads(cached_corr)
        except Exception:
            pass
        if correlation_matrix is None:
            correlation_matrix = await asyncio.to_thread(_compute_correlation_matrix)
            # Dynamic TTL: shorter during high-volatility / extreme market conditions
            corr_ttl = 1800  # default 30 minutes
            _mb = getattr(self, '_market_breadth', None)
            if _mb:
                pos_pct = _mb.get("positive_pct", 50)
                if pos_pct > 80 or pos_pct < 20:
                    corr_ttl = 600  # 10 minutes during extreme breadth
            _fmb = None
            try:
                _fmb_raw = await asyncio.to_thread(self.redis.get, "market:breadth:full")
                if _fmb_raw:
                    _fmb = json.loads(_fmb_raw)
            except Exception:
                pass
            if _fmb:
                pos_pct = _fmb.get("positive_pct", 50)
                if pos_pct > 80 or pos_pct < 20:
                    corr_ttl = 600
            try:
                await asyncio.to_thread(
                    self.redis.setex, corr_cache_key, corr_ttl, json.dumps(correlation_matrix)
                )
            except Exception:
                pass

        perf = await asyncio.to_thread(self._compute_performance_metrics)
        trade_pattern_analysis = await asyncio.to_thread(self._compute_trade_pattern_analysis)

        # --- Composite opportunity score (trend + sentiment) ---
        composite_scores: Dict[str, float] = {}
        for sym in sample_pairs:
            trend = symbol_trend_scores.get(sym, 0.0)
            # Normalise sentiment compound to 0-1 (assuming range -1..1)
            base_sym = sym.split("/")[0] if "/" in sym else sym
            sent = news_sentiment.get(base_sym, {}).get("avg_compound", 0.0) if news_sentiment else 0.0
            sentiment_score = (sent + 1.0) / 2.0  # map -1..1 to 0..1
            composite = 0.6 * trend + 0.4 * sentiment_score
            composite_scores[sym] = round(composite, 3)

        # Build a shortlist for the LLM: all symbols sorted by composite score,
        # plus any currently held symbols and historically best symbols.
        sorted_by_composite = sorted(sample_pairs, key=lambda s: composite_scores.get(s, 0), reverse=True)
        shortlist = sorted_by_composite

        # Always include currently held symbols (they must be managed)
        for entry in self.current_symbols:
            sym = entry["symbol"]
            if sym in sample_pairs and sym not in shortlist:
                shortlist.append(sym)

        # Always include historically best symbols (from trade pattern analysis)
        if trade_pattern_analysis:
            best_syms = [item["symbol"] for item in trade_pattern_analysis.get("best_symbols", [])]
            for sym in best_syms:
                if sym in sample_pairs and sym not in shortlist:
                    shortlist.append(sym)

        # Always include the configured ETFs
        for etf in settings.ALWAYS_INCLUDE_ETFS:
            pair = f"{etf}/{self.base_currency}"
            if pair in sample_pairs and pair not in shortlist:
                shortlist.append(pair)

        # Always include ALL discovered ETFs for the LLM to consider
        for sym in etf_pairs:
            if sym not in shortlist:
                shortlist.append(sym)

        # Always include all BTPs for the LLM to consider
        for sym in btp_pairs:
            if sym not in shortlist:
                shortlist.append(sym)

        # Deduplicate shortlist while preserving order. The sample_pairs
        # reconstruction (stocks + ETFs + BTPs) can introduce duplicates
        # when a symbol appears in multiple category lists, and those
        # duplicates propagate into shortlist via the initial
        # sorted_by_composite assignment.
        seen = set()
        shortlist = [s for s in shortlist if not (s in seen or seen.add(s))]

        sample_pairs = shortlist
        logger.info(f"LLM candidate list: {len(sample_pairs)} symbols (will be evaluated in chunks)")

        # --- Ensure tickers dict covers all symbols in the final shortlist ---
        # The tickers dict was built from the original sample_pairs before
        # shortlist added ETFs, BTPs, and historical best symbols.  Re-fetch
        # quotes for any shortlist entries missing from tickers so the LLM
        # prompt includes them in the ticker_summary section.
        missing_tickers = [s for s in sample_pairs if s not in tickers or not tickers.get(s, {}).get('last')]
        if missing_tickers:
            missing_plain = [s.split("/")[0] for s in missing_tickers]
            try:
                extra_raw = await asyncio.to_thread(get_quotes_cached, missing_plain)
                for pair in missing_tickers:
                    base = pair.split("/")[0]
                    if base in extra_raw and extra_raw[base].get('last'):
                        tickers[pair] = extra_raw[base]
            except Exception as e:
                logger.warning(f"Failed to fetch missing tickers for shortlist: {e}")

        # --- Detect upcoming corporate events from news (parallelized) ---
        symbol_events: Dict[str, Dict[str, Any]] = {}
        if settings.NEWS_ENABLED and detect_upcoming_events is not None:
            async def _detect_event(sym: str):
                try:
                    event = await asyncio.to_thread(detect_upcoming_events, sym)
                    if event:
                        return sym, event
                except Exception:
                    pass
                return sym, None

            event_tasks = [_detect_event(sym) for sym in sample_pairs]
            event_results = await asyncio.gather(*event_tasks)
            for sym, event in event_results:
                if event:
                    symbol_events[sym] = event
        session_info = self._get_session_info()

        # Market breadth: percentage of candidate stocks with positive 24h change
        positive_count = sum(1 for sym in sample_pairs if (tickers.get(sym, {}).get('percentage') or 0) > 0)
        total_count = len(sample_pairs)
        market_breadth = {
            "positive_pct": round(positive_count / total_count * 100, 1) if total_count > 0 else 0.0,
            "positive_count": positive_count,
            "total_count": total_count,
        }
        self._market_breadth = market_breadth

        # Read full market breadth from Redis (computed by background task)
        full_market_breadth = None
        try:
            full_breadth_raw = await asyncio.to_thread(self.redis.get, "market:breadth:full")
            if full_breadth_raw:
                full_market_breadth = json.loads(full_breadth_raw)
        except Exception:
            pass

        vix = await self._fetch_vix()
        # Store market status in Redis for the web dashboard
        market_status = {
            "vix": vix,
            "market_breadth": market_breadth,
            "full_market_breadth": full_market_breadth,
            "spy_price": market_trend["last"] if market_trend else None,
            "timestamp": time.time(),
        }
        await asyncio.to_thread(self.redis.setex, "market:status", 3600, json.dumps(market_status))

        # Check if trading is currently paused
        trading_paused_raw = await asyncio.to_thread(self.redis.get, "trading:paused")
        trading_paused_bool = trading_paused_raw is not None and trading_paused_raw == "1"

        # Compute symbol tenure for the prompt
        symbol_tenure = {}
        for sym, first_seen in self._symbol_first_seen.items():
            symbol_tenure[sym] = round(now - first_seen)

        # Compute current max tenure per symbol for the prompt
        symbol_max_tenure = {}
        for entry in self.current_symbols:
            if 'max_tenure_hours' in entry:
                symbol_max_tenure[entry['symbol']] = entry['max_tenure_hours']

        # --- Warn if trading was recently auto-resumed ---
        auto_resume_note = ""
        last_auto_resume_raw = await asyncio.to_thread(self.redis.get, "trading:last_auto_resume")
        if last_auto_resume_raw:
            try:
                last_auto_resume_ts = float(last_auto_resume_raw)
                seconds_since = now - last_auto_resume_ts
                if seconds_since < self._symbol_reevaluation_interval * 2:
                    minutes_since = seconds_since / 60
                    auto_resume_note = (
                        f"\n**NOTE:** Trading was auto‑resumed {minutes_since:.1f} minutes ago after a pause. "
                        "Market conditions may not have changed significantly. "
                        "Consider whether conditions have actually improved enough to justify trading. "
                        "If you decide to pause again, set a longer `pause_duration_seconds` (e.g., 1800–7200) "
                        "to allow conditions to evolve; a very short pause will likely lead to the same outcome.\n"
                    )
            except (ValueError, TypeError):
                pass

        # Compute OHLCV summary for the prompt (do not pass raw candles to the LLM)
        ohlcv_summary = {}
        if ohlcv_data:
            for symbol in sample_pairs:
                if symbol in ohlcv_data:
                    tf_data = ohlcv_data[symbol]
                    summary = {}
                    for tf, candles in tf_data.items():
                        if not candles:
                            continue
                        open_price = candles[0][1]
                        close_price = candles[-1][4]
                        high = max(c[2] for c in candles)
                        low = min(c[3] for c in candles)
                        volume = sum(c[5] for c in candles)
                        change_pct = ((close_price - open_price) / open_price) * 100 if open_price else 0
                        summary[tf] = {
                            "change_pct": round(change_pct, 2),
                            "high": high,
                            "low": low,
                            "volume": volume,
                        }
                    ohlcv_summary[symbol] = summary

        max_retries = 2
        # Compute prompt complexity for temperature selection
        _st_values = [abs(v) for v in sentiment_trend.values() if v is not None]
        _st_mag = max(_st_values) if _st_values else None
        symbol_selection_complexity = self._compute_prompt_complexity(
            num_candidates=len(sample_pairs),
            market_breadth=market_breadth,
            fear_greed=None,
            volatility_percentile=None,
            sentiment_trend_magnitude=_st_mag,
            conflicting_signals=False,
            is_critical=False,
        )
        effective_temp = self._get_effective_temperature("mind", symbol_selection_complexity)

        # --- Chunked LLM evaluation ---
        CHUNK_SIZE = settings.LLM_CHUNK_SIZE
        chunk_results: List[Dict[str, Any]] = []
        chunks = [sample_pairs[i:i + CHUNK_SIZE] for i in range(0, len(sample_pairs), CHUNK_SIZE)]
        total_steps = 10 + len(chunks) + 2
        logger.info("Re-evaluation step 11/%d: Evaluating %d chunks of ~%d symbols each...", total_steps, len(chunks), CHUNK_SIZE)

        for chunk_idx, chunk_symbols in enumerate(chunks):
            chunk_set = set(chunk_symbols)

            # Filter per-symbol data to chunk symbols
            chunk_tickers = {s: tickers.get(s, {}) for s in chunk_symbols}
            chunk_ohlcv_summary = {s: ohlcv_summary.get(s, {}) for s in chunk_symbols if s in ohlcv_summary}
            chunk_symbol_indicators = {s: symbol_indicators.get(s, {}) for s in chunk_symbols if s in symbol_indicators}
            chunk_market_limits = {s: market_limits.get(s, {}) for s in chunk_symbols if s in market_limits}
            chunk_symbol_events = {s: symbol_events.get(s, {}) for s in chunk_symbols if s in symbol_events}
            chunk_symbol_trend_scores = {s: symbol_trend_scores.get(s, 0.0) for s in chunk_symbols}
            chunk_sentiment_trend = {s.split("/")[0]: sentiment_trend.get(s.split("/")[0]) for s in chunk_symbols if s.split("/")[0] in sentiment_trend}

            # Filter correlation matrix to chunk symbols
            chunk_corr = {}
            if correlation_matrix:
                for sym_a, row in correlation_matrix.items():
                    if sym_a in chunk_set:
                        chunk_corr[sym_a] = {sym_b: v for sym_b, v in row.items() if sym_b in chunk_set}

            # Build chunk prompt
            chunk_prompt = await asyncio.to_thread(
                build_stock_selection_prompt,
                available_symbols=chunk_symbols,
                current_symbols=self.current_symbols,
                max_symbols=self.effective_max_symbols,
                base_currency=self.base_currency,
                tickers=chunk_tickers,
                base_balance=base_balance,
                per_symbol_budget=per_symbol_budget,
                market_limits=chunk_market_limits,
                performance=perf,
                ohlcv_summary=chunk_ohlcv_summary,
                market_trend=market_trend,
                symbol_indicators=chunk_symbol_indicators,
                daily_pnl=perf["equity_curve"].get("daily_pnl"),
                correlation_matrix=chunk_corr if chunk_corr else None,
                session_info=session_info,
                sentiment_trend=chunk_sentiment_trend,
                trading_paused=trading_paused_bool,
                open_positions=self.positions,
                symbol_tenure=symbol_tenure,
                symbol_max_tenure=symbol_max_tenure,
                vix=vix,
                trade_pattern_analysis=trade_pattern_analysis,
                symbol_events=chunk_symbol_events,
                symbol_trend_scores=chunk_symbol_trend_scores,
                market_breadth=market_breadth,
                min_viable_trade_amount=min_viable_amount,
            )
            if auto_resume_note:
                chunk_prompt += "\n" + auto_resume_note

            # Build market snapshot for caching
            chunk_market_snapshot = {
                "chunk_idx": chunk_idx,
                "available_pairs": chunk_symbols,
                "tickers": chunk_tickers,
                "ohlcv_data": {s: ohlcv_data.get(s, {}) for s in chunk_symbols},
                "symbol_indicators": chunk_symbol_indicators,
                "performance": perf,
                "session_info": session_info,
                "market_breadth": market_breadth,
                "trading_paused": trading_paused_bool,
                "open_positions": self.positions,
                "base_balance": base_balance,
                "per_symbol_budget": per_symbol_budget,
                "current_symbols": self.current_symbols,
            }
            chunk_market_hash = compute_market_hash(chunk_market_snapshot)

            # Call LLM for this chunk
            chunk_response = None
            for attempt in range(max_retries + 1):
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            get_cached_llm_response,
                            compact_prompt(chunk_prompt),
                            COMPACTED_SYSTEM_PROMPT,
                            300,
                            market_hash=chunk_market_hash,
                            model_type="mind",
                            temperature=effective_temp,
                        ),
                        timeout=settings.LLM_TIMEOUT
                    )
                    chunk_response = result["response"]
                    break
                except asyncio.TimeoutError:
                    if attempt < max_retries:
                        logger.warning(f"Chunk {chunk_idx + 1}/{len(chunks)} LLM timed out (attempt {attempt + 1}). Retrying...")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        logger.warning(f"Chunk {chunk_idx + 1}/{len(chunks)} LLM timed out after all retries. Skipping.")
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Chunk {chunk_idx + 1}/{len(chunks)} LLM failed: {e}. Retrying...")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        logger.error(f"Chunk {chunk_idx + 1}/{len(chunks)} LLM failed after all retries: {e}")

            if chunk_response:
                try:
                    chunk_parsed = json.loads(chunk_response)
                    chunk_results.append(chunk_parsed)
                    logger.info("Chunk %d/%d: received %d symbol selections", chunk_idx + 1, len(chunks), len(chunk_parsed.get("stocks", [])))
                except json.JSONDecodeError:
                    logger.warning(f"Chunk {chunk_idx + 1}/{len(chunks)}: invalid JSON, retrying with correction.")
                    correction = (
                        "Your previous response was not valid JSON. "
                        "You MUST output ONLY a single JSON object, with no markdown fences, no explanations, no extra text. "
                        "Here is the original request:\n\n" + chunk_prompt
                    )
                    try:
                        correction_result = await asyncio.wait_for(
                            asyncio.to_thread(
                                get_cached_llm_response, compact_prompt(correction), COMPACTED_SYSTEM_PROMPT, 120,
                                model_type="actuator", temperature=effective_temp,
                            ),
                            timeout=settings.LLM_TIMEOUT
                        )
                        chunk_parsed = json.loads(correction_result["response"])
                        chunk_results.append(chunk_parsed)
                        logger.info("Chunk %d/%d: corrected, received %d selections", chunk_idx + 1, len(chunks), len(chunk_parsed.get("stocks", [])))
                    except Exception as e:
                        logger.warning(f"Chunk {chunk_idx + 1}/{len(chunks)}: correction also failed: {e}")
            else:
                logger.warning(f"Chunk {chunk_idx + 1}/{len(chunks)}: no response, skipping.")

            await asyncio.sleep(1)

        # --- Final selection call ---
        logger.info("Re-evaluation step %d/%d: Calling LLM for final selection from %d chunk results...", total_steps - 1, total_steps, len(chunk_results))

        response = None
        llm_provider = None
        llm_model = None

        if not chunk_results:
            logger.warning("All chunk LLM calls failed. Will use fallback selection.")
        else:
            final_prompt = await asyncio.to_thread(
                build_final_selection_prompt,
                chunk_results=chunk_results,
                current_symbols=self.current_symbols,
                max_symbols=self.effective_max_symbols,
                base_currency=self.base_currency,
                base_balance=base_balance,
                per_symbol_budget=per_symbol_budget,
                performance=perf,
                open_positions=self.positions,
                market_breadth=market_breadth,
                full_market_breadth=full_market_breadth,
                market_trend=market_trend,
                session_info=session_info,
                trading_paused=trading_paused_bool,
                symbol_tenure=symbol_tenure,
                symbol_max_tenure=symbol_max_tenure,
                trade_pattern_analysis=trade_pattern_analysis,
                daily_pnl=perf["equity_curve"].get("daily_pnl"),
                vix=vix,
                min_viable_trade_amount=min_viable_amount,
                available_timeframes=settings.OHLCV_TIMEFRAMES,
                market_limits=market_limits,
                available_timeframes_by_symbol=available_timeframes_by_symbol,
            )
            if auto_resume_note:
                final_prompt += "\n" + auto_resume_note

            for attempt in range(max_retries + 1):
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            get_cached_llm_response,
                            compact_prompt(final_prompt),
                            COMPACTED_SYSTEM_PROMPT,
                            300,
                            model_type="mind",
                            temperature=effective_temp,
                        ),
                        timeout=settings.LLM_TIMEOUT
                    )
                    response = result["response"]
                    llm_provider = result["provider"]
                    llm_model = result["model"]
                    break
                except asyncio.TimeoutError:
                    if attempt < max_retries:
                        logger.warning(f"Final selection LLM timed out (attempt {attempt + 1}). Retrying...")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        logger.warning("Final selection LLM timed out after all retries.")
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Final selection LLM failed: {e}. Retrying...")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        logger.error(f"Final selection LLM failed after all retries: {e}")

            # Fallback: merge all chunk selections if final call failed
            if response is None and chunk_results:
                logger.warning("Final selection LLM call failed. Merging all chunk selections as fallback.")
                merged_stocks = []
                for chunk in chunk_results:
                    for stock in chunk.get("stocks", []):
                        if isinstance(stock, dict) and "symbol" in stock:
                            merged_stocks.append(stock)
                seen = set()
                deduped = []
                for s in merged_stocks:
                    if s["symbol"] not in seen:
                        seen.add(s["symbol"])
                        deduped.append(s)
                response = json.dumps({
                    "stocks": deduped[:self.effective_max_symbols],
                    "max_stocks": min(len(deduped), self.effective_max_symbols),
                    "reasoning": "Fallback: merged all chunk selections (final LLM call failed)",
                })
                llm_provider = "fallback"
                llm_model = "merged_chunks"

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
                logger.warning("LLM symbol selection response was not valid JSON. Retrying with correction prompt.")
                correction_prompt = (
                    "Your previous response was not valid JSON. "
                    "You MUST output ONLY a single JSON object, with no markdown fences, no explanations, no extra text. "
                    "Here is the original request:\n\n" + final_prompt
                )
                try:
                    correction_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            get_cached_llm_response, compact_prompt(correction_prompt), COMPACTED_SYSTEM_PROMPT, 120,
                            model_type="actuator",
                            temperature=effective_temp,
                        ),
                        timeout=settings.LLM_TIMEOUT
                    )
                    response = correction_result["response"]
                    llm_provider = correction_result["provider"]
                    llm_model = correction_result["model"]
                    json.loads(response)  # validate the retry response
                except Exception as e:
                    logger.error(f"LLM symbol selection still invalid after retry: {e}")
                    response = None

        if response is not None:
            try:
                parsed = json.loads(response)
                new_symbols: List[Dict[str, str]] = []

                if isinstance(parsed, dict):
                    # New format: {"stocks": [...], "max_stocks": N}
                    stocks_list = parsed.get("stocks", [])
                    llm_max_stocks = parsed.get("max_stocks")
                    if not isinstance(stocks_list, list):
                        logger.error("LLM symbol selection 'stocks' field is not a list.")
                        stocks_list = []
                    for item in stocks_list:
                        if isinstance(item, dict) and "symbol" in item:
                            sym = item["symbol"]
                            normalized = self._normalize_llm_symbol(sym, sample_pairs)
                            if normalized:
                                sym = normalized
                                tf = item.get("timeframe")
                                if tf not in settings.OHLCV_TIMEFRAMES:
                                    tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                                entry = {"symbol": sym, "timeframe": tf}
                                sector = item.get("sector")
                                if sector:
                                    entry["sector"] = sector
                                mth = item.get("max_tenure_hours")
                                if mth is not None:
                                    entry["max_tenure_hours"] = mth
                                new_symbols.append(entry)
                        elif isinstance(item, str):
                            normalized = self._normalize_llm_symbol(item, sample_pairs)
                            if normalized:
                                default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                                new_symbols.append({"symbol": normalized, "timeframe": default_tf})
                elif isinstance(parsed, list):
                    # Old format: plain list of objects or strings
                    for item in parsed:
                        if isinstance(item, dict) and "symbol" in item:
                            sym = item["symbol"]
                            normalized = self._normalize_llm_symbol(sym, sample_pairs)
                            if normalized:
                                sym = normalized
                                tf = item.get("timeframe")
                                if tf not in settings.OHLCV_TIMEFRAMES:
                                    tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                                entry = {"symbol": sym, "timeframe": tf}
                                sector = item.get("sector")
                                if sector:
                                    entry["sector"] = sector
                                mth = item.get("max_tenure_hours")
                                if mth is not None:
                                    entry["max_tenure_hours"] = mth
                                new_symbols.append(entry)
                        elif isinstance(item, str):
                            normalized = self._normalize_llm_symbol(item, sample_pairs)
                            if normalized:
                                default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                                new_symbols.append({"symbol": normalized, "timeframe": default_tf})
                else:
                    logger.error("LLM symbol selection response is neither a list nor a dict.")

                # Deduplicate by symbol, keeping first occurrence
                seen = set()
                deduped = []
                for entry in new_symbols:
                    sym = entry["symbol"]
                    if sym not in seen:
                        seen.add(sym)
                        deduped.append(entry)

                # Remove excluded pairs
                deduped = [
                    e for e in deduped
                    if not self._is_excluded(e["symbol"], e["timeframe"])
                ]

                # Validate that each selected symbol/timeframe has OHLCV data;
                # fall back to an available timeframe or skip the symbol entirely
                validated_deduped = []
                for entry in deduped:
                    sym = entry["symbol"]
                    tf = entry["timeframe"]
                    sym_data = ohlcv_data.get(sym, {})
                    if tf in sym_data and sym_data[tf]:
                        validated_deduped.append(entry)
                    else:
                        available_tfs = [t for t in settings.OHLCV_TIMEFRAMES if t in sym_data and sym_data[t]]
                        if available_tfs:
                            entry["timeframe"] = available_tfs[0]
                            validated_deduped.append(entry)
                            logger.info(f"No OHLCV data for {sym} on {tf}, falling back to {available_tfs[0]}")
                        else:
                            logger.warning(f"Skipping {sym}: no OHLCV data available for any timeframe")
                deduped = validated_deduped

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

                # --- Enforce minimum symbols (unless LLM explicitly paused) ---
                if (
                    settings.MIN_SYMBOLS > 0
                    and pause_trading is not True
                    and self.effective_max_symbols < settings.MIN_SYMBOLS
                    and len(deduped) >= settings.MIN_SYMBOLS
                ):
                    logger.info(
                        f"LLM selected {self.effective_max_symbols} symbols; "
                        f"enforcing MIN_SYMBOLS={settings.MIN_SYMBOLS}"
                    )
                    self.effective_max_symbols = settings.MIN_SYMBOLS

                # --- Fallback: fill remaining slots if LLM returned fewer than MIN_SYMBOLS ---
                if (
                    settings.MIN_SYMBOLS > 0
                    and pause_trading is not True
                    and len(deduped) < settings.MIN_SYMBOLS
                ):
                    # Try to fill remaining slots from composite-score-sorted sample_pairs
                    existing_syms = {e["symbol"] for e in deduped}
                    default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
                    needed = settings.MIN_SYMBOLS - len(deduped)
                    filled = 0
                    for sym in sorted_by_composite:
                        if filled >= needed:
                            break
                        if sym in existing_syms:
                            continue
                        if self._is_excluded(sym, default_tf):
                            continue
                        # Check if we can afford the minimum trade cost
                        min_cost = market_limits.get(sym, {}).get("min_cost", 0)
                        if base_balance >= min_cost:
                            deduped.append({"symbol": sym, "timeframe": default_tf})
                            existing_syms.add(sym)
                            filled += 1
                    if filled > 0:
                        logger.info(
                            f"LLM returned only {len(deduped) - filled} symbols; "
                            f"filled {filled} additional slots from composite scores to reach MIN_SYMBOLS={settings.MIN_SYMBOLS}"
                        )
                        self.effective_max_symbols = max(self.effective_max_symbols, len(deduped))

                # Parse max_positions_per_sector from LLM
                max_positions_per_sector = parsed.get("max_positions_per_sector")
                if max_positions_per_sector is not None and isinstance(max_positions_per_sector, int) and max_positions_per_sector > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:max_positions_per_sector", 7 * 24 * 3600, str(max_positions_per_sector))
                    logger.info(f"LLM set max positions per sector to {max_positions_per_sector}")
                else:
                    # Fallback if not provided: remove the limit
                    await asyncio.to_thread(self.redis.delete, "trading:max_positions_per_sector")

                # Parse LLM-decided portfolio risk thresholds
                max_port_exp = parsed.get("max_portfolio_exposure_pct")
                if max_port_exp is not None and isinstance(max_port_exp, (int, float)) and 0.0 <= float(max_port_exp) <= 1.0:
                    await asyncio.to_thread(self.redis.setex, "trading:max_portfolio_exposure_pct", 7 * 24 * 3600, str(float(max_port_exp)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:max_portfolio_exposure_pct")

                max_port_risk = parsed.get("max_portfolio_stop_risk_pct")
                if max_port_risk is not None and isinstance(max_port_risk, (int, float)) and 0.0 <= float(max_port_risk) <= 1.0:
                    await asyncio.to_thread(self.redis.setex, "trading:max_portfolio_stop_risk_pct", 7 * 24 * 3600, str(float(max_port_risk)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:max_portfolio_stop_risk_pct")

                min_rr = parsed.get("min_risk_reward_ratio")
                if min_rr is not None and isinstance(min_rr, (int, float)) and min_rr > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:min_risk_reward_ratio", 7 * 24 * 3600, str(float(min_rr)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:min_risk_reward_ratio")

                conf_rejection = parsed.get("confidence_rejection_threshold")
                if conf_rejection is not None and isinstance(conf_rejection, (int, float)) and 0.0 <= float(conf_rejection) <= 1.0:
                    await asyncio.to_thread(self.redis.setex, "trading:confidence_rejection_threshold", 7 * 24 * 3600, str(float(conf_rejection)))
                    logger.info(f"LLM set confidence rejection threshold to {float(conf_rejection):.2f}")
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:confidence_rejection_threshold")

                # Parse LLM-controlled limit price max distance
                limit_price_max_dist = parsed.get("limit_price_max_distance_pct")
                if limit_price_max_dist is not None and isinstance(limit_price_max_dist, (int, float)) and 0.0 <= float(limit_price_max_dist) <= 1.0:
                    await asyncio.to_thread(self.redis.setex, "trading:limit_price_max_distance_pct", 7 * 24 * 3600, str(float(limit_price_max_dist)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:limit_price_max_distance_pct")

                # Parse LLM-controlled minimum viable trade amount
                min_viable = parsed.get("min_viable_trade_amount")
                if min_viable is not None and isinstance(min_viable, (int, float)) and min_viable > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:min_viable_trade_amount", 7 * 24 * 3600, str(float(min_viable)))
                    logger.info(f"LLM set min viable trade amount to {float(min_viable):.2f}")
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:min_viable_trade_amount")

                # Parse LLM evaluation skip thresholds
                skip_price_mult = parsed.get("skip_eval_price_change_atr_mult")
                if skip_price_mult is not None and isinstance(skip_price_mult, (int, float)) and skip_price_mult > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:skip_eval_price_change_atr_mult", 7 * 24 * 3600, str(float(skip_price_mult)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:skip_eval_price_change_atr_mult")

                skip_rsi = parsed.get("skip_eval_rsi_change")
                if skip_rsi is not None and isinstance(skip_rsi, (int, float)) and skip_rsi > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:skip_eval_rsi_change", 7 * 24 * 3600, str(float(skip_rsi)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:skip_eval_rsi_change")

                skip_rsi_oversold = parsed.get("skip_eval_rsi_oversold")
                if skip_rsi_oversold is not None and isinstance(skip_rsi_oversold, (int, float)) and skip_rsi_oversold > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:skip_eval_rsi_oversold", 7 * 24 * 3600, str(float(skip_rsi_oversold)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:skip_eval_rsi_oversold")

                skip_rsi_overbought = parsed.get("skip_eval_rsi_overbought")
                if skip_rsi_overbought is not None and isinstance(skip_rsi_overbought, (int, float)) and skip_rsi_overbought > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:skip_eval_rsi_overbought", 7 * 24 * 3600, str(float(skip_rsi_overbought)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:skip_eval_rsi_overbought")

                skip_macd = parsed.get("skip_eval_macd_hist_change")
                if skip_macd is not None and isinstance(skip_macd, (int, float)) and skip_macd > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:skip_eval_macd_hist_change", 7 * 24 * 3600, str(float(skip_macd)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:skip_eval_macd_hist_change")

                # Parse LLM-driven market regime thresholds
                regime_adx_strong = parsed.get("regime_adx_strong")
                if regime_adx_strong is not None and isinstance(regime_adx_strong, (int, float)) and regime_adx_strong > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:regime_adx_strong", 7 * 24 * 3600, str(float(regime_adx_strong)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:regime_adx_strong")

                regime_adx_moderate = parsed.get("regime_adx_moderate")
                if regime_adx_moderate is not None and isinstance(regime_adx_moderate, (int, float)) and regime_adx_moderate > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:regime_adx_moderate", 7 * 24 * 3600, str(float(regime_adx_moderate)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:regime_adx_moderate")

                regime_vol_high = parsed.get("regime_volatility_high_pct")
                if regime_vol_high is not None and isinstance(regime_vol_high, (int, float)) and regime_vol_high > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:regime_volatility_high_pct", 7 * 24 * 3600, str(float(regime_vol_high)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:regime_volatility_high_pct")

                regime_vol_low = parsed.get("regime_volatility_low_pct")
                if regime_vol_low is not None and isinstance(regime_vol_low, (int, float)) and regime_vol_low > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:regime_volatility_low_pct", 7 * 24 * 3600, str(float(regime_vol_low)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:regime_volatility_low_pct")

                regime_bb_squeeze = parsed.get("regime_bb_squeeze_width")
                if regime_bb_squeeze is not None and isinstance(regime_bb_squeeze, (int, float)) and regime_bb_squeeze > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:regime_bb_squeeze_width", 7 * 24 * 3600, str(float(regime_bb_squeeze)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:regime_bb_squeeze_width")

                regime_bb_expansion = parsed.get("regime_bb_expansion_width")
                if regime_bb_expansion is not None and isinstance(regime_bb_expansion, (int, float)) and regime_bb_expansion > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:regime_bb_expansion_width", 7 * 24 * 3600, str(float(regime_bb_expansion)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:regime_bb_expansion_width")

                min_stop_atr_mult = parsed.get("min_stop_loss_atr_mult")
                if min_stop_atr_mult is not None and isinstance(min_stop_atr_mult, (int, float)) and min_stop_atr_mult > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:min_stop_loss_atr_mult", 7 * 24 * 3600, str(float(min_stop_atr_mult)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:min_stop_loss_atr_mult")

                min_hold_time_mult = parsed.get("min_max_hold_time_mult")
                if min_hold_time_mult is not None and isinstance(min_hold_time_mult, (int, float)) and min_hold_time_mult > 0:
                    await asyncio.to_thread(self.redis.setex, "trading:min_max_hold_time_mult", 7 * 24 * 3600, str(float(min_hold_time_mult)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:min_max_hold_time_mult")

                max_sl_reviews = parsed.get("max_stop_loss_reviews")
                if max_sl_reviews is not None and isinstance(max_sl_reviews, int) and 1 <= max_sl_reviews <= 20:
                    await asyncio.to_thread(self.redis.setex, "trading:max_stop_loss_reviews", 7 * 24 * 3600, str(max_sl_reviews))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:max_stop_loss_reviews")

                max_tp_reviews = parsed.get("max_take_profit_reviews")
                if max_tp_reviews is not None and isinstance(max_tp_reviews, int) and 1 <= max_tp_reviews <= 20:
                    await asyncio.to_thread(self.redis.setex, "trading:max_take_profit_reviews", 7 * 24 * 3600, str(max_tp_reviews))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:max_take_profit_reviews")

                min_llm_pause = parsed.get("min_llm_pause_duration_seconds")
                if min_llm_pause is not None and isinstance(min_llm_pause, int) and 300 <= min_llm_pause <= 14400:
                    await asyncio.to_thread(self.redis.setex, "trading:min_llm_pause_duration", 7 * 24 * 3600, str(min_llm_pause))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:min_llm_pause_duration")

                pause_max_keep = parsed.get("pause_max_consecutive_keep")
                if pause_max_keep is not None and isinstance(pause_max_keep, int) and 1 <= pause_max_keep <= 10:
                    await asyncio.to_thread(self.redis.setex, "trading:pause_max_consecutive_keep", 7 * 24 * 3600, str(pause_max_keep))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:pause_max_consecutive_keep")

                pause_force_mult = parsed.get("pause_force_resume_risk_multiplier")
                if pause_force_mult is not None and isinstance(pause_force_mult, (int, float)) and 0.0 <= float(pause_force_mult) <= 1.0:
                    await asyncio.to_thread(self.redis.setex, "trading:pause_force_resume_risk_multiplier", 7 * 24 * 3600, str(float(pause_force_mult)))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:pause_force_resume_risk_multiplier")

                max_partial_tp = parsed.get("max_partial_tp_reviews")
                if max_partial_tp is not None and isinstance(max_partial_tp, int) and 1 <= max_partial_tp <= 20:
                    await asyncio.to_thread(self.redis.setex, "trading:max_partial_tp_reviews", 7 * 24 * 3600, str(max_partial_tp))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:max_partial_tp_reviews")

                max_dust_sweep = parsed.get("max_dust_sweep_reviews")
                if max_dust_sweep is not None and isinstance(max_dust_sweep, int) and 1 <= max_dust_sweep <= 20:
                    await asyncio.to_thread(self.redis.setex, "trading:max_dust_sweep_reviews", 7 * 24 * 3600, str(max_dust_sweep))
                else:
                    await asyncio.to_thread(self.redis.delete, "trading:max_dust_sweep_reviews")

                # Optional: LLM can set the global symbol re-evaluation interval
                new_interval = parsed.get("stock_revaluation_interval_seconds")
                if new_interval is not None:
                    if isinstance(new_interval, (int, float)) and new_interval >= 3600:
                        clamped = max(new_interval, MIN_SYMBOL_REEVALUATION_INTERVAL)
                        self._symbol_reevaluation_interval = clamped
                        logger.info(f"LLM set symbol re-evaluation interval to {clamped}s (requested {new_interval}s)")
                    else:
                        logger.warning(f"Invalid stock_revaluation_interval_seconds: {new_interval} (must be >= 3600)")

                # Optional: LLM can request to pause/resume trading
                # (pause_trading was already extracted above, before MIN_SYMBOLS enforcement)
                pause_reason = parsed.get("pause_reason", "")
                pause_duration = parsed.get("pause_duration_seconds")

                # --- Auto-resume cooldown: ignore pause requests shortly after an auto-resume ---
                cooldown_active = await asyncio.to_thread(self.redis.get, "trading:auto_resume_cooldown")
                if cooldown_active and pause_trading is True:
                    logger.info(
                        "Ignoring LLM pause request because auto‑resume cooldown is active "
                        "(trading was recently auto‑resumed)."
                    )
                    pause_trading = None  # treat as "no decision"

                skip_resume = False
                if pause_trading is not None:
                    if isinstance(pause_trading, bool):
                        if pause_trading:
                            # Only pause if not already manually paused
                            current_source = await asyncio.to_thread(self.redis.get, "trading:pause_source")
                            if current_source and current_source == "manual":
                                logger.info("LLM pause request ignored because trading is manually paused.")
                            else:
                                await asyncio.to_thread(self.redis.set, "trading:paused", "1")
                                await asyncio.to_thread(self.redis.set, "trading:pause_source", "llm")
                                await asyncio.to_thread(self.redis.set, "trading:pause_start", str(time.time()))
                                await asyncio.to_thread(self.redis.set, "trading:llm_pause_time", str(time.time()))
                                # Fallback if LLM did not provide pause_duration_seconds
                                if pause_duration is None:
                                    _min_pause = settings.MIN_LLM_PAUSE_DURATION
                                    try:
                                        raw = await asyncio.to_thread(self.redis.get, "trading:min_llm_pause_duration")
                                        if raw:
                                            _min_pause = int(raw)
                                    except Exception:
                                        pass
                                    pause_duration = _min_pause
                                    await asyncio.to_thread(
                                        self.redis.setex, "trading:pause_duration", 7 * 24 * 3600, str(int(pause_duration))
                                    )
                                if pause_reason:
                                    await asyncio.to_thread(self.redis.set, "trading:pause_reason", pause_reason)
                                logger.info("LLM requested to pause trading.")
                        else:
                            # LLM requests resume – only allowed if the pause was LLM-initiated
                            current_source = await asyncio.to_thread(self.redis.get, "trading:pause_source")
                            if current_source and current_source != "llm":
                                logger.info("LLM resume request ignored because pause was not initiated by LLM.")
                            else:
                                if trading_paused_bool:
                                    # Determine the required pause duration:
                                    # - the LLM-set pause_duration_seconds (if any) stored in Redis
                                    # - but never less than MIN_LLM_PAUSE_DURATION
                                    pause_start_raw = await asyncio.to_thread(self.redis.get, "trading:pause_start")
                                    pause_duration_raw = await asyncio.to_thread(self.redis.get, "trading:pause_duration")
                                    required_pause = settings.MIN_LLM_PAUSE_DURATION
                                    try:
                                        raw = await asyncio.to_thread(self.redis.get, "trading:min_llm_pause_duration")
                                        if raw:
                                            required_pause = int(raw)
                                    except Exception:
                                        pass
                                    if pause_duration_raw:
                                        try:
                                            llm_set_duration = int(pause_duration_raw)
                                            required_pause = max(settings.MIN_LLM_PAUSE_DURATION, llm_set_duration)
                                        except (ValueError, TypeError):
                                            pass

                                    if pause_start_raw:
                                        try:
                                            pause_start = float(pause_start_raw)
                                            elapsed = time.time() - pause_start
                                            if elapsed < required_pause:
                                                remaining = required_pause - elapsed
                                                logger.info(
                                                    f"Ignoring LLM resume request: required pause duration "
                                                    f"({required_pause}s) not yet elapsed ({remaining:.0f}s remaining)."
                                                )
                                                skip_resume = True
                                        except (ValueError, TypeError):
                                            pass
                                    if not skip_resume:
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
                                        logger.info("LLM requested to resume trading.")
                                        self._reeval_trigger.set()
                                else:
                                    # Trading is already active – LLM confirms to keep it active
                                    logger.info("LLM decided to keep trading active (already active).")
                    else:
                        logger.warning(f"Invalid pause_trading value: {pause_trading}")

                # Store LLM-provided pause duration in Redis (if not already stored by pause logic)
                if pause_duration is not None and isinstance(pause_duration, (int, float)) and pause_duration > 0:
                    await asyncio.to_thread(
                        self.redis.setex, "trading:pause_duration", 7 * 24 * 3600, str(int(pause_duration))
                    )
                    logger.info(f"LLM set pause duration: {pause_duration}s")
                elif pause_duration is not None:
                    logger.warning(f"Invalid pause_duration_seconds: {pause_duration}")

                # Optional: LLM can set a global risk multiplier to scale all position sizes
                global_risk_mult = parsed.get("global_risk_multiplier")
                if global_risk_mult is not None:
                    if isinstance(global_risk_mult, (int, float)) and 0.0 <= global_risk_mult <= 1.0:
                        await self._set_global_risk_multiplier(global_risk_mult)
                        logger.info(f"LLM set global risk multiplier: {global_risk_mult}")
                    else:
                        logger.warning(f"Invalid global_risk_multiplier: {global_risk_mult}")

                # Only replace current_symbols if the LLM returned at least one valid symbol.
                # If the LLM returned no symbols (empty list or max_stocks=0), keep the
                # previously tracked symbols so the bot continues to generate signals for them.
                if deduped and self.effective_max_symbols > 0:
                    existing_symbols = {c['symbol']: c for c in self.current_symbols}
                    for entry in deduped[: self.effective_max_symbols]:
                        sym = entry['symbol']
                        new_tf = entry['timeframe']
                        if sym in existing_symbols:
                            old_entry = existing_symbols[sym]
                            if 'entry_time' in old_entry:
                                entry['entry_time'] = old_entry['entry_time']
                            else:
                                entry['entry_time'] = time.time()
                            
                            # Preserve max_tenure_hours from existing symbol if LLM didn't specify it
                            if 'max_tenure_hours' not in entry and 'max_tenure_hours' in old_entry:
                                entry['max_tenure_hours'] = old_entry['max_tenure_hours']
                                
                            # Check if timeframe changed for an existing symbol
                            old_tf = old_entry.get('timeframe')
                            if old_tf != new_tf:
                                logger.info(f"Timeframe changed for {sym}: {old_tf} -> {new_tf}")
                                if sym in self.positions:
                                    self.positions[sym]['timeframe'] = new_tf
                                    # Clear max hold expired flags since the timeframe context changed
                                    self.positions[sym].pop("_max_hold_expired", None)
                                    self.positions[sym].pop("_max_hold_expired_count", None)
                        else:
                            entry['entry_time'] = time.time()
                    self.current_symbols = deduped[: self.effective_max_symbols]
                else:
                    # LLM returned no symbols – keep previously tracked symbols
                    if old_symbols:
                        logger.info("LLM selected 0 symbols. Keeping previously tracked symbols for signal generation.")
                        self.current_symbols = old_symbols
                        self.effective_max_symbols = max(len(old_symbols), 1)
                    else:
                        self.current_symbols = []
                        self.effective_max_symbols = 0
                        logger.info("LLM selected 0 symbols – pausing trading until next evaluation.")

            except json.JSONDecodeError:
                logger.error("Failed to parse symbol selection response.")

        # Fallback: if LLM returned no symbols AND did NOT explicitly pause, pick top affordable symbols by composite score
        if not self.current_symbols and pause_trading is not True:
            logger.warning("LLM returned no symbols without pausing – using composite-score-based fallback.")
            if self.notifier:
                await self.notifier.send_notification(
                    "⚠️ LLM returned no symbols. Using composite-score-based fallback selection.",
                    summary={
                        "action": "FALLBACK",
                        "reason": "LLM returned no symbols, using fallback",
                        "model_type": "mind",
                    }
                )
            # Sort sample_pairs by composite score (already computed above)
            sorted_pairs = sorted(sample_pairs, key=lambda s: composite_scores.get(s, 0), reverse=True)
            fallback_symbols: List[Dict[str, str]] = []
            default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
            for sym in sorted_pairs:
                if composite_scores.get(sym, 0) < settings.FALLBACK_MIN_COMPOSITE_SCORE:
                    continue
                # Apply minimum 24h volume filter if configured
                if settings.FALLBACK_MIN_24H_VOLUME > 0:
                    vol = _volume(sym)
                    if vol < settings.FALLBACK_MIN_24H_VOLUME:
                        continue
                min_cost = market_limits.get(sym, {}).get('min_cost', 0)
                # Use total base_balance, not per_symbol_budget, since the LLM
                # allocates capital dynamically (not equal split)
                if base_balance >= min_cost:
                    if self._is_excluded(sym, default_tf):
                        continue
                    fallback_symbols.append({"symbol": sym, "timeframe": default_tf})
                if len(fallback_symbols) >= self.effective_max_symbols:
                    break
            if fallback_symbols:
                existing_symbols = {c['symbol']: c for c in self.current_symbols}
                for entry in fallback_symbols:
                    if entry['symbol'] in existing_symbols and 'entry_time' in existing_symbols[entry['symbol']]:
                        entry['entry_time'] = existing_symbols[entry['symbol']]['entry_time']
                    else:
                        entry['entry_time'] = time.time()
                self.current_symbols = fallback_symbols
            elif old_symbols:
                logger.warning("Fallback found no symbols. Keeping previously tracked symbols.")
                self.current_symbols = old_symbols
                self.effective_max_symbols = max(len(old_symbols), 1)

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
            for entry in new_symbols:
                sym = entry["symbol"]
                logger.info(f"Triggering immediate news fetch for newly selected symbol {sym}")
                asyncio.create_task(self._fetch_and_store_news_for_symbol(sym))

        # Build formatted symbol labels with stock names (parallelized)
        async def _fetch_label(c):
            name = await self._get_stock_name(c['symbol'])
            return self._format_symbol_display(c['symbol'], name, c['timeframe'])
        symbol_labels = await asyncio.gather(*[_fetch_label(c) for c in self.current_symbols])
        logger.info(f"Selected symbols: {symbol_labels}")

        # Build a pause/resume message if the LLM provided a decision
        pause_msg = ""
        if isinstance(pause_trading, bool):
            if pause_trading:
                if trading_paused_bool:
                    pause_msg = "⏸️ LLM decided to keep trading paused"
                else:
                    pause_msg = "⏸️ LLM decided to pause trading"
            else:
                if trading_paused_bool:
                    pause_msg = "▶️ LLM decided to resume trading"
                else:
                    pause_msg = "▶️ LLM decided to keep trading active"
            if pause_reason:
                pause_msg += f" – {pause_reason}"

        # Include pause duration if set
        if pause_duration is not None and isinstance(pause_duration, (int, float)) and pause_duration > 0:
            minutes = pause_duration / 60
            if minutes >= 1:
                duration_str = f"{minutes:.0f} min"
            else:
                duration_str = f"{pause_duration:.0f}s"
            if pause_msg:
                pause_msg += f" (auto‑resume in {duration_str})"
            else:
                pause_msg = f"⏱️ LLM set pause duration: {duration_str}"

        if force:
            market_open = await self._is_market_open()
            if not market_open:
                status_str = "paused"
                emoji = "⏸️"
            else:
                if trading_paused_bool:
                    if isinstance(pause_trading, bool) and not pause_trading:
                        status_str = "resumed"
                        emoji = "▶️"
                    else:
                        status_str = "paused"
                        emoji = "⏸️"
                else:
                    status_str = "active"
                    emoji = "▶️"
            forced_by = "manually forced" if is_user_forced else "forced by market conditions"
            pause_msg = f"{emoji} Reevaluation has been {forced_by} – Bot is currently {status_str}"
            if pause_reason:
                pause_msg += f" – {pause_reason}"

        if not self.current_symbols:
            logger.warning("No symbols selected after evaluation. Bot will idle until next cycle.")
            if self.notifier:
                msg = f"⚠️ No stocks selected. Bot will idle.\n"
                msg += f"Balance: {base_balance:.2f} {self.base_currency}, "
                msg += f"Per-symbol budget: {per_symbol_budget:.2f}"
                if pause_msg:
                    msg = pause_msg + "\n" + msg
                await self.notifier.send_notification(
                    msg,
                    summary={
                        "action": "HOLD",
                        "reason": "No stocks selected",
                        "base_balance": base_balance,
                        "per_symbol_budget": per_symbol_budget,
                        "pause_decision": pause_trading if isinstance(pause_trading, bool) else None,
                        "pause_reason": pause_reason,
                        "model_type": "mind",
                        "llm_provider": llm_provider,
                        "llm_model": llm_model,
                    }
                )
        elif self.notifier:
            stock_reasoning = parsed.get("reasoning", "") if isinstance(parsed, dict) else ""
            if stock_reasoning:
                msg = f"🔄 Tickers Updated: {', '.join(symbol_labels)}\n💡 {stock_reasoning}"
            else:
                msg = f"🔄 Tickers Updated: {', '.join(symbol_labels)}"
            if pause_msg:
                msg = pause_msg + "\n" + msg
            await self.notifier.send_notification(
                msg,
                summary={
                    "action": "INFO",
                    "reason": "Symbols updated",
                    "stocks": [c["symbol"] for c in self.current_symbols],
                    "stock_reasoning": stock_reasoning,
                    "pause_decision": pause_trading if isinstance(pause_trading, bool) else None,
                    "pause_reason": pause_reason,
                    "model_type": "mind",
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                }
            )

        # If no symbols were selected, shorten the re‑evaluation interval to retry sooner.
        if not self.current_symbols:
            self._symbol_reevaluation_interval = max(self._symbol_reevaluation_interval, MIN_SYMBOL_REEVALUATION_INTERVAL)
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
        async with self._symbol_reeval_lock:
            # Only run if actually paused
            paused_raw = await asyncio.to_thread(self.redis.get, "trading:paused")
            if not paused_raw or paused_raw != "1":
                return

            # Only handle LLM-initiated pauses. Manual pauses are not subject to auto-resume logic.
            source_raw = await asyncio.to_thread(self.redis.get, "trading:pause_source")
            source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")
            if source != "llm":
                logger.info("Pause/resume check skipped: pause was not initiated by LLM (source=%s).", source or "unknown")
                return

            # Read LLM-decided pause recovery settings from Redis
            max_keep = settings.PAUSE_MAX_CONSECUTIVE_KEEP
            force_resume_mult = settings.PAUSE_FORCE_RESUME_RISK_MULTIPLIER
            try:
                raw = await asyncio.to_thread(self.redis.get, "trading:pause_max_consecutive_keep")
                if raw:
                    max_keep = int(raw)
                raw = await asyncio.to_thread(self.redis.get, "trading:pause_force_resume_risk_multiplier")
                if raw:
                    force_resume_mult = float(raw)
            except Exception:
                pass

            # Gather minimal market context
            benchmark_price = None
            try:
                tickers_map = await self._get_quotes_async([settings.BENCHMARK_SYMBOL], timeout=45.0)
                benchmark_ticker = tickers_map.get(settings.BENCHMARK_SYMBOL)
                benchmark_price = benchmark_ticker.get("last") if benchmark_ticker else None
            except Exception:
                pass

            # Market breadth from Redis (already computed by background task)
            full_market_breadth = None
            try:
                raw = await asyncio.to_thread(self.redis.get, "market:breadth:full")
                if raw:
                    full_market_breadth = json.loads(raw)
            except Exception:
                pass
            market_breadth = getattr(self, '_market_breadth', None)

            # Current pause reason
            reason_raw = await asyncio.to_thread(self.redis.get, "trading:pause_reason")
            pause_reason = reason_raw.decode() if isinstance(reason_raw, bytes) else (reason_raw or "")

            # --- Consecutive "keep paused" counter ---
            keep_key = "trading:pause:keep_count"
            keep_count_raw = await asyncio.to_thread(self.redis.get, keep_key)
            try:
                keep_count = int(keep_count_raw) if keep_count_raw else 0
            except (ValueError, TypeError):
                keep_count = 0

            # Build a richer prompt with performance context
            perf = await asyncio.to_thread(self._compute_performance_metrics)
            daily_pnl = perf["equity_curve"].get("daily_pnl", 0.0)
            total_pnl = perf["equity_curve"].get("total_pnl", 0.0)
            consecutive_losses = perf["equity_curve"].get("consecutive_losses", 0)
            drawdown_pct = perf["equity_curve"].get("drawdown_pct", 0.0)

            prompt_parts = [
                "Trading is currently paused.",
            ]
            if pause_reason:
                prompt_parts.append(f"Pause reason: {pause_reason}")
            prompt_parts.append(f"Account P&L: daily={daily_pnl:.4f}, total={total_pnl:.4f}, drawdown={drawdown_pct:.2f}%")
            if consecutive_losses > 0:
                prompt_parts.append(f"Consecutive losing trades: {consecutive_losses}")
            if benchmark_price is not None:
                prompt_parts.append(f"Benchmark ({settings.BENCHMARK_SYMBOL}) price: {benchmark_price}")
            if market_breadth:
                prompt_parts.append(f"Market breadth (top stocks): {market_breadth['positive_pct']}% positive")
            if full_market_breadth:
                prompt_parts.append(f"Full market breadth: {full_market_breadth['positive_pct']}% positive")

            # Check if this is a recent auto-resume situation
            last_auto_resume_raw = await asyncio.to_thread(self.redis.get, "trading:last_auto_resume")
            if last_auto_resume_raw:
                try:
                    last_auto_resume_ts = float(last_auto_resume_raw)
                    seconds_since = time.time() - last_auto_resume_ts
                    if seconds_since < 3600:  # within the last hour
                        minutes_since = seconds_since / 60
                        prompt_parts.append(
                            f"Trading was auto‑resumed {minutes_since:.1f} minutes ago. "
                            "Market conditions may not have changed significantly. "
                            "Only resume if there is clear, concrete improvement in the data above."
                        )
                except (ValueError, TypeError):
                    pass

            # --- Consecutive keep warning and recovery nudge ---
            if keep_count > 0:
                prompt_parts.append(
                    f"You have chosen to keep trading paused {keep_count} time(s) in a row. "
                    f"If you keep it paused {max_keep} times consecutively, the engine will "
                    f"force‑resume trading with a reduced global risk multiplier of "
                    f"{force_resume_mult} to attempt recovery."
                )

            prompt_parts.append(
                "If the account is in drawdown or has consecutive losses, consider resuming "
                "with a **reduced global risk multiplier** (e.g., 0.3–0.5) instead of staying "
                "paused indefinitely. This allows the bot to cautiously seek small profitable "
                "trades to recover, while limiting downside. You can provide an optional "
                "`global_risk_multiplier` field in your JSON response (0.0–1.0) to set the "
                "risk level upon resume. If you omit it, the current multiplier (or 1.0) will be used. "
                "If you are even slightly uncertain, **resume trading** with a very small "
                "`global_risk_multiplier` (e.g., 0.15–0.25). This lets the bot probe for "
                "opportunities with minimal risk. Staying paused forever guarantees zero recovery."
            )

            prompt = (
                "\n".join(prompt_parts)
                + "\n\nShould we resume trading now? Reply with a JSON object: "
                '{"resume_trading": true/false, "reason": "short explanation", '
                '"global_risk_multiplier": 0.0-1.0 (optional)}'
                + "\n\n**Important:** Only resume if you see specific, high‑confidence opportunities. "
                "If conditions are still poor, you may keep trading paused, but remember that "
                "staying paused forever prevents any recovery. A cautious resume with a low risk "
                "multiplier is often better than doing nothing."
            )

            pause_resume_complexity = self._compute_prompt_complexity(
                num_candidates=0,
                market_breadth=market_breadth,
                fear_greed=None,
                volatility_percentile=None,
                sentiment_trend_magnitude=None,
                conflicting_signals=False,
                is_critical=False,
            )
            effective_temp = self._get_effective_temperature("actuator", pause_resume_complexity)

            try:
                pause_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        get_cached_llm_response, compact_prompt(prompt), COMPACTED_SYSTEM_PROMPT, 120,
                        model_type="actuator",
                        temperature=effective_temp,
                    ),
                    timeout=settings.LLM_TIMEOUT
                )
                response = pause_result["response"]
                llm_provider = pause_result["provider"]
                llm_model = pause_result["model"]
                decision = json.loads(response)
            except Exception as e:
                logger.warning(f"Pause/resume LLM call failed: {e}")
                # Track consecutive failures in Redis
                fail_key = "trading:pause:llm_fail_count"
                current_fails = await asyncio.to_thread(self.redis.incr, fail_key)
                await asyncio.to_thread(self.redis.expire, fail_key, 3600)
                _min_pause = settings.MIN_LLM_PAUSE_DURATION
                try:
                    raw = await asyncio.to_thread(self.redis.get, "trading:min_llm_pause_duration")
                    if raw:
                        _min_pause = int(raw)
                except Exception:
                    pass
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⚠️ Could not reach LLM to decide pause/resume (failure #{current_fails}). "
                        f"Auto‑resume will be attempted after {_min_pause}s if LLM stays silent.",
                        summary={"action": "INFO", "reason": "LLM pause-resume call failed"}
                    )
                # If we failed 3 times in a row, force‑resume (optional but safe)
                if current_fails >= 3:
                    # Double-check source before force-resuming
                    fail_source = await asyncio.to_thread(self.redis.get, "trading:pause_source")
                    if fail_source and (fail_source.decode() if isinstance(fail_source, bytes) else fail_source) != "llm":
                        logger.warning("Force-resume on LLM failure skipped: pause source is not LLM.")
                        return
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
                    await asyncio.to_thread(self.redis.delete, fail_key)
                    # --- Also reset keep counter and set force‑resume risk multiplier ---
                    await asyncio.to_thread(self.redis.delete, keep_key)
                    await self._set_global_risk_multiplier(force_resume_mult)
                    self._reeval_trigger.set()
                    if self.notifier:
                        await self.notifier.send_notification(
                            "▶️ Trading auto‑resumed because LLM could not be reached for pause decision. "
                            f"Global risk multiplier set to {force_resume_mult}.",
                            summary={"action": "RESUME", "reason": "LLM pause-resume failures exceeded limit"}
                        )
                return

            resume_trading = decision.get("resume_trading")
            reason = decision.get("reason", "")

            if resume_trading is True:
                # Source is already verified as "llm" by the early check at the top of this method.

                # Check minimum LLM pause duration
                llm_pause_time_raw = await asyncio.to_thread(self.redis.get, "trading:llm_pause_time")
                if llm_pause_time_raw:
                    try:
                        llm_pause_time = float(llm_pause_time_raw)
                        _min_pause = settings.MIN_LLM_PAUSE_DURATION
                        try:
                            raw = await asyncio.to_thread(self.redis.get, "trading:min_llm_pause_duration")
                            if raw:
                                _min_pause = int(raw)
                        except Exception:
                            pass
                        if time.time() - llm_pause_time < _min_pause:
                            remaining = _min_pause - (time.time() - llm_pause_time)
                            logger.info(f"Ignoring LLM resume request: minimum pause duration not elapsed ({remaining:.0f}s remaining).")
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"⏸️ LLM resume request ignored: minimum pause duration "
                                    f"({_min_pause}s) not yet elapsed ({remaining:.0f}s remaining).",
                                    summary={"action": "RESUME", "reason": f"LLM resume blocked by minimum pause duration ({_min_pause}s)", "model_type": "actuator"}
                                )
                            return
                    except (ValueError, TypeError):
                        pass

                # --- Apply optional global_risk_multiplier from LLM ---
                global_mult_raw = decision.get("global_risk_multiplier")
                applied_mult = None
                if global_mult_raw is not None:
                    try:
                        mult_val = float(global_mult_raw)
                        if 0.0 <= mult_val <= 1.0:
                            await self._set_global_risk_multiplier(mult_val)
                            logger.info(f"LLM set global risk multiplier on resume: {mult_val}")
                            applied_mult = mult_val
                        else:
                            logger.warning(f"Invalid global_risk_multiplier in resume decision: {global_mult_raw}")
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid global_risk_multiplier value: {global_mult_raw}")

                # Resume trading
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
                # Reset the keep counter
                await asyncio.to_thread(self.redis.delete, keep_key)
                logger.info("LLM decided to resume trading.")
                self._reeval_trigger.set()
                if self.notifier:
                    reason_text = f" – {reason}" if reason else ""
                    mult_text = f" (risk multiplier: {applied_mult})" if applied_mult is not None else ""
                    await self.notifier.send_notification(
                        f"▶️ Trading resumed by LLM decision{reason_text}{mult_text}",
                        summary={"action": "RESUME", "reason": f"LLM resume request: {reason}" if reason else "LLM resume request", "model_type": "actuator", "llm_provider": llm_provider, "llm_model": llm_model}
                    )
            elif resume_trading is False:
                # LLM wants to stay paused – optionally update reason
                if reason:
                    await asyncio.to_thread(self.redis.set, "trading:pause_reason", reason)

                # Increment consecutive keep counter
                new_keep_count = await asyncio.to_thread(self.redis.incr, keep_key)
                # Set a TTL so it doesn't persist forever (e.g., 24h)
                await asyncio.to_thread(self.redis.expire, keep_key, 86400)

                if new_keep_count >= max_keep:
                    # Double-check that the pause is still LLM-initiated (should always be true here)
                    current_source = await asyncio.to_thread(self.redis.get, "trading:pause_source")
                    if current_source and (current_source.decode() if isinstance(current_source, bytes) else current_source) != "llm":
                        logger.warning("Force-resume skipped: pause source changed to non-LLM.")
                        return

                    # --- Drawdown circuit breaker: do not force-resume in significant drawdown ---
                    max_drawdown = settings.PAUSE_FORCE_RESUME_MAX_DRAWDOWN_PCT
                    if drawdown_pct >= max_drawdown:
                        logger.warning(
                            f"Force-resume blocked: account drawdown {drawdown_pct:.2f}% "
                            f"exceeds circuit breaker threshold {max_drawdown:.2f}%. "
                            f"Keeping trading paused for safety."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⛔ Force-resume blocked: drawdown {drawdown_pct:.2f}% exceeds "
                                f"circuit breaker threshold {max_drawdown:.2f}%. "
                                f"Trading stays paused to protect capital. "
                                f"LLM has kept it paused {new_keep_count} time(s).",
                                summary={
                                    "action": "PAUSE",
                                    "reason": f"Force-resume blocked by drawdown circuit breaker ({drawdown_pct:.2f}% >= {max_drawdown:.2f}%)",
                                    "model_type": "actuator",
                                    "llm_provider": llm_provider,
                                    "llm_model": llm_model,
                                }
                            )
                        # Do NOT force-resume; let the LLM continue deciding
                        return

                    logger.warning(
                        f"LLM kept trading paused {new_keep_count} times consecutively – "
                        f"forcing resume with risk multiplier {force_resume_mult}."
                    )
                    # Force resume
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
                    await asyncio.to_thread(self.redis.delete, keep_key)
                    await self._set_global_risk_multiplier(force_resume_mult)
                    self._reeval_trigger.set()
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"▶️ Trading force‑resumed after {new_keep_count} consecutive pauses. "
                            f"Global risk multiplier set to {force_resume_mult}.",
                            summary={
                                "action": "RESUME",
                                "reason": f"Force resume after {new_keep_count} consecutive keep-paused decisions",
                                "model_type": "actuator",
                            }
                        )
                else:
                    logger.info(f"LLM decided to keep trading paused. Reason: {reason} (keep count: {new_keep_count}/{max_keep})")
                    if self.notifier:
                        reason_text = f" – {reason}" if reason else ""
                        await self.notifier.send_notification(
                            f"⏸️ LLM decided to keep trading paused{reason_text} "
                            f"({new_keep_count}/{max_keep} consecutive keeps)",
                            summary={"action": "PAUSE", "reason": f"LLM keep paused: {reason}" if reason else "LLM keep paused", "model_type": "actuator", "llm_provider": llm_provider, "llm_model": llm_model}
                        )
            else:
                logger.warning(f"Invalid resume_trading value in LLM response: {resume_trading}")

    async def _compute_portfolio_exposure_summary(self, base_balance: float) -> Dict[str, float]:
        """Compute portfolio exposure, stop-loss risk, and available capital for the prompt."""
        portfolio_total_value = base_balance
        portfolio_exposure = 0.0
        portfolio_stop_risk = 0.0
        pos_tickers = await self._get_all_position_tickers()
        for sym, pos in self.positions.items():
            try:
                t = pos_tickers.get(sym)
                price = t['last'] if t and t.get('last') else 0.0
                pos_value = pos['amount'] * price
                portfolio_exposure += pos_value
                portfolio_total_value += pos_value
                stop_loss = pos.get('stop_loss')
                if stop_loss is not None and price > 0:
                    loss_if_stop = pos_value * (price - stop_loss) / price
                    portfolio_stop_risk += max(0, loss_if_stop)
            except Exception:
                pass
        portfolio_exposure_pct = (portfolio_exposure / portfolio_total_value * 100) if portfolio_total_value > 0 else 0.0
        portfolio_stop_risk_pct = (portfolio_stop_risk / portfolio_total_value * 100) if portfolio_total_value > 0 else 0.0
        async with self._cycle_spent_lock:
            portfolio_available_capital = max(0.0, base_balance - self._cycle_spent)
        return {
            "portfolio_total_value": portfolio_total_value,
            "portfolio_exposure": portfolio_exposure,
            "portfolio_stop_risk": portfolio_stop_risk,
            "portfolio_exposure_pct": portfolio_exposure_pct,
            "portfolio_stop_risk_pct": portfolio_stop_risk_pct,
            "portfolio_available_capital": portfolio_available_capital,
        }

    async def _compute_multi_tf_indicators(
        self, symbol: str, ohlcv_data: Dict[str, List[List]], assigned_tf: str
    ) -> Dict[str, Any]:
        """Batch-fetch indicators from DB and extract assigned-timeframe values.

        Returns a dict with keys: multi_tf_indicators, multi_tf_raw_candles,
        atr, rsi, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower,
        ema_9, ema_21, stochastic_k, stochastic_d, adx, plus_di, minus_di,
        obv, mfi, cci, williams_r, ichimoku, donchian_channels, parabolic_sar,
        keltner_channels, vwap, daily_pivot_points.
        """
        multi_tf_indicators: Dict[str, Dict[str, Any]] = {}
        multi_tf_raw_candles: Dict[str, List[List]] = {}
        atr = rsi = macd = macd_signal = macd_hist = None
        bb_upper = bb_middle = bb_lower = ema_9 = ema_21 = None
        stochastic_k = stochastic_d = adx = plus_di = minus_di = None
        obv = mfi = cci = williams_r = ichimoku = donchian_channels = None
        parabolic_sar = keltner_channels = vwap = daily_pivot_points = None

        batch_inds = await asyncio.to_thread(get_indicators_for_symbols, [symbol], settings.OHLCV_TIMEFRAMES)
        symbol_inds = batch_inds.get(symbol, {})

        for tf in settings.OHLCV_TIMEFRAMES:
            if tf in ohlcv_data and ohlcv_data[tf]:
                candles = ohlcv_data[tf]
                multi_tf_raw_candles[tf] = candles
                ind = symbol_inds.get(tf)
                if ind:
                    multi_tf_indicators[tf] = ind
                    if tf == assigned_tf:
                        atr = ind.get('atr')
                        rsi = ind.get('rsi')
                        macd = ind.get('macd')
                        macd_signal = ind.get('macd_signal')
                        macd_hist = ind.get('macd_hist')
                        bb_upper = ind.get('bb_upper')
                        bb_middle = ind.get('bb_middle')
                        bb_lower = ind.get('bb_lower')
                        ema_9 = ind.get('ema_9')
                        ema_21 = ind.get('ema_21')
                        stochastic_k = ind.get('stochastic_k')
                        stochastic_d = ind.get('stochastic_d')
                        adx = ind.get('adx')
                        plus_di = ind.get('plus_di')
                        minus_di = ind.get('minus_di')
                        obv = ind.get('obv')
                        mfi = ind.get('mfi')
                        cci = ind.get('cci')
                        williams_r = ind.get('williams_r')
                        ichimoku = ind.get('ichimoku')
                        donchian_channels = ind.get('donchian_channels')
                        parabolic_sar = ind.get('parabolic_sar')
                        keltner_channels = ind.get('keltner_channels')
                        vwap = compute_vwap(candles)

        # Compute daily pivot points from the 1d timeframe (if available)
        if "1d" in multi_tf_raw_candles and len(multi_tf_raw_candles["1d"]) >= 2:
            daily_candles = multi_tf_raw_candles["1d"]
            prev_daily = daily_candles[-2]
            daily_pivot_points = compute_pivot_points(prev_daily[2], prev_daily[3], prev_daily[4])

        return {
            "multi_tf_indicators": multi_tf_indicators,
            "multi_tf_raw_candles": multi_tf_raw_candles,
            "atr": atr, "rsi": rsi, "macd": macd, "macd_signal": macd_signal,
            "macd_hist": macd_hist, "bb_upper": bb_upper, "bb_middle": bb_middle,
            "bb_lower": bb_lower, "ema_9": ema_9, "ema_21": ema_21,
            "stochastic_k": stochastic_k, "stochastic_d": stochastic_d,
            "adx": adx, "plus_di": plus_di, "minus_di": minus_di,
            "obv": obv, "mfi": mfi, "cci": cci, "williams_r": williams_r,
            "ichimoku": ichimoku, "donchian_channels": donchian_channels,
            "parabolic_sar": parabolic_sar, "keltner_channels": keltner_channels,
            "vwap": vwap, "daily_pivot_points": daily_pivot_points,
        }

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
        # ATR multi-TF
        atr_multi_tf: Dict[str, float] = {}
        for tf in settings.OHLCV_TIMEFRAMES:
            ind = multi_tf_indicators.get(tf, {})
            tf_atr = ind.get('atr')
            if tf_atr is not None and tf_atr > 0:
                atr_multi_tf[tf] = tf_atr

        # ATR Percentile (volatility context)
        atr_percentile = None
        if atr is not None and atr > 0:
            atr_percentile_key = f"atr_percentile:{symbol}"
            try:
                stored_atr = await asyncio.to_thread(self.redis.get, atr_percentile_key)
                if stored_atr:
                    atr_history = json.loads(stored_atr)
                else:
                    atr_history = []
                atr_history.append(atr)
                atr_history = atr_history[-100:]
                await asyncio.to_thread(self.redis.setex, atr_percentile_key, 7 * 24 * 3600, json.dumps(atr_history))
                if len(atr_history) >= 5:
                    sorted_atr = sorted(atr_history)
                    rank = sum(1 for v in sorted_atr if v <= atr)
                    atr_percentile = round(rank / len(sorted_atr) * 100, 1)
            except Exception as e:
                logger.info(f"ATR percentile computation failed for {symbol}: {e}")

        # Market regime classification
        market_regime = await self._classify_market_regime(
            adx=adx, plus_di=plus_di, minus_di=minus_di,
            ema_9=ema_9, ema_21=ema_21,
            bb_upper=bb_upper, bb_lower=bb_lower, bb_middle=bb_middle,
            atr=atr, atr_percentile=atr_percentile,
            current_price=ticker['last'],
        )

        # Extract raw candles for the assigned timeframe
        raw_candles = multi_tf_raw_candles.get(assigned_tf)

        # Fetch historical OHLCV from DB for backtest analysis
        historical_ohlcv = None
        try:
            since_ms = int(time.time() * 1000) - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000
            hist_limit = int((settings.OHLCV_RETENTION_DAYS * 86400) / tf_seconds) + 100
            db_candles = await asyncio.to_thread(
                get_ohlcv, symbol, assigned_tf, since_ms=since_ms, limit=hist_limit
            )
            if db_candles:
                historical_ohlcv = [
                    [c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
                    for c in db_candles
                ]
                if len(historical_ohlcv) >= 2:
                    interval_ms = self._timeframe_to_ms(assigned_tf)
                    timestamps = [c[0] for c in historical_ohlcv]
                    has_gap = False
                    for i in range(len(timestamps) - 1):
                        if timestamps[i+1] - timestamps[i] > interval_ms * 1.5:
                            has_gap = True
                            break
                    if has_gap:
                        logger.warning(
                            f"Historical OHLCV for {symbol} {assigned_tf} contains gaps; "
                            f"passing data to LLM anyway for backtesting."
                        )
        except Exception as e:
            logger.warning(f"Failed to fetch historical OHLCV for {symbol} {assigned_tf}: {e}")

        # Unrealized P&L for current position
        unrealized_pnl = None
        position_info = None
        if symbol in self.positions:
            pos = self.positions[symbol]
            position_info = pos
            current_price = ticker['last']
            entry_price = pos['price']
            amount = pos['amount']
            unrealized_pnl = (current_price - entry_price) * amount

        # Recent trade outcomes (last 5 closed trades)
        recent_trades = [t for t in self.trade_history if t.get("side") == "sell"][-5:]
        recent_trades_summary = [
            {
                "symbol": t["symbol"],
                "realized_pnl": t.get("realized_pnl", 0.0),
                "strategy": t.get("strategy_type", "unknown"),
            }
            for t in recent_trades
        ]

        # Fetch minimum order size
        try:
            asset = await self._get_asset_info(symbol)
            min_order_amount = float(asset.min_order_size) if asset.min_order_size else None
        except Exception:
            min_order_amount = None
        current_price = ticker['last']
        if min_order_amount is not None and current_price:
            min_order_cost = min_order_amount * current_price
        else:
            min_order_cost = None

        # Past trades for this specific symbol (last 10 closed sells)
        past_trades = [
            t for t in self.trade_history
            if t.get("symbol") == symbol and t.get("side") == "sell"
        ][-10:]

        # Fetch historical backtest results for this symbol
        historical_backtest_results = await asyncio.to_thread(
            get_backtest_results_for_symbol, symbol, assigned_tf, 10
        )

        # Fetch aggregate sentiment
        aggregate_sentiment = None
        if settings.NEWS_ENABLED:
            try:
                aggregate_sentiment = await self._get_cached_sentiment(symbol)
            except Exception as e:
                logger.info(f"Could not fetch aggregate sentiment for {symbol}: {e}")

        # Sentiment trend
        sentiment_trend_val = None
        if aggregate_sentiment:
            base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
            current_compound = aggregate_sentiment.get("avg_compound")
            prev_key = f"sentiment:prev:{base_symbol}"
            prev_raw = await asyncio.to_thread(self.redis.get, prev_key)
            prev_compound = float(prev_raw) if prev_raw else None
            if current_compound is not None:
                await asyncio.to_thread(self.redis.setex, prev_key, settings.NEWS_CACHE_TTL_SECONDS, str(current_compound))
            if current_compound is not None and prev_compound is not None:
                sentiment_trend_val = round(current_compound - prev_compound, 4)

        # Volume trend
        volume_trend_val = None
        current_volume = ticker.get('quoteVolume', 0) or 0
        if current_volume > 0:
            volume_trend_val = await self._compute_volume_trend(symbol, current_volume, timeframe=assigned_tf)

        # Full market breadth from Redis
        full_market_breadth = None
        try:
            full_breadth_raw = await asyncio.to_thread(self.redis.get, "market:breadth:full")
            if full_breadth_raw:
                full_market_breadth = json.loads(full_breadth_raw)
        except Exception:
            pass
        session_info = self._get_session_info()

        # Compute minutes until market close
        now_rome = datetime.now(timezone.utc).astimezone(ZoneInfo(settings.MARKET_TIMEZONE))
        weekday = now_rome.weekday()
        if weekday < 5:
            rome_minutes = now_rome.hour * 60 + now_rome.minute
            close_minutes = settings.MARKET_CLOSE_HOUR * 60 + settings.MARKET_CLOSE_MINUTE
            minutes_to_market_close = close_minutes - rome_minutes
            if minutes_to_market_close < 0:
                minutes_to_market_close = 0
        else:
            minutes_to_market_close = None

        # Global risk multiplier
        global_risk_mult = await self._get_global_risk_multiplier()

        # Portfolio risk thresholds
        max_port_exp = None
        max_port_risk = None
        try:
            raw = await asyncio.to_thread(self.redis.get, "trading:max_portfolio_exposure_pct")
            if raw:
                max_port_exp = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:max_portfolio_stop_risk_pct")
            if raw:
                max_port_risk = float(raw)
        except Exception:
            pass

        partial_tp_executed_levels = self.positions[symbol].get("partial_tp_levels_triggered", []) if symbol in self.positions else []

        # Validator multipliers
        min_stop_atr_mult = 1.0
        min_hold_time_mult = 1.0
        global_min_rr = None
        try:
            raw = await asyncio.to_thread(self.redis.get, "trading:min_stop_loss_atr_mult")
            if raw:
                min_stop_atr_mult = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:min_max_hold_time_mult")
            if raw:
                min_hold_time_mult = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:min_risk_reward_ratio")
            if raw:
                global_min_rr = float(raw)
        except Exception:
            pass

        return {
            "atr_multi_tf": atr_multi_tf,
            "atr_percentile": atr_percentile,
            "market_regime": market_regime,
            "raw_candles": raw_candles,
            "historical_ohlcv": historical_ohlcv,
            "unrealized_pnl": unrealized_pnl,
            "position_info": position_info,
            "recent_trades_summary": recent_trades_summary,
            "min_order_amount": min_order_amount,
            "min_order_cost": min_order_cost,
            "past_trades": past_trades,
            "aggregate_sentiment": aggregate_sentiment,
            "sentiment_trend_val": sentiment_trend_val,
            "volume_trend_val": volume_trend_val,
            "full_market_breadth": full_market_breadth,
            "session_info": session_info,
            "minutes_to_market_close": minutes_to_market_close,
            "global_risk_mult": global_risk_mult,
            "max_port_exp": max_port_exp,
            "max_port_risk": max_port_risk,
            "partial_tp_executed_levels": partial_tp_executed_levels,
            "min_stop_atr_mult": min_stop_atr_mult,
            "min_hold_time_mult": min_hold_time_mult,
            "global_min_rr": global_min_rr,
            "historical_backtest_results": historical_backtest_results,
        }

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
        """Run backtests and the Step 2 LLM call to produce the final signal.

        Returns (final_signal, combined_backtest_summary, llm_provider, llm_model).
        """
        combined_bt_summary = ""
        llm_provider = None
        llm_model = None

        if preliminary_signal.action in ("BUY", "HOLD"):
            # Determine which variant param sets to backtest
            variants_to_test = []
            if preliminary_signal.backtest_variants:
                variants_to_test = list(preliminary_signal.backtest_variants)
            else:
                # Fallback: use the preliminary signal's own params as a single variant
                variants_to_test.append(preliminary_signal.strategy_params or {})
            # --- Deduplicate variants with identical key risk parameters ---
            variants_to_test = self._deduplicate_variants(variants_to_test)
            # Safety cap: limit to configured max variants to prevent excessive backtest time
            if len(variants_to_test) > settings.MAX_BACKTEST_VARIANTS:
                logger.warning(
                    f"LLM returned {len(variants_to_test)} backtest variants for {symbol}, "
                    f"capping to {settings.MAX_BACKTEST_VARIANTS}"
                )
                variants_to_test = variants_to_test[:settings.MAX_BACKTEST_VARIANTS]

            # Limit number of variants based on available data length
            source_candles = historical_ohlcv or raw_candles or []
            if source_candles and len(source_candles) < 50:
                variants_to_test = variants_to_test[:2]
            elif source_candles and len(source_candles) < 100:
                variants_to_test = variants_to_test[:3]

            # Run backtest variants in parallel (concurrency-limited by semaphore)
            async def _run_single_variant(vp: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    bt_stats, bt_summary = await self._run_backtest_variant(
                        symbol=symbol,
                        variant_params=vp,
                        preliminary_signal=preliminary_signal,
                        atr=atr,
                        current_price=current_price,
                        tf_secs=tf_seconds,
                        assigned_tf=assigned_tf,
                        historical_ohlcv=historical_ohlcv,
                        raw_candles=raw_candles,
                        base_balance=base_balance,
                        is_btp=is_btp,
                    )
                    if bt_stats is not None:
                        return {"variant_params": vp, "summary": bt_summary, "stats": bt_stats}
                    else:
                        return {"variant_params": vp, "summary": bt_summary or "Insufficient data for backtest.", "stats": {}}
                except Exception as e:
                    logger.warning(f"Backtest variant failed for {symbol}: {e}")
                    return {"variant_params": vp, "summary": f"Backtest error: {e}", "stats": {}}

            backtest_results = list(await asyncio.gather(*[_run_single_variant(vp) for vp in variants_to_test]))

            # Log results after all variants complete
            for i, r in enumerate(backtest_results):
                if r["stats"]:
                    logger.info(f"Backtest variant {i+1}/{len(variants_to_test)} for {symbol}: {r['summary']}")
                else:
                    logger.info(f"Backtest variant {i+1}/{len(variants_to_test)} for {symbol}: insufficient data")

            # Build combined backtest summary for notifications
            combined_bt_summary = " | ".join(
                f"V{i+1}: {r['summary']}" for i, r in enumerate(backtest_results)
            ) if backtest_results else "No backtest performed"

            if backtest_results:
                # Build Step 2 prompt with ALL backtest results
                total_variants_proposed = len(preliminary_signal.backtest_variants) if preliminary_signal.backtest_variants else 1
                historical_bt_results = await asyncio.to_thread(
                    get_backtest_results_for_symbol, symbol, assigned_tf, 10
                )
                step2_prompt = build_final_decision_prompt(
                    symbol=symbol,
                    ticker=ticker,
                    preliminary_decision={
                        "action": preliminary_signal.action,
                        "confidence": preliminary_signal.confidence,
                        "reasoning": preliminary_signal.reasoning,
                        "strategy_params": preliminary_signal.strategy_params,
                        "timeframe": assigned_tf,
                    },
                    backtest_results=backtest_results,
                    base_currency=self.base_currency,
                    trading_paused=trading_paused,
                    total_variants_proposed=total_variants_proposed,
                    historical_backtest_results=historical_bt_results,
                )
                # Append position info if exists
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    step2_prompt += (
                        f"\n**Existing Position:** You already hold {pos['amount']:.6f} "
                        f"at entry {pos['price']:.4f}. A BUY will ADD to this position (scale in).\n"
                    )

                # Call LLM for Step 2
                try:
                    step2_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            get_cached_llm_response,
                            compact_prompt(step2_prompt),
                            COMPACTED_SYSTEM_PROMPT,
                            60,
                            model_type=strategy_model_type,
                            temperature=effective_temp,
                        ),
                        timeout=settings.LLM_TIMEOUT
                    )
                    step2_response = step2_result["response"]
                    llm_provider = step2_result["provider"]
                    llm_model = step2_result["model"]
                    logger.info(f"LLM Step 2 call completed for {symbol} (provider={llm_provider}, model={llm_model})")

                    # Parse Step 2 response
                    try:
                        final_strategy = create_strategy_from_llm(step2_response)
                    except ValueError:
                        # Retry with correction prompt
                        correction = (
                            "Your previous response was not valid JSON. "
                            "Output ONLY a single JSON object. "
                            "Here is the request:\n\n" + step2_prompt
                        )
                        try:
                            retry_result = await asyncio.wait_for(
                                asyncio.to_thread(
                                    get_cached_llm_response,
                                    compact_prompt(correction),
                                    COMPACTED_SYSTEM_PROMPT, 30,
                                    model_type="actuator",
                                    temperature=effective_temp,
                                ),
                                timeout=settings.LLM_TIMEOUT
                            )
                            final_strategy = create_strategy_from_llm(retry_result["response"])
                            step2_response = retry_result["response"]
                            llm_provider = retry_result["provider"]
                            llm_model = retry_result["model"]
                        except Exception:
                            logger.error(f"Step 2 JSON parse retry failed for {symbol}. Using preliminary decision.")
                            final_strategy = None

                    if final_strategy is not None:
                        signal = final_strategy.generate_signal({})
                        signal.model_type = strategy_model_type
                        signal.llm_provider = llm_provider
                        signal.llm_model = llm_model
                        signal.backtest_summary = combined_bt_summary
                    else:
                        signal = preliminary_signal
                        signal.backtest_summary = combined_bt_summary
                    # Carry over execution-critical fields from Step 1 if not provided in Step 2
                    if signal.action == "BUY":
                        # Execution parameters
                        if signal.entry_condition is None and preliminary_signal.entry_condition is not None:
                            signal.entry_condition = preliminary_signal.entry_condition
                        if signal.order_type is None and preliminary_signal.order_type is not None:
                            signal.order_type = preliminary_signal.order_type
                        if signal.limit_price is None and preliminary_signal.limit_price is not None:
                            signal.limit_price = preliminary_signal.limit_price
                        if signal.stop_price is None and preliminary_signal.stop_price is not None:
                            signal.stop_price = preliminary_signal.stop_price
                        if signal.stop_loss_order_type is None and preliminary_signal.stop_loss_order_type is not None:
                            signal.stop_loss_order_type = preliminary_signal.stop_loss_order_type
                        if signal.stop_loss_stop_price is None and preliminary_signal.stop_loss_stop_price is not None:
                            signal.stop_loss_stop_price = preliminary_signal.stop_loss_stop_price
                        if signal.stop_loss_limit_price is None and preliminary_signal.stop_loss_limit_price is not None:
                            signal.stop_loss_limit_price = preliminary_signal.stop_loss_limit_price
                        if signal.stop_loss_trail_offset is None and preliminary_signal.stop_loss_trail_offset is not None:
                            signal.stop_loss_trail_offset = preliminary_signal.stop_loss_trail_offset
                        if signal.take_profit_order_type is None and preliminary_signal.take_profit_order_type is not None:
                            signal.take_profit_order_type = preliminary_signal.take_profit_order_type
                        if signal.take_profit_limit_price is None and preliminary_signal.take_profit_limit_price is not None:
                            signal.take_profit_limit_price = preliminary_signal.take_profit_limit_price
                        if signal.trail_offset is None and preliminary_signal.trail_offset is not None:
                            signal.trail_offset = preliminary_signal.trail_offset
                        # Risk parameters — carry over from Step 1 if missing in Step 2
                        if signal.strategy_params:
                            prelim_params = preliminary_signal.strategy_params or {}
                            for risk_key in (
                                "cooldown_after_loss_seconds",
                                "max_hold_time_seconds",
                                "stop_loss_method",
                                "stop_loss_atr_multiple",
                                "trailing_stop_distance_pct",
                                "trailing_stop_atr_multiple",
                                "trailing_stop_activation_pct",
                                "partial_take_profit_levels",
                                "partial_take_profit_pct",
                                "partial_take_profit_fraction",
                                "breakeven_activation_pct",
                                "trailing_take_profit",
                                "trailing_take_profit_distance_pct",
                                "max_unrealized_loss_pct",
                                "news_sentiment_exit_threshold",
                                "max_risk_per_trade_pct",
                                "max_portfolio_risk_pct",
                                "min_profit_per_trade",
                                "min_risk_reward_ratio",
                                "min_confidence",
                                "position_size_multiplier",
                                "strategy_interval_seconds",
                                "backtest_period_days",
                                "order_fill_timeout_seconds",
                                "time_in_force",
                            ):
                                if risk_key not in signal.strategy_params and risk_key in prelim_params:
                                    signal.strategy_params[risk_key] = prelim_params[risk_key]
                        else:
                            # Step 2 returned no params at all — use Step 1's params
                            signal.strategy_params = preliminary_signal.strategy_params
                except Exception as e:
                    logger.error(f"LLM Step 2 call failed for {symbol}: {e}. Using preliminary decision.")
                    signal = preliminary_signal
                    signal.backtest_summary = combined_bt_summary
                    # Preserve provider/model from Step 1b as fallback
                    if llm_provider is None:
                        llm_provider = preliminary_signal.llm_provider
                    if llm_model is None:
                        llm_model = preliminary_signal.llm_model
            else:
                logger.info(f"Insufficient data for any backtest for {symbol}. Using preliminary decision.")
                signal = preliminary_signal
        else:
            # For SELL or HOLD, no backtest needed, use preliminary decision
            signal = preliminary_signal

        return signal, combined_bt_summary, llm_provider, llm_model

    async def _handle_triggered_flags(
        self,
        symbol: str,
        display_symbol: str,
        signal: Signal,
        validated: Signal,
        assigned_tf: str,
        current_price: float,
        atr: Optional[float],
        ticker: Dict[str, Any],
        max_hold_expired: bool,
        stop_loss_triggered: bool,
        take_profit_triggered: bool,
        partial_tp_triggered: bool,
        dust_sweep_triggered: bool,
        strategy_model_type: str,
        llm_provider: Optional[str],
        llm_model: Optional[str],
    ) -> bool:
        """Handle triggered position flags (max hold, stop loss, take profit, partial TP, dust sweep).

        Returns True if the caller should return immediately (flag was handled).
        Returns False if the caller should continue with normal execution.
        """
        params = signal.strategy_params or {}

        # --- Handle max‑hold‑expired LLM decision ---
        if max_hold_expired and signal.action == "HOLD":
            new_max_hold = params.get("max_hold_time_seconds") if params else None
            if new_max_hold is not None and new_max_hold > 0:
                logger.info(f"LLM extended max hold time for {symbol} to {new_max_hold}s")
                if symbol in self.positions:
                    async with self._positions_lock:
                        self.positions[symbol]["max_hold_time_seconds"] = new_max_hold
                        self.positions[symbol]["timestamp"] = int(time.time() * 1000)
                        self.positions[symbol].pop("_max_hold_expired", None)
                        self.positions[symbol].pop("_max_hold_expired_count", None)
                for symbol_entry in self.current_symbols:
                    if symbol_entry["symbol"] == symbol:
                        symbol_entry["entry_time"] = time.time()
                        break
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⏰ Max hold time for {display_symbol} extended to {new_max_hold}s by LLM.\n"
                        f"Reasoning: {validated.reasoning}",
                        summary={
                            "symbol": symbol,
                            "action": "HOLD",
                            "reason": validated.reasoning,
                            "new_max_hold_seconds": new_max_hold,
                            "model_type": strategy_model_type,
                            "llm_provider": llm_provider,
                            "llm_model": llm_model,
                        }
                    )
                await self._update_position_params(
                    symbol, params, signal.indicator_config, assigned_tf, current_price, atr,
                )
                self._state_dirty = True
            else:
                logger.warning(
                    f"LLM returned HOLD without new max_hold_time_seconds for {symbol} "
                    f"after max hold expiry – forcing SELL."
                )
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⏰ LLM did not extend hold time for {display_symbol} – closing position.",
                        summary={
                            "symbol": symbol, "action": "SELL",
                            "reason": "Max hold expired, LLM did not extend",
                            "exit_reason": "max_hold_time_llm_no_extend",
                            "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                await self._execute_signal(
                    symbol, Signal(action="SELL", confidence=1.0, reasoning="Max hold expired, LLM did not extend"),
                    exit_reason="max_hold_time_llm_no_extend"
                )
            return True

        # --- Handle stop-loss-triggered LLM decision ---
        if stop_loss_triggered and signal.action == "HOLD":
            new_params = signal.strategy_params or {}
            new_stop_method = new_params.get("stop_loss_method", "fixed")
            new_stop_pct = None
            if new_stop_method == "atr_multiple" and atr is not None and atr > 0:
                atr_mult = new_params.get("stop_loss_atr_multiple")
                if atr_mult is not None:
                    new_stop_pct = (atr_mult * atr) / current_price
            else:
                new_stop_pct = new_params.get("stop_loss_pct")

            if new_stop_pct is not None and new_stop_pct > 0:
                logger.info(
                    f"LLM decided to hold {symbol} after stop-loss trigger, "
                    f"new stop_loss_pct={new_stop_pct:.4%}"
                )
                if symbol in self.positions:
                    async with self._positions_lock:
                        self.positions[symbol]["stop_loss"] = current_price * (1 - new_stop_pct)
                        self.positions[symbol].pop("_stop_loss_triggered", None)
                        self.positions[symbol].pop("_stop_loss_review_count", None)
                    await self._update_position_params(
                        symbol, new_params, signal.indicator_config, assigned_tf, current_price, atr,
                    )
                    self._state_dirty = True
                if self.notifier:
                    await self.notifier.send_notification(
                        f"🔄 {display_symbol}: LLM adjusted stop-loss to {new_stop_pct:.4%} – holding.\n"
                        f"Reasoning: {validated.reasoning}",
                        summary={
                            "symbol": symbol, "action": "HOLD", "reason": validated.reasoning,
                            "new_stop_loss_pct": new_stop_pct, "model_type": strategy_model_type,
                            "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                return True
            else:
                logger.warning(
                    f"LLM returned HOLD for {symbol} after stop-loss trigger but did not provide "
                    f"a new stop-loss. Forcing SELL."
                )
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⛔ {display_symbol}: LLM did not provide new stop-loss – selling.",
                        summary={
                            "symbol": symbol, "action": "SELL",
                            "reason": "Stop-loss triggered, LLM did not provide new stop",
                            "exit_reason": "stop_loss_llm_no_action",
                            "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                await self._execute_signal(
                    symbol, Signal(action="SELL", confidence=1.0, reasoning="Stop-loss triggered, LLM did not provide new stop"),
                    exit_reason="stop_loss_llm_no_action"
                )
                return True

        elif stop_loss_triggered and signal.action == "SELL":
            if symbol in self.positions:
                async with self._positions_lock:
                    self.positions[symbol].pop("_stop_loss_triggered", None)
                    self.positions[symbol].pop("_stop_loss_review_count", None)
            # Continue to normal SELL execution

        # --- Handle take-profit-triggered LLM decision ---
        if take_profit_triggered and signal.action == "HOLD":
            new_params = signal.strategy_params or {}
            new_tp_pct = new_params.get("take_profit_pct")
            if new_tp_pct is not None and new_tp_pct > 0:
                logger.info(
                    f"LLM decided to hold {symbol} after take-profit trigger, "
                    f"new take_profit_pct={new_tp_pct:.4%}"
                )
                if symbol in self.positions:
                    async with self._positions_lock:
                        self.positions[symbol]["take_profit"] = current_price * (1 + new_tp_pct)
                        self.positions[symbol].pop("_take_profit_triggered", None)
                        self.positions[symbol].pop("_take_profit_review_count", None)
                    await self._update_position_params(
                        symbol, new_params, signal.indicator_config, assigned_tf, current_price, atr,
                    )
                    self._state_dirty = True
                if self.notifier:
                    await self.notifier.send_notification(
                        f"🔄 {display_symbol}: LLM adjusted take-profit to {new_tp_pct:.4%} – holding.\n"
                        f"Reasoning: {validated.reasoning}",
                        summary={
                            "symbol": symbol, "action": "HOLD", "reason": validated.reasoning,
                            "new_take_profit_pct": new_tp_pct, "model_type": strategy_model_type,
                            "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                return True
            else:
                logger.warning(
                    f"LLM returned HOLD for {symbol} after take-profit trigger but did not provide "
                    f"a new take-profit. Forcing SELL."
                )
                if self.notifier:
                    await self.notifier.send_notification(
                        f"🎯 {display_symbol}: LLM did not provide new take-profit – selling.",
                        summary={
                            "symbol": symbol, "action": "SELL",
                            "reason": "Take-profit triggered, LLM did not provide new take-profit",
                            "exit_reason": "take_profit_llm_no_action",
                            "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                await self._execute_signal(
                    symbol, Signal(action="SELL", confidence=1.0, reasoning="Take-profit triggered, LLM did not provide new take-profit"),
                    exit_reason="take_profit_llm_no_action"
                )
                return True

        elif take_profit_triggered and signal.action == "SELL":
            if symbol in self.positions:
                async with self._positions_lock:
                    self.positions[symbol].pop("_take_profit_triggered", None)
                    self.positions[symbol].pop("_take_profit_review_count", None)
            # Continue to normal SELL execution

        # --- Handle partial TP triggered ---
        if partial_tp_triggered and signal.action == "HOLD":
            new_levels = params.get("partial_take_profit_levels") if params else None
            if new_levels is not None:
                async with self._positions_lock:
                    self.positions[symbol]["partial_take_profit_levels"] = new_levels
                    self.positions[symbol].pop("_partial_tp_triggered", None)
                    self.positions[symbol].pop("_partial_tp_triggered_single", None)
                    self.positions[symbol].pop("_partial_tp_review_count", None)
                    self.positions[symbol].pop("_partial_tp_single_review_count", None)
                    self.positions[symbol].pop("_partial_tp_triggered_levels", None)
                    self.positions[symbol]["partial_tp_levels_triggered"] = []
                    self.positions[symbol]["partial_tp_depth_wait_start"] = {}
                logger.info(f"LLM updated partial TP levels for {symbol}")
                await self._update_position_params(
                    symbol, params, signal.indicator_config, assigned_tf, current_price, atr,
                )
                self._state_dirty = True
                if self.notifier:
                    await self.notifier.send_notification(
                        f"🔄 {display_symbol}: LLM adjusted partial TP levels – holding.",
                        summary={"symbol": symbol, "action": "HOLD", "reason": "Partial TP levels adjusted by LLM", "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model}
                    )
                return True
            else:
                logger.info(f"LLM did not update partial TP levels for {symbol}, executing triggered level(s)")
                if self.positions[symbol].get("_partial_tp_triggered_single"):
                    await self._execute_partial_tp_single(symbol, current_price, None, ticker)
                    async with self._positions_lock:
                        self.positions[symbol].pop("_partial_tp_triggered_single", None)
                        self.positions[symbol].pop("_partial_tp_single_review_count", None)
                if self.positions[symbol].get("_partial_tp_triggered"):
                    for lvl in self.positions[symbol].get("_partial_tp_triggered_levels", []):
                        await self._execute_partial_tp_level(symbol, lvl, current_price, None, ticker)
                    async with self._positions_lock:
                        self.positions[symbol].pop("_partial_tp_triggered", None)
                        self.positions[symbol].pop("_partial_tp_review_count", None)
                        self.positions[symbol].pop("_partial_tp_triggered_levels", None)
                return True

        elif partial_tp_triggered and signal.action == "SELL":
            async with self._positions_lock:
                self.positions[symbol].pop("_partial_tp_triggered", None)
                self.positions[symbol].pop("_partial_tp_triggered_single", None)
                self.positions[symbol].pop("_partial_tp_review_count", None)
                self.positions[symbol].pop("_partial_tp_single_review_count", None)
                self.positions[symbol].pop("_partial_tp_triggered_levels", None)
            # Continue to normal SELL execution

        # --- Handle dust sweep triggered ---
        if dust_sweep_triggered and signal.action == "HOLD":
            async with self._positions_lock:
                self.positions[symbol].pop("_dust_sweep_triggered", None)
                if self.positions[symbol].get("_dust_keep_since") is None:
                    self.positions[symbol]["_dust_keep_since"] = time.time()
            self._state_dirty = True
            logger.info(f"LLM decided to hold dust for {symbol}")
            if self.notifier:
                await self.notifier.send_notification(
                    f"🧹 {display_symbol}: LLM decided to keep dust – holding.",
                    summary={"symbol": symbol, "action": "HOLD", "reason": "Dust kept by LLM"}
                )
            return True
        elif dust_sweep_triggered and signal.action == "SELL":
            async with self._positions_lock:
                self.positions[symbol].pop("_dust_sweep_triggered", None)
                self.positions[symbol].pop("_dust_sweep_review_count", None)
            logger.info(f"LLM decided to sell dust for {symbol}")
            await self._sweep_dust(symbol)
            return True

        return False

    async def _fetch_symbol_market_data(self, symbol: str, assigned_tf: str) -> Optional[Dict[str, Any]]:
        """Fetch all raw market data for a symbol: ticker, fundamentals, balance, OHLCV, and multi-TF indicators.

        Returns a dict with all fetched data, or None if ticker is unavailable.
        Both _process_symbol and _prepare_simulation_data use this to avoid duplication.
        """
        base_symbol = symbol.split("/")[0]
        is_btp = re.match(r'^IT[A-Z0-9]{10}$', base_symbol) is not None
        tf_seconds = self._timeframe_to_seconds(assigned_tf)

        # --- Fetch ticker ---
        async with self._exchange_semaphore:
            quotes = await self._get_quotes_async([base_symbol], timeout=45.0)
            ticker = quotes.get(base_symbol)
        if ticker is None:
            return None
        current_price = ticker['last']

        # --- Yahoo Finance fallback for missing bid/ask ---
        if ticker is not None and not is_btp:
            bid = ticker.get('bid')
            ask = ticker.get('ask')
            if bid is None or ask is None:
                yahoo = await asyncio.to_thread(get_yahoo_quote, base_symbol)
                if yahoo:
                    if bid is None:
                        ticker['bid'] = yahoo.get('bid')
                    if ask is None:
                        ticker['ask'] = yahoo.get('ask')
                    logger.info(f"Yahoo Finance quote merged for {symbol}: bid={ticker.get('bid')}, ask={ticker.get('ask')}")

        # --- Fetch fundamental data ---
        fundamentals = None
        if settings.YAHOO_FINANCE_ENABLED and not is_btp:
            fundamentals = await asyncio.to_thread(get_yahoo_fundamentals, base_symbol)

        # --- Fetch balance ---
        balance = await self._get_cached_balance()
        base_balance = balance.get(self.base_currency, 0.0)

        # --- Fetch OHLCV from database ---
        ohlcv_data = {}
        if settings.OHLCV_TIMEFRAMES:
            async def _fetch_ohlcv_tf(tf):
                try:
                    db_candles = await asyncio.to_thread(get_ohlcv, symbol, tf, limit=100)
                    if db_candles:
                        return tf, [[c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in db_candles]
                except Exception as e:
                    logger.debug(f"DB OHLCV fetch failed for {symbol} {tf}: {e}")
                return tf, None
            ohlcv_results = await asyncio.gather(*[_fetch_ohlcv_tf(tf) for tf in settings.OHLCV_TIMEFRAMES])
            for tf, candles in ohlcv_results:
                if candles:
                    ohlcv_data[tf] = candles

        # --- Compute multi-TF indicators ---
        _inds = await self._compute_multi_tf_indicators(symbol, ohlcv_data, assigned_tf)

        return {
            "ticker": ticker,
            "current_price": current_price,
            "fundamentals": fundamentals,
            "balance": balance,
            "base_balance": base_balance,
            "ohlcv_data": ohlcv_data,
            "is_btp": is_btp,
            "tf_seconds": tf_seconds,
            "multi_tf_indicators": _inds["multi_tf_indicators"],
            "multi_tf_raw_candles": _inds["multi_tf_raw_candles"],
            "atr": _inds["atr"],
            "rsi": _inds["rsi"],
            "macd": _inds["macd"],
            "macd_signal": _inds["macd_signal"],
            "macd_hist": _inds["macd_hist"],
            "bb_upper": _inds["bb_upper"],
            "bb_middle": _inds["bb_middle"],
            "bb_lower": _inds["bb_lower"],
            "ema_9": _inds["ema_9"],
            "ema_21": _inds["ema_21"],
            "stochastic_k": _inds["stochastic_k"],
            "stochastic_d": _inds["stochastic_d"],
            "adx": _inds["adx"],
            "plus_di": _inds["plus_di"],
            "minus_di": _inds["minus_di"],
            "obv": _inds["obv"],
            "mfi": _inds["mfi"],
            "cci": _inds["cci"],
            "williams_r": _inds["williams_r"],
            "ichimoku": _inds["ichimoku"],
            "donchian_channels": _inds["donchian_channels"],
            "parabolic_sar": _inds["parabolic_sar"],
            "keltner_channels": _inds["keltner_channels"],
            "vwap": _inds["vwap"],
            "daily_pivot_points": _inds["daily_pivot_points"],
        }

    async def _process_symbol(self, symbol_entry: Dict[str, str], trading_paused: bool = False):
        """Fetch market data, get LLM strategy, validate, and execute."""
        symbol = symbol_entry["symbol"]
        assigned_tf = symbol_entry["timeframe"]
        tf_seconds = self._timeframe_to_seconds(assigned_tf)

        # Pre‑compute the display symbol for all notifications in this method
        stock_name = await self._get_stock_name(symbol)
        display_symbol = self._format_symbol_display(symbol, stock_name, assigned_tf)

        base_symbol = symbol.split("/")[0]
        is_btp = re.match(r'^IT[A-Z0-9]{10}$', base_symbol) is not None

        # Read min viable trade amount (LLM override or settings default)
        min_viable_amount = settings.MIN_VIABLE_TRADE_AMOUNT
        try:
            raw = await asyncio.to_thread(self.redis.get, "trading:min_viable_trade_amount")
            if raw:
                min_viable_amount = float(raw)
        except Exception:
            pass

        # --- Maximum symbol tenure (per-symbol, set by LLM) ---
        max_tenure_hours = symbol_entry.get('max_tenure_hours')
        if max_tenure_hours is not None and max_tenure_hours > 0 and 'entry_time' in symbol_entry:
            tenure_seconds = max_tenure_hours * 3600
            if time.time() - symbol_entry['entry_time'] > tenure_seconds:
                logger.info(f"Max symbol tenure reached for {symbol} ({max_tenure_hours:.1f}h), forcing sell")
                signal = Signal(action="SELL", confidence=1.0, reasoning="Max symbol tenure reached")
                await self._execute_signal(symbol, signal, exit_reason="max_tenure")
                self._force_eval.pop(symbol, None)
                return

        # --- Cooldown after a losing trade (LLM-defined) ---
        # Only apply cooldown if there is NO open position for this symbol.
        # An open position must be managed regardless of cooldown.
        if symbol not in self.positions:
            last_loss = self.last_loss_time.get(symbol)
            if last_loss is not None:
                cooldown = self.cooldown_durations.get(symbol, 0)
                if cooldown > 0:
                    elapsed = time.time() - last_loss
                    if elapsed < cooldown:
                        remaining = cooldown - elapsed
                        logger.info(
                            f"Skipping {symbol}: cooldown active ({remaining:.0f}s remaining after loss)"
                        )
                        self._force_eval.pop(symbol, None)
                        return

        # Skip if there is already a queued order for this symbol
        async with self._queued_orders_lock:
            has_queued = any(q['symbol'] == symbol for q in self.queued_orders)
        if has_queued:
            logger.info(f"Skipping {display_symbol}: order already queued.")
            self._force_eval.pop(symbol, None)
            return

        # --- Max hold expired flag ---
        max_hold_expired = False
        max_hold_expired_count = 0
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos.get("_max_hold_expired"):
                max_hold_expired = True
                max_hold_expired_count = pos.get("_max_hold_expired_count", 1)

        # --- Stop-loss triggered flag ---
        stop_loss_triggered = False
        stop_loss_review_count = 0
        # --- Take-profit triggered flag ---
        take_profit_triggered = False
        take_profit_review_count = 0
        # --- Partial TP and dust sweep triggers ---
        partial_tp_triggered = False
        partial_tp_review_count = 0
        partial_tp_triggered_levels = []
        dust_sweep_triggered = False
        dust_sweep_review_count = 0
        if symbol in self.positions:
            pos = self.positions[symbol]
            stop_loss_triggered = pos.get("_stop_loss_triggered", False)
            stop_loss_review_count = pos.get("_stop_loss_review_count", 0)
            take_profit_triggered = pos.get("_take_profit_triggered", False)
            take_profit_review_count = pos.get("_take_profit_review_count", 0)
            partial_tp_triggered = pos.get("_partial_tp_triggered", False) or pos.get("_partial_tp_triggered_single", False)
            partial_tp_review_count = pos.get("_partial_tp_review_count", 0) or pos.get("_partial_tp_single_review_count", 0)
            partial_tp_triggered_levels = pos.get("_partial_tp_triggered_levels", [])
            dust_sweep_triggered = pos.get("_dust_sweep_triggered", False)
            dust_sweep_review_count = pos.get("_dust_sweep_review_count", 0)

        # Read LLM-decided review limits for the prompt
        max_sl_reviews_prompt = settings.MAX_STOP_LOSS_REVIEWS
        max_tp_reviews_prompt = settings.MAX_TAKE_PROFIT_REVIEWS
        try:
            raw = await asyncio.to_thread(self.redis.get, "trading:max_stop_loss_reviews")
            if raw:
                max_sl_reviews_prompt = int(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:max_take_profit_reviews")
            if raw:
                max_tp_reviews_prompt = int(raw)
        except Exception:
            pass

        max_partial_tp_reviews_prompt = settings.MAX_PARTIAL_TP_REVIEWS
        max_dust_sweep_reviews_prompt = settings.MAX_DUST_SWEEP_REVIEWS
        try:
            raw = await asyncio.to_thread(self.redis.get, "trading:max_partial_tp_reviews")
            if raw:
                max_partial_tp_reviews_prompt = int(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:max_dust_sweep_reviews")
            if raw:
                max_dust_sweep_reviews_prompt = int(raw)
        except Exception:
            pass

        # Scale stop-loss review limit for long-term timeframes
        if tf_seconds >= settings.LONG_TERM_TF_SECONDS:  # >= 1 month
            max_sl_reviews_prompt = min(max_sl_reviews_prompt, settings.LONG_TERM_MAX_STOP_LOSS_REVIEWS)
        elif tf_seconds >= 604_800:  # >= 1 week
            max_sl_reviews_prompt = min(max_sl_reviews_prompt, settings.WEEKLY_MAX_STOP_LOSS_REVIEWS)

        try:
            symbol_data = await self._fetch_symbol_market_data(symbol, assigned_tf)
            if symbol_data is None:
                logger.warning(f"No ticker data for {symbol}, skipping.")
                return
            ticker = symbol_data["ticker"]
            current_price = symbol_data["current_price"]
            fundamentals = symbol_data["fundamentals"]
            balance = symbol_data["balance"]
            base_balance = symbol_data["base_balance"]
            ohlcv_data = symbol_data["ohlcv_data"]
            is_btp = symbol_data["is_btp"]
            tf_seconds = symbol_data["tf_seconds"]
            multi_tf_indicators = symbol_data["multi_tf_indicators"]
            multi_tf_raw_candles = symbol_data["multi_tf_raw_candles"]
            atr = symbol_data["atr"]
            rsi = symbol_data["rsi"]
            macd = symbol_data["macd"]
            macd_signal = symbol_data["macd_signal"]
            macd_hist = symbol_data["macd_hist"]
            bb_upper = symbol_data["bb_upper"]
            bb_middle = symbol_data["bb_middle"]
            bb_lower = symbol_data["bb_lower"]
            ema_9 = symbol_data["ema_9"]
            ema_21 = symbol_data["ema_21"]
            stochastic_k = symbol_data["stochastic_k"]
            stochastic_d = symbol_data["stochastic_d"]
            adx = symbol_data["adx"]
            plus_di = symbol_data["plus_di"]
            minus_di = symbol_data["minus_di"]
            obv = symbol_data["obv"]
            mfi = symbol_data["mfi"]
            cci = symbol_data["cci"]
            williams_r = symbol_data["williams_r"]
            ichimoku = symbol_data["ichimoku"]
            donchian_channels = symbol_data["donchian_channels"]
            parabolic_sar = symbol_data["parabolic_sar"]
            keltner_channels = symbol_data["keltner_channels"]
            vwap = symbol_data["vwap"]
            daily_pivot_points = symbol_data["daily_pivot_points"]
            has_position = symbol in self.positions
            # If we have an open position, we must continue evaluating it for SELL signals
            # even when base_balance is 0 (all capital deployed) or effective_max_symbols is 0.
            if not has_position and (base_balance <= 0 or self.effective_max_symbols == 0):
                logger.warning(
                    f"Skipping {symbol}: {self.base_currency} balance={base_balance:.2f}, "
                    f"effective_max_symbols={self.effective_max_symbols}"
                )
                return
            # For positions we still need to manage, use a per_symbol_budget of 0
            # so the LLM knows no new capital is available for scaling in.
            if base_balance <= 0:
                logger.info(
                    f"Evaluating {symbol} for position management only "
                    f"(base_balance={base_balance:.2f}, no new capital available)."
                )

            # --- Skip if no meaningful market data is available ---
            # If we have no OHLCV candles at all, there is nothing for the LLM to analyse.
            # Skip to save costs and noise.
            no_ohlcv = (
                not ohlcv_data
                or all(len(candles) == 0 for candles in ohlcv_data.values())
            )
            if no_ohlcv:
                logger.info(
                    f"Skipping {symbol}: no OHLCV data – market data unavailable."
                )
                # Find the most recent OHLCV timestamp across all timeframes
                last_data_ts = None
                last_data_tf = None
                for tf in settings.OHLCV_TIMEFRAMES:
                    try:
                        ts = await asyncio.to_thread(get_latest_ohlcv_timestamp, symbol, tf)
                        if ts is not None and (last_data_ts is None or ts > last_data_ts):
                            last_data_ts = ts
                            last_data_tf = tf
                    except Exception:
                        pass

                if last_data_ts is not None:
                    age_seconds = time.time() - (last_data_ts / 1000.0)
                    if age_seconds < 3600:
                        age_str = f"{age_seconds/60:.0f} minutes ago"
                    elif age_seconds < 86400:
                        age_str = f"{age_seconds/3600:.1f} hours ago"
                    else:
                        age_str = f"{age_seconds/86400:.1f} days ago"
                    msg = (
                        f"⚠️ Skipping {display_symbol}: no OHLCV data available. "
                        f"Last data: {last_data_tf} candle from {age_str}. "
                        f"Try a manual force-download via the dashboard or Telegram."
                    )
                else:
                    msg = (
                        f"⚠️ Skipping {display_symbol}: no OHLCV data available. "
                        f"No historical data found in database. "
                        f"Run a force-download via the dashboard or Telegram to populate market data."
                    )

                no_ohlcv_notify_key = f"trading:no_ohlcv_notify:{symbol}"
                should_notify = True
                try:
                    last_notify_raw = await asyncio.to_thread(self.redis.get, no_ohlcv_notify_key)
                    if last_notify_raw:
                        if (time.time() - float(last_notify_raw)) < 3600:
                            should_notify = False
                except Exception:
                    pass

                if should_notify and self.notifier:
                    await self.notifier.send_notification(
                        msg,
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "No OHLCV data",
                            "last_data_timestamp": last_data_ts,
                            "last_data_timeframe": last_data_tf,
                        }
                    )
                    try:
                        await asyncio.to_thread(self.redis.setex, no_ohlcv_notify_key, 3600, str(time.time()))
                    except Exception:
                        pass

                self._force_eval.pop(symbol, None)
                return

            open_positions = [
                pos for pos in self.positions.values() if pos.get("symbol") == symbol
            ]

            # Compute per-symbol budget for this symbol
            per_symbol_budget = base_balance / self.effective_max_symbols if self.effective_max_symbols > 0 else 0.0

            perf = await asyncio.to_thread(self._compute_performance_metrics)
            trade_pattern_analysis = await asyncio.to_thread(self._compute_trade_pattern_analysis)

            # --- Detect upcoming corporate events for this symbol ---
            symbol_event = None
            if settings.NEWS_ENABLED and detect_upcoming_events is not None:
                try:
                    symbol_event = await asyncio.to_thread(detect_upcoming_events, symbol)
                except Exception:
                    pass

            # --- Compute additional metrics for the LLM ---
            # Build indicator config from position or defaults
            ind_cfg = self.positions.get(symbol, {}).get('indicator_config') if symbol in self.positions else None

            _ctx = await self._gather_prompt_context(
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
            atr_multi_tf = _ctx["atr_multi_tf"]
            atr_percentile = _ctx["atr_percentile"]
            market_regime = _ctx["market_regime"]
            raw_candles = _ctx["raw_candles"]
            historical_ohlcv = _ctx["historical_ohlcv"]
            unrealized_pnl = _ctx["unrealized_pnl"]
            position_info = _ctx["position_info"]
            recent_trades_summary = _ctx["recent_trades_summary"]
            min_order_amount = _ctx["min_order_amount"]
            min_order_cost = _ctx["min_order_cost"]
            past_trades = _ctx["past_trades"]
            aggregate_sentiment = _ctx["aggregate_sentiment"]
            sentiment_trend_val = _ctx["sentiment_trend_val"]
            volume_trend_val = _ctx["volume_trend_val"]
            full_market_breadth = _ctx["full_market_breadth"]
            session_info = _ctx["session_info"]
            minutes_to_market_close = _ctx["minutes_to_market_close"]
            global_risk_mult = _ctx["global_risk_mult"]
            max_port_exp = _ctx["max_port_exp"]
            max_port_risk = _ctx["max_port_risk"]
            partial_tp_executed_levels = _ctx["partial_tp_executed_levels"]
            min_stop_atr_mult = _ctx["min_stop_atr_mult"]
            min_hold_time_mult = _ctx["min_hold_time_mult"]
            global_min_rr = _ctx["global_min_rr"]
            historical_backtest_results = _ctx["historical_backtest_results"]

            # --- Compute portfolio exposure summary for the prompt ---
            _portfolio = await self._compute_portfolio_exposure_summary(base_balance)
            portfolio_total_value = _portfolio["portfolio_total_value"]
            portfolio_exposure = _portfolio["portfolio_exposure"]
            portfolio_stop_risk = _portfolio["portfolio_stop_risk"]
            portfolio_exposure_pct = _portfolio["portfolio_exposure_pct"]
            portfolio_stop_risk_pct = _portfolio["portfolio_stop_risk_pct"]
            portfolio_available_capital = _portfolio["portfolio_available_capital"]

            async with self._cycle_spent_lock:
                remaining = max(0.0, base_balance - self._cycle_spent)

            # --- Step 1a: Analysis call (focused on market analysis only) ---
            analysis_prompt = await asyncio.to_thread(
                build_analysis_prompt,
                symbol=symbol,
                ticker=ticker,
                balance=balance,
                open_positions=open_positions,
                per_symbol_budget=per_symbol_budget,
                max_symbols=self.effective_max_symbols,
                base_currency=self.base_currency,
                performance=perf,
                ohlcv_data=ohlcv_data,
                assigned_timeframe=assigned_tf,
                atr=atr,
                atr_multi_tf=atr_multi_tf,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                macd_hist=macd_hist,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                ema_9=ema_9,
                ema_21=ema_21,
                stochastic_k=stochastic_k,
                stochastic_d=stochastic_d,
                adx=adx,
                plus_di=plus_di,
                minus_di=minus_di,
                obv=obv,
                mfi=mfi,
                cci=cci,
                williams_r=williams_r,
                unrealized_pnl=unrealized_pnl,
                position_info=position_info,
                drawdown_pct=perf.get("equity_curve", {}).get("drawdown_pct"),
                raw_candles=raw_candles,
                recent_trades=recent_trades_summary,
                historical_ohlcv=historical_ohlcv,
                min_order_amount=min_order_amount,
                min_order_cost=min_order_cost,
                all_symbols=self.current_symbols,
                past_trades=past_trades,
                cycle_spent=self._cycle_spent,
                remaining_balance=remaining,
                market_regime=market_regime,
                multi_tf_raw_candles=multi_tf_raw_candles,
                multi_tf_indicators=multi_tf_indicators,
                session_info=session_info,
                sentiment_trend=sentiment_trend_val,
                volume_trend=volume_trend_val,
                ichimoku=ichimoku,
                market_breadth=getattr(self, '_market_breadth', None),
                full_market_breadth=full_market_breadth,
                parabolic_sar=parabolic_sar,
                keltner_channels=keltner_channels,
                donchian_channels=donchian_channels,
                atr_percentile=atr_percentile,
                global_risk_multiplier=global_risk_mult,
                trading_paused=trading_paused,
                max_hold_expired=max_hold_expired,
                max_hold_expired_count=max_hold_expired_count,
                stop_loss_triggered=stop_loss_triggered,
                stop_loss_review_count=stop_loss_review_count,
                take_profit_triggered=take_profit_triggered,
                take_profit_review_count=take_profit_review_count,
                partial_tp_triggered=partial_tp_triggered,
                partial_tp_review_count=partial_tp_review_count,
                partial_tp_triggered_levels=partial_tp_triggered_levels if partial_tp_triggered_levels else None,
                partial_tp_executed_levels=partial_tp_executed_levels,
                dust_sweep_triggered=dust_sweep_triggered,
                dust_sweep_review_count=dust_sweep_review_count,
                max_stop_loss_reviews=max_sl_reviews_prompt,
                max_take_profit_reviews=max_tp_reviews_prompt,
                max_partial_tp_reviews=max_partial_tp_reviews_prompt,
                max_dust_sweep_reviews=max_dust_sweep_reviews_prompt,
                portfolio_exposure_pct=portfolio_exposure_pct,
                portfolio_stop_risk_pct=portfolio_stop_risk_pct,
                portfolio_total_value=portfolio_total_value,
                portfolio_open_count=len(self.positions),
                portfolio_available_capital=portfolio_available_capital,
                last_decision=self._last_decisions.get(symbol),
                minutes_to_market_close=minutes_to_market_close,
                current_strategy_interval_seconds=self._strategy_intervals.get(symbol, tf_seconds),
                max_portfolio_exposure_pct=max_port_exp,
                max_portfolio_stop_risk_pct=max_port_risk,
                trade_pattern_analysis=trade_pattern_analysis,
                symbol_event=symbol_event,
                queued_orders=self.queued_orders,
                fundamentals=fundamentals,
                vwap=vwap,
                daily_pivot_points=daily_pivot_points,
                min_hold_time_mult=min_hold_time_mult,
                min_stop_atr_mult=min_stop_atr_mult,
                min_viable_trade_amount=min_viable_amount,
                historical_backtest_results=historical_backtest_results,
            )
            # Add quote staleness warning if the price data is outdated
            staleness_warning = self._get_quote_staleness_warning(ticker)
            if staleness_warning:
                analysis_prompt += staleness_warning
            # Add auto-resume note so the LLM sees this context in per-symbol decisions
            last_auto_resume_raw = await asyncio.to_thread(self.redis.get, "trading:last_auto_resume")
            if last_auto_resume_raw:
                try:
                    last_auto_resume_ts = float(last_auto_resume_raw)
                    seconds_since = time.time() - last_auto_resume_ts
                    if seconds_since < self._symbol_reevaluation_interval * 2:
                        minutes_since = seconds_since / 60
                        analysis_prompt += (
                            f"\n**NOTE:** Trading was auto‑resumed {minutes_since:.1f} minutes ago after a pause. "
                            "Market conditions may not have changed significantly. "
                            "Consider whether conditions have actually improved enough to justify trading. "
                            "If you decide to pause again, set a longer `pause_duration_seconds` (e.g., 1800–7200) "
                            "to allow conditions to evolve; a very short pause will likely lead to the same outcome.\n"
                        )
                except (ValueError, TypeError):
                    pass
            logger.info(f"LLM Step 1a analysis prompt for {symbol}: {len(analysis_prompt)} chars")
            # Build a market snapshot dict for caching (per-symbol)
            market_snapshot = {
                "symbol": symbol,
                "ticker": ticker,
                "staleness_warning": staleness_warning,
                "balance": balance,
                "open_positions": open_positions,
                "per_symbol_budget": per_symbol_budget,
                "max_symbols": self.effective_max_symbols,
                "performance": perf,
                "ohlcv_data": ohlcv_data,
                "assigned_timeframe": assigned_tf,
                "atr": atr,
                "atr_multi_tf": atr_multi_tf,
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_hist": macd_hist,
                "bb_upper": bb_upper,
                "bb_middle": bb_middle,
                "bb_lower": bb_lower,
                "ema_9": ema_9,
                "ema_21": ema_21,
                "stochastic_k": stochastic_k,
                "stochastic_d": stochastic_d,
                "adx": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "obv": obv,
                "mfi": mfi,
                "cci": cci,
                "williams_r": williams_r,
                "ichimoku": ichimoku,
                "donchian_channels": donchian_channels,
                "drawdown_pct": perf.get("equity_curve", {}).get("drawdown_pct"),
                "raw_candles": raw_candles,
                "recent_trades": recent_trades_summary,
                "historical_ohlcv": historical_ohlcv,
                "min_order_amount": min_order_amount,
                "min_order_cost": min_order_cost,
                "all_symbols": self.current_symbols,
                "past_trades": past_trades,
                "aggregate_sentiment": aggregate_sentiment,
                "cycle_spent": self._cycle_spent,
                "remaining_balance": remaining,
                "market_regime": market_regime,
                "multi_tf_raw_candles": multi_tf_raw_candles,
                "multi_tf_indicators": multi_tf_indicators,
                "session_info": session_info,
                "sentiment_trend": sentiment_trend_val,
                "volume_trend": volume_trend_val,
                "market_breadth": getattr(self, '_market_breadth', None),
                "full_market_breadth": full_market_breadth,
                "parabolic_sar": parabolic_sar,
                "keltner_channels": keltner_channels,
                "atr_percentile": atr_percentile,
                "global_risk_multiplier": global_risk_mult,
                "trading_paused": trading_paused,
                "last_decision": self._last_decisions.get(symbol),
            }
            market_hash = compute_market_hash(market_snapshot)
            # Determine whether we even need to call the LLM, and if so which model to use
            is_critical = max_hold_expired or stop_loss_triggered or take_profit_triggered or partial_tp_triggered or dust_sweep_triggered
            has_position = symbol in self.positions

            if await self._should_skip_llm_eval(
                symbol=symbol,
                current_price=current_price,
                atr=atr,
                rsi=rsi,
                macd_hist=macd_hist,
                atr_percentile=atr_percentile,
                market_regime=market_regime,
                sentiment_trend_val=sentiment_trend_val,
                timeframe_seconds=tf_seconds,
                has_position=has_position,
                is_critical=is_critical,
            ):
                logger.info(f"Skipping LLM for {symbol}: market unchanged, no strong signals.")
                # Do NOT update the snapshot here. Keeping the last actual LLM evaluation
                # values ensures the 3× interval safety net fires and cumulative change
                # detection works. Updating the snapshot on every skip would reset the
                # clock and prevent forced re-evaluation from ever triggering.
                # Clear any force‑eval flag for this symbol
                self._force_eval.pop(symbol, None)
                return

            strategy_model_type = self._choose_model_tier(
                atr=atr,
                atr_percentile=atr_percentile,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                macd_hist=macd_hist,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                ema_9=ema_9,
                ema_21=ema_21,
                stochastic_k=stochastic_k,
                adx=adx,
                plus_di=plus_di,
                minus_di=minus_di,
                mfi=mfi,
                cci=cci,
                williams_r=williams_r,
                ichimoku=ichimoku,
                market_regime=market_regime,
                market_breadth=getattr(self, '_market_breadth', None),
                full_market_breadth=full_market_breadth,
                sentiment_trend_val=sentiment_trend_val,
                volume_trend=volume_trend_val,
                unrealized_pnl=unrealized_pnl,
                drawdown_pct=perf.get("equity_curve", {}).get("drawdown_pct"),
                portfolio_exposure_pct=portfolio_exposure_pct,
                portfolio_stop_risk_pct=portfolio_stop_risk_pct,
                is_critical=is_critical,
                trading_paused=trading_paused,
                symbol_event=symbol_event,
                fundamentals=fundamentals,
                consecutive_losses=perf.get("equity_curve", {}).get("consecutive_losses", 0),
                current_price=ticker['last'],
            )

            # Compute prompt complexity for temperature selection
            _conflicting = False
            if rsi is not None and macd_hist is not None:
                if (rsi < 30 and macd_hist < 0) or (rsi > 70 and macd_hist > 0):
                    _conflicting = True
            strategy_complexity = self._compute_prompt_complexity(
                num_candidates=len(self.current_symbols),
                volatility_percentile=atr_percentile,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                macd_hist=macd_hist,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                ema_9=ema_9,
                ema_21=ema_21,
                stochastic_k=stochastic_k,
                adx=adx,
                plus_di=plus_di,
                minus_di=minus_di,
                mfi=mfi,
                cci=cci,
                williams_r=williams_r,
                ichimoku=ichimoku,
                market_breadth=getattr(self, '_market_breadth', None),
                full_market_breadth=full_market_breadth,
                sentiment_trend_magnitude=abs(sentiment_trend_val) if sentiment_trend_val is not None else None,
                volume_trend=volume_trend_val,
                market_regime=market_regime,
                unrealized_pnl=unrealized_pnl,
                drawdown_pct=perf.get("equity_curve", {}).get("drawdown_pct"),
                portfolio_exposure_pct=portfolio_exposure_pct,
                portfolio_stop_risk_pct=portfolio_stop_risk_pct,
                is_critical=is_critical,
                trading_paused=trading_paused,
                symbol_event=symbol_event,
                fundamentals=fundamentals,
                consecutive_losses=perf.get("equity_curve", {}).get("consecutive_losses", 0),
                current_price=ticker['last'],
                conflicting_signals=_conflicting,
            )
            effective_temp = self._get_effective_temperature(strategy_model_type, strategy_complexity)

            # --- Step 1a: Call LLM for analysis ---
            analysis_result = None
            llm_provider = None
            llm_model = None
            try:
                step1a_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        get_cached_llm_response,
                        compact_prompt(analysis_prompt),
                        COMPACTED_SYSTEM_PROMPT,
                        60,
                        market_hash=market_hash,
                        model_type=strategy_model_type,
                        temperature=effective_temp,
                    ),
                    timeout=settings.LLM_TIMEOUT
                )
                step1a_response = step1a_result["response"]
                llm_provider = step1a_result["provider"]
                llm_model = step1a_result["model"]
                logger.info(f"LLM Step 1a (analysis) completed for {symbol} (provider={llm_provider}, model={llm_model})")
                analysis_result = self._parse_analysis_response(step1a_response)
                if analysis_result is None:
                    logger.warning(f"Failed to parse Step 1a analysis response for {symbol}. Retrying with correction.")
                    correction_prompt = (
                        "Your previous response was not valid JSON. "
                        "You MUST output ONLY a single JSON object with fields: "
                        '"action", "confidence", "reasoning", "strategy_direction". '
                        "No markdown fences, no explanations, no extra text. "
                        "Here is the original request:\n\n" + analysis_prompt
                    )
                    retry_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            get_cached_llm_response,
                            compact_prompt(correction_prompt),
                            COMPACTED_SYSTEM_PROMPT, 30,
                            model_type="actuator",
                            temperature=effective_temp,
                        ),
                        timeout=settings.LLM_TIMEOUT
                    )
                    analysis_result = self._parse_analysis_response(retry_result["response"])
                    llm_provider = retry_result["provider"]
                    llm_model = retry_result["model"]
                # Update snapshot after a real LLM call
                self._update_last_eval_snapshot(symbol, current_price, rsi, macd_hist)
                self._force_eval.pop(symbol, None)
            except asyncio.TimeoutError:
                logger.warning(f"LLM Step 1a (analysis) timed out for {symbol}.")
                if is_critical:
                    reason = "LLM timeout"
                    if max_hold_expired:
                        reason = "Max hold expired, LLM timeout"
                    elif stop_loss_triggered:
                        reason = "Stop-loss triggered, LLM timeout"
                    elif take_profit_triggered:
                        reason = "Take-profit triggered, LLM timeout"
                    elif partial_tp_triggered:
                        reason = "Partial TP triggered, LLM timeout"
                    elif dust_sweep_triggered:
                        reason = "Dust sweep triggered, LLM timeout"
                    logger.warning(f"Forcing SELL for {symbol} due to {reason}")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⏱️ LLM timeout for {display_symbol} with critical flag – forcing SELL.",
                            summary={"symbol": symbol, "action": "SELL", "reason": reason, "model_type": strategy_model_type}
                        )
                    await self._execute_signal(
                        symbol,
                        Signal(action="SELL", confidence=1.0, reasoning=reason),
                        exit_reason=reason.replace(" ", "_").lower()
                    )
                    return
                # Non-critical timeout: fall through to fallback HOLD
                self._force_eval.pop(symbol, None)
                # Fall through to fallback HOLD below
            except Exception as e:
                logger.error(f"LLM Step 1a failed for {symbol}: {e}")
                self._force_eval.pop(symbol, None)
                # Fall through to fallback HOLD below

            if analysis_result is None:
                logger.warning(f"Step 1a analysis failed for {symbol} after all retries. Using fallback HOLD.")
                self._force_eval.pop(symbol, None)
                # Create a fallback HOLD signal so the bot continues functioning
                preliminary_signal = self._create_fallback_hold_signal(
                    symbol, "LLM Step 1a analysis failed after retries", strategy_model_type
                )
                signal = preliminary_signal
                llm_provider = "fallback"
                llm_model = "default_hold"
                combined_bt_summary = ""
                _skip_backtest = True
            # If analysis says HOLD with no position, skip parameter selection entirely
            elif analysis_result.get("action") == "HOLD" and not has_position:
                logger.info(f"Step 1a analysis returned HOLD with no position for {symbol}. Skipping Step 1b.")
                # Create a minimal preliminary signal for the notification flow
                preliminary_signal = Signal(
                    action="HOLD",
                    confidence=analysis_result.get("confidence", 0.0),
                    reasoning=analysis_result.get("reasoning", ""),
                )
                preliminary_signal.model_type = strategy_model_type
                preliminary_signal.llm_provider = llm_provider
                preliminary_signal.llm_model = llm_model
                # Skip backtests and Step 2 — go directly to notification
                signal = preliminary_signal
                combined_bt_summary = ""
                _skip_backtest = True
            else:
                _skip_backtest = False

            if not _skip_backtest:
                # --- Step 1b: Call LLM for backtest variants and parameters ---
                variants_prompt = await asyncio.to_thread(
                    build_backtest_variants_prompt,
                    symbol=symbol,
                    analysis=analysis_result,
                    ticker=ticker,
                    current_price=current_price,
                    atr=atr,
                    assigned_timeframe=assigned_tf,
                    base_currency=self.base_currency,
                    base_balance=base_balance,
                    per_symbol_budget=per_symbol_budget,
                    min_order_amount=min_order_amount,
                    min_order_cost=min_order_cost,
                    remaining_balance=remaining,
                    portfolio_total_value=portfolio_total_value,
                    portfolio_exposure_pct=portfolio_exposure_pct,
                    portfolio_stop_risk_pct=portfolio_stop_risk_pct,
                    portfolio_available_capital=portfolio_available_capital,
                    max_portfolio_exposure_pct=max_port_exp,
                    max_portfolio_stop_risk_pct=max_port_risk,
                    global_risk_multiplier=global_risk_mult,
                    min_stop_atr_mult=min_stop_atr_mult,
                    min_hold_time_mult=min_hold_time_mult,
                    trading_paused=trading_paused,
                    has_position=has_position,
                )
                logger.info(f"LLM Step 1b variants prompt for {symbol}: {len(variants_prompt)} chars")

                # Use a different market hash for Step 1b (include analysis to differentiate)
                variants_market_hash = compute_market_hash({
                    **market_snapshot,
                    "step": "1b",
                    "analysis": analysis_result,
                })

                try:
                    step1b_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            get_cached_llm_response,
                            compact_prompt(variants_prompt),
                            COMPACTED_SYSTEM_PROMPT,
                            60,
                            market_hash=variants_market_hash,
                            model_type=strategy_model_type,
                            temperature=effective_temp,
                        ),
                        timeout=settings.LLM_TIMEOUT
                    )
                    step1b_response = step1b_result["response"]
                    llm_provider = step1b_result["provider"]
                    llm_model = step1b_result["model"]
                    logger.info(f"LLM Step 1b (variants) completed for {symbol} (provider={llm_provider}, model={llm_model})")
                except asyncio.TimeoutError:
                    logger.warning(f"LLM Step 1b (variants) timed out for {symbol}. Using Step 1a analysis as fallback.")
                    step1b_response = json.dumps({
                        "action": analysis_result.get("action", "HOLD"),
                        "confidence": analysis_result.get("confidence", 0.0),
                        "reasoning": analysis_result.get("reasoning", ""),
                        "strategy": {
                            "type": "fallback",
                            "parameters": {},
                        },
                    })
                except Exception as e:
                    logger.error(f"LLM Step 1b failed for {symbol}: {e}. Using Step 1a analysis as fallback.")
                    step1b_response = json.dumps({
                        "action": analysis_result.get("action", "HOLD"),
                        "confidence": analysis_result.get("confidence", 0.0),
                        "reasoning": analysis_result.get("reasoning", ""),
                        "strategy": {
                            "type": "fallback",
                            "parameters": {},
                        },
                    })

                # --- Parse Step 1b response ---
                try:
                    preliminary_strategy = create_strategy_from_llm(step1b_response)
                except ValueError as e:
                    logger.warning(f"LLM Step 1b response parse failed for {symbol}: {e}. Retrying with correction prompt.")
                    correction_prompt = (
                        "Your previous response was not valid JSON. "
                        "You MUST output ONLY a single JSON object, with no markdown fences, no explanations, no extra text. "
                        "Here is the original request:\n\n" + variants_prompt
                    )
                    try:
                        response2 = await asyncio.wait_for(
                            asyncio.to_thread(
                                get_cached_llm_response, compact_prompt(correction_prompt), COMPACTED_SYSTEM_PROMPT, 30,
                                model_type="actuator",
                                temperature=effective_temp,
                            ),
                            timeout=settings.LLM_TIMEOUT
                        )
                        preliminary_strategy = create_strategy_from_llm(response2["response"])
                        llm_provider = response2["provider"]
                        llm_model = response2["model"]
                    except Exception as e2:
                        logger.error(f"LLM Step 1b response still invalid after retry for {symbol}: {e2}")
                        preliminary_strategy = LLMStrategy(self._create_fallback_hold_signal(
                            symbol, "Failed to parse LLM Step 1b response after retry", strategy_model_type
                        ))

                preliminary_signal = preliminary_strategy.generate_signal({})
                preliminary_signal.model_type = strategy_model_type
                preliminary_signal.llm_provider = llm_provider
                preliminary_signal.llm_model = llm_model

                # --- Step 2: Run backtest(s) and ask LLM for final decision ---
                signal, combined_bt_summary, llm_provider, llm_model = await self._run_backtest_and_final_decision(
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

            current_price = ticker['last']

            validated = validate_signal(
                signal,
                atr=atr,
                price=current_price,
                timeframe_seconds=tf_seconds,
                min_stop_atr_mult=min_stop_atr_mult,
                min_hold_time_mult=min_hold_time_mult,
                global_min_risk_reward_ratio=global_min_rr,
            )
            validated.model_type = getattr(signal, 'model_type', None)
            validated.backtest_summary = getattr(signal, 'backtest_summary', None)

            # Clear _needs_risk_params flag if the LLM has now provided risk parameters
            if symbol in self.positions:
                _pos = self.positions[symbol]
                if _pos.get("_needs_risk_params"):
                    if _pos.get("stop_loss") is not None and _pos.get("take_profit") is not None:
                        _pos.pop("_needs_risk_params", None)
                        _pos.pop("_needs_risk_params_attempts", None)
                        logger.info(f"Risk parameters obtained for {symbol}; cleared _needs_risk_params flag.")

            # Log and notify the decision
            logger.info(f"Decision for {symbol}: {validated.action} (confidence: {validated.confidence:.2f})")
            # Store the last decision for the next prompt cycle
            params = signal.strategy_params
            self._last_decisions[symbol] = {
                "action": validated.action,
                "confidence": validated.confidence,
                "reasoning": validated.reasoning[:300],
                "strategy_type": signal.strategy_type,
                "timestamp": time.time(),
                "stop_loss_pct": params.get("stop_loss_pct") if params else None,
                "take_profit_pct": params.get("take_profit_pct") if params else None,
                "position_size_fraction": params.get("position_size_fraction") if params else None,
                "stop_loss_method": params.get("stop_loss_method") if params else None,
            }
            self._state_dirty = True
            # Compute trade amount for display in the signals card
            _params = signal.strategy_params or {}
            _psf = _params.get("position_size_fraction")
            if validated.action == "BUY" and _psf is not None:
                _trade_amount = base_balance * float(_psf)
            elif validated.action == "SELL" and symbol in self.positions:
                _pos = self.positions[symbol]
                _trade_amount = _pos.get("amount", 0) * current_price
            else:
                _trade_amount = 0.0

            # Extract strategy parameters for the signal detail modal
            _sig_params = signal.strategy_params or {}
            _entry_cond_str = None
            if validated.entry_condition:
                _ec = validated.entry_condition
                _etype = _ec.get("type", "")
                if _etype == "limit_price":
                    _entry_cond_str = f"Wait for price to drop to {_ec.get('price', '?')} (timeout: {_ec.get('timeout_seconds', '?')}s)"
                elif _etype == "rsi_threshold":
                    _entry_cond_str = f"Wait for RSI(14) to fall below {_ec.get('rsi_below', '?')} (timeout: {_ec.get('timeout_seconds', '?')}s)"
                elif _etype == "delay":
                    _entry_cond_str = f"Wait {_ec.get('delay_seconds', '?')}s before executing"
                elif _etype == "indicator_combo":
                    _conds = _ec.get("conditions", [])
                    _cond_strs = []
                    for c in _conds:
                        _cond_strs.append(f"{c.get('indicator','?')} {c.get('direction','?')} {c.get('threshold','?')}")
                    _entry_cond_str = f"Wait for ALL: {', '.join(_cond_strs)} (timeout: {_ec.get('timeout_seconds', '?')}s)"
            _sl_method = _sig_params.get("stop_loss_method", "fixed")
            _sl_str = ""
            if _sl_method == "atr_multiple":
                _sl_str = f"ATR × {_sig_params.get('stop_loss_atr_multiple', '?')} (fallback: {_sig_params.get('stop_loss_pct', '?')})"
            else:
                _sl_str = f"{_sig_params.get('stop_loss_pct', '?')}"
            _tp_str = ""
            if _sig_params.get("take_profit_atr_multiple"):
                _tp_str = f"ATR × {_sig_params.get('take_profit_atr_multiple', '?')} (fallback: {_sig_params.get('take_profit_pct', '?')})"
            else:
                _tp_str = f"{_sig_params.get('take_profit_pct', '?')}"

            # Record signal for the web dashboard
            self.recent_signals.append({
                "symbol": symbol,
                "display_symbol": display_symbol,
                "stock_name": stock_name,
                "timeframe": assigned_tf,
                "action": validated.action,
                "confidence": validated.confidence,
                "reasoning": validated.reasoning or "",
                "strategy_type": signal.strategy_type,
                "model_type": getattr(validated, 'model_type', None),
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "trade_amount": round(_trade_amount, 2),
                "base_currency": self.base_currency,
                "timestamp": time.time(),
                "entry_condition": _entry_cond_str,
                "stop_loss": _sl_str,
                "take_profit": _tp_str,
                "position_size_fraction": _sig_params.get("position_size_fraction"),
                "trailing_stop": _sig_params.get("trailing_stop"),
                "trailing_stop_distance_pct": _sig_params.get("trailing_stop_distance_pct"),
                "max_hold_time_seconds": _sig_params.get("max_hold_time_seconds"),
                "cooldown_after_loss_seconds": _sig_params.get("cooldown_after_loss_seconds"),
                "order_type": signal.order_type,
                "limit_price": _sig_params.get("limit_price"),
            })
            # Keep only the last 50 signals
            if len(self.recent_signals) > 50:
                self.recent_signals = self.recent_signals[-50:]
            params = signal.strategy_params or {}
            # --- Format symbol for notification ---
            stock_name = await self._get_stock_name(symbol)
            display_symbol = self._format_symbol_display(symbol, stock_name, assigned_tf)
            if self.notifier:
                emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}.get(validated.action, "❓")
                paused_tag = " (PAUSED)" if trading_paused and validated.action == "BUY" else ""
                # Build a short indicator summary
                ind_parts = []
                if rsi is not None:
                    ind_parts.append(f"RSI={rsi:.1f}")
                if macd is not None and macd_signal is not None:
                    ind_parts.append(f"MACD={macd:.4f}/{macd_signal:.4f}")
                    if macd_hist is not None:
                        ind_parts.append(f"Hist={macd_hist:.4f}")
                if bb_upper is not None:
                    ind_parts.append(f"BB={bb_lower:.2f}/{bb_middle:.2f}/{bb_upper:.2f}")
                if ema_9 is not None and ema_21 is not None:
                    ind_parts.append(f"EMA9/21={ema_9:.2f}/{ema_21:.2f}")
                if stochastic_k is not None:
                    ind_parts.append(f"StochK={stochastic_k:.1f}")
                    if stochastic_d is not None:
                        ind_parts.append(f"StochD={stochastic_d:.1f}")
                if adx is not None:
                    ind_parts.append(f"ADX={adx:.1f}")
                    if plus_di is not None and minus_di is not None:
                        ind_parts.append(f"+DI={plus_di:.1f}/-DI={minus_di:.1f}")
                if atr is not None:
                    ind_parts.append(f"ATR={atr:.4f}")
                if obv is not None:
                    ind_parts.append(f"OBV={obv:.2f}")
                if mfi is not None:
                    ind_parts.append(f"MFI={mfi:.2f}")
                if cci is not None:
                    ind_parts.append(f"CCI={cci:.2f}")
                if williams_r is not None:
                    ind_parts.append(f"WR={williams_r:.2f}")
                if ichimoku is not None:
                    ind_parts.append(f"Ichi T={ichimoku['tenkan_sen']:.2f}/K={ichimoku['kijun_sen']:.2f}")
                    ind_parts.append(f"Cloud={ichimoku['cloud_bottom']:.2f}-{ichimoku['cloud_top']:.2f}")
                if donchian_channels is not None:
                    ind_parts.append(f"Donch={donchian_channels['lower']:.2f}/{donchian_channels['middle']:.2f}/{donchian_channels['upper']:.2f}")
                if parabolic_sar is not None:
                    ind_parts.append(f"SAR={parabolic_sar:.4f}")
                if keltner_channels is not None:
                    ind_parts.append(f"Kelt={keltner_channels['lower']:.4f}/{keltner_channels['middle']:.4f}/{keltner_channels['upper']:.4f}")
                indicator_str = " | ".join(ind_parts) if ind_parts else "No indicators (insufficient OHLCV data)"
                sentiment_str = await self._get_sentiment_str(symbol)
                reasoning_str = f" – {validated.reasoning}" if validated.reasoning else ""
                msg = f"{emoji} {display_symbol}: {validated.action} (confidence: {validated.confidence:.2f}){reasoning_str}{paused_tag}"
                if sentiment_str:
                    msg += f"\n{sentiment_str}"
                if getattr(validated, 'backtest_summary', None):
                    msg += f"\n📈 Backtest: {validated.backtest_summary}"
                msg += f"\n📊 {indicator_str}"
                # Build summary dict for logging
                decision_summary = {
                    "symbol": symbol,
                    "action": validated.action,
                    "confidence": validated.confidence,
                    "reason": validated.reasoning[:200],
                    "sentiment": aggregate_sentiment,
                    "indicators": {
                        "rsi": rsi,
                        "macd": macd,
                        "macd_signal": macd_signal,
                        "atr": atr,
                        "adx": adx,
                        "bb_upper": bb_upper,
                        "bb_lower": bb_lower,
                        "ema_9": ema_9,
                        "ema_21": ema_21,
                        "stochastic_k": stochastic_k,
                        "mfi": mfi,
                        "cci": cci,
                        "williams_r": williams_r,
                        "ichimoku": ichimoku,
                        "donchian_channels": donchian_channels,
                    },
                    "backtest": getattr(validated, 'backtest_summary', None),
                    "strategy_type": signal.strategy_type,
                    "market_regime": market_regime,
                    "model_type": getattr(validated, 'model_type', None),
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                }
                await self.notifier.send_notification(msg, summary=decision_summary)

            params = signal.strategy_params or {}

            # --- Handle triggered position flags (max hold, stop loss, take profit, partial TP, dust sweep) ---
            if await self._handle_triggered_flags(
                symbol=symbol,
                display_symbol=display_symbol,
                signal=signal,
                validated=validated,
                assigned_tf=assigned_tf,
                current_price=current_price,
                atr=atr,
                ticker=ticker,
                max_hold_expired=max_hold_expired,
                stop_loss_triggered=stop_loss_triggered,
                take_profit_triggered=take_profit_triggered,
                partial_tp_triggered=partial_tp_triggered,
                dust_sweep_triggered=dust_sweep_triggered,
                strategy_model_type=strategy_model_type,
                llm_provider=llm_provider,
                llm_model=llm_model,
            ):
                return

            # --- LLM‑controlled trade filters ---

            # Compute stop-loss percentage for max risk cap (needed for slippage check)
            sl_pct = None
            if validated.action == "BUY":
                stop_method = params.get("stop_loss_method", "fixed")
                if stop_method == "atr_multiple" and atr is not None and atr > 0:
                    atr_mult = params["stop_loss_atr_multiple"]
                    sl_pct = (atr_mult * atr) / current_price
                else:
                    sl_pct = params.get("stop_loss_pct")

            # --- Global confidence rejection threshold (set during stock selection) ---
            if validated.action == "BUY":
                conf_rejection_raw = await asyncio.to_thread(self.redis.get, "trading:confidence_rejection_threshold")
                if conf_rejection_raw:
                    try:
                        conf_threshold = float(conf_rejection_raw)
                        if conf_threshold > 0 and validated.confidence < conf_threshold:
                            logger.info(f"Skipping {symbol}: confidence {validated.confidence:.2f} below global rejection threshold {conf_threshold:.2f}")
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"⚠️ Skipping {display_symbol}: confidence {validated.confidence:.2f} below threshold {conf_threshold:.2f}",
                                    summary={
                                        "symbol": symbol,
                                        "action": "SKIP",
                                        "reason": "Confidence below rejection threshold",
                                        "confidence": validated.confidence,
                                        "threshold": conf_threshold,
                                    }
                                )
                            return
                    except (ValueError, TypeError):
                        pass

            min_conf = params.get("min_confidence")
            if min_conf is not None and validated.confidence < min_conf:
                logger.info(f"Skipping {symbol}: confidence {validated.confidence:.2f} below LLM min {min_conf:.2f}")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⚠️ Skipping {display_symbol}: confidence too low ({validated.confidence:.2f})",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "Confidence too low",
                            "confidence": validated.confidence,
                            "min_confidence": min_conf,
                        }
                    )
                return

            # Prevent SELL without an open position (no shorting)
            if validated.action == "SELL" and symbol not in self.positions:
                logger.info(f"Skipping SELL for {symbol}: no open position.")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⚠️ Skipping SELL for {display_symbol}: no open position.",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "No open position",
                        }
                    )
                return

            # Apply any updated risk parameters from the LLM to the open position
            if symbol in self.positions and signal.strategy_params:
                await self._update_position_params(
                    symbol,
                    signal.strategy_params,
                    signal.indicator_config,
                    assigned_tf,
                    current_price,
                    atr,
                )

            if validated.action != "HOLD":
                # --- Sector concentration limit check (only for BUY) ---
                if validated.action == "BUY":
                    current_sector = None
                    for entry in self.current_symbols:
                        if entry["symbol"] == symbol:
                            current_sector = entry.get("sector")
                            break
                    
                    if current_sector:
                        max_positions_per_sector_raw = await asyncio.to_thread(self.redis.get, "trading:max_positions_per_sector")
                        if max_positions_per_sector_raw:
                            try:
                                max_positions_per_sector = int(max_positions_per_sector_raw)
                            except ValueError:
                                max_positions_per_sector = None
                        else:
                            max_positions_per_sector = None
                        
                        if max_positions_per_sector is not None and max_positions_per_sector > 0:
                            sector_count = 0
                            for pos_sym in self.positions.keys():
                                for entry in self.current_symbols:
                                    if entry["symbol"] == pos_sym and entry.get("sector") == current_sector:
                                        sector_count += 1
                                        break
                            
                            if sector_count >= max_positions_per_sector:
                                logger.info(
                                    f"Skipping BUY {symbol}: sector '{current_sector}' already has "
                                    f"{sector_count} open positions (max {max_positions_per_sector})"
                                )
                                if self.notifier:
                                    stock_name = await self._get_stock_name(symbol)
                                    display_symbol = self._format_symbol_display(symbol, stock_name, assigned_tf)
                                    await self.notifier.send_notification(
                                        f"⚠️ Skipping BUY {display_symbol}: sector '{current_sector}' concentration limit reached ({sector_count}/{max_positions_per_sector})",
                                        summary={
                                            "symbol": symbol,
                                            "action": "SKIP",
                                            "reason": "Sector concentration limit",
                                            "sector": current_sector,
                                            "sector_count": sector_count,
                                            "max_positions_per_sector": max_positions_per_sector,
                                        }
                                    )
                                return

                # --- Entry condition check (only for BUY) ---
                if validated.action == "BUY" and validated.entry_condition is not None and not trading_paused:
                    etype = validated.entry_condition.get("type")
                    if etype == "delay":
                        # Delay entries are simple time-based waits – schedule directly
                        delay_sec = validated.entry_condition.get("delay_seconds", 0)
                        logger.info(f"Scheduling delayed BUY for {symbol} in {delay_sec}s")
                        task = asyncio.create_task(
                            self._execute_delayed_entry(symbol, validated, assigned_tf, delay_sec)
                        )
                        self._delayed_entry_tasks.add(task)
                        task.add_done_callback(self._delayed_entry_tasks.discard)
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⏳ Delayed entry for {display_symbol} – executing in {delay_sec}s.",
                                summary={
                                    "symbol": symbol,
                                    "action": "WAIT",
                                    "reason": "Delay entry scheduled",
                                    "delay_seconds": delay_sec,
                                }
                            )
                        return  # do NOT execute now

                    timeout = validated.entry_condition.get("timeout_seconds", 600)
                    # Enforce a minimum based on the candle timeframe
                    min_timeout = max(300, int(settings.ENTRY_CONDITION_MIN_TIMEOUT_MULT * tf_seconds))
                    # Cap the minimum timeout to avoid absurd values for very long timeframes
                    # (e.g., 2 × 31,536,000 = ~730 days for 1Y candles).
                    # 180 days is a reasonable maximum wait for an entry condition in medium/long-term
                    # trading — it accommodates 1M (60d natural) and 3M (180d natural) candles
                    # while still capping 6M, 1Y, 3Y, and 5Y candles.
                    min_timeout = min(min_timeout, 15_552_000)  # 180 days
                    if timeout < min_timeout:
                        logger.info(
                            f"Entry condition timeout for {symbol} too short ({timeout}s), "
                            f"clamping to minimum {min_timeout}s (timeframe={assigned_tf})"
                        )
                        timeout = min_timeout
                    deadline = time.time() + timeout
                    # Store for background checking – do NOT block the main loop
                    self._pending_entries[symbol] = {
                        "signal": validated,
                        "deadline": deadline,
                        "timeframe": assigned_tf,
                        "condition": validated.entry_condition,
                    }
                    logger.info(
                        f"Queued entry condition for {symbol} (type={etype}, deadline in {timeout}s). "
                        f"Will monitor in background."
                    )
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⏳ Waiting for entry condition on {display_symbol} "
                            f"(type={etype}, timeout {timeout}s).",
                            summary={
                                "symbol": symbol,
                                "action": "WAIT",
                                "reason": "Entry condition pending",
                            }
                        )
                    return  # do NOT execute now

                await self._execute_signal(symbol, validated, timeframe=assigned_tf, atr=atr)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            if self.notifier:
                await self.notifier.send_notification(
                    f"❌ Error processing {display_symbol}: {e}",
                    summary={
                        "symbol": symbol,
                        "action": "ERROR",
                        "reason": str(e)[:200],
                    }
                )

    async def get_profit_summary(self) -> Dict[str, Any]:
        """Return profit/loss summary including queued orders."""
        balance = await asyncio.to_thread(self.trader.fetch_balance)
        current_balance = balance.get(self.base_currency, 0.0)

        # --- Early exit: no positions and no queued orders → nothing to compute ---
        if not self.positions and not self.queued_orders:
            return {
                "initial_balance": self.initial_balance,
                "current_balance": current_balance,
                "effective_balance": current_balance,
                "open_value": 0.0,
                "total_pnl": current_balance - self.initial_balance,
                "pnl_percent": ((current_balance - self.initial_balance) / self.initial_balance * 100) if self.initial_balance else 0.0,
                "total_fees": 0.0,
                "total_fees_display": "0.000000",
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "base_currency": self.base_currency,
                "queued_buy_count": 0,
                "queued_sell_count": 0,
                "queued_buy_quote_total": 0.0,
                "queued_sell_base_total": 0.0,
                "queued_sell_value": 0.0,
            }

        open_value = 0.0
        pos_tickers = await asyncio.to_thread(self._get_all_position_tickers_sync)
        for sym, pos in self.positions.items():
            try:
                t = pos_tickers.get(sym)
                price = t['last'] if t and t.get('last') else 0.0
                open_value += pos['amount'] * price
            except Exception:
                pass

        # --- Queued orders ---
        queued_buy_count = 0
        queued_sell_count = 0
        queued_buy_quote_total = 0.0
        queued_sell_base_total = 0.0
        queued_sell_value = 0.0

        # Collect symbols for queued sells to fetch prices
        queued_sell_symbols = []
        for q in self.queued_orders:
            if q['side'] == 'buy':
                queued_buy_count += 1
                # 'amount' is the remaining quote to spend
                queued_buy_quote_total += q.get('amount', 0.0)
            elif q['side'] == 'sell':
                queued_sell_count += 1
                queued_sell_base_total += q.get('amount', 0.0)
                queued_sell_symbols.append(q['symbol'])

        if queued_sell_symbols:
            sell_tickers = await asyncio.to_thread(self._get_tickers_for_symbols_sync, queued_sell_symbols)
        else:
            sell_tickers = {}
        for q in self.queued_orders:
            if q['side'] == 'sell':
                sym = q['symbol']
                t = sell_tickers.get(sym) if sell_tickers else None
                price = t['last'] if t and t.get('last') else 0.0
                queued_sell_value += q.get('amount', 0.0) * price

        effective_balance = current_balance - queued_buy_quote_total

        total_fees = 0.0
        for t in self.trade_history:
            fee = t.get('fee', {})
            fee_cost = float(fee.get('cost', 0) or 0)
            fee_currency = fee.get('currency', '')
            if fee_cost == 0.0:
                continue
            if fee_currency == self.base_currency:
                total_fees += fee_cost
            else:
                # fee is in the base symbol (e.g., BTC) → convert using trade price
                price = t.get('price', 0.0)
                total_fees += fee_cost * price
        total_value = current_balance + open_value
        pnl = total_value - self.initial_balance
        pnl_percent = (pnl / self.initial_balance * 100) if self.initial_balance else 0.0

        # Win/Loss stats
        wins = 0
        losses = 0
        for t in self.trade_history:
            if t.get('side') == 'sell' and 'realized_pnl' in t:
                pnl_val = t['realized_pnl']
                if pnl_val > 0:
                    wins += 1
                elif pnl_val < 0:
                    losses += 1
        total_closed = wins + losses
        win_rate = (wins / total_closed) if total_closed > 0 else 0.0

        return {
            "initial_balance": self.initial_balance,
            "current_balance": current_balance,
            "effective_balance": effective_balance,
            "open_value": open_value,
            "total_pnl": pnl,
            "pnl_percent": pnl_percent,
            "total_fees": round(total_fees, 6),
            "total_fees_display": f"{total_fees:.6f}",
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "base_currency": self.base_currency,
            "queued_buy_count": queued_buy_count,
            "queued_sell_count": queued_sell_count,
            "queued_buy_quote_total": queued_buy_quote_total,
            "queued_sell_base_total": queued_sell_base_total,
            "queued_sell_value": queued_sell_value,
        }

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
        balance = await asyncio.to_thread(self.trader.fetch_balance)
        total_balance = balance.get(self.base_currency, 0.0)

        pnl = total_balance - self.initial_balance
        pnl_pct = (pnl / self.initial_balance * 100) if self.initial_balance else 0.0

        # Open positions exposure and stop‑loss risk
        exposure = 0.0
        position_exposures = []
        total_stop_risk = 0.0
        pos_tickers = await asyncio.to_thread(self._get_all_position_tickers_sync)
        for sym, pos in self.positions.items():
            try:
                t = pos_tickers.get(sym)
                price = t['last'] if t and t.get('last') else 0.0
                pos_value = pos['amount'] * price
                exposure += pos_value
                position_exposures.append(pos_value)
                stop_loss = pos.get('stop_loss')
                if stop_loss is not None and price > 0:
                    loss_if_stop = pos_value * (price - stop_loss) / price
                    total_stop_risk += loss_if_stop
            except Exception:
                pass

        total_portfolio_value = total_balance + exposure
        largest_position_exposure_pct = (
            (max(position_exposures) / total_portfolio_value * 100)
            if position_exposures and total_portfolio_value > 0
            else 0.0
        )

        # Drawdown from performance metrics
        perf = await asyncio.to_thread(self._compute_performance_metrics)
        max_drawdown_pct = perf.get('equity_curve', {}).get('drawdown_pct', 0.0)

        # Trade statistics
        wins = []
        losses = []
        for t in self.trade_history:
            if t.get('side') == 'sell' and 'realized_pnl' in t:
                pnl_val = t['realized_pnl']
                if pnl_val > 0:
                    wins.append(pnl_val)
                elif pnl_val < 0:
                    losses.append(abs(pnl_val))
        total_trades = len(wins) + len(losses)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        avg_win = (gross_profit / len(wins)) if wins else 0.0
        avg_loss = (gross_loss / len(losses)) if losses else 0.0

        # Sanitize non-finite floats for JSON serialization
        def _sanitize_float(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        profit_factor = _sanitize_float(profit_factor)
        avg_win = _sanitize_float(avg_win)
        avg_loss = _sanitize_float(avg_loss)

        return {
            'current_balance': total_balance,
            'initial_balance': self.initial_balance,
            'total_pnl': pnl,
            'total_pnl_pct': pnl_pct,
            'open_positions_count': len(self.positions),
            'total_exposure': exposure,
            'base_currency': self.base_currency,
            'max_drawdown_pct': max_drawdown_pct,
            'largest_position_exposure_pct': largest_position_exposure_pct,
            'total_stop_loss_risk': total_stop_risk,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': total_trades,
        }

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
                trade["realized_pnl"] = 0.0
                trade["cost_basis"] = 0.0
                trade["exit_reason"] = "manual_sell"

                # Update virtual cash balance even if position wasn't tracked
                self.trader._balances[base] = self.trader._balances.get(base, 0.0) - quantity
                self.trader._balances[quote] = self.trader._balances.get(quote, 0.0) + (cost - fee)
                self.trader._balances_dirty = True
                await asyncio.to_thread(self.trader._save_balances)

        self._append_trade(trade)
        await asyncio.to_thread(insert_trade, trade)
        await self._save_state(force=True)
        logger.info(f"Manual trade logged: {side} {quantity} {symbol} @ {price:.4f}")
        return {"status": "ok", "trade": trade}

    async def _record_position_pnl_snapshots(self):
        """Record P&L snapshots for all open positions to the database."""
        if not self.positions:
            return
        pos_tickers = await self._get_all_position_tickers_sync()
        now_ms = int(time.time() * 1000)
        for symbol, pos in self.positions.items():
            try:
                t = pos_tickers.get(symbol)
                current_price = t['last'] if t and t.get('last') else pos.get('price', 0.0)
                amount = pos.get('amount', 0.0)
                entry_price = pos.get('price', 0.0)
                cost_basis = pos.get('cost_basis', amount * entry_price)
                position_value = amount * current_price
                unrealized_pnl = (current_price - entry_price) * amount
                pnl_pct = (unrealized_pnl / cost_basis) if cost_basis > 0 else 0.0
                # Realized P&L: sum of all closed sell trades for this symbol
                realized_pnl = sum(
                    t.get("realized_pnl", 0.0)
                    for t in self.trade_history
                    if t.get("symbol") == symbol and t.get("side") == "sell"
                )
                await asyncio.to_thread(
                    insert_position_pnl_snapshot,
                    symbol=symbol,
                    timestamp=now_ms,
                    unrealized_pnl=round(unrealized_pnl, 6),
                    realized_pnl=round(realized_pnl, 6),
                    position_value=round(position_value, 6),
                    cost_basis=round(cost_basis, 6),
                    amount=amount,
                    current_price=current_price,
                    pnl_pct=round(pnl_pct, 6),
                )
            except Exception as e:
                logger.debug(f"Failed to record P&L snapshot for {symbol}: {e}")

    async def _process_native_exit_fill(
        self,
        symbol: str,
        order_id: str,
        order_obj: Any,
        pos: Dict[str, Any],
        exit_reason: str,
    ):
        """Process a filled native exit order (stop-loss or take-profit) inline.

        This avoids the race condition where a native exit order fills between
        the OCO cancellation and a manual _execute_signal call, which would
        result in a double sell.
        """
        # Find and remove the queued entry under the lock, but do NOT call
        # _handle_queued_sell_fill inside the lock — it internally acquires
        # _queued_orders_lock via _cancel_exit_orders, which would deadlock.
        async with self._queued_orders_lock:
            queued = next((q for q in self.queued_orders if q.get("order_id") == order_id), None)
            if queued:
                self.queued_orders = [q for q in self.queued_orders if q.get("order_id") != order_id]

        if queued:
            filled_qty = float(order_obj.filled_qty) if order_obj.filled_qty else 0.0
            filled_avg_price = float(order_obj.filled_avg_price) if order_obj.filled_avg_price else 0.0
            if filled_qty > 0:
                delta_cost = filled_qty * filled_avg_price
                from src.exchanges.fees import calculate_transaction_costs
                _quote_ccy = symbol.split("/")[1] if "/" in symbol else self.base_currency
                _fee_costs = calculate_transaction_costs("SELL", filled_avg_price, filled_qty, symbol=symbol)
                trade_dict = {
                    'id': str(order_obj.id),
                    'symbol': symbol,
                    'side': 'sell',
                    'amount': filled_qty,
                    'price': filled_avg_price,
                    'cost': delta_cost,
                    'fee': {'cost': _fee_costs["total_costs"], 'currency': _quote_ccy},
                    'status': 'closed',
                    'timestamp': int(time.time() * 1000),
                }
                await self._handle_queued_sell_fill(trade_dict, queued, partial=False)

        # Cancel the OCO pair if it still exists
        oco_pair_id = queued.get("oco_pair") if queued else None
        if oco_pair_id:
            try:
                await asyncio.to_thread(self.trader.cancel_order, oco_pair_id)
            except Exception:
                pass
            async with self._queued_orders_lock:
                self.queued_orders = [q for q in self.queued_orders if q.get("order_id") != oco_pair_id]
        pos.pop("stop_loss_order_id", None)
        pos.pop("take_profit_order_id", None)
        pos.pop("stop_loss_order_type", None)
        pos.pop("_native_stop_price", None)

    async def _check_risk_management(self):
        """Check open positions and close if stop-loss, take-profit, or trailing stop is hit."""
        # --- Notify mode: no automated risk management ---
        if settings.TRADING_MODE == "notify":
            return

        # Read LLM-decided review limits from Redis once (before the per-position loop)
        max_sl_reviews = settings.MAX_STOP_LOSS_REVIEWS
        max_tp_reviews = settings.MAX_TAKE_PROFIT_REVIEWS
        max_partial_tp_reviews = settings.MAX_PARTIAL_TP_REVIEWS
        max_dust_sweep_reviews = settings.MAX_DUST_SWEEP_REVIEWS
        try:
            raw = await asyncio.to_thread(self.redis.get, "trading:max_stop_loss_reviews")
            if raw:
                max_sl_reviews = int(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:max_take_profit_reviews")
            if raw:
                max_tp_reviews = int(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:max_partial_tp_reviews")
            if raw:
                max_partial_tp_reviews = int(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:max_dust_sweep_reviews")
            if raw:
                max_dust_sweep_reviews = int(raw)
        except Exception:
            pass

        # Batch-fetch missing tickers once before the per-position loop
        risk_tickers: Dict[str, Dict[str, Any]] = {}
        missing_risk: List[str] = []
        for sym in self.positions:
            missing_risk.append(sym.split("/")[0])
        if missing_risk:
            try:
                raw = await self._get_quotes_batched(missing_risk, timeout_per_chunk=45.0)
                for sym in self.positions:
                    base = sym.split("/")[0]
                    if base in raw:
                        risk_tickers[sym] = raw[base]
            except Exception as e:
                logger.warning(f"Batch quote fetch failed in risk management: {e}")

        for symbol, pos in list(self.positions.items()):
            try:
                # Skip if there is already a queued order for this symbol
                async with self._queued_orders_lock:
                    has_queued = any(q['symbol'] == symbol for q in self.queued_orders)
                if has_queued:
                    continue

                ticker = risk_tickers.get(symbol)
                if ticker is None:
                    continue  # no real-time data yet, skip this check
                current_price = ticker['last']

                # --- Format symbol for notifications ---
                stock_name = await self._get_stock_name(symbol)
                display_symbol = self._format_symbol_display(symbol, stock_name, pos.get("timeframe"))

                # --- Hard stop: maximum total loss regardless of LLM decisions ---
                # Checked BEFORE the stop_loss/take_profit skip so positions
                # awaiting LLM risk parameters (_needs_risk_params) are still
                # protected against catastrophic loss during the re-evaluation window.
                entry_price = pos["price"]
                if entry_price > 0:
                    unrealized_loss_pct = (entry_price - current_price) / entry_price
                    if unrealized_loss_pct >= settings.HARD_MAX_LOSS_PCT:
                        logger.warning(
                            f"Hard max loss threshold reached for {symbol}: "
                            f"unrealized loss {unrealized_loss_pct:.2%} >= {settings.HARD_MAX_LOSS_PCT:.2%}. Forcing SELL."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⛔ Hard stop for {display_symbol}: unrealized loss {unrealized_loss_pct:.2%} "
                                f"exceeds maximum {settings.HARD_MAX_LOSS_PCT:.2%} – force selling.",
                                summary={
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "reason": "Hard maximum loss threshold",
                                    "price": current_price,
                                    "unrealized_loss_pct": round(unrealized_loss_pct, 4),
                                    "exit_reason": "hard_max_loss",
                                }
                            )
                        await self._execute_signal(
                            symbol,
                            Signal(action="SELL", confidence=1.0, reasoning="Hard maximum loss threshold exceeded"),
                            exit_reason="hard_max_loss"
                        )
                        continue

                # Skip positions that don't have LLM-defined risk parameters yet
                if pos.get("stop_loss") is None or pos.get("take_profit") is None:
                    continue

                # Trailing stop update (only if enabled)
                # Skip if a native trailing-stop order is already handling it
                if pos.get("trailing_stop") and pos.get("stop_loss_order_type") != "trailing_stop":
                    # Check activation threshold
                    activation_pct = pos.get("trailing_stop_activation_pct")
                    activated = True
                    if activation_pct is not None:
                        entry_price = pos["price"]
                        profit_pct = (current_price - entry_price) / entry_price
                        if profit_pct < activation_pct:
                            activated = False

                    if activated:
                        # Track highest price since activation.
                        # Use both the current ticker price AND the highest high
                        # from recent OHLCV candles to capture intra-check price
                        # spikes (the risk check only runs every
                        # RISK_CHECK_INTERVAL_SECONDS, so the ticker price alone
                        # may miss brief highs between checks).
                        candidate_prices = [current_price]
                        tf = pos.get("timeframe")
                        if not tf:
                            # Fallback to the assigned timeframe from current_symbols
                            for entry in self.current_symbols:
                                if entry["symbol"] == symbol:
                                    tf = entry.get("timeframe")
                                    break
                        if tf:
                            try:
                                last_check_ts = pos.get("_last_trailing_check_ts", 0)
                                tf_secs = self._timeframe_to_seconds(tf)
                                now_ts = time.time()
                                # Skip OHLCV fetch for very long timeframes (>= 1 month).
                                # OHLCV data is too sparse (2-10 candles) to provide meaningful
                                # intra-check price spikes. The ticker price alone is sufficient.
                                if tf_secs >= 2_592_000:
                                    if last_check_ts == 0:
                                        async with self._positions_lock:
                                            pos["_last_trailing_check_ts"] = now_ts
                                else:
                                    # Throttle OHLCV fetches: only fetch every ~10% of the
                                    # timeframe interval, clamped between 5 min and 1 hour.
                                    fetch_interval = max(300, min(3600, int(tf_secs * 0.1)))
                                    # On first check (last_check_ts == 0), initialize
                                    # timestamp but don't fetch (avoids using pre-entry
                                    # candles, matching the original _load_state behavior).
                                    if last_check_ts == 0:
                                        async with self._positions_lock:
                                            pos["_last_trailing_check_ts"] = now_ts
                                    elif (now_ts - last_check_ts) >= fetch_interval:
                                        since_ms = int(last_check_ts * 1000)
                                        db_candles = await asyncio.to_thread(get_ohlcv, symbol, tf, since_ms=since_ms, limit=200)
                                        if db_candles:
                                            candle_high = max(c["high"] for c in db_candles)
                                            candidate_prices.append(candle_high)
                                        async with self._positions_lock:
                                            pos["_last_trailing_check_ts"] = now_ts
                            except Exception as e:
                                logger.debug(f"Failed to fetch OHLCV for trailing stop on {symbol}: {e}")

                        best_high = max(candidate_prices)
                        async with self._positions_lock:
                            if "_highest_price" not in pos or best_high > pos["_highest_price"]:
                                pos["_highest_price"] = best_high

                        highest_price = pos["_highest_price"]
                        new_stop = None

                        # ATR-based trailing stop (Chandelier Exit)
                        atr_mult = pos.get("trailing_stop_atr_multiple")
                        if atr_mult is not None and atr_mult > 0:
                            # Determine the position timeframe for ATR reliability check
                            tf_for_atr = pos.get("timeframe")
                            if not tf_for_atr:
                                for entry in self.current_symbols:
                                    if entry["symbol"] == symbol:
                                        tf_for_atr = entry.get("timeframe")
                                        break
                            tf_secs_atr = self._timeframe_to_seconds(tf_for_atr) if tf_for_atr else 0
                            # For very long timeframes (>= 1 month), ATR is computed from
                            # too few candles (2-10) to be statistically reliable.
                            # Skip ATR fetch and fall back to fixed percentage trailing stop.
                            skip_atr = tf_secs_atr >= 2_592_000

                            if not skip_atr:
                                # Fetch ATR from DB if we don't have it in this loop
                                if "_current_atr" not in pos or time.time() - pos.get("_atr_fetched_at", 0) > 300:
                                    tf = pos.get("timeframe")
                                    if tf:
                                        try:
                                            ind = await asyncio.to_thread(get_indicators, symbol, tf)
                                            if ind and ind.get("atr") and ind["atr"] > 0:
                                                # Check indicator staleness: if the latest candle
                                                # used to compute ATR is older than 2× the timeframe
                                                # interval, the ATR may not reflect current volatility.
                                                ind_ts = ind.get("_indicator_timestamp")
                                                atr_is_stale = False
                                                if ind_ts is not None:
                                                    tf_secs = self._timeframe_to_seconds(tf)
                                                    # Cap max age at 1 day so long timeframes
                                                    # (5Y, 3Y, etc.) don't use stale ATR values
                                                    # for trailing stop calculations.
                                                    max_age_secs = min(tf_secs * 2, 86400)
                                                    # The indicator timestamp is the candle's
                                                    # start time.  The candle covers a period
                                                    # of tf_secs, so the most recent data is
                                                    # tf_secs more recent than the timestamp.
                                                    # Subtract the candle duration to get the
                                                    # effective age of the data.
                                                    age_secs = (time.time() * 1000 - ind_ts) / 1000
                                                    effective_age = max(0, age_secs - tf_secs)
                                                    if effective_age > max_age_secs:
                                                        logger.info(
                                                            f"ATR for {symbol} {tf} is stale "
                                                            f"(indicator data {effective_age/86400:.1f}d old, "
                                                            f"max {max_age_secs/86400:.1f}d). "
                                                            f"Falling back to fixed-percentage trailing stop."
                                                        )
                                                        atr_is_stale = True
                                                async with self._positions_lock:
                                                    if not atr_is_stale:
                                                        pos["_current_atr"] = ind["atr"]
                                                    else:
                                                        pos["_current_atr"] = None
                                                    pos["_atr_fetched_at"] = time.time()
                                        except Exception as e:
                                            logger.warning(f"Failed to fetch ATR for trailing stop on {symbol}: {e}")

                            current_atr = pos.get("_current_atr") if not skip_atr else None
                            if current_atr is not None and current_atr > 0:
                                new_stop = highest_price - (current_atr * atr_mult)
                            else:
                                # Fallback to fixed percentage if ATR fetch failed, is stale,
                                # or was skipped due to very long timeframe
                                distance = pos.get("trailing_stop_distance_pct")
                                if distance is not None:
                                    new_stop = highest_price * (1 - distance)
                        else:
                            # Fixed percentage trailing stop
                            distance = pos.get("trailing_stop_distance_pct")
                            if distance is not None:
                                new_stop = highest_price * (1 - distance)

                        if new_stop is not None:
                            async with self._positions_lock:
                                if new_stop > pos["stop_loss"]:
                                    # Only update trailing stop if the improvement is at least 0.1%
                                    # to avoid over-tightening on micro-movements (medium/long-term)
                                    min_improvement = pos["stop_loss"] * 0.001
                                    if new_stop - pos["stop_loss"] >= min_improvement:
                                        pos["stop_loss"] = new_stop
                                        logger.info(f"Trailing stop updated for {symbol}: new stop {new_stop:.4f}")

                # --- Trailing take-profit ---
                if pos.get("trailing_take_profit") and pos.get("trailing_take_profit_distance_pct"):
                    ttp_dist = pos["trailing_take_profit_distance_pct"]
                    new_tp = current_price * (1 + ttp_dist)
                    async with self._positions_lock:
                        if new_tp > pos["take_profit"]:
                            pos["take_profit"] = new_tp
                            logger.info(f"Trailing take-profit updated for {symbol}: new TP {new_tp:.4f}")

                # --- Breakeven stop ---
                breakeven_activation = pos.get("breakeven_activation_pct")
                if breakeven_activation is not None and breakeven_activation > 0:
                    entry_price = pos["price"]
                    if current_price >= entry_price * (1 + breakeven_activation):
                        # Compute exact break-even price that covers exit fee
                        breakeven_price = entry_price
                        async with self._positions_lock:
                            if breakeven_price > pos["stop_loss"]:
                                pos["stop_loss"] = breakeven_price
                                logger.info(f"Breakeven stop activated for {symbol}: new stop {breakeven_price:.4f}")

                # --- Lock profit feature removed (was scalping-specific) ---

                # --- Update native stop order if stop price changed ---
                if (pos.get("stop_loss_order_id")
                        and pos.get("stop_loss_order_type") in ("stop", "stop_limit")):
                    # Compare current stop_loss with the order's original stop price
                    original_stop = pos.get("_native_stop_price")
                    if original_stop is None:
                        # First time – store the current stop_loss as the baseline
                        async with self._positions_lock:
                            pos["_native_stop_price"] = pos["stop_loss"]
                    else:
                        # Check if stop_loss has moved by more than a tick
                        tick = 0.01 if pos["stop_loss"] >= 1.0 else 0.0001
                        if abs(pos["stop_loss"] - original_stop) > tick * 0.5:
                            logger.info(
                                f"Stop price changed for {symbol}: {original_stop:.4f} -> {pos['stop_loss']:.4f}. "
                                f"Replacing native stop order."
                            )
                            await self._replace_native_stop_order(
                                symbol, pos, original_stop, pos["stop_loss"]
                            )
                            # Update the stored baseline
                            async with self._positions_lock:
                                pos["_native_stop_price"] = pos["stop_loss"]

                # --- Partial take-profit ---
                partial_levels = pos.get("partial_take_profit_levels")
                if partial_levels:
                    # Multiple levels
                    triggered = pos.get("partial_tp_levels_triggered", [])
                    original_amount = pos.get("original_amount", pos["amount"])
                    for i, level in enumerate(partial_levels):
                        if i in triggered:
                            continue
                        if i in pos.get("_partial_tp_triggered_levels", []):
                            continue
                        lvl_pct = level["take_profit_pct"]
                        lvl_frac = level["fraction"]
                        entry_price = pos["price"]
                        # Time‑based cancellation
                        max_time = level.get("max_time_seconds")
                        if max_time is not None:
                            entry_ts = pos.get("timestamp", 0) / 1000.0
                            if time.time() - entry_ts > max_time:
                                logger.info(f"Partial TP level {i} for {symbol} expired (max {max_time}s). Cancelling.")
                                triggered.append(i)
                                async with self._positions_lock:
                                    pos["partial_tp_levels_triggered"] = triggered
                                continue
                        if current_price >= entry_price * (1 + lvl_pct):
                            # --- Instead of executing immediately, set a trigger flag for LLM review ---
                            # Check if we are already waiting for LLM on this level
                            async with self._positions_lock:
                                triggered_levels = pos.setdefault("_partial_tp_triggered_levels", [])
                                already_pending = i in triggered_levels
                                review_count = pos.get("_partial_tp_review_count", 0) + 1
                            if already_pending:
                                continue  # already pending

                            if review_count > max_partial_tp_reviews:
                                # Force execute
                                logger.info(f"Partial TP level {i} for {symbol}: max reviews reached, executing.")
                                await self._execute_partial_tp_level(symbol, i, current_price, None, ticker)
                                # After execution, the level is marked triggered; clear the review flags for this level
                                async with self._positions_lock:
                                    pos.pop("_partial_tp_triggered", None)
                                    pos.pop("_partial_tp_review_count", None)
                                    pos["_partial_tp_triggered_levels"] = [x for x in pos.get("_partial_tp_triggered_levels", []) if x != i]
                                continue

                            # Set trigger and ask LLM
                            async with self._positions_lock:
                                pos["_partial_tp_triggered"] = True
                                pos["_partial_tp_review_count"] = review_count
                                triggered_levels.append(i)
                            self._last_strategy_eval.pop(symbol, None)  # force immediate re‑eval
                            logger.info(f"Partial TP level {i} triggered for {symbol} – asking LLM (review {review_count})")
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"🔸 Partial TP level {i} triggered for {display_symbol} – consulting LLM...",
                                    summary={"symbol": symbol, "action": "HOLD", "reason": f"Partial TP level {i} triggered – awaiting LLM"}
                                )
                            break  # only handle one new trigger per cycle; others will be picked up after LLM responds
                else:
                    # Single partial TP – trigger LLM review instead of immediate execution
                    partial_tp_pct = pos.get("partial_take_profit_pct")
                    partial_tp_fraction = pos.get("partial_take_profit_fraction")
                    if (
                        partial_tp_pct is not None
                        and partial_tp_fraction is not None
                        and not pos.get("partial_tp_triggered", False)
                        and not pos.get("_partial_tp_triggered_single")
                    ):
                        entry_price = pos["price"]
                        if current_price >= entry_price * (1 + partial_tp_pct):
                            review_count = pos.get("_partial_tp_single_review_count", 0) + 1
                            if review_count > max_partial_tp_reviews:
                                logger.info(f"Single partial TP for {symbol}: max reviews reached, executing.")
                                await self._execute_partial_tp_single(symbol, current_price, None, ticker)
                                async with self._positions_lock:
                                    pos.pop("_partial_tp_triggered_single", None)
                                    pos.pop("_partial_tp_single_review_count", None)
                            else:
                                async with self._positions_lock:
                                    pos["_partial_tp_triggered_single"] = True
                                    pos["_partial_tp_single_review_count"] = review_count
                                self._last_strategy_eval.pop(symbol, None)
                                logger.info(f"Single partial TP triggered for {symbol} – asking LLM (review {review_count})")
                                if self.notifier:
                                    await self.notifier.send_notification(
                                        f"🔸 Partial TP triggered for {display_symbol} – consulting LLM...",
                                        summary={"symbol": symbol, "action": "HOLD", "reason": "Partial TP triggered – awaiting LLM"}
                                    )

                # --- Dust sweep check (if not already triggered) ---
                if not pos.get("_dust_sweep_triggered"):
                    base = symbol.split("/")[0]
                    amount = pos["amount"]
                    is_dust = False
                    try:
                        asset = await self._get_asset_info(symbol)
                        min_amount = float(asset.min_order_size) if asset.min_order_size else None
                    except Exception:
                        min_amount = None
                    if min_amount is not None and amount < min_amount:
                        is_dust = True

                    if is_dust:
                        # Check if dust has been kept past the timeout
                        dust_keep_since = pos.get("_dust_keep_since")
                        if dust_keep_since is not None and (time.time() - dust_keep_since) > settings.DUST_KEEP_TIMEOUT_SECONDS:
                            logger.info(
                                f"Dust keep timeout reached for {symbol} "
                                f"(kept for {(time.time() - dust_keep_since) / 3600:.1f}h), force-selling."
                            )
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"🧹 Dust keep timeout for {display_symbol} – auto-selling "
                                    f"after {settings.DUST_KEEP_TIMEOUT_SECONDS // 3600:.0f}h.",
                                    summary={
                                        "symbol": symbol,
                                        "action": "SELL",
                                        "reason": "Dust keep timeout",
                                        "exit_reason": "dust_keep_timeout",
                                    }
                                )
                            await self._sweep_dust(symbol)
                            continue
                        review_count = pos.get("_dust_sweep_review_count", 0) + 1
                        if review_count > max_dust_sweep_reviews:
                            logger.info(f"Dust sweep max reviews reached for {symbol}, force sweeping.")
                            await self._sweep_dust(symbol)
                        else:
                            async with self._positions_lock:
                                pos["_dust_sweep_triggered"] = True
                                pos["_dust_sweep_review_count"] = review_count
                            self._last_strategy_eval.pop(symbol, None)
                            logger.info(f"Dust condition triggered for {symbol} – asking LLM (review {review_count})")
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"🧹 Dust sweep triggered for {display_symbol} – consulting LLM...",
                                    summary={"symbol": symbol, "action": "HOLD", "reason": "Dust sweep triggered – awaiting LLM"}
                                )
                else:
                    # If dust was previously triggered but condition no longer holds, clear it
                    base = symbol.split("/")[0]
                    amount = pos["amount"]
                    is_dust = False
                    try:
                        asset = await self._get_asset_info(symbol)
                        min_amount = float(asset.min_order_size) if asset.min_order_size else None
                    except Exception:
                        min_amount = None
                    if min_amount is not None and amount < min_amount:
                        is_dust = True
                    if not is_dust:
                        async with self._positions_lock:
                            pos.pop("_dust_sweep_triggered", None)
                            pos.pop("_dust_sweep_review_count", None)
                            pos.pop("_dust_keep_since", None)
                        logger.info(f"Dust condition cleared for {symbol}")

                # --- News sentiment exit ---
                # Skip for long-term timeframes (>= 1 week): short-term sentiment
                # (15–30 min TTL) should not trigger exits on positions held
                # for weeks or months.
                news_threshold = pos.get("news_sentiment_exit_threshold")
                if news_threshold is not None and settings.NEWS_ENABLED:
                    pos_tf = pos.get("timeframe")
                    if pos_tf and self._timeframe_to_seconds(pos_tf) >= 604_800:
                        logger.debug(
                            f"Skipping news sentiment exit for {symbol}: "
                            f"long-term timeframe ({pos_tf}) ignores short-term sentiment."
                        )
                    else:
                        # Clamp to non-positive: a positive threshold would trigger
                        # an exit even when sentiment is mildly positive, which is
                        # almost certainly not the LLM's intent.  Only negative
                        # compound scores should trigger a sentiment-based exit.
                        effective_threshold = min(float(news_threshold), 0.0)
                        try:
                            agg = await self._get_cached_sentiment(symbol)
                            if agg and agg["avg_compound"] < effective_threshold:
                                logger.info(
                                    f"News sentiment exit for {symbol}: compound {agg['avg_compound']:.2f} < threshold {effective_threshold}"
                                )
                                if self.notifier:
                                    await self.notifier.send_notification(
                                        f"📰 Negative news exit for {display_symbol} (sentiment {agg['avg_compound']:.2f})",
                                        summary={
                                            "symbol": symbol,
                                            "action": "SELL",
                                            "reason": "News sentiment exit",
                                            "sentiment": agg,
                                            "exit_reason": "news_sentiment_exit",
                                        }
                                    )
                                await self._execute_signal(
                                    symbol,
                                    Signal(action="SELL", confidence=1.0, reasoning="News sentiment exit"),
                                    exit_reason="news_sentiment_exit"
                                )
                                continue  # skip further checks for this symbol
                        except Exception as e:
                            logger.info(f"News sentiment check failed for {symbol}: {e}")

                # --- Soft stop: max unrealized loss ---
                max_ul_pct = pos.get("max_unrealized_loss_pct")
                if max_ul_pct is not None and max_ul_pct > 0:
                    entry_price = pos["price"]
                    if current_price <= entry_price * (1 - max_ul_pct):
                        logger.info(f"Max unrealized loss reached for {symbol} ({max_ul_pct:.2%}). Closing position.")
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"📉 Soft stop triggered for {display_symbol} at {current_price:.4f} (max loss {max_ul_pct:.2%})",
                                summary={
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "reason": "Max unrealized loss",
                                    "price": current_price,
                                    "exit_reason": "max_unrealized_loss",
                                }
                            )
                        await self._execute_signal(
                            symbol,
                            Signal(action="SELL", confidence=1.0, reasoning="Max unrealized loss"),
                            exit_reason="max_unrealized_loss"
                        )
                        continue

                # --- Max hold time expired → ask LLM instead of auto‑closing ---
                max_hold = pos.get("max_hold_time_seconds")
                if max_hold is not None and max_hold > 0:
                    entry_ts = pos.get("timestamp", 0) / 1000.0  # convert ms to seconds
                    if time.time() - entry_ts > max_hold:
                        # Already waiting for LLM – do not re‑trigger
                        if pos.get("_max_hold_expired"):
                            continue
                        # First expiry – ask LLM
                        expired_count = pos.get("_max_hold_expired_count", 0) + 1
                        async with self._positions_lock:
                            pos["_max_hold_expired"] = True
                            pos["_max_hold_expired_count"] = expired_count

                        # Force re‑evaluation on the next main loop tick
                        self._last_strategy_eval.pop(symbol, None)

                        logger.info(
                            f"Max hold time expired for {symbol} (attempt {expired_count}) – asking LLM to decide."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⏰ Max hold time expired for {display_symbol} – asking LLM whether to sell or extend.",
                                summary={
                                    "symbol": symbol,
                                    "action": "HOLD",
                                    "reason": "Max hold time expired – awaiting LLM decision",
                                }
                            )
                        continue   # skip further checks for this symbol in this cycle

                if pos.get("stop_loss_order_id") or pos.get("take_profit_order_id"):
                    # Native exit orders are active – skip manual stop/tp triggers.
                    # But proactively cancel the OCO pair when the trigger price
                    # has been reached, instead of waiting up to 120s for the
                    # queued-order polling loop to notice.
                    sl_order_id = pos.get("stop_loss_order_id")
                    tp_order_id = pos.get("take_profit_order_id")
                    sl_order_type = pos.get("stop_loss_order_type", "stop")

                    # Stop price reached → cancel take-profit OCO pair
                    if (sl_order_id and tp_order_id
                            and sl_order_type in ("stop", "stop_limit")
                            and pos.get("stop_loss") is not None
                            and current_price <= pos["stop_loss"]):
                        try:
                            await asyncio.to_thread(self.trader.cancel_order, tp_order_id)
                            logger.info(
                                f"Risk check: stop price reached for {symbol}, "
                                f"cancelled OCO take-profit {tp_order_id}"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to cancel OCO TP {tp_order_id} for {symbol}: {e}")
                        async with self._queued_orders_lock:
                            self.queued_orders = [
                                q for q in self.queued_orders
                                if q.get("order_id") != tp_order_id
                            ]
                            for q in self.queued_orders:
                                if q.get("order_id") == sl_order_id:
                                    q["oco_pair"] = None
                                    break
                        async with self._positions_lock:
                            pos.pop("take_profit_order_id", None)
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"🛑 Stop triggered for {display_symbol} at {current_price:.4f}, "
                                f"take‑profit order cancelled.",
                                summary={
                                    "symbol": symbol,
                                    "action": "CANCEL",
                                    "reason": "Stop triggered, OCO pair cancelled (risk check)",
                                }
                            )
                        # Process the stop-loss order: check if it has already filled
                        # (race condition prevention — the order may have filled between
                        # cancelling the TP and now).  Calling get_order will also trigger
                        # the fill if the stop price has been reached, which is the
                        # desired behaviour: we process the native fill instead of
                        # executing a duplicate manual sell.
                        sl_filled = False
                        sl_order_obj = None
                        try:
                            sl_order_obj = await asyncio.to_thread(self.trader.get_order, sl_order_id)
                            if sl_order_obj is not None and sl_order_obj.status == "filled":
                                sl_filled = True
                        except Exception:
                            pass

                        if sl_filled:
                            # The native stop-loss order filled — process the fill to
                            # update positions and trade history, avoiding a double sell.
                            logger.info(f"Stop-loss order {sl_order_id} filled for {symbol}, processing native fill.")
                            await self._process_native_exit_fill(symbol, sl_order_id, sl_order_obj, pos, "stop_loss")
                        else:
                            # Stop-loss not yet filled — cancel it and execute manual sell
                            try:
                                await asyncio.to_thread(self.trader.cancel_order, sl_order_id)
                            except Exception:
                                pass
                            async with self._queued_orders_lock:
                                self.queued_orders = [
                                    q for q in self.queued_orders
                                    if q.get("order_id") != sl_order_id
                                ]
                            async with self._positions_lock:
                                pos.pop("stop_loss_order_id", None)
                                pos.pop("stop_loss_order_type", None)
                                pos.pop("_native_stop_price", None)
                            await self._execute_signal(
                                symbol,
                                Signal(action="SELL", confidence=1.0, reasoning="Stop-loss triggered (risk check)"),
                                exit_reason="stop_loss"
                            )
                        continue  # position has been closed, move to next

                    # Take-profit price reached → cancel stop OCO pair
                    if (sl_order_id and tp_order_id
                            and pos.get("take_profit") is not None
                            and current_price >= pos["take_profit"]):
                        try:
                            await asyncio.to_thread(self.trader.cancel_order, sl_order_id)
                            logger.info(
                                f"Risk check: take-profit price reached for {symbol}, "
                                f"cancelled OCO stop {sl_order_id}"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to cancel OCO stop {sl_order_id} for {symbol}: {e}")
                        async with self._queued_orders_lock:
                            self.queued_orders = [
                                q for q in self.queued_orders
                                if q.get("order_id") != sl_order_id
                            ]
                            for q in self.queued_orders:
                                if q.get("order_id") == tp_order_id:
                                    q["oco_pair"] = None
                                    break
                        async with self._positions_lock:
                            pos.pop("stop_loss_order_id", None)
                            pos.pop("stop_loss_order_type", None)
                            pos.pop("_native_stop_price", None)
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"🎯 Take‑profit reached for {display_symbol} at {current_price:.4f}, "
                                f"stop order cancelled.",
                                summary={
                                    "symbol": symbol,
                                    "action": "CANCEL",
                                    "reason": "Take-profit reached, OCO pair cancelled (risk check)",
                                }
                            )
                        # Process the take-profit order: check if it has already
                        # filled (race condition prevention — the order may have
                        # filled between cancelling the SL and now).  Calling
                        # get_order will also trigger the fill if the TP price has
                        # been reached.
                        tp_filled = False
                        tp_order_obj = None
                        try:
                            tp_order_obj = await asyncio.to_thread(self.trader.get_order, tp_order_id)
                            if tp_order_obj is not None and tp_order_obj.status == "filled":
                                tp_filled = True
                        except Exception:
                            pass

                        if tp_filled:
                            logger.info(f"Take-profit order {tp_order_id} filled for {symbol}, processing native fill.")
                            await self._process_native_exit_fill(symbol, tp_order_id, tp_order_obj, pos, "take_profit")
                        else:
                            # TP not yet filled — cancel it and execute manual sell
                            try:
                                await asyncio.to_thread(self.trader.cancel_order, tp_order_id)
                            except Exception:
                                pass
                            async with self._queued_orders_lock:
                                self.queued_orders = [
                                    q for q in self.queued_orders
                                    if q.get("order_id") != tp_order_id
                                ]
                            async with self._positions_lock:
                                pos.pop("take_profit_order_id", None)
                            await self._execute_signal(
                                symbol,
                                Signal(action="SELL", confidence=1.0, reasoning="Take-profit triggered (risk check)"),
                                exit_reason="take_profit"
                            )
                        continue  # position has been closed, move to next
                elif current_price <= pos["stop_loss"]:
                    # Instead of immediately selling, ask the LLM whether to sell or adjust the stop.
                    # Scale max reviews based on position timeframe to prevent excessive
                    # loss accumulation in long-term positions.
                    effective_max_sl_reviews = max_sl_reviews
                    pos_tf = pos.get("timeframe")
                    if pos_tf:
                        pos_tf_secs = self._timeframe_to_seconds(pos_tf)
                        if pos_tf_secs >= settings.LONG_TERM_TF_SECONDS:  # >= 1 month
                            effective_max_sl_reviews = min(effective_max_sl_reviews, settings.LONG_TERM_MAX_STOP_LOSS_REVIEWS)
                        elif pos_tf_secs >= 604_800:  # >= 1 week
                            effective_max_sl_reviews = min(effective_max_sl_reviews, settings.WEEKLY_MAX_STOP_LOSS_REVIEWS)
                    review_count = pos.get("_stop_loss_review_count", 0)
                    if review_count >= effective_max_sl_reviews:
                        # Fallback: force-sell after too many reviews
                        logger.warning(
                            f"Stop-loss triggered for {symbol} at {current_price} – "
                            f"review count {review_count} >= {effective_max_sl_reviews}, forcing SELL."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⛔ Stop‑loss triggered for {display_symbol} at {current_price:.4f} – "
                                f"max reviews reached ({effective_max_sl_reviews}), selling.",
                                summary={
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "reason": "Stop-loss (max reviews)",
                                    "price": current_price,
                                    "exit_reason": "stop_loss_max_reviews",
                                }
                            )
                        await self._execute_signal(
                            symbol,
                            Signal(action="SELL", confidence=1.0, reasoning="Stop-loss (max reviews)"),
                            exit_reason="stop_loss_max_reviews"
                        )
                    else:
                        # First or repeated trigger: set flag and ask LLM
                        if not pos.get("_stop_loss_triggered"):
                            async with self._positions_lock:
                                pos["_stop_loss_triggered"] = True
                                pos["_stop_loss_review_count"] = review_count + 1
                            # Force immediate strategy re-evaluation for this symbol
                            self._last_strategy_eval.pop(symbol, None)
                            logger.info(
                                f"Stop-loss triggered for {symbol} at {current_price} – "
                                f"asking LLM (review {pos['_stop_loss_review_count']}/{effective_max_sl_reviews})."
                            )
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"⛔ Stop‑loss hit for {display_symbol} at {current_price:.4f} – consulting LLM...",
                                    summary={
                                        "symbol": symbol,
                                        "action": "HOLD",
                                        "reason": "Stop-loss triggered – awaiting LLM decision",
                                        "price": current_price,
                                    }
                                )
                        else:
                            # Already waiting for LLM; do nothing (avoid re-triggering)
                            logger.debug(
                                f"Stop-loss still triggered for {symbol}, waiting for LLM response "
                                f"(review {review_count}/{effective_max_sl_reviews})."
                            )
                elif current_price >= pos["take_profit"]:
                    # Always ask the LLM whether to sell or adjust the take-profit, but cap reviews.
                    review_count = pos.get("_take_profit_review_count", 0)
                    if review_count >= max_tp_reviews:
                        logger.warning(
                            f"Take-profit triggered for {symbol} at {current_price} – "
                            f"review count {review_count} >= {max_tp_reviews}, forcing SELL."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"🎯 Take‑profit triggered for {display_symbol} at {current_price:.4f} – "
                                f"max reviews reached, selling.",
                                summary={
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "reason": "Take-profit (max reviews)",
                                    "price": current_price,
                                    "exit_reason": "take_profit_max_reviews",
                                }
                            )
                        await self._execute_signal(
                            symbol,
                            Signal(action="SELL", confidence=1.0, reasoning="Take-profit (max reviews)"),
                            exit_reason="take_profit_max_reviews"
                        )
                        continue
                    # First or repeated trigger: set flag and ask LLM
                    if not pos.get("_take_profit_triggered"):
                        async with self._positions_lock:
                            pos["_take_profit_triggered"] = True
                            pos["_take_profit_review_count"] = review_count + 1
                        # Force immediate strategy re-evaluation for this symbol
                        self._last_strategy_eval.pop(symbol, None)
                        logger.info(
                            f"Take-profit triggered for {symbol} at {current_price} – "
                            f"asking LLM (review {pos['_take_profit_review_count']}/{max_tp_reviews})."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"🎯 Take‑profit hit for {display_symbol} at {current_price:.4f} – consulting LLM...",
                                summary={
                                    "symbol": symbol,
                                    "action": "HOLD",
                                    "reason": "Take-profit triggered – awaiting LLM decision",
                                    "price": current_price,
                                }
                            )
                    else:
                        # Already waiting for LLM; do nothing
                        logger.debug(
                            f"Take-profit still triggered for {symbol}, waiting for LLM response "
                            f"(review {review_count}/{max_tp_reviews})."
                        )
            except Exception as e:
                logger.error(f"Risk check failed for {symbol}: {e}")

        # Record position-level P&L snapshots for all open positions
        await self._record_position_pnl_snapshots()

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
        balance = await self._get_cached_balance()

        if signal.action == "BUY":
            # Safety: never buy when trading is paused
            paused = await asyncio.to_thread(self.redis.get, "trading:paused")
            if paused:
                logger.info(f"Ignoring BUY {symbol}: trading is paused (safety check).")
                return
            # Extract known parameters from the LLM's strategy_params (if any)
            params = signal.strategy_params or {}
            fill_timeout = params.get("order_fill_timeout_seconds", settings.ORDER_FILL_TIMEOUT_SECONDS)

            # Fetch current price early for position sizing and stop calculations
            base = symbol.split("/")[0]
            quotes = await self._get_quotes_async([base], timeout=45.0)
            ticker = quotes.get(base)
            current_price = ticker['last'] if ticker else None
            if current_price is None or current_price <= 0:
                logger.warning(f"Cannot execute BUY for {symbol}: no valid current price.")
                return

            # Use LLM-provided risk parameters directly (no hardcoded minimums)
            # Determine take-profit percentage based on method
            if "take_profit_atr_multiple" in params and atr is not None and atr > 0 and current_price > 0:
                tp_atr_mult = params["take_profit_atr_multiple"]
                tp_pct = (tp_atr_mult * atr) / current_price
                logger.info(f"ATR-based take-profit: ATR={atr}, multiplier={tp_atr_mult}, take_profit_pct={tp_pct:.4%}")
            else:
                if "take_profit_atr_multiple" in params:
                    logger.warning(f"ATR unavailable for {symbol}, falling back to fixed take_profit_pct from LLM params.")
                tp_pct = params.get("take_profit_pct")
                if tp_pct is None or tp_pct <= 0:
                    logger.warning(f"Cannot execute BUY for {symbol}: take_profit_pct missing/invalid and ATR unavailable.")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⚠️ Skipping BUY {display_symbol}: missing take_profit_pct and ATR unavailable.",
                            summary={"symbol": symbol, "action": "SKIP", "reason": "Missing take_profit_pct and ATR unavailable"}
                        )
                    return
            trailing_stop = params["trailing_stop"]
            trailing_stop_distance_pct = params.get("trailing_stop_distance_pct")

            # Determine stop-loss percentage based on method
            stop_method = params.get("stop_loss_method", "fixed")
            if stop_method == "atr_multiple" and atr is not None and atr > 0:
                atr_mult = params["stop_loss_atr_multiple"]
                sl_pct = (atr_mult * atr) / current_price
                logger.info(f"ATR-based stop: ATR={atr}, multiplier={atr_mult}, stop_loss_pct={sl_pct:.4%}")
            else:
                if stop_method == "atr_multiple":
                    logger.warning(f"ATR unavailable for {symbol}, falling back to fixed stop_loss_pct from LLM params.")
                sl_pct = params.get("stop_loss_pct")
                if sl_pct is None:
                    logger.warning(f"Cannot execute BUY for {symbol}: stop_loss_pct missing and ATR method not applicable/available.")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⚠️ Skipping BUY {display_symbol}: missing stop_loss_pct and ATR unavailable.",
                            summary={"symbol": symbol, "action": "SKIP", "reason": "Missing stop_loss_pct and ATR unavailable"}
                        )
                    return

            quote_balance = balance.get(quote, 0.0)
            position_fraction = params["position_size_fraction"]

            # Desired amount based on fraction of total available quote balance
            desired_amount = quote_balance * position_fraction

            # Apply confidence-based position sizing (LLM-decided weight)
            confidence_sizing_weight = params.get("confidence_sizing_weight", 0.0)
            if confidence_sizing_weight is not None:
                try:
                    confidence_sizing_weight = float(confidence_sizing_weight)
                except (TypeError, ValueError):
                    confidence_sizing_weight = 0.0
            if confidence_sizing_weight > 0 and signal.confidence < 1.0:
                confidence_multiplier = 1.0 - confidence_sizing_weight * (1.0 - signal.confidence)
                desired_amount *= confidence_multiplier
                logger.info(
                    f"Confidence sizing applied: weight={confidence_sizing_weight}, "
                    f"confidence={signal.confidence:.2f}, multiplier={confidence_multiplier:.4f}, "
                    f"adjusted amount={desired_amount:.2f}"
                )

            # --- Consolidated position sizing: single hard ceiling from all caps ---
            # All risk caps are computed into one hard_max. The LLM's desired_amount
            # (position_size_fraction × balance × confidence_sizing × global_mult ×
            # per_symbol_mult) is then capped at hard_max. This replaces 8+ sequential
            # multiplier/cap layers with a single min() check the LLM can reason about.
            pos_tickers = await self._get_all_position_tickers()

            # Compute current portfolio state once
            total_value = quote_balance
            total_open_exposure = 0.0
            total_open_stop_risk = 0.0
            for sym, pos in self.positions.items():
                try:
                    t = pos_tickers.get(sym)
                    price = t['last'] if t and t.get('last') else 0.0
                    pos_value = pos['amount'] * price
                    total_open_exposure += pos_value
                    total_value += pos_value
                    stop_loss = pos.get('stop_loss')
                    if stop_loss is not None and price > 0:
                        loss_if_stop = pos_value * (price - stop_loss) / price
                        total_open_stop_risk += max(0, loss_if_stop)
                except Exception:
                    pass

            # Apply global risk multiplier to desired amount (scales all positions)
            global_mult = await self._get_global_risk_multiplier()
            if global_mult is not None and 0.0 <= global_mult <= 1.0:
                desired_amount *= global_mult

            # Apply per-symbol position size multiplier to desired amount
            per_symbol_mult = params.get("position_size_multiplier")
            if per_symbol_mult is not None:
                try:
                    per_symbol_mult = float(per_symbol_mult)
                    if 0.0 <= per_symbol_mult <= 1.0:
                        desired_amount *= per_symbol_mult
                except (ValueError, TypeError):
                    pass

            # Compute single hard ceiling from all risk caps
            hard_max = float('inf')

            # Cap 1: max_risk_per_trade_pct (per-trade risk from LLM strategy params)
            max_risk_pct = params.get("max_risk_per_trade_pct")
            if max_risk_pct is not None and sl_pct > 0:
                hard_max = min(hard_max, (total_value * max_risk_pct) / sl_pct)

            # Cap 2: max_portfolio_risk_pct (portfolio risk from LLM strategy params)
            max_portfolio_risk_pct = params.get("max_portfolio_risk_pct")
            if max_portfolio_risk_pct is not None and sl_pct > 0:
                available_risk_budget = max(0.0, (total_value * max_portfolio_risk_pct) - total_open_stop_risk)
                hard_max = min(hard_max, available_risk_budget / sl_pct)

            # Cap 3: max_portfolio_exposure_pct (global LLM setting from stock selection)
            max_port_exp_raw = await asyncio.to_thread(self.redis.get, "trading:max_portfolio_exposure_pct")
            max_port_exp = float(max_port_exp_raw) if max_port_exp_raw else None
            if max_port_exp is not None and total_value > 0:
                available_exposure = max(0.0, (max_port_exp * total_value) - total_open_exposure)
                hard_max = min(hard_max, available_exposure)

            # Cap 4: max_portfolio_stop_risk_pct (global LLM setting from stock selection)
            max_port_risk_raw = await asyncio.to_thread(self.redis.get, "trading:max_portfolio_stop_risk_pct")
            max_port_risk = float(max_port_risk_raw) if max_port_risk_raw else None
            if max_port_risk is not None and sl_pct > 0 and total_value > 0:
                available_stop_risk_budget = max(0.0, (total_value * max_port_risk) - total_open_stop_risk)
                hard_max = min(hard_max, available_stop_risk_budget / sl_pct)

            # Cap at remaining cycle budget
            available = max(0.0, quote_balance - self._cycle_spent)
            hard_max = min(hard_max, available)

            # Final amount: min of LLM's desired amount and the single hard ceiling
            amount = min(desired_amount, hard_max)

            if amount <= 0:
                logger.info(f"Skipping BUY {symbol}: position size reduced to 0 by portfolio constraints")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⚠️ Skipping BUY {display_symbol}: portfolio constraints leave no room for new position",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "Portfolio constraints exhausted",
                            "desired_amount": desired_amount,
                            "hard_max": 0.0,
                        }
                    )
                return

            if amount < desired_amount:
                # Single consolidated notification about which cap was binding
                cap_reasons = []
                if max_risk_pct is not None and sl_pct > 0:
                    cap_reasons.append(f"max_risk_per_trade={max_risk_pct:.2%}")
                if max_portfolio_risk_pct is not None:
                    cap_reasons.append(f"max_portfolio_risk={max_portfolio_risk_pct:.2%}")
                if max_port_exp is not None:
                    cap_reasons.append(f"max_exposure={max_port_exp:.2%}")
                if max_port_risk is not None:
                    cap_reasons.append(f"max_stop_risk={max_port_risk:.2%}")
                if global_mult is not None and global_mult < 1.0:
                    cap_reasons.append(f"global_risk_mult={global_mult:.2f}")
                if per_symbol_mult is not None and per_symbol_mult < 1.0:
                    cap_reasons.append(f"position_size_mult={per_symbol_mult:.2f}")
                reason_str = ", ".join(cap_reasons) if cap_reasons else "portfolio constraints"
                logger.info(
                    f"Position size capped for {symbol}: {desired_amount:.2f} -> {amount:.2f} "
                    f"({reason_str})"
                )
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⚠️ {display_symbol}: position capped {desired_amount:.2f} → {amount:.2f} ({reason_str})",
                        summary={
                            "symbol": symbol,
                            "action": "INFO",
                            "reason": f"Position size capped: {reason_str}",
                            "desired_amount": desired_amount,
                            "capped_amount": amount,
                        }
                    )

            # --- Minimum absolute profit check (LLM‑defined) ---
            if settings.ENFORCE_MIN_PROFIT_PER_TRADE:
                min_profit = params.get("min_profit_per_trade")
                if min_profit is not None and min_profit > 0:
                    expected_gross_profit = amount * tp_pct
                    if expected_gross_profit < min_profit:
                        logger.info(
                            f"Skipping BUY {symbol}: expected gross profit {expected_gross_profit:.4f} {quote} "
                            f"below LLM minimum {min_profit:.4f}"
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⚠️ Skipping BUY {display_symbol}: profit too small ({expected_gross_profit:.4f} {quote})",
                                summary={
                                    "symbol": symbol,
                                    "action": "SKIP",
                                    "reason": "Expected profit below minimum",
                                    "expected_profit": expected_gross_profit,
                                    "min_profit": min_profit,
                                }
                            )
                        return

            # No hardcoded minimum viable trade amount gate.
            # The LLM decides the trade amount dynamically.
            # Only exchange minimums (checked below) are hard limits.

            # Check minimum order size and adjust upward if needed
            try:
                price = current_price
                base_amount = amount / price
                # Fetch minimum order size from asset info
                try:
                    asset = await self._get_asset_info(symbol)
                    min_amount_limit = float(asset.min_order_size) if asset.min_order_size else None
                    if not asset.fractionable and (min_amount_limit is None or min_amount_limit < 1.0):
                        min_amount_limit = 1.0
                except Exception:
                    min_amount_limit = None
                # Compute min cost from min amount and current price
                if min_amount_limit is not None and price:
                    min_cost_limit = min_amount_limit * price
                else:
                    min_cost_limit = None

                # Determine the required minimum quote amount
                required_quote = amount
                if min_amount_limit is not None:
                    min_base = float(min_amount_limit)
                    required_quote = max(required_quote, min_base * price)
                if min_cost_limit is not None:
                    required_quote = max(required_quote, float(min_cost_limit))

                if required_quote > amount:
                    # If the required minimum exceeds the risk-limited desired_amount, skip
                    if required_quote > desired_amount:
                        logger.info(
                            f"Skipping BUY {symbol}: exchange minimum {required_quote:.2f} "
                            f"exceeds risk-limited amount {desired_amount:.2f}"
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⚠️ Skipping BUY {display_symbol}: exchange minimum exceeds risk limit",
                                summary={
                                    "symbol": symbol,
                                    "action": "SKIP",
                                    "reason": "Exchange minimum exceeds risk limit",
                                    "required_quote": required_quote,
                                    "desired_amount": desired_amount,
                                }
                            )
                        return
                    # Adjust amount upward to meet the minimum
                    old_amount = amount
                    amount = required_quote
                    # Check if the adjusted amount exceeds remaining cycle budget
                    if amount > available:
                        logger.info(
                            f"BUY amount adjusted from {old_amount:.2f} to {amount:.2f} {quote} "
                            f"to meet minimum, but exceeds remaining cycle budget ({available:.2f}). Skipping."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⚠️ BUY skipped for {display_symbol}: amount adjusted to {amount:.2f} but insufficient remaining budget",
                                summary={
                                    "symbol": symbol,
                                    "action": "SKIP",
                                    "reason": "Adjusted amount exceeds remaining budget",
                                    "adjusted_amount": amount,
                                }
                            )
                        return
                    logger.info(
                        f"BUY amount adjusted from {old_amount:.2f} to {amount:.2f} {quote} "
                        f"to meet exchange minimum"
                    )
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"ℹ️ {display_symbol}: buy amount adjusted to {amount:.2f} {quote} to meet minimum",
                            summary={
                                "symbol": symbol,
                                "action": "INFO",
                                "reason": "Buy amount adjusted to meet minimum",
                                "adjusted_amount": amount,
                            }
                        )
                    # Recalculate base_amount for the order
                    base_amount = amount / price
            except Exception as e:
                logger.warning(f"Could not verify/adjust min order size for {symbol}: {e}")

            need_limit = not self._is_regular_hours()
            limit_price = None
            time_in_force = "day"
            # If LLM provided a limit_price, use it even during regular hours
            llm_limit_price = params.get("limit_price")
            if llm_limit_price is not None and llm_limit_price > 0:
                limit_price = llm_limit_price
                time_in_force = params.get("time_in_force", "day")
                need_limit = True  # force limit order path
                # Validate that the limit price is within a reasonable distance from the market
                # Read LLM-controlled limit price max distance (fallback to static setting)
                max_distance = settings.LIMIT_PRICE_MAX_DISTANCE_PCT
                try:
                    raw = await asyncio.to_thread(self.redis.get, "trading:limit_price_max_distance_pct")
                    if raw:
                        max_distance = float(raw)
                except Exception:
                    pass
                if ticker and ticker.get('ask') and max_distance > 0:
                    ask = ticker['ask']
                    if limit_price < ask * (1 - max_distance):
                        logger.warning(
                            f"LLM limit_price {limit_price} for {symbol} is >{max_distance*100:.0f}% below ask {ask}. "
                            f"Rejecting BUY to avoid indefinite queuing."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⚠️ Skipping BUY {display_symbol}: limit price {limit_price} too far below ask {ask}.",
                                summary={"symbol": symbol, "action": "SKIP", "reason": "Limit price too far from market"}
                            )
                        return
            elif need_limit:
                limit_price = self._default_limit_price(symbol, "BUY", ticker)
                time_in_force = params.get("time_in_force", "day")
                if limit_price is None:
                    logger.error(f"Cannot place limit order for {symbol}: no limit price available.")
                    return

            if limit_price is not None:
                # Round to valid tick size ($0.01 for >=$1, $0.0001 for <$1)
                if limit_price >= 1.0:
                    limit_price = round(limit_price, 2)
                else:
                    limit_price = round(limit_price, 4)

            if limit_price is not None and limit_price <= 0:
                logger.error(f"Invalid limit_price {limit_price} for {symbol}, skipping.")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"❌ Invalid limit price for {display_symbol}, skipping.",
                        summary={"symbol": symbol, "action": "SKIP", "reason": "Invalid limit price"}
                    )
                return

            # --- Determine order type ---
            order_type = signal.order_type
            if order_type not in ("market", "limit", "stop", "stop_limit", "trailing_stop"):
                # Fallback to existing behaviour: limit if limit_price provided, else market
                if limit_price is not None:
                    order_type = "limit"
                else:
                    order_type = "market"

            try:
                if order_type == "market":
                    order = await asyncio.to_thread(
                        self.trader.create_market_buy_order, symbol, amount, fill_timeout,
                        limit_price=None, time_in_force='day'
                    )
                elif order_type == "limit":
                    order = await asyncio.to_thread(
                        self.trader.create_market_buy_order, symbol, amount, fill_timeout,
                        limit_price=limit_price, time_in_force=time_in_force
                    )
                elif order_type == "stop":
                    stop_price = signal.stop_price
                    if stop_price is None or stop_price <= 0:
                        raise ValueError("Missing or invalid stop_price for stop order")
                    order = await asyncio.to_thread(
                        self.trader.create_stop_buy_order, symbol, amount, stop_price,
                        time_in_force=time_in_force, timeout=fill_timeout
                    )
                elif order_type == "stop_limit":
                    stop_price = signal.stop_price
                    limit_price_sl = limit_price  # LLM-provided limit_price for stop_limit
                    if stop_price is None or stop_price <= 0:
                        raise ValueError("Missing or invalid stop_price for stop_limit order")
                    if limit_price_sl is None or limit_price_sl <= 0:
                        raise ValueError("Missing or invalid limit_price for stop_limit order")
                    order = await asyncio.to_thread(
                        self.trader.create_stop_limit_buy_order, symbol, amount,
                        stop_price, limit_price_sl,
                        time_in_force=time_in_force, timeout=fill_timeout
                    )
                elif order_type == "trailing_stop":
                    trail_offset = signal.trail_offset
                    if trail_offset is None or trail_offset <= 0:
                        raise ValueError("Missing or invalid trail_offset for trailing_stop order")
                    order = await asyncio.to_thread(
                        self.trader.create_trailing_stop_buy_order, symbol, amount,
                        trail_offset,
                        time_in_force=time_in_force, timeout=fill_timeout
                    )
                else:
                    raise ValueError(f"Unknown order_type: {order_type}")
                if order.get('status') == 'open':
                    price_str = f" at {limit_price}" if limit_price is not None else ""
                    logger.info(f"BUY {order_type} order for {symbol} queued{price_str}")
                    queued_entry = {
                        'symbol': symbol,
                        'side': 'buy',
                        'amount': amount,
                        'original_amount': amount,
                        'limit_price': limit_price if order_type in ("limit", "stop_limit") else None,
                        'stop_price': signal.stop_price if order_type in ("stop", "stop_limit") else None,
                        'trail_offset': signal.trail_offset if order_type == "trailing_stop" else None,
                        'order_type': order_type,
                        'time_in_force': time_in_force,
                        'signal': asdict(signal),
                        'timeframe': timeframe,
                        'atr': atr,
                        'order_id': order['id'],
                        'queued_at': time.time(),
                        'filled_qty': 0,
                        'filled_cost': 0.0,
                    }
                    async with self._queued_orders_lock:
                        self.queued_orders.append(queued_entry)
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⏳ BUY {order_type} order for {display_symbol} queued{price_str}",
                            summary={"symbol": symbol, "action": "QUEUE", "reason": "Order not yet filled"}
                        )
                    return
                if order.get('status') == 'rejected':
                    logger.warning(f"BUY order rejected for {symbol}")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"❌ BUY order rejected for {display_symbol}",
                            summary={"symbol": symbol, "action": "REJECT", "reason": "Order rejected by simulator"}
                        )
                    return
                logger.info(f"BUY {symbol}: {order}")
                # Queue remaining partial market order for polling
                if order.get("remaining_order_id"):
                    queued_entry = {
                        'symbol': symbol,
                        'side': 'buy',
                        'amount': amount - order['cost'],
                        'original_amount': amount - order['cost'],
                        'limit_price': order['price'],
                        'stop_price': None,
                        'trail_offset': None,
                        'order_type': 'limit',
                        'time_in_force': 'day',
                        'signal': asdict(signal),
                        'timeframe': timeframe,
                        'atr': atr,
                        'order_id': order['remaining_order_id'],
                        'queued_at': time.time(),
                        'filled_qty': 0,
                        'filled_cost': 0.0,
                    }
                    async with self._queued_orders_lock:
                        self.queued_orders.append(queued_entry)
                async with self._cycle_spent_lock:
                    self._cycle_spent += order['cost']
                # Update or create position
                # Extract fee info for cost basis tracking
                fee = order.get('fee', {})
                fee_cost = float(fee.get('cost', 0.0) or 0.0)
                fee_currency = fee.get('currency', '')

                cost_basis = order['cost'] + (fee_cost if fee_currency == quote else 0.0)
                net_base = order['amount'] - (fee_cost if fee_currency == base else 0.0)

                # Risk parameters are guaranteed by the validator
                # sl_pct, tp_pct, trailing_stop, trailing_stop_distance_pct are set above

                if symbol in self.positions:
                    # Accumulate: weighted average price with cost basis
                    old_cost_basis = self.positions[symbol].get("cost_basis", self.positions[symbol]["amount"] * self.positions[symbol]["price"])
                    old_net_base = self.positions[symbol].get("net_base", self.positions[symbol]["amount"])
                    new_cost_basis = old_cost_basis + cost_basis
                    new_net_base = old_net_base + net_base
                    new_price = new_cost_basis / new_net_base if new_net_base > 0 else 0.0
                    self.positions[symbol]["amount"] = new_net_base
                    self.positions[symbol]["price"] = new_price
                    self.positions[symbol]["cost_basis"] = new_cost_basis
                    self.positions[symbol]["net_base"] = new_net_base
                    # Preserve existing absolute SL/TP prices when scaling in.
                    # Recalculating based on the new weighted average would shift
                    # them from where the LLM originally intended. The LLM can
                    # still update SL/TP via _update_position_params (which uses
                    # current_price, not the new average).
                    self.positions[symbol]["take_profit_atr_multiple"] = params.get("take_profit_atr_multiple")
                    self.positions[symbol]["trailing_stop"] = trailing_stop
                    self.positions[symbol]["trailing_stop_distance_pct"] = trailing_stop_distance_pct
                    self.positions[symbol]["trailing_stop_atr_multiple"] = params.get("trailing_stop_atr_multiple")
                    self.positions[symbol]["max_hold_time_seconds"] = params.get("max_hold_time_seconds")
                    self.positions[symbol]["trailing_stop_activation_pct"] = params.get("trailing_stop_activation_pct")
                    self.positions[symbol]["trailing_take_profit"] = params.get("trailing_take_profit", False)
                    self.positions[symbol]["trailing_take_profit_distance_pct"] = params.get("trailing_take_profit_distance_pct")
                    self.positions[symbol]["breakeven_activation_pct"] = params.get("breakeven_activation_pct")
                    # Multiple partial take-profit levels
                    partial_levels = params.get("partial_take_profit_levels")
                    if partial_levels:
                        self.positions[symbol]["partial_take_profit_levels"] = partial_levels
                        self.positions[symbol]["partial_tp_levels_triggered"] = []
                        self.positions[symbol]["partial_tp_depth_wait_start"] = {}
                        # Clear single-level fields to avoid confusion
                        self.positions[symbol]["partial_take_profit_pct"] = None
                        self.positions[symbol]["partial_take_profit_fraction"] = None
                        self.positions[symbol]["partial_tp_triggered"] = None
                    else:
                        self.positions[symbol]["partial_take_profit_pct"] = params.get("partial_take_profit_pct")
                        self.positions[symbol]["partial_take_profit_fraction"] = params.get("partial_take_profit_fraction")
                        self.positions[symbol]["partial_tp_triggered"] = False
                    self.positions[symbol]["cooldown_after_loss_seconds"] = params["cooldown_after_loss_seconds"]
                    self.positions[symbol]["news_sentiment_exit_threshold"] = params.get("news_sentiment_exit_threshold")
                    self.positions[symbol]["max_unrealized_loss_pct"] = params.get("max_unrealized_loss_pct")
                    custom_interval = params.get("strategy_interval_seconds")
                    if custom_interval is not None:
                        self._strategy_intervals[symbol] = custom_interval
                    self.positions[symbol]["timeframe"] = timeframe
                    self.positions[symbol]["indicator_config"] = signal.indicator_config
                    self.positions[symbol]["entry_order_type"] = order_type
                    self.positions[symbol]["buy_confidence"] = signal.confidence
                    self.positions[symbol]["buy_reasoning"] = (signal.reasoning or "")[:200]
                else:
                    entry_price = cost_basis / net_base if net_base > 0 else order["price"]
                    self.positions[symbol] = {
                        "symbol": symbol,
                        "side": "buy",
                        "amount": net_base,
                        "price": entry_price,
                        "timestamp": order["timestamp"],
                        "stop_loss": entry_price * (1 - sl_pct),
                        "take_profit": entry_price * (1 + tp_pct),
                        "take_profit_atr_multiple": params.get("take_profit_atr_multiple"),
                        "cost_basis": cost_basis,
                        "net_base": net_base,
                        "buy_confidence": signal.confidence,
                        "buy_reasoning": (signal.reasoning or "")[:200],
                        "trailing_stop": trailing_stop,
                        "trailing_stop_distance_pct": trailing_stop_distance_pct,
                        "trailing_stop_atr_multiple": params.get("trailing_stop_atr_multiple"),
                        "max_hold_time_seconds": params.get("max_hold_time_seconds"),
                        "trailing_stop_activation_pct": params.get("trailing_stop_activation_pct"),
                        "trailing_take_profit": params.get("trailing_take_profit", False),
                        "trailing_take_profit_distance_pct": params.get("trailing_take_profit_distance_pct"),
                        "breakeven_activation_pct": params.get("breakeven_activation_pct"),
                        "partial_take_profit_levels": params.get("partial_take_profit_levels"),
                        "partial_tp_levels_triggered": [],
                        "partial_tp_depth_wait_start": {},
                        "original_amount": net_base,
                        "partial_take_profit_pct": params.get("partial_take_profit_pct") if not params.get("partial_take_profit_levels") else None,
                        "partial_take_profit_fraction": params.get("partial_take_profit_fraction") if not params.get("partial_take_profit_levels") else None,
                        "partial_tp_triggered": False if not params.get("partial_take_profit_levels") else None,
                        "cooldown_after_loss_seconds": params["cooldown_after_loss_seconds"],
                        "news_sentiment_exit_threshold": params.get("news_sentiment_exit_threshold"),
                        "max_unrealized_loss_pct": params.get("max_unrealized_loss_pct"),
                        "timeframe": timeframe,
                        "indicator_config": signal.indicator_config,
                        "entry_order_type": order_type,
                    }
                    custom_interval = params.get("strategy_interval_seconds")
                    if custom_interval is not None:
                        self._strategy_intervals[symbol] = custom_interval
                # --- Place native exit orders (OCO) if LLM specified them ---
                current_entry = self.positions[symbol]["price"]
                exit_prices = self._compute_exit_order_prices(
                    entry_price=current_entry,
                    signal=signal,
                    atr=atr,
                )
                await self._place_exit_orders(symbol, signal, exit_prices, timeframe)
                order["strategy_type"] = signal.strategy_type
                order["timeframe"] = timeframe
                order["buy_confidence"] = signal.confidence
                order["buy_reasoning"] = (signal.reasoning or "")[:200]
                if hasattr(signal, 'backtest_summary') and signal.backtest_summary:
                    order["backtest_summary"] = signal.backtest_summary
                self._append_trade(order)
                self._balance_cache = None  # force refresh on next fetch
                await asyncio.to_thread(insert_trade, order)
                await self._save_state(force=True)
                if self.notifier:
                    # --- Format symbol for notification ---
                    stock_name = await self._get_stock_name(symbol)
                    display_symbol = self._format_symbol_display(symbol, stock_name, timeframe)
                    buy_msg = f"🟢 BUY {display_symbol}: {order['amount']:.6f} @ {order['price']:.4f}"
                    buy_summary = {
                        "symbol": symbol,
                        "action": "BUY",
                        "price": order["price"],
                        "amount": order["amount"],
                        "confidence": signal.confidence,
                        "reason": signal.reasoning[:200],
                        "strategy_type": signal.strategy_type,
                        "indicators": {
                            "atr": atr,
                        },
                    }
                    if signal.model_type:
                        buy_summary["model_type"] = signal.model_type
                    if signal.llm_provider:
                        buy_summary["llm_provider"] = signal.llm_provider
                    if signal.llm_model:
                        buy_summary["llm_model"] = signal.llm_model
                    await self.notifier.send_notification(
                        buy_msg,
                        summary=buy_summary,
                    )
            except Exception as e:
                logger.error(f"Buy order failed for {symbol}: {e}")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"❌ Buy order failed for {display_symbol}: {e}",
                        summary={
                            "symbol": symbol,
                            "action": "ERROR",
                            "reason": f"Buy order failed: {e}"[:200],
                        }
                    )

        elif signal.action == "SELL":
            # Cancel any native exit orders before selling
            if symbol in self.positions:
                await self._cancel_exit_orders(symbol)
            params = signal.strategy_params or {}
            fill_timeout = params.get("order_fill_timeout_seconds", settings.ORDER_FILL_TIMEOUT_SECONDS)
            # Determine the amount of base currency to sell
            pos = self.positions.get(symbol)
            if pos:
                gross_amount = pos["amount"]
            else:
                gross_amount = balance.get(base, 0.0)

            # Guard against overselling: cap sell amount to actual balance
            actual_base_balance = balance.get(base, 0.0)
            if pos and gross_amount > actual_base_balance:
                logger.warning(
                    f"Tracked position amount {gross_amount} exceeds actual balance "
                    f"{actual_base_balance} for {symbol}. Capping sell amount to actual balance."
                )
                gross_amount = actual_base_balance

            if gross_amount <= 0:
                logger.info(f"No {base} to sell for {symbol}")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"⚠️ No {base} to sell for {display_symbol}",
                        summary={
                            "symbol": symbol,
                            "action": "SKIP",
                            "reason": "No base balance to sell",
                        }
                    )
                return

            # Check minimum sell size
            ticker = None
            try:
                base = symbol.split("/")[0]
                quotes = await self._get_quotes_async([base], timeout=45.0)
                ticker = quotes.get(base)
                price = ticker['last']
                # Fetch minimum order size from asset info
                try:
                    asset = await self._get_asset_info(symbol)
                    min_amount_limit = float(asset.min_order_size) if asset.min_order_size else None
                    if not asset.fractionable and (min_amount_limit is None or min_amount_limit < 1.0):
                        min_amount_limit = 1.0
                except Exception:
                    min_amount_limit = None
                if min_amount_limit is not None and price:
                    min_cost_limit = min_amount_limit * price
                else:
                    min_cost_limit = None
                if min_amount_limit is not None and gross_amount < float(min_amount_limit):
                    logger.info(f"SELL amount {gross_amount:.6f} {base} below min amount {min_amount_limit} for {symbol}, skipping")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⚠️ SELL skipped for {display_symbol}: amount too small",
                            summary={
                                "symbol": symbol,
                                "action": "SKIP",
                                "reason": "Sell amount below minimum",
                            }
                        )
                    return
                if min_cost_limit is not None and gross_amount * price < float(min_cost_limit):
                    logger.info(f"SELL cost {gross_amount * price:.2f} {quote} below min cost {min_cost_limit} for {symbol}, skipping")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⚠️ SELL skipped for {display_symbol}: cost too small",
                            summary={
                                "symbol": symbol,
                                "action": "SKIP",
                                "reason": "Sell cost below minimum",
                            }
                        )
                    return
            except Exception as e:
                logger.warning(f"Could not verify min sell size for {symbol}: {e}")

            need_limit = not self._is_regular_hours()
            limit_price = None
            time_in_force = "day"
            # If LLM provided a limit_price, use it even during regular hours
            llm_limit_price = params.get("limit_price")
            if llm_limit_price is not None and llm_limit_price > 0:
                limit_price = llm_limit_price
                time_in_force = params.get("time_in_force", "day")
                need_limit = True  # force limit order path
            elif need_limit:
                limit_price = self._default_limit_price(symbol, "SELL", ticker)
                time_in_force = params.get("time_in_force", "day")
                if limit_price is None:
                    logger.error(f"Cannot place limit order for {symbol}: no limit price available.")
                    return

            if limit_price is not None:
                # Round to valid tick size ($0.01 for >=$1, $0.0001 for <$1)
                if limit_price >= 1.0:
                    limit_price = round(limit_price, 2)
                else:
                    limit_price = round(limit_price, 4)

            if limit_price is not None and limit_price <= 0:
                logger.error(f"Invalid limit_price {limit_price} for {symbol}, skipping.")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"❌ Invalid limit price for {display_symbol}, skipping.",
                        summary={"symbol": symbol, "action": "SKIP", "reason": "Invalid limit price"}
                    )
                return

            if limit_price is not None:
                # Read LLM-controlled limit price max distance (fallback to static setting)
                max_distance = settings.LIMIT_PRICE_MAX_DISTANCE_PCT
                try:
                    raw = await asyncio.to_thread(self.redis.get, "trading:limit_price_max_distance_pct")
                    if raw:
                        max_distance = float(raw)
                except Exception:
                    pass
                # For a sell, the limit must not be too far above the bid
                if max_distance > 0 and ticker and ticker.get('bid'):
                    bid = ticker['bid']
                    if limit_price > bid * (1 + max_distance):
                        logger.warning(
                            f"LLM limit_price {limit_price} for SELL {symbol} is >{max_distance*100:.0f}% above bid {bid}. "
                            f"Rejecting SELL to avoid indefinite queuing."
                        )
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⚠️ Skipping SELL {display_symbol}: limit price {limit_price} too far above bid {bid}.",
                                summary={"symbol": symbol, "action": "SKIP", "reason": "Limit price too far from market"}
                            )
                        return

            # --- Determine order type for SELL ---
            order_type = signal.order_type
            if order_type not in ("market", "limit", "stop", "stop_limit", "trailing_stop"):
                # Fallback: limit if limit_price provided, else market
                if limit_price is not None:
                    order_type = "limit"
                else:
                    order_type = "market"

            try:
                if order_type == "market":
                    order = await asyncio.to_thread(
                        self.trader.create_market_sell_order, symbol, gross_amount, fill_timeout,
                        limit_price=None, time_in_force='day'
                    )
                elif order_type == "limit":
                    order = await asyncio.to_thread(
                        self.trader.create_market_sell_order, symbol, gross_amount, fill_timeout,
                        limit_price=limit_price, time_in_force=time_in_force
                    )
                elif order_type == "stop":
                    stop_price = signal.stop_price
                    if stop_price is None or stop_price <= 0:
                        raise ValueError("Missing or invalid stop_price for stop order")
                    order = await asyncio.to_thread(
                        self.trader.create_stop_sell_order, symbol, gross_amount, stop_price,
                        time_in_force=time_in_force, timeout=fill_timeout
                    )
                elif order_type == "stop_limit":
                    stop_price = signal.stop_price
                    limit_price_sl = limit_price  # LLM-provided limit_price for stop_limit
                    if stop_price is None or stop_price <= 0:
                        raise ValueError("Missing or invalid stop_price for stop_limit order")
                    if limit_price_sl is None or limit_price_sl <= 0:
                        raise ValueError("Missing or invalid limit_price for stop_limit order")
                    order = await asyncio.to_thread(
                        self.trader.create_stop_limit_sell_order, symbol, gross_amount,
                        stop_price, limit_price_sl,
                        time_in_force=time_in_force, timeout=fill_timeout
                    )
                elif order_type == "trailing_stop":
                    trail_offset = signal.trail_offset
                    if trail_offset is None or trail_offset <= 0:
                        raise ValueError("Missing or invalid trail_offset for trailing_stop order")
                    order = await asyncio.to_thread(
                        self.trader.create_trailing_stop_sell_order, symbol, gross_amount,
                        trail_offset,
                        time_in_force=time_in_force, timeout=fill_timeout
                    )
                else:
                    raise ValueError(f"Unknown order_type: {order_type}")
                if order.get('status') == 'open':
                    order_type_str = "limit" if limit_price is not None else "market"
                    # Override with actual order_type if explicitly set
                    if signal.order_type in ("market", "limit", "stop", "stop_limit", "trailing_stop"):
                        order_type_str = signal.order_type
                    price_str = f" at {limit_price}" if limit_price is not None else ""
                    logger.info(f"SELL {order_type_str} order for {symbol} queued{price_str}")
                    _sell_queued_entry = {
                        'symbol': symbol,
                        'side': 'sell',
                        'amount': gross_amount,
                        'original_amount': gross_amount,
                        'limit_price': limit_price if order_type in ("limit", "stop_limit") else None,
                        'stop_price': signal.stop_price if order_type in ("stop", "stop_limit") else None,
                        'trail_offset': signal.trail_offset if order_type == "trailing_stop" else None,
                        'order_type': order_type_str,
                        'time_in_force': time_in_force,
                        'signal': asdict(signal),
                        'timeframe': timeframe,
                        'atr': atr,
                        'exit_reason': exit_reason,
                        'order_id': order['id'],
                        'queued_at': time.time(),
                        'filled_qty': 0,
                        'filled_cost': 0.0,
                    }
                    async with self._queued_orders_lock:
                        self.queued_orders.append(_sell_queued_entry)
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"⏳ SELL {order_type_str} order for {display_symbol} queued{price_str}",
                            summary={"symbol": symbol, "action": "QUEUE", "reason": "Order not yet filled"}
                        )
                    return
                if order.get('status') == 'rejected':
                    logger.warning(f"SELL order rejected for {symbol}")
                    if self.notifier:
                        await self.notifier.send_notification(
                            f"❌ SELL order rejected for {display_symbol}",
                            summary={"symbol": symbol, "action": "REJECT", "reason": "Order rejected by simulator"}
                        )
                    return
                logger.info(f"SELL {symbol}: {order}")
                # Queue remaining partial market order for polling
                if order.get("remaining_order_id"):
                    _sell_queued_entry = {
                        'symbol': symbol,
                        'side': 'sell',
                        'amount': gross_amount - order['amount'],
                        'original_amount': gross_amount - order['amount'],
                        'limit_price': order['price'],
                        'stop_price': None,
                        'trail_offset': None,
                        'order_type': 'limit',
                        'time_in_force': 'day',
                        'signal': asdict(signal),
                        'timeframe': timeframe,
                        'atr': atr,
                        'exit_reason': exit_reason,
                        'order_id': order['remaining_order_id'],
                        'queued_at': time.time(),
                        'filled_qty': 0,
                        'filled_cost': 0.0,
                    }
                    async with self._queued_orders_lock:
                        self.queued_orders.append(_sell_queued_entry)
                # Compute realized P&L
                fee = order.get('fee', {})
                fee_cost = float(fee.get('cost', 0.0) or 0.0)
                fee_currency = fee.get('currency', '')

                net_quote = order['cost'] - (fee_cost if fee_currency == quote else 0.0)
                is_partial_sell = order.get("remaining_order_id") is not None
                if pos:
                    cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                    net_base = pos.get("net_base", pos["amount"])
                    if is_partial_sell and net_base > 0:
                        # Prorate cost basis for the sold portion
                        prorated_cost_basis = cost_basis * (order['amount'] / net_base)
                        realized_pnl = net_quote - prorated_cost_basis
                        order["cost_basis"] = prorated_cost_basis
                    else:
                        realized_pnl = net_quote - cost_basis
                        order["cost_basis"] = cost_basis
                else:
                    realized_pnl = 0.0
                    order["cost_basis"] = 0.0
                order["realized_pnl"] = realized_pnl
                # Track loss timestamps for cooldown
                if realized_pnl < 0:
                    self.last_loss_time[symbol] = time.time()
                    cd = pos.get("cooldown_after_loss_seconds", 0) if pos else 0
                    self.cooldown_durations[symbol] = cd
                tf = timeframe or (pos.get("timeframe") if pos else None)
                order["timeframe"] = tf
                order["strategy_type"] = signal.strategy_type
                if pos:
                    order["buy_confidence"] = pos.get("buy_confidence", 0.0)
                    order["buy_reasoning"] = pos.get("buy_reasoning", "")
                order["exit_reason"] = exit_reason
                order["exit_price"] = order["price"]
                if pos and "timestamp" in pos:
                    hold_time = (order["timestamp"] - pos["timestamp"]) / 1000.0
                    order["hold_time_seconds"] = hold_time
                else:
                    order["hold_time_seconds"] = None
                # Clear any stop-loss review flags
                if pos:
                    pos.pop("_stop_loss_triggered", None)
                    pos.pop("_stop_loss_review_count", None)

                if is_partial_sell and pos:
                    # Partial sell: reduce position instead of removing it
                    remaining_amount = pos["amount"] - order['amount']
                    remaining_cost_basis = cost_basis - order["cost_basis"]
                    remaining_net_base = net_base - order['amount']
                    if remaining_amount <= 0 or remaining_net_base <= 0:
                        async with self._positions_lock:
                            self.positions.pop(symbol, None)
                        self._strategy_intervals.pop(symbol, None)
                        self._last_strategy_eval.pop(symbol, None)
                        self._last_decisions.pop(symbol, None)
                        self._pending_entries.pop(symbol, None)
                        await self._remove_symbol_if_paused(symbol)
                    else:
                        async with self._positions_lock:
                            self.positions[symbol]["amount"] = remaining_amount
                            self.positions[symbol]["cost_basis"] = remaining_cost_basis
                            self.positions[symbol]["net_base"] = remaining_net_base
                            self.positions[symbol]["price"] = remaining_cost_basis / remaining_net_base if remaining_net_base > 0 else 0.0
                        # Place replacement exit orders for the remaining position
                        from src.strategies.base import Signal as _Signal
                        _dummy_params = {
                            "trailing_take_profit": self.positions[symbol].get("trailing_take_profit", False),
                            "partial_take_profit_levels": self.positions[symbol].get("partial_take_profit_levels"),
                            "partial_take_profit_pct": self.positions[symbol].get("partial_take_profit_pct"),
                        }
                        _dummy_signal = _Signal(
                            action="BUY",
                            confidence=1.0,
                            reasoning="Replacing exit orders after partial sell",
                            stop_loss_order_type=self.positions[symbol].get("stop_loss_order_type"),
                            stop_loss_stop_price=self.positions[symbol].get("stop_loss"),
                            stop_loss_limit_price=None,
                            take_profit_order_type=self.positions[symbol].get("take_profit_order_type"),
                            take_profit_limit_price=self.positions[symbol].get("take_profit"),
                            strategy_params=_dummy_params,
                        )
                        _exit_prices = {
                            "stop_loss_price": self.positions[symbol].get("stop_loss"),
                            "take_profit_price": self.positions[symbol].get("take_profit"),
                        }
                        try:
                            await self._place_exit_orders(symbol, _dummy_signal, _exit_prices, self.positions[symbol].get("timeframe"))
                        except Exception as _e:
                            logger.warning(f"Failed to place replacement exit orders after partial sell for {symbol}: {_e}")
                else:
                    # Full sell: remove position
                    async with self._positions_lock:
                        self.positions.pop(symbol, None)
                    self._strategy_intervals.pop(symbol, None)
                    self._last_strategy_eval.pop(symbol, None)
                    self._last_decisions.pop(symbol, None)
                    self._pending_entries.pop(symbol, None)
                    await self._remove_symbol_if_paused(symbol)
                self._append_trade(order)
                self._balance_cache = None
                await asyncio.to_thread(insert_trade, order)
                await self._save_state(force=True)
                if self.notifier:
                    # Human-readable labels for common exit reasons
                    reason_labels = {
                        "manual_sell": "🖐️ Manual",
                        "manual_sell_all": "🖐️ Manual (Sell All)",
                        "stop_loss": "⛔ Stop-Loss",
                        "take_profit": "✅ Take-Profit",
                        "max_hold_time": "⏰ Max Hold Time",
                        "news_sentiment_exit": "📰 News Sentiment",
                        "force_close": "🔻 Force Close",
                        "external_sell": "🔄 External Sell",
                        "delisted": "🗑️ Delisted",
                    }
                    reason_label = reason_labels.get(exit_reason, exit_reason) if exit_reason else None
                    reason_str = f" [{reason_label}]" if reason_label else ""
                    # --- Format symbol for notification ---
                    stock_name = await self._get_stock_name(symbol)
                    # Use the timeframe from the position or the passed parameter
                    tf = timeframe or (pos.get("timeframe") if pos else None)
                    display_symbol = self._format_symbol_display(symbol, stock_name, tf)
                    partial_str = " (partial)" if is_partial_sell else ""
                    sell_msg = f"🔴 SELL{reason_str}{partial_str} {display_symbol}: {order['amount']:.6f} @ {order['price']:.4f}"
                    # Add profit/loss info
                    if pos:
                        pnl_pct = (realized_pnl / order["cost_basis"] * 100) if order["cost_basis"] > 0 else 0.0
                        sell_msg += f" | P&L: {realized_pnl:+.4f} ({pnl_pct:+.2f}%)"
                    sell_summary = {
                        "symbol": symbol,
                        "action": "SELL",
                        "price": order["price"],
                        "amount": order["amount"],
                        "confidence": signal.confidence,
                        "reason": signal.reasoning[:200],
                        "exit_reason": exit_reason,
                        "realized_pnl": realized_pnl,
                        "strategy_type": signal.strategy_type,
                        "indicators": {
                            "atr": atr,
                        },
                    }
                    if signal.model_type:
                        sell_summary["model_type"] = signal.model_type
                    if signal.llm_provider:
                        sell_summary["llm_provider"] = signal.llm_provider
                    if signal.llm_model:
                        sell_summary["llm_model"] = signal.llm_model
                    await self.notifier.send_notification(
                        sell_msg,
                        summary=sell_summary,
                    )
            except Exception as e:
                logger.error(f"Sell order failed for {symbol}: {e}")
                if self.notifier:
                    await self.notifier.send_notification(
                        f"❌ Sell order failed for {display_symbol}: {e}",
                        summary={
                            "symbol": symbol,
                            "action": "ERROR",
                            "reason": f"Sell order failed: {e}"[:200],
                        }
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
        """Return True if it’s safe to skip the LLM call and just HOLD."""
        # If a force evaluation was requested (entry signal detected), never skip
        if self._force_eval.get(symbol, False):
            return False
        # Never skip critical situations (max hold, stop-loss, take-profit triggered)
        if is_critical:
            return False

        # ATR is used for price-change comparison but is not strictly required.
        # When ATR is None (common for long timeframes like 1Y/3Y/5Y), we fall
        # back to a fixed percentage threshold so the skip logic still works
        # and we don't waste LLM calls every cycle.

        snapshot = self._last_eval_snapshot.get(symbol)
        if snapshot is None:
            # First evaluation – must call
            return False

        now = time.time()
        last_time = snapshot.get("timestamp", 0)
        last_price = snapshot.get("price", 0)

        # Always call if enough time has passed (3× the effective interval)
        # For medium/long-term, be more patient before forcing an evaluation
        effective_interval = timeframe_seconds * settings.STRATEGY_INTERVAL_MULTIPLIER
        # Cap the safety net at the configured max skip interval so the bot
        # never skips LLM evaluations indefinitely, even for very long
        # timeframes (e.g., 1Y, 3Y, 5Y where 3× the interval would be ~3 years).
        # Cap the safety net at a value proportional to the timeframe,
        # but never less than the configured MAX_SKIP_INTERVAL_SECONDS.
        # This prevents excessively frequent forced evaluations for long
        # timeframes (e.g., 5Y candles should not be forced every 7 days).
        max_skip = max(settings.MAX_SKIP_INTERVAL_SECONDS, int(timeframe_seconds))
        if now - last_time > min(3 * effective_interval, max_skip):
            return False

        # Fetch LLM-driven skip thresholds from Redis.
        # If the LLM has not configured the skip logic, do not skip – always
        # call the LLM so it can decide based on the current market data.
        skip_price_mult_raw = await asyncio.to_thread(self.redis.get, "trading:skip_eval_price_change_atr_mult")
        skip_price_mult = float(skip_price_mult_raw) if skip_price_mult_raw else None

        skip_rsi_raw = await asyncio.to_thread(self.redis.get, "trading:skip_eval_rsi_change")
        skip_rsi = float(skip_rsi_raw) if skip_rsi_raw else None

        skip_macd_raw = await asyncio.to_thread(self.redis.get, "trading:skip_eval_macd_hist_change")
        skip_macd = float(skip_macd_raw) if skip_macd_raw else None

        # If the core skip thresholds are missing, the LLM has not configured
        # the skip logic – always call the LLM.
        if skip_price_mult is None or skip_rsi is None or skip_macd is None:
            return False

        # Price change since last evaluation
        if last_price > 0:
            price_change_pct = abs(current_price - last_price) / last_price
            # If price moved less than skip_price_mult × ATR (in %), it’s boring
            atr_pct = (atr / current_price) if (atr and atr > 0) else 0.005
            if price_change_pct > atr_pct * skip_price_mult:
                return False   # enough movement to warrant a new look

        # Indicator changes
        last_rsi = snapshot.get("rsi")
        last_macd_hist = snapshot.get("macd_hist")
        if rsi is not None and last_rsi is not None:
            if abs(rsi - last_rsi) > skip_rsi:
                return False
        if macd_hist is not None and last_macd_hist is not None:
            if abs(macd_hist - last_macd_hist) > skip_macd:
                return False

        # MACD histogram sign change (crossover) — momentum shift
        if macd_hist is not None and last_macd_hist is not None:
            if (macd_hist > 0) != (last_macd_hist > 0):
                return False

        # If we have no open position and nothing is screaming, skip
        if not has_position:
            # Only call if there is a potential entry signal (extreme RSI, MACD crossover, etc.)
            # RSI extreme? (thresholds are LLM-decided)
            # RSI extremes are optional – only use them if the LLM has set them.
            rsi_oversold = None
            rsi_overbought = None
            try:
                raw = await asyncio.to_thread(self.redis.get, "trading:skip_eval_rsi_oversold")
                if raw:
                    rsi_oversold = float(raw)
                raw = await asyncio.to_thread(self.redis.get, "trading:skip_eval_rsi_overbought")
                if raw:
                    rsi_overbought = float(raw)
            except Exception:
                pass
            if (
                rsi is not None
                and rsi_oversold is not None
                and rsi_overbought is not None
                and (rsi < rsi_oversold or rsi > rsi_overbought)
            ):
                return False
            # MACD histogram direction change? (harder to detect without previous sign – skip for simplicity)
            # Otherwise, no strong signal → skip
            return True

        # Have an open position – skip if price far from stop/tp and indicators calm
        # (the risk management loop will handle stop/tp)
        return True

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
                    cooldown = getattr(settings, 'ENTRY_SIGNAL_COOLDOWN_SECONDS', 30)
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
        # Fetch pre-computed indicators from DB
        ind = await asyncio.to_thread(get_indicators, symbol, timeframe)

        # Still need candles for volume EMA computation and fallback
        # indicator computation
        db_candles = await asyncio.to_thread(
            get_ohlcv, symbol, timeframe, limit=50
        )
        if len(db_candles) < 26:
            return False

        # If DB indicators are missing (common for long timeframes like
        # 5Y/3Y/1Y where indicators may not be stored), compute them
        # on-the-fly from the OHLCV candles so entry signal detection
        # still works.
        if not ind:
            raw_candles = [
                [c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
                for c in db_candles
            ]
            try:
                ind = await asyncio.to_thread(compute_all_indicators, raw_candles)
            except Exception as e:
                logger.debug(
                    f"Failed to compute indicators on-the-fly for {symbol} {timeframe}: {e}"
                )
                return False
            if not ind:
                return False

        closes = [c["close"] for c in db_candles]
        volumes = [c["volume"] for c in db_candles]

        # Retrieve previous state
        prev = self._entry_signal_state.get(symbol, {})

        # Current values
        rsi = ind.get("rsi")
        macd_hist = ind.get("macd_hist")
        macd_val = ind.get("macd")
        macd_signal = ind.get("macd_signal")
        stoch_k = ind.get("stochastic_k")
        adx = ind.get("adx")
        plus_di = ind.get("plus_di")
        minus_di = ind.get("minus_di")
        bb_upper = ind.get("bb_upper")
        bb_lower = ind.get("bb_lower")
        bb_middle = ind.get("bb_middle")
        ema_9 = ind.get("ema_9")
        ema_21 = ind.get("ema_21")
        parabolic_sar = ind.get("parabolic_sar")
        ichimoku = ind.get("ichimoku")
        current_close = closes[-1] if closes else None

        # Volume EMA for spike detection (using talib via compute_ema).
        # Exclude the latest candle (which may be incomplete for intraday
        # timeframes) by using the second-to-last EMA value.
        volume_ema_list = compute_ema(volumes, 20)
        volume_ema = volume_ema_list[-2] if len(volume_ema_list) >= 2 else 0.0

        # Store current state for next cycle
        new_state = {
            "rsi": rsi,
            "macd_hist": macd_hist,
            "macd_val": macd_val,
            "macd_signal": macd_signal,
            "stoch_k": stoch_k,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_middle": bb_middle,
            "ema_9": ema_9,
            "ema_21": ema_21,
            "parabolic_sar": parabolic_sar,
            "ichimoku_cloud_top": ichimoku["cloud_top"] if ichimoku else None,
            "ichimoku_cloud_bottom": ichimoku["cloud_bottom"] if ichimoku else None,
            "close": current_close,
            "volume_ema": volume_ema,
        }
        self._entry_signal_state[symbol] = new_state

        # --- Read LLM-defined thresholds from Redis (fallback to defaults) ---
        rsi_oversold = 30.0
        rsi_overbought = 70.0
        adx_moderate = 25.0
        bb_squeeze_width = 0.02
        try:
            raw = await asyncio.to_thread(self.redis.get, "trading:skip_eval_rsi_oversold")
            if raw:
                rsi_oversold = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:skip_eval_rsi_overbought")
            if raw:
                rsi_overbought = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:regime_adx_moderate")
            if raw:
                adx_moderate = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:regime_bb_squeeze_width")
            if raw:
                bb_squeeze_width = float(raw)
        except Exception:
            pass

        prev_close = prev.get("close")

        # --- Long-term timeframe detection (>= 1 month) ---
        # Use trend reversal and regime shift logic instead of short-term
        # crossovers, which fire on every candle for long timeframes because
        # indicators change dramatically between candles (e.g., RSI 20→80).
        tf_seconds = self._timeframe_to_seconds(timeframe)
        if tf_seconds >= 2_592_000:  # >= 1 month (1M, 3M, 6M, 1Y, 3Y, 5Y)
            # 1. Trend direction reversal: +DI crosses above -DI
            prev_plus_di = prev.get("plus_di")
            prev_minus_di = prev.get("minus_di")
            if (prev_plus_di is not None and prev_minus_di is not None
                    and plus_di is not None and minus_di is not None
                    and prev_plus_di <= prev_minus_di and plus_di > minus_di):
                return True

            # 2. Trend initiation: ADX crosses above moderate threshold
            prev_adx = prev.get("adx")
            if (prev_adx is not None and adx is not None
                    and prev_adx <= adx_moderate and adx > adx_moderate
                    and plus_di is not None and minus_di is not None
                    and plus_di > minus_di):
                return True

            # 3. Major breakout: price breaks above Donchian upper channel
            donchian = ind.get("donchian_channels")
            if (donchian is not None and prev_close is not None
                    and current_close is not None):
                dc_upper = donchian.get("upper")
                if dc_upper is not None and prev_close <= dc_upper and current_close > dc_upper:
                    return True

            # 4. Ichimoku cloud breakout: price crosses above cloud top
            prev_cloud_top = prev.get("ichimoku_cloud_top")
            if (prev_cloud_top is not None and ichimoku is not None
                    and prev_close is not None and current_close is not None):
                cloud_top = ichimoku.get("cloud_top")
                if cloud_top is not None and prev_close <= cloud_top and current_close > cloud_top:
                    return True

            # 5. MACD zero-line crossover (long-term momentum shift)
            prev_macd_val = prev.get("macd_val")
            if (prev_macd_val is not None and macd_val is not None
                    and prev_macd_val <= 0 and macd_val > 0):
                return True

            # 6. EMA golden cross (valid for long timeframes — major trend shift)
            prev_ema_9 = prev.get("ema_9")
            prev_ema_21 = prev.get("ema_21")
            if (prev_ema_9 is not None and prev_ema_21 is not None
                    and ema_9 is not None and ema_21 is not None
                    and prev_ema_9 <= prev_ema_21 and ema_9 > ema_21):
                return True

            # No long-term entry signal detected
            return False

        # --- Condition checks ---
        # 1. RSI oversold
        if rsi is not None and rsi < rsi_oversold:
            return True

        # 2. MACD histogram bullish crossover (was negative, now positive)
        prev_macd_hist = prev.get("macd_hist")
        if (prev_macd_hist is not None and macd_hist is not None
                and prev_macd_hist <= 0 and macd_hist > 0):
            return True

        # 3. RSI leaving oversold (momentum shift)
        prev_rsi = prev.get("rsi")
        if (prev_rsi is not None and rsi is not None
                and prev_rsi < rsi_oversold and rsi >= rsi_oversold):
            return True

        # 4. MACD line crossing above signal line (bullish crossover)
        prev_macd_val = prev.get("macd_val")
        prev_macd_signal = prev.get("macd_signal")
        if (prev_macd_val is not None and prev_macd_signal is not None
                and macd_val is not None and macd_signal is not None
                and prev_macd_val <= prev_macd_signal and macd_val > macd_signal):
            return True

        # 6. ADX rising above moderate threshold and +DI > -DI (trend start)
        prev_adx = prev.get("adx")
        if (adx is not None and plus_di is not None and minus_di is not None
                and plus_di > minus_di
                and prev_adx is not None and prev_adx <= adx_moderate and adx > adx_moderate):
            return True

        # 7. Bollinger Band squeeze breakout
        prev_bb_upper = prev.get("bb_upper")
        prev_bb_lower = prev.get("bb_lower")
        prev_bb_middle = prev.get("bb_middle")
        if (prev_bb_upper is not None and prev_bb_lower is not None and prev_bb_middle is not None
                and bb_upper is not None and bb_lower is not None and bb_middle is not None
                and prev_bb_middle > 0 and bb_middle > 0):
            prev_width = (prev_bb_upper - prev_bb_lower) / prev_bb_middle
            curr_width = (bb_upper - bb_lower) / bb_middle
            if prev_width < bb_squeeze_width and current_close is not None and current_close > bb_upper:
                return True

        # 8. Volume spike (last COMPLETE candle volume > 3 * EMA of volume)
        # Use the second-to-last candle to avoid false signals from the
        # latest candle which may still be forming (incomplete volume).
        if len(volumes) >= 2 and volume_ema > 0 and volumes[-2] > 3.0 * volume_ema:
            return True

        # 9. EMA9 crossing above EMA21 (golden cross)
        prev_ema_9 = prev.get("ema_9")
        prev_ema_21 = prev.get("ema_21")
        if (prev_ema_9 is not None and prev_ema_21 is not None
                and ema_9 is not None and ema_21 is not None
                and prev_ema_9 <= prev_ema_21 and ema_9 > ema_21):
            return True

        # 11. Parabolic SAR flip (from above price to below price → uptrend)
        prev_sar = prev.get("parabolic_sar")
        if (prev_sar is not None and parabolic_sar is not None
                and prev_close is not None and current_close is not None
                and prev_sar > prev_close and parabolic_sar < current_close):
            return True

        # 12. Ichimoku: price crossing above cloud
        prev_cloud_top = prev.get("ichimoku_cloud_top")
        prev_cloud_bottom = prev.get("ichimoku_cloud_bottom")
        if (prev_cloud_top is not None and prev_cloud_bottom is not None
                and ichimoku is not None
                and prev_close is not None and current_close is not None):
            cloud_top = ichimoku["cloud_top"]
            cloud_bottom = ichimoku["cloud_bottom"]
            # Previous close was below or inside cloud, current close above cloud top
            if prev_close <= cloud_top and current_close > cloud_top:
                return True

        return False

    async def _check_pending_entries(self):
        """Periodically check pending entry conditions and execute if met."""
        await asyncio.sleep(10)  # short initial delay
        while self._running:
            try:
                now = time.time()
                for symbol in list(self._pending_entries.keys()):
                    entry = self._pending_entries.get(symbol)
                    if entry is None:
                        continue
                    entry_tf = entry.get("timeframe")
                    stock_name = await self._get_stock_name(symbol)
                    display_symbol = self._format_symbol_display(symbol, stock_name, entry_tf)
                    if now >= entry["deadline"]:
                        # Timeout – clear and notify
                        logger.info(f"Entry condition timeout for {symbol}")
                        if self.notifier:
                            await self.notifier.send_notification(
                                f"⏭️ Entry condition timeout for {display_symbol} – skipping BUY.",
                                summary={
                                    "symbol": symbol,
                                    "action": "SKIP",
                                    "reason": "Entry condition timeout",
                                }
                            )
                        del self._pending_entries[symbol]
                        self._state_dirty = True
                        continue

                    # Check the condition (non‑blocking)
                    condition_met = await self._check_entry_condition_once(
                        symbol, entry["condition"], entry["timeframe"]
                    )
                    if condition_met:
                        logger.info(f"Entry condition met for {symbol}, executing BUY")
                        # Remove from pending before executing to avoid re‑trigger
                        signal = entry["signal"]
                        del self._pending_entries[symbol]
                        self._state_dirty = True
                        # Check trading pause again (may have changed)
                        paused = await asyncio.to_thread(self.redis.get, "trading:paused")
                        if paused:
                            logger.info(f"Ignoring queued BUY {symbol}: trading is now paused.")
                            if self.notifier:
                                await self.notifier.send_notification(
                                    f"⏸️ Queued BUY for {display_symbol} skipped – trading paused.",
                                    summary={"symbol": symbol, "action": "SKIP", "reason": "Trading paused"}
                                )
                        else:
                            await self._execute_signal(
                                symbol,
                                signal,
                                timeframe=entry["timeframe"],
                                atr=None,
                            )
            except Exception as e:
                logger.error(f"Error checking pending entries: {e}", exc_info=True)
            await asyncio.sleep(60)  # check every 60 seconds (medium/long-term)

    async def _check_entry_condition_once(
        self, symbol: str, condition: Dict[str, Any], timeframe: str
    ) -> bool:
        """Check a single entry condition immediately. Return True if met."""
        etype = condition.get("type")
        if etype == "limit_price":
            target_price = condition["price"]
            try:
                tickers_map = await self._get_quotes_async([symbol.split("/")[0]], timeout=45.0)
                ticker = tickers_map.get(symbol.split("/")[0])
            except Exception:
                return False
            current_price = ticker.get("last", 0) if ticker else 0
            return current_price > 0 and current_price <= target_price

        elif etype == "rsi_threshold":
            target_rsi = condition["rsi_below"]
            ind = await asyncio.to_thread(get_indicators, symbol, timeframe)
            if ind:
                rsi = ind.get("rsi")
                return rsi is not None and rsi <= target_rsi
            return False

        elif etype == "delay":
            # Delay conditions are handled by _execute_delayed_entry, not the
            # pending-entries system. If we somehow reach here, treat as not met
            # so the deadline handler can deal with it.
            return False

        elif etype == "indicator_combo":
            conditions = condition["conditions"]
            ind = await asyncio.to_thread(get_indicators, symbol, timeframe)
            if not ind:
                return False
            # Mapping of indicator names the LLM can use to DB keys.
            # All scalar indicators stored in the indicators table are supported.
            _INDICATOR_KEYS = {
                "rsi": "rsi",
                "macd": "macd",
                "macd_signal": "macd_signal",
                "macd_hist": "macd_hist",
                "bb_upper": "bb_upper",
                "bb_middle": "bb_middle",
                "bb_lower": "bb_lower",
                "ema_9": "ema_9",
                "ema_21": "ema_21",
                "stochastic_k": "stochastic_k",
                "stochastic_d": "stochastic_d",
                "adx": "adx",
                "plus_di": "plus_di",
                "minus_di": "minus_di",
                "obv": "obv",
                "mfi": "mfi",
                "cci": "cci",
                "williams_r": "williams_r",
                "parabolic_sar": "parabolic_sar",
                "atr": "atr",
            }
            for cond in conditions:
                indicator_name = cond["indicator"]
                thresh = cond["threshold"]
                direction = cond["direction"]
                db_key = _INDICATOR_KEYS.get(indicator_name)
                if db_key is None:
                    logger.warning(
                        f"Unsupported indicator '{indicator_name}' in indicator_combo "
                        f"entry condition for {symbol}"
                    )
                    return False
                val = ind.get(db_key)
                if val is None:
                    return False
                if direction == "below" and val > thresh:
                    return False
                if direction == "above" and val < thresh:
                    return False
            return True

        return False

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
        # Volatility
        atr: Optional[float] = None,
        atr_percentile: Optional[float] = None,
        # Core indicators
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
        # Market context
        market_regime: str = "",
        market_breadth: Optional[Dict[str, Any]] = None,
        full_market_breadth: Optional[Dict[str, Any]] = None,
        sentiment_trend_val: Optional[float] = None,
        volume_trend: Optional[float] = None,
        # Portfolio context
        unrealized_pnl: Optional[float] = None,
        drawdown_pct: Optional[float] = None,
        portfolio_exposure_pct: Optional[float] = None,
        portfolio_stop_risk_pct: Optional[float] = None,
        # Position state
        is_critical: bool = False,
        trading_paused: bool = False,
        # Events & fundamentals
        symbol_event: Optional[Dict[str, Any]] = None,
        fundamentals: Optional[Dict[str, Any]] = None,
        # Past performance
        consecutive_losses: int = 0,
        # Current price
        current_price: Optional[float] = None,
    ) -> str:
        """Return "mind" or "actuator" based on market complexity.

        Considers the same parameters as Step 1 (build_strategy_prompt) to ensure
        model selection is calibrated to the full market context.
        """
        if is_critical:
            return "mind"

        score = 0.0
        max_score = 0.0

        # === Critical factors (weight 2.0) — directly affect risk/decision quality ===
        # Conflicting technicals: RSI extreme vs MACD direction
        if rsi is not None and macd_hist is not None:
            max_score += 2.0
            if (rsi < 30 and macd_hist < 0) or (rsi > 70 and macd_hist > 0):
                score += 2.0

        # EMA alignment conflict with ADX/DI trend
        if all(v is not None for v in (ema_9, ema_21, adx, plus_di, minus_di)):
            max_score += 2.0
            if (ema_9 > ema_21) != (plus_di > minus_di) and adx > 25:
                score += 2.0

        # Account drawdown
        if drawdown_pct is not None:
            max_score += 2.0
            if drawdown_pct > 10:
                score += 2.0

        # Symbol event detected (earnings, FDA, M&A, etc.)
        if symbol_event is not None:
            max_score += 2.0
            if symbol_event.get("has_event"):
                score += 2.0

        # Consecutive losses
        max_score += 2.0
        if consecutive_losses >= 3:
            score += 2.0

        # === Significant factors (weight 1.5) — market structure changes ===
        # Volatility extremes (ATR percentile)
        if atr_percentile is not None:
            max_score += 1.5
            if atr_percentile > 80 or atr_percentile < 20:
                score += 1.5

        # Turbulent market regime
        if market_regime:
            max_score += 1.5
            if any(kw in market_regime for kw in ("high volatility", "squeeze", "expansion", "ranging")):
                score += 1.5

        # Strong sentiment swing
        if sentiment_trend_val is not None:
            max_score += 1.5
            if abs(sentiment_trend_val) > 0.2:
                score += 1.5

        # Bollinger Band squeeze or expansion
        if all(v is not None for v in (bb_upper, bb_lower, bb_middle)) and bb_middle > 0:
            max_score += 1.5
            bb_width = (bb_upper - bb_lower) / bb_middle
            if bb_width < 0.02 or bb_width > 0.08:
                score += 1.5

        # Portfolio stress: high exposure
        if portfolio_exposure_pct is not None:
            max_score += 1.5
            if portfolio_exposure_pct > 70:
                score += 1.5

        # Portfolio stress: high stop risk
        if portfolio_stop_risk_pct is not None:
            max_score += 1.5
            if portfolio_stop_risk_pct > 8:
                score += 1.5

        # Large unrealized loss (position under stress)
        if unrealized_pnl is not None:
            max_score += 1.5
            if unrealized_pnl < 0:
                score += 1.5

        # Market breadth extremes (candidate stocks)
        if market_breadth is not None:
            max_score += 1.5
            pos_pct = market_breadth.get("positive_pct", 50)
            if pos_pct > 80 or pos_pct < 20:
                score += 1.5

        # Market breadth extremes (full universe)
        if full_market_breadth is not None:
            max_score += 1.5
            pos_pct = full_market_breadth.get("positive_pct", 50)
            if pos_pct > 80 or pos_pct < 20:
                score += 1.5

        # === Standard factors (weight 1.0) — supplementary indicators ===
        # MACD crossover nearby (lines very close → indecision)
        if macd is not None and macd_signal is not None and macd != 0:
            max_score += 1.0
            if abs(macd - macd_signal) < 0.0001 * abs(macd):
                score += 1.0

        # Stochastic extremes
        if stochastic_k is not None:
            max_score += 1.0
            if stochastic_k < 20 or stochastic_k > 80:
                score += 1.0

        # MFI extremes
        if mfi is not None:
            max_score += 1.0
            if mfi < 20 or mfi > 80:
                score += 1.0

        # CCI extremes
        if cci is not None:
            max_score += 1.0
            if cci < -100 or cci > 100:
                score += 1.0

        # Williams %R extremes
        if williams_r is not None:
            max_score += 1.0
            if williams_r < -80 or williams_r > -20:
                score += 1.0

        # Ichimoku cloud conflict (price inside cloud = uncertainty)
        if ichimoku is not None and current_price is not None:
            cloud_top = ichimoku.get("cloud_top")
            cloud_bottom = ichimoku.get("cloud_bottom")
            if cloud_top is not None and cloud_bottom is not None:
                max_score += 1.0
                if cloud_bottom <= current_price <= cloud_top:
                    score += 1.0

        # Volume spike
        if volume_trend is not None:
            max_score += 1.0
            if volume_trend > 3.0:
                score += 1.0

        # Extreme fundamentals (very high P/E or negative margins)
        if fundamentals is not None:
            pe = fundamentals.get("pe_ratio")
            if pe is not None:
                max_score += 1.0
                if pe > 50 or pe < 0:
                    score += 1.0
            margins = fundamentals.get("profit_margins")
            if margins is not None:
                max_score += 1.0
                if margins < 0:
                    score += 1.0

        # Normalize score against the maximum achievable given available data,
        # then compare against the configurable threshold.
        if max_score == 0:
            return "actuator"
        normalized_score = score / max_score
        return "mind" if normalized_score >= settings.LLM_MIND_MODEL_THRESHOLD else "actuator"

    def _compute_prompt_complexity(
        self,
        # Candidate/portfolio context
        num_candidates: int = 0,
        # Volatility
        volatility_percentile: Optional[float] = None,
        # Core indicators
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
        # Market context
        market_breadth: Optional[Dict[str, Any]] = None,
        full_market_breadth: Optional[Dict[str, Any]] = None,
        sentiment_trend_magnitude: Optional[float] = None,
        volume_trend: Optional[float] = None,
        market_regime: str = "",
        # Portfolio context
        unrealized_pnl: Optional[float] = None,
        drawdown_pct: Optional[float] = None,
        portfolio_exposure_pct: Optional[float] = None,
        portfolio_stop_risk_pct: Optional[float] = None,
        # Position state
        is_critical: bool = False,
        trading_paused: bool = False,
        # Events & fundamentals
        symbol_event: Optional[Dict[str, Any]] = None,
        fundamentals: Optional[Dict[str, Any]] = None,
        # Past performance
        consecutive_losses: int = 0,
        # Current price
        current_price: Optional[float] = None,
        # Legacy params (kept for backward compat with other call sites)
        fear_greed: Optional[Dict[str, Any]] = None,
        conflicting_signals: bool = False,
    ) -> float:
        """Return a complexity score between 0.0 (simple) and 1.0 (very complex).

        Considers the same parameters as Step 1 (build_strategy_prompt) to ensure
        temperature selection is calibrated to the full market context.
        """
        # Category-based scoring: within each category, take the MAX contribution
        # (not the sum) so that multiple factors in the same category don't
        # dominate the score.  Category maxima sum to 1.0 exactly.

        # === Category 1: Technical indicator extremes (max 0.25) ===
        tech_score = 0.0
        if rsi is not None and (rsi < 30 or rsi > 70):
            tech_score = max(tech_score, 0.15)
        if stochastic_k is not None and (stochastic_k < 20 or stochastic_k > 80):
            tech_score = max(tech_score, 0.12)
        if mfi is not None and (mfi < 20 or mfi > 80):
            tech_score = max(tech_score, 0.12)
        if cci is not None and (cci < -100 or cci > 100):
            tech_score = max(tech_score, 0.12)
        if williams_r is not None and (williams_r < -80 or williams_r > -20):
            tech_score = max(tech_score, 0.12)
        if macd is not None and macd_signal is not None and macd != 0:
            if abs(macd - macd_signal) < 0.0001 * abs(macd):
                tech_score = max(tech_score, 0.10)
        if bb_upper is not None and bb_lower is not None and bb_middle is not None and bb_middle > 0:
            bb_width = (bb_upper - bb_lower) / bb_middle
            if bb_width < 0.02 or bb_width > 0.08:
                tech_score = max(tech_score, 0.15)
        if ichimoku is not None and current_price is not None:
            cloud_top = ichimoku.get("cloud_top")
            cloud_bottom = ichimoku.get("cloud_bottom")
            if cloud_top is not None and cloud_bottom is not None:
                if cloud_bottom <= current_price <= cloud_top:
                    tech_score = max(tech_score, 0.12)

        # === Category 2: Conflicting signals (max 0.20) ===
        conflict_score = 0.0
        if rsi is not None and macd_hist is not None:
            if (rsi < 30 and macd_hist < 0) or (rsi > 70 and macd_hist > 0):
                conflict_score = max(conflict_score, 0.20)
        if ema_9 is not None and ema_21 is not None and adx is not None and plus_di is not None and minus_di is not None:
            ema_bullish = ema_9 > ema_21
            di_bullish = plus_di > minus_di
            if ema_bullish != di_bullish and adx > 25:
                conflict_score = max(conflict_score, 0.15)
        if conflicting_signals:
            conflict_score = max(conflict_score, 0.10)

        # === Category 3: Market context (max 0.20) ===
        market_score = 0.0
        if volatility_percentile is not None and (volatility_percentile > 80 or volatility_percentile < 20):
            market_score = max(market_score, 0.15)
        if market_regime and any(kw in market_regime for kw in ("high volatility", "squeeze", "expansion", "ranging")):
            market_score = max(market_score, 0.12)
        if market_breadth:
            pos_pct = market_breadth.get("positive_pct", 50)
            if pos_pct > 80 or pos_pct < 20:
                market_score = max(market_score, 0.12)
        if full_market_breadth:
            pos_pct = full_market_breadth.get("positive_pct", 50)
            if pos_pct > 80 or pos_pct < 20:
                market_score = max(market_score, 0.10)
        if sentiment_trend_magnitude is not None and sentiment_trend_magnitude > 0.2:
            market_score = max(market_score, 0.12)
        if volume_trend is not None and volume_trend > 3.0:
            market_score = max(market_score, 0.10)

        # === Category 4: Portfolio stress (max 0.15) ===
        portfolio_score = 0.0
        if portfolio_exposure_pct is not None and portfolio_exposure_pct > 70:
            portfolio_score = max(portfolio_score, 0.15)
        if portfolio_stop_risk_pct is not None and portfolio_stop_risk_pct > 8:
            portfolio_score = max(portfolio_score, 0.15)
        if drawdown_pct is not None and drawdown_pct > 10:
            portfolio_score = max(portfolio_score, 0.15)
        if unrealized_pnl is not None and unrealized_pnl < 0:
            portfolio_score = max(portfolio_score, 0.10)
        if consecutive_losses >= 3:
            portfolio_score = max(portfolio_score, 0.12)

        # === Category 5: Critical & events (max 0.15) ===
        critical_score = 0.0
        if is_critical:
            critical_score = max(critical_score, 0.15)
        if symbol_event is not None and symbol_event.get("has_event"):
            critical_score = max(critical_score, 0.10)
        if fundamentals is not None:
            pe = fundamentals.get("pe_ratio")
            if pe is not None and (pe > 50 or pe < 0):
                critical_score = max(critical_score, 0.08)
            margins = fundamentals.get("profit_margins")
            if margins is not None and margins < 0:
                critical_score = max(critical_score, 0.08)
        if trading_paused:
            critical_score = max(critical_score, 0.05)

        # === Category 6: Candidate count (max 0.03) ===
        candidate_score = 0.0
        if num_candidates > 20:
            candidate_score = 0.03
        elif num_candidates > 10:
            candidate_score = 0.02

        # === Category 7: Legacy fear_greed (max 0.02) ===
        legacy_score = 0.0
        if fear_greed:
            fg = fear_greed.get("value", 50)
            if fg <= 25 or fg >= 75:
                legacy_score = 0.02

        total = tech_score + conflict_score + market_score + portfolio_score + critical_score + candidate_score + legacy_score
        return min(1.0, total)

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

    async def _update_position_params(
        self,
        symbol: str,
        params: Dict[str, Any],
        indicator_config: Optional[Dict[str, Any]],
        timeframe: str,
        current_price: float,
        atr: Optional[float],
    ):
        """Update risk parameters of an open position from LLM strategy_params."""
        async with self._positions_lock:
            pos = self.positions.get(symbol)
        if not pos:
            return

        # --- Stop-loss (supports fixed pct and ATR multiple) ---
        stop_method = params.get("stop_loss_method", "fixed")
        if stop_method == "atr_multiple" and atr is not None and atr > 0:
            atr_mult = params.get("stop_loss_atr_multiple")
            if atr_mult is not None:
                sl_pct = (atr_mult * atr) / current_price
                pos["stop_loss"] = current_price * (1 - sl_pct)
        elif "stop_loss_pct" in params:
            sl_pct = params["stop_loss_pct"]
            try:
                sl_pct = float(sl_pct)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid stop_loss_pct=%r for %s",
                    params["stop_loss_pct"], symbol,
                )
                sl_pct = None
            if sl_pct is not None and sl_pct > 0 and current_price > 0:
                pos["stop_loss"] = current_price * (1 - sl_pct)
            elif sl_pct is not None:
                logger.warning(
                    "Ignoring invalid stop_loss_pct=%s for %s (current_price=%s)",
                    sl_pct, symbol, current_price,
                )

        # --- Take-profit (supports fixed pct and ATR multiple) ---
        if "take_profit_atr_multiple" in params and atr is not None and atr > 0 and current_price > 0:
            atr_mult = params["take_profit_atr_multiple"]
            if atr_mult is not None:
                tp_pct = (atr_mult * atr) / current_price
                pos["take_profit"] = current_price * (1 + tp_pct)
        elif "take_profit_pct" in params:
            tp_pct = params["take_profit_pct"]
            try:
                tp_pct = float(tp_pct)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid take_profit_pct=%r for %s",
                    params["take_profit_pct"], symbol,
                )
                tp_pct = None
            if tp_pct is not None and tp_pct > 0 and current_price > 0:
                pos["take_profit"] = current_price * (1 + tp_pct)
            elif tp_pct is not None:
                logger.warning(
                    "Ignoring invalid take_profit_pct=%s for %s (current_price=%s)",
                    tp_pct, symbol, current_price,
                )

        # --- Trailing stop ---
        if "trailing_stop" in params:
            pos["trailing_stop"] = params["trailing_stop"]
        if "trailing_stop_distance_pct" in params:
            pos["trailing_stop_distance_pct"] = params["trailing_stop_distance_pct"]
        if "trailing_stop_activation_pct" in params:
            pos["trailing_stop_activation_pct"] = params["trailing_stop_activation_pct"]

        # --- Trailing take-profit ---
        if "trailing_take_profit" in params:
            pos["trailing_take_profit"] = params["trailing_take_profit"]
        if "trailing_take_profit_distance_pct" in params:
            pos["trailing_take_profit_distance_pct"] = params["trailing_take_profit_distance_pct"]

        # --- Breakeven / lock-profit ---
        if "breakeven_activation_pct" in params:
            pos["breakeven_activation_pct"] = params["breakeven_activation_pct"]
        # --- Time-based exits ---
        if "max_hold_time_seconds" in params:
            pos["max_hold_time_seconds"] = params["max_hold_time_seconds"]
            # If the LLM explicitly sets a new hold time, clear any expiry flag
            pos.pop("_max_hold_expired", None)
            pos.pop("_max_hold_expired_count", None)

        # --- Cooldown after loss ---
        if "cooldown_after_loss_seconds" in params:
            pos["cooldown_after_loss_seconds"] = params["cooldown_after_loss_seconds"]

        # --- News sentiment exit ---
        if "news_sentiment_exit_threshold" in params:
            pos["news_sentiment_exit_threshold"] = params["news_sentiment_exit_threshold"]

        # --- Max unrealized loss ---
        if "max_unrealized_loss_pct" in params:
            pos["max_unrealized_loss_pct"] = params["max_unrealized_loss_pct"]

        # --- Partial take-profit levels ---
        if "partial_take_profit_levels" in params:
            pos["partial_take_profit_levels"] = params["partial_take_profit_levels"]
            pos["partial_tp_levels_triggered"] = []
            pos["partial_tp_depth_wait_start"] = {}
            # Clear single-level fields to avoid confusion
            pos["partial_take_profit_pct"] = None
            pos["partial_take_profit_fraction"] = None
            pos["partial_tp_triggered"] = None
        else:
            if "partial_take_profit_pct" in params:
                pos["partial_take_profit_pct"] = params["partial_take_profit_pct"]
            if "partial_take_profit_fraction" in params:
                pos["partial_take_profit_fraction"] = params["partial_take_profit_fraction"]
            if "partial_tp_triggered" not in pos:
                pos["partial_tp_triggered"] = False

        # --- Strategy interval ---
        if "strategy_interval_seconds" in params:
            self._strategy_intervals[symbol] = params["strategy_interval_seconds"]

        # --- Indicator config ---
        if indicator_config is not None:
            pos["indicator_config"] = indicator_config

        # --- Timeframe (if changed) ---
        if timeframe:
            pos["timeframe"] = timeframe

        logger.info(f"Updated risk parameters for {symbol} from LLM strategy_params")
        self._state_dirty = True

    def _compute_exit_order_prices(
        self,
        entry_price: float,
        signal: Signal,
        atr: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Return a dict with keys:
          - stop_loss_price: the trigger/limit price for the stop-loss order
          - take_profit_price: the limit price for the take-profit order
        Uses the LLM's exit order type fields; falls back to the standard
        stop_loss_pct / take_profit_pct if exit order types are not provided.
        """
        params = signal.strategy_params or {}
        stop_loss_pct = params.get("stop_loss_pct")
        take_profit_pct = params.get("take_profit_pct")

        # --- Stop-loss price ---
        sl_ot = signal.stop_loss_order_type
        if sl_ot == "stop":
            sl_price = signal.stop_loss_stop_price
            if sl_price is None and stop_loss_pct is not None:
                sl_price = entry_price * (1 - stop_loss_pct)
        elif sl_ot == "stop_limit":
            sl_price = signal.stop_loss_stop_price
            if sl_price is None and stop_loss_pct is not None:
                sl_price = entry_price * (1 - stop_loss_pct)
        elif sl_ot == "trailing_stop":
            sl_price = None  # not a fixed price
        else:
            sl_price = None

        # --- Take-profit price ---
        tp_ot = signal.take_profit_order_type
        if tp_ot == "limit":
            tp_price = signal.take_profit_limit_price
            if tp_price is None and take_profit_pct is not None:
                tp_price = entry_price * (1 + take_profit_pct)
        elif tp_ot == "market":
            tp_price = None  # will be handled by risk loop later
        else:
            tp_price = None

        return {
            "stop_loss_price": sl_price,
            "take_profit_price": tp_price,
        }

    async def _cancel_exit_orders(self, symbol: str):
        """Cancel any native stop-loss and take-profit orders for a symbol."""
        pos = self.positions.get(symbol)
        if not pos:
            return
        for order_id_key in ("stop_loss_order_id", "take_profit_order_id"):
            order_id = pos.pop(order_id_key, None)
            if order_id:
                try:
                    await asyncio.to_thread(self.trader.cancel_order, order_id)
                    logger.info(f"Cancelled exit order {order_id} for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to cancel exit order {order_id}: {e}")
                async with self._queued_orders_lock:
                    self.queued_orders = [
                        q for q in self.queued_orders
                        if q.get("order_id") != order_id
                    ]
        pos.pop("stop_loss_order_type", None)
        pos.pop("_native_stop_price", None)

    async def _place_exit_orders(
        self,
        symbol: str,
        signal: Signal,
        exit_prices: Dict[str, Optional[float]],
        timeframe: Optional[str] = None,
    ):
        """Place native stop-loss and take-profit orders for a position."""
        pos = self.positions.get(symbol)
        if not pos:
            return

        # --- Cancel any existing exit orders for this position ---
        old_sl_id = pos.get("stop_loss_order_id")
        old_tp_id = pos.get("take_profit_order_id")
        for old_id in (old_sl_id, old_tp_id):
            if old_id:
                try:
                    await asyncio.to_thread(self.trader.cancel_order, old_id)
                    logger.info(f"Cancelled old exit order {old_id} for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to cancel old exit order {old_id}: {e}")
                # Remove from queued_orders
                async with self._queued_orders_lock:
                    self.queued_orders = [
                        q for q in self.queued_orders
                        if q.get("order_id") != old_id
                    ]
        # Clear the stored IDs so they are not reused
        pos.pop("stop_loss_order_id", None)
        pos.pop("take_profit_order_id", None)
        pos.pop("stop_loss_order_type", None)
        pos.pop("_native_stop_price", None)

        base, quote = symbol.split("/")
        qty = pos["amount"]  # base quantity to sell
        if qty <= 0:
            return

        # --- Stop-loss order ---
        sl_ot = signal.stop_loss_order_type
        sl_price = exit_prices.get("stop_loss_price")
        sl_order_id = None
        actual_sl_ot = None
        if sl_ot in ("stop", "stop_limit") and sl_price is not None:
            actual_sl_ot = sl_ot
            try:
                if sl_ot == "stop" or (sl_ot == "stop_limit" and signal.stop_loss_limit_price is None):
                    # Fall back to a regular stop order when no explicit limit
                    # price is provided for stop_limit. Defaulting the limit to
                    # the stop price defeats the purpose of a stop-limit order.
                    if sl_ot == "stop_limit":
                        actual_sl_ot = "stop"
                    order = await asyncio.to_thread(
                        self.trader.create_stop_sell_order,
                        symbol, qty, sl_price,
                        time_in_force="gtc", timeout=60.0
                    )
                else:  # stop_limit with explicit limit price
                    limit_price = signal.stop_loss_limit_price
                    order = await asyncio.to_thread(
                        self.trader.create_stop_limit_sell_order,
                        symbol, qty, sl_price, limit_price,
                        time_in_force="gtc", timeout=60.0
                    )
                sl_order_id = order["id"]
                _sl_queued = {
                    "symbol": symbol,
                    "side": "sell",
                    "amount": qty,
                    "original_amount": qty,
                    "limit_price": order.get("limit_price"),
                    "stop_price": order.get("stop_price"),
                    "trail_offset": order.get("trail_offset"),
                    "order_type": actual_sl_ot,
                    "time_in_force": "gtc",
                    "signal": asdict(signal),
                    "timeframe": timeframe,
                    "atr": None,
                    "exit_reason": "stop_loss",
                    "order_id": sl_order_id,
                    "queued_at": time.time(),
                    "filled_qty": 0,
                    "filled_cost": 0.0,
                    "is_exit_order": True,
                    "oco_pair": None,
                }
                async with self._queued_orders_lock:
                    self.queued_orders.append(_sl_queued)
            except Exception as e:
                logger.error(f"Failed to place stop-loss order for {symbol}: {e}")

        elif sl_ot == "trailing_stop":
            trail_offset = signal.stop_loss_trail_offset
            if trail_offset is not None and trail_offset > 0:
                try:
                    order = await asyncio.to_thread(
                        self.trader.create_trailing_stop_sell_order,
                        symbol, qty, trail_offset,
                        time_in_force="gtc", timeout=60.0
                    )
                    sl_order_id = order["id"]
                    _trail_queued = {
                        "symbol": symbol,
                        "side": "sell",
                        "amount": qty,
                        "original_amount": qty,
                        "limit_price": None,
                        "stop_price": None,
                        "trail_offset": trail_offset,
                        "order_type": "trailing_stop",
                        "time_in_force": "gtc",
                        "signal": asdict(signal),
                        "timeframe": timeframe,
                        "atr": None,
                        "exit_reason": "stop_loss",
                        "order_id": sl_order_id,
                        "queued_at": time.time(),
                        "filled_qty": 0,
                        "filled_cost": 0.0,
                        "is_exit_order": True,
                        "oco_pair": None,
                    }
                    async with self._queued_orders_lock:
                        self.queued_orders.append(_trail_queued)
                except Exception as e:
                    logger.error(f"Failed to place trailing-stop order for {symbol}: {e}")

        # --- Take-profit order ---
        # If trailing_take_profit or partial take-profit is enabled, do not place a
        # native limit order because the take-profit price will move or only a
        # fraction of the position should be sold. The risk management loop will
        # handle these cases instead.
        params = signal.strategy_params or {}
        trailing_tp = params.get("trailing_take_profit", False)
        partial_tp_levels = params.get("partial_take_profit_levels")
        partial_tp_pct = params.get("partial_take_profit_pct")

        tp_ot = signal.take_profit_order_type
        tp_price = exit_prices.get("take_profit_price")
        tp_order_id = None
        if (tp_ot == "limit" and tp_price is not None
                and not trailing_tp
                and not partial_tp_levels
                and not partial_tp_pct):
            try:
                order = await asyncio.to_thread(
                    self.trader.create_market_sell_order,
                    symbol, qty, 60.0,
                    limit_price=tp_price, time_in_force="gtc"
                )
                tp_order_id = order["id"]
                _tp_queued = {
                    "symbol": symbol,
                    "side": "sell",
                    "amount": qty,
                    "original_amount": qty,
                    "limit_price": tp_price,
                    "stop_price": None,
                    "trail_offset": None,
                    "order_type": "limit",
                    "time_in_force": "gtc",
                    "signal": asdict(signal),
                    "timeframe": timeframe,
                    "atr": None,
                    "exit_reason": "take_profit",
                    "order_id": tp_order_id,
                    "queued_at": time.time(),
                    "filled_qty": 0,
                    "filled_cost": 0.0,
                    "is_exit_order": True,
                    "oco_pair": None,
                }
                async with self._queued_orders_lock:
                    self.queued_orders.append(_tp_queued)
            except Exception as e:
                logger.error(f"Failed to place take-profit order for {symbol}: {e}")

        # --- Link OCO pair ---
        if sl_order_id and tp_order_id:
            for q in self.queued_orders:
                if q.get("order_id") == sl_order_id:
                    q["oco_pair"] = tp_order_id
                elif q.get("order_id") == tp_order_id:
                    q["oco_pair"] = sl_order_id

        # Store order IDs in position for risk management
        pos["stop_loss_order_id"] = sl_order_id
        # Store order type for risk management decisions
        if actual_sl_ot:
            pos["stop_loss_order_type"] = actual_sl_ot
        pos["take_profit_order_id"] = tp_order_id

        # Notify user
        if self.notifier:
            stock_name = await self._get_stock_name(symbol)
            display_symbol = self._format_symbol_display(symbol, stock_name, pos.get("timeframe"))
            msg = f"🛡️ Exit orders placed for {display_symbol}:\n"
            if sl_order_id:
                sl_type = actual_sl_ot or "stop"
                if actual_sl_ot == "trailing_stop":
                    msg += f"  🛑 Trailing stop: offset ${trail_offset:.2f}\n"
                elif actual_sl_ot == "stop_limit":
                    msg += f"  🛑 Stop-limit: stop ${sl_price:.2f}, limit ${signal.stop_loss_limit_price:.2f}\n"
                else:
                    msg += f"  🛑 Stop: ${sl_price:.2f}\n"
            if tp_order_id:
                msg += f"  🎯 Take-profit: limit ${tp_price:.2f}\n"
            if sl_order_id and tp_order_id:
                msg += "  (OCO – one cancels the other)"
            await self.notifier.send_notification(
                msg,
                summary={
                    "symbol": symbol,
                    "action": "INFO",
                    "reason": "Exit orders placed",
                    "stop_loss_order_id": sl_order_id,
                    "take_profit_order_id": tp_order_id,
                }
            )

    async def _replace_native_stop_order(
        self,
        symbol: str,
        pos: Dict[str, Any],
        old_stop_price: float,
        new_stop_price: float,
    ):
        """Cancel the existing native stop order and place a new one with the updated stop price."""
        old_order_id = pos.get("stop_loss_order_id")
        if not old_order_id:
            return

        # Cancel the old order
        try:
            await asyncio.to_thread(self.trader.cancel_order, old_order_id)
            logger.info(f"Cancelled old stop order {old_order_id} for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to cancel old stop order {old_order_id}: {e}")
            # Continue anyway – the old order may still be open, but we'll place a new one.
            # The OCO logic will eventually cancel the old one if the new one fills.

        # Capture the old queued entry's limit price BEFORE removing it
        async with self._queued_orders_lock:
            old_queued = next(
                (q for q in self.queued_orders if q.get("order_id") == old_order_id),
                None
            )
            old_limit_price = old_queued.get("limit_price") if old_queued else None

            # Remove the old queued entry
            self.queued_orders = [
                q for q in self.queued_orders
                if q.get("order_id") != old_order_id
            ]

        # Place a new stop order
        qty = pos["amount"]
        sl_ot = pos.get("stop_loss_order_type", "stop")
        new_order_id = None
        try:
            if sl_ot == "stop":
                order = await asyncio.to_thread(
                    self.trader.create_stop_sell_order,
                    symbol, qty, new_stop_price,
                    time_in_force="gtc", timeout=60.0
                )
            else:  # stop_limit
                # For stop_limit, use the original limit price from the old
                # queued entry, or fall back to the new stop price.
                limit_price = old_limit_price if old_limit_price is not None else new_stop_price
                order = await asyncio.to_thread(
                    self.trader.create_stop_limit_sell_order,
                    symbol, qty, new_stop_price, limit_price,
                    time_in_force="gtc", timeout=60.0
                )
            new_order_id = order["id"]
            _replace_queued = {
                "symbol": symbol,
                "side": "sell",
                "amount": qty,
                "original_amount": qty,
                "limit_price": order.get("limit_price"),
                "stop_price": order.get("stop_price"),
                "trail_offset": order.get("trail_offset"),
                "order_type": sl_ot,
                "time_in_force": "gtc",
                "signal": {},  # no original signal for replacement
                "timeframe": pos.get("timeframe"),
                "atr": None,
                "exit_reason": "stop_loss",
                "order_id": new_order_id,
                "queued_at": time.time(),
                "filled_qty": 0,
                "filled_cost": 0.0,
                "is_exit_order": True,
                "oco_pair": pos.get("take_profit_order_id"),  # maintain OCO link
            }
            async with self._queued_orders_lock:
                self.queued_orders.append(_replace_queued)
                # Update OCO link on the take-profit order if it exists
                tp_order_id = pos.get("take_profit_order_id")
                if tp_order_id:
                    for q in self.queued_orders:
                        if q.get("order_id") == tp_order_id:
                            q["oco_pair"] = new_order_id
                            break
            # Update position
            pos["stop_loss_order_id"] = new_order_id
            logger.info(f"Placed new stop order {new_order_id} for {symbol} at {new_stop_price:.4f}")

            # Notify user
            if self.notifier:
                stock_name = await self._get_stock_name(symbol)
                display_symbol = self._format_symbol_display(symbol, stock_name, pos.get("timeframe"))
                msg = f"🔄 Stop order updated for {display_symbol}: {old_stop_price:.4f} → {new_stop_price:.4f}"
                await self.notifier.send_notification(
                    msg,
                    summary={
                        "symbol": symbol,
                        "action": "INFO",
                        "reason": "Stop order replaced",
                        "old_stop_price": old_stop_price,
                        "new_stop_price": new_stop_price,
                    }
                )
        except Exception as e:
            logger.error(f"Failed to place replacement stop order for {symbol}: {e}")

    async def _execute_partial_tp_single(
        self,
        symbol: str,
        current_price: float,
        atr: Optional[float],
        ticker: Dict[str, Any],
    ) -> None:
        """Execute a single partial take-profit sell for a position."""
        pos = self.positions.get(symbol)
        if not pos:
            logger.warning(f"Cannot execute partial TP for {symbol}: no position.")
            return

        stock_name = await self._get_stock_name(symbol)
        tf = pos.get("timeframe") if pos else None
        display_symbol = self._format_symbol_display(symbol, stock_name, tf)

        fraction = pos.get("partial_take_profit_fraction")
        if fraction is None or fraction <= 0 or fraction >= 1:
            logger.warning(f"Invalid partial_take_profit_fraction for {symbol}: {fraction}")
            return

        sell_amount = pos["amount"] * fraction
        base, quote = symbol.split("/")

        # Check minimum sell size
        # Fetch minimum order size from asset info
        try:
            asset = await self._get_asset_info(symbol)
            min_amount = float(asset.min_order_size) if asset.min_order_size else None
            if not asset.fractionable and (min_amount is None or min_amount < 1.0):
                min_amount = 1.0
        except Exception:
            min_amount = None
        if min_amount is not None and sell_amount < float(min_amount):
            logger.info(f"Partial TP sell amount {sell_amount:.6f} below min {min_amount} for {symbol}, skipping.")
            return

        if not await self._is_market_open():
            logger.info(f"Partial TP (single) for {symbol} skipped: market closed.")
            return

        need_limit = not self._is_regular_hours()
        limit_price = None
        time_in_force = "day"
        if need_limit:
            limit_price = self._default_limit_price(symbol, "SELL", ticker)
            if limit_price is None:
                logger.error(f"Cannot place limit order for partial TP on {symbol}: no limit price.")
                return

        if limit_price is not None and limit_price <= 0:
            logger.error(f"Invalid limit_price for partial TP on {symbol}, skipping.")
            return

        try:
            order = await asyncio.to_thread(
                self.trader.create_market_sell_order, symbol, sell_amount,
                settings.ORDER_FILL_TIMEOUT_SECONDS, limit_price, time_in_force
            )
            logger.info(f"Partial TP SELL {symbol}: {sell_amount:.6f} @ {order.get('price', current_price):.4f}")

            # Use actual filled amount from the order
            filled_amount = order.get("amount", sell_amount)

            # Compute fee
            fee = order.get("fee", {})
            fee_cost = float(fee.get("cost", 0.0) or 0.0)
            fee_currency = fee.get("currency", "")
            # Prorated cost basis for the sold portion
            cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
            net_base = pos.get("net_base", pos["amount"])
            prorated_cost_basis = cost_basis * (filled_amount / net_base) if net_base > 0 else 0.0

            net_quote = order["cost"] - (fee_cost if fee_currency == quote else 0.0)
            realized_pnl = net_quote - prorated_cost_basis

            order["realized_pnl"] = realized_pnl
            order["cost_basis"] = prorated_cost_basis
            order["exit_reason"] = "partial_take_profit"
            order["strategy_type"] = pos.get("strategy_type", "unknown")
            order["timeframe"] = pos.get("timeframe")
            if "timestamp" in pos:
                order["hold_time_seconds"] = (order["timestamp"] - pos["timestamp"]) / 1000.0

            # Update position: reduce amount, cost_basis, net_base
            remaining_amount = pos["amount"] - filled_amount
            remaining_cost_basis = cost_basis - prorated_cost_basis
            remaining_net_base = net_base - filled_amount

            # Cancel old exit orders because quantity changed
            await self._cancel_exit_orders(symbol)

            if remaining_amount <= 0 or remaining_net_base <= 0:
                # Position fully closed (shouldn't normally happen with partial, but handle gracefully)
                async with self._positions_lock:
                    self.positions.pop(symbol, None)
                self._strategy_intervals.pop(symbol, None)
                self._last_strategy_eval.pop(symbol, None)
                self._pending_entries.pop(symbol, None)
                await self._remove_symbol_if_paused(symbol)
            else:
                async with self._positions_lock:
                    self.positions[symbol]["amount"] = remaining_amount
                    self.positions[symbol]["cost_basis"] = remaining_cost_basis
                    self.positions[symbol]["net_base"] = remaining_net_base
                    self.positions[symbol]["price"] = remaining_cost_basis / remaining_net_base if remaining_net_base > 0 else 0.0
                    # Clear single partial TP flags
                    self.positions[symbol].pop("partial_tp_triggered", None)
                    self.positions[symbol].pop("_partial_tp_triggered_single", None)
                    self.positions[symbol].pop("_partial_tp_single_review_count", None)

                # Check if remaining amount is dust
                is_dust = False
                if min_amount is not None and remaining_amount < float(min_amount):
                    is_dust = True
                if is_dust:
                    logger.info(f"Remaining {remaining_amount:.6f} {base} is dust after partial TP for {symbol}, sweeping.")
                    await self._sweep_dust(symbol)
                else:
                    # Replace exit orders for the remaining amount
                    from src.strategies.base import Signal
                    dummy_params = {
                        "trailing_take_profit": self.positions[symbol].get("trailing_take_profit", False),
                        "partial_take_profit_levels": self.positions[symbol].get("partial_take_profit_levels"),
                        "partial_take_profit_pct": self.positions[symbol].get("partial_take_profit_pct"),
                    }
                    dummy_signal = Signal(
                        action="BUY",
                        confidence=1.0,
                        reasoning="Replacing exit orders after partial TP",
                        stop_loss_order_type=self.positions[symbol].get("stop_loss_order_type"),
                        stop_loss_stop_price=self.positions[symbol].get("stop_loss"),
                        stop_loss_limit_price=None,
                        take_profit_order_type=self.positions[symbol].get("take_profit_order_type"),
                        take_profit_limit_price=self.positions[symbol].get("take_profit"),
                        strategy_params=dummy_params,
                    )
                    exit_prices = {
                        "stop_loss_price": self.positions[symbol].get("stop_loss"),
                        "take_profit_price": self.positions[symbol].get("take_profit"),
                    }
                    await self._place_exit_orders(symbol, dummy_signal, exit_prices, self.positions[symbol].get("timeframe"))

            self._append_trade(order)
            await asyncio.to_thread(insert_trade, order)
            await self._save_state(force=True)

            if self.notifier:
                pnl_pct = (realized_pnl / prorated_cost_basis * 100) if prorated_cost_basis > 0 else 0.0
                await self.notifier.send_notification(
                    f"🔸 Partial TP SELL {display_symbol}: {filled_amount:.6f} @ {order.get('price', current_price):.4f} "
                    f"| P&L: {realized_pnl:+.4f} ({pnl_pct:+.2f}%)",
                    summary={
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": "Partial take-profit",
                        "amount": filled_amount,
                        "price": order.get("price", current_price),
                        "realized_pnl": realized_pnl,
                        "exit_reason": "partial_take_profit",
                    }
                )
        except Exception as e:
            logger.error(f"Partial TP sell failed for {symbol}: {e}")
            if self.notifier:
                await self.notifier.send_notification(
                    f"❌ Partial TP sell failed for {display_symbol}: {e}",
                    summary={"symbol": symbol, "action": "ERROR", "reason": f"Partial TP sell failed: {e}"[:200]}
                )

    async def _execute_partial_tp_level(
        self,
        symbol: str,
        level_index: int,
        current_price: float,
        atr: Optional[float],
        ticker: Dict[str, Any],
    ) -> None:
        """Execute a partial take-profit sell for a specific level."""
        pos = self.positions.get(symbol)
        if not pos:
            logger.warning(f"Cannot execute partial TP level for {symbol}: no position.")
            return

        stock_name = await self._get_stock_name(symbol)
        tf = pos.get("timeframe") if pos else None
        display_symbol = self._format_symbol_display(symbol, stock_name, tf)

        levels = pos.get("partial_take_profit_levels")
        if not levels or level_index >= len(levels):
            logger.warning(f"Invalid partial TP level index {level_index} for {symbol}")
            return

        level = levels[level_index]
        fraction = level.get("fraction")
        if fraction is None or fraction <= 0 or fraction >= 1:
            logger.warning(f"Invalid fraction for partial TP level {level_index} of {symbol}: {fraction}")
            return

        sell_amount = pos["amount"] * fraction
        base, quote = symbol.split("/")

        # Check minimum sell size
        # Fetch minimum order size from asset info
        try:
            asset = await self._get_asset_info(symbol)
            min_amount = float(asset.min_order_size) if asset.min_order_size else None
            if not asset.fractionable and (min_amount is None or min_amount < 1.0):
                min_amount = 1.0
        except Exception:
            min_amount = None
        if min_amount is not None and sell_amount < float(min_amount):
            logger.info(f"Partial TP level {level_index} sell amount {sell_amount:.6f} below min for {symbol}, skipping.")
            return

        if not await self._is_market_open():
            logger.info(f"Partial TP level {level_index} for {symbol} skipped: market closed.")
            return

        need_limit = not self._is_regular_hours()
        limit_price = None
        time_in_force = "day"
        if need_limit:
            limit_price = self._default_limit_price(symbol, "SELL", ticker)
            if limit_price is None:
                logger.error(f"Cannot place limit order for partial TP level on {symbol}: no limit price.")
                return

        if limit_price is not None and limit_price <= 0:
            logger.error(f"Invalid limit_price for partial TP level on {symbol}, skipping.")
            return

        try:
            order = await asyncio.to_thread(
                self.trader.create_market_sell_order, symbol, sell_amount,
                settings.ORDER_FILL_TIMEOUT_SECONDS, limit_price, time_in_force
            )
            logger.info(f"Partial TP level {level_index} SELL {symbol}: {sell_amount:.6f} @ {order.get('price', current_price):.4f}")

            # Use actual filled amount from the order
            filled_amount = order.get("amount", sell_amount)

            # Compute fee
            fee = order.get("fee", {})
            fee_cost = float(fee.get("cost", 0.0) or 0.0)
            fee_currency = fee.get("currency", "")
            # Prorated cost basis for the sold portion
            cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
            net_base = pos.get("net_base", pos["amount"])
            prorated_cost_basis = cost_basis * (filled_amount / net_base) if net_base > 0 else 0.0

            net_quote = order["cost"] - (fee_cost if fee_currency == quote else 0.0)
            realized_pnl = net_quote - prorated_cost_basis

            order["realized_pnl"] = realized_pnl
            order["cost_basis"] = prorated_cost_basis
            order["exit_reason"] = f"partial_take_profit_level_{level_index}"
            order["strategy_type"] = pos.get("strategy_type", "unknown")
            order["timeframe"] = pos.get("timeframe")
            if "timestamp" in pos:
                order["hold_time_seconds"] = (order["timestamp"] - pos["timestamp"]) / 1000.0

            # Mark this level as triggered
            if symbol in self.positions:
                triggered = self.positions[symbol].get("partial_tp_levels_triggered", [])
                if level_index not in triggered:
                    triggered.append(level_index)
                    self.positions[symbol]["partial_tp_levels_triggered"] = triggered
                # Clear depth wait state for this level
                if "partial_tp_depth_wait_start" in self.positions[symbol]:
                    self.positions[symbol]["partial_tp_depth_wait_start"].pop(level_index, None)

            # Update position: reduce amount, cost_basis, net_base
            remaining_amount = pos["amount"] - filled_amount
            remaining_cost_basis = cost_basis - prorated_cost_basis
            remaining_net_base = net_base - filled_amount

            # Cancel old exit orders because quantity changed
            await self._cancel_exit_orders(symbol)

            if remaining_amount <= 0 or remaining_net_base <= 0:
                # Position fully closed
                self.positions.pop(symbol, None)
                self._strategy_intervals.pop(symbol, None)
                self._last_strategy_eval.pop(symbol, None)
                self._pending_entries.pop(symbol, None)
                await self._remove_symbol_if_paused(symbol)
            else:
                async with self._positions_lock:
                    self.positions[symbol]["amount"] = remaining_amount
                    self.positions[symbol]["cost_basis"] = remaining_cost_basis
                    self.positions[symbol]["net_base"] = remaining_net_base
                    self.positions[symbol]["price"] = remaining_cost_basis / remaining_net_base if remaining_net_base > 0 else 0.0
                    # Clear partial TP review flags for this level
                    self.positions[symbol].pop("_partial_tp_triggered", None)
                    self.positions[symbol].pop("_partial_tp_review_count", None)
                    triggered_levels = self.positions[symbol].get("_partial_tp_triggered_levels", [])
                    self.positions[symbol]["_partial_tp_triggered_levels"] = [
                        x for x in triggered_levels if x != level_index
                    ]

                # Check if remaining amount is dust
                is_dust = False
                if min_amount is not None and remaining_amount < float(min_amount):
                    is_dust = True
                if is_dust:
                    logger.info(f"Remaining {remaining_amount:.6f} {base} is dust after partial TP for {symbol}, sweeping.")
                    await self._sweep_dust(symbol)
                else:
                    # Replace exit orders for the remaining amount
                    from src.strategies.base import Signal
                    dummy_params = {
                        "trailing_take_profit": self.positions[symbol].get("trailing_take_profit", False),
                        "partial_take_profit_levels": self.positions[symbol].get("partial_take_profit_levels"),
                        "partial_take_profit_pct": self.positions[symbol].get("partial_take_profit_pct"),
                    }
                    dummy_signal = Signal(
                        action="BUY",
                        confidence=1.0,
                        reasoning="Replacing exit orders after partial TP",
                        stop_loss_order_type=self.positions[symbol].get("stop_loss_order_type"),
                        stop_loss_stop_price=self.positions[symbol].get("stop_loss"),
                        stop_loss_limit_price=None,
                        take_profit_order_type=self.positions[symbol].get("take_profit_order_type"),
                        take_profit_limit_price=self.positions[symbol].get("take_profit"),
                        strategy_params=dummy_params,
                    )
                    exit_prices = {
                        "stop_loss_price": self.positions[symbol].get("stop_loss"),
                        "take_profit_price": self.positions[symbol].get("take_profit"),
                    }
                    await self._place_exit_orders(symbol, dummy_signal, exit_prices, self.positions[symbol].get("timeframe"))

            self._append_trade(order)
            await asyncio.to_thread(insert_trade, order)
            await self._save_state(force=True)

            if self.notifier:
                pnl_pct = (realized_pnl / prorated_cost_basis * 100) if prorated_cost_basis > 0 else 0.0
                await self.notifier.send_notification(
                    f"🔸 Partial TP level {level_index} SELL {display_symbol}: {filled_amount:.6f} @ {order.get('price', current_price):.4f} "
                    f"| P&L: {realized_pnl:+.4f} ({pnl_pct:+.2f}%)",
                    summary={
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": f"Partial take-profit level {level_index}",
                        "amount": filled_amount,
                        "price": order.get("price", current_price),
                        "realized_pnl": realized_pnl,
                        "exit_reason": f"partial_take_profit_level_{level_index}",
                        "level_index": level_index,
                    }
                )
        except Exception as e:
            logger.error(f"Partial TP level {level_index} sell failed for {symbol}: {e}")
            if self.notifier:
                await self.notifier.send_notification(
                    f"❌ Partial TP level {level_index} sell failed for {display_symbol}: {e}",
                    summary={"symbol": symbol, "action": "ERROR", "reason": f"Partial TP level sell failed: {e}"[:200]}
                )

    async def _sweep_dust(self, symbol: str):
        """Sell any remaining dust balance of a symbol after a partial sell."""
        base = symbol.split("/")[0]
        try:
            balance = await asyncio.to_thread(self.trader.get_balance, base)
        except Exception as e:
            logger.warning(f"Dust sweep: could not fetch balance for {base}: {e}")
            return
        if balance <= 0:
            return

        stock_name = await self._get_stock_name(symbol)
        tf = self.positions.get(symbol, {}).get("timeframe") if symbol in self.positions else None
        display_symbol = self._format_symbol_display(symbol, stock_name, tf)

        try:
            base = symbol.split("/")[0]
            quotes = await self._get_quotes_async([base], timeout=45.0)
            ticker = quotes.get(base)
            price = ticker["last"]
        except Exception as e:
            logger.warning(f"Dust sweep: could not fetch price for {symbol}: {e}")
            return

        # Fetch minimum order size from asset info
        try:
            asset = await self._get_asset_info(symbol)
            min_amount = float(asset.min_order_size) if asset.min_order_size else None
            if not asset.fractionable and (min_amount is None or min_amount < 1.0):
                min_amount = 1.0
        except Exception:
            min_amount = None
        if min_amount is not None and balance < float(min_amount):
            logger.info(f"Dust sweep: {balance} {base} below min amount {min_amount}, cannot sell.")
            return

        if not await self._is_market_open():
            logger.info(f"Dust sweep for {symbol} skipped: market closed.")
            return

        need_limit = not self._is_regular_hours()
        limit_price = None
        time_in_force = "day"
        if need_limit:
            limit_price = self._default_limit_price(symbol, "SELL", ticker)
            if limit_price is None:
                logger.error(f"Cannot place limit order for dust sweep on {symbol}: no limit price.")
                return

        if limit_price is not None and limit_price <= 0:
            logger.error(f"Invalid limit_price for dust sweep on {symbol}, skipping.")
            return

        try:
            order = await asyncio.to_thread(
                self.trader.create_market_sell_order, symbol, balance,
                settings.ORDER_FILL_TIMEOUT_SECONDS, limit_price, time_in_force
            )
            logger.info(f"Dust sweep: sold {balance} {base} from {symbol} – order {order.get('id')}")

            # Record the dust sale in trade history for consistency
            fee = order.get('fee', {})
            fee_cost = float(fee.get('cost', 0.0) or 0.0)
            fee_currency = fee.get('currency', '')
            pos = self.positions.get(symbol)
            if pos:
                cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                net_quote = order['cost'] - (fee_cost if fee_currency == symbol.split('/')[1] else 0.0)
                realized_pnl = net_quote - cost_basis
                order["realized_pnl"] = realized_pnl
                order["cost_basis"] = cost_basis
                order["exit_reason"] = "dust_sweep"
                order["strategy_type"] = pos.get("strategy_type", "unknown")
                order["timeframe"] = pos.get("timeframe")
                if "timestamp" in pos:
                    order["hold_time_seconds"] = (order["timestamp"] - pos["timestamp"]) / 1000.0
                self._append_trade(order)
                await asyncio.to_thread(insert_trade, order)
                await self._save_state(force=True)

            # Cancel any remaining exit orders before removing the position
            await self._cancel_exit_orders(symbol)

            # Remove the now-empty position
            async with self._positions_lock:
                self.positions.pop(symbol, None)
            self._strategy_intervals.pop(symbol, None)
            self._last_strategy_eval.pop(symbol, None)
            await self._remove_symbol_if_paused(symbol)

            if self.notifier:
                await self.notifier.send_notification(
                    f"🧹 Dust sweep: sold remaining {balance} {base} from {display_symbol}",
                    summary={
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": "Dust sweep",
                        "amount": balance,
                        "exit_reason": "dust_sweep",
                    }
                )
        except Exception as e:
            logger.error(f"Dust sweep failed for {symbol}: {e}")

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
                    order_id = queued.get('order_id')
                    if not order_id:
                        # Old queued entries without order_id – remove them safely
                        logger.warning(f"Queued order for {queued['symbol']} missing order_id, removing.")
                        self.queued_orders.remove(queued)
                        continue

                    # --- Timeout check: cancel stale queued limit orders ---
                    # Exit orders (stop-loss, take-profit, trailing-stop) are exempt
                    # from the timeout because they must remain active until triggered
                    # or explicitly cancelled when the position is closed.
                    queued_at = queued.get('queued_at', 0)
                    if not queued.get("is_exit_order"):
                        # Scale timeout based on the assigned timeframe.
                        # Long-term timeframes may have limit orders that need
                        # days or weeks to fill — a fixed 15-minute timeout would
                        # prematurely cancel them.
                        queued_tf = queued.get('timeframe')
                        base_timeout = settings.QUEUED_ORDER_TIMEOUT_SECONDS
                        if queued_tf:
                            tf_secs = self._timeframe_to_seconds(queued_tf)
                            # Use 50% of the timeframe as the timeout, with a
                            # minimum of the base timeout and a maximum of
                            # 180 days (same cap as entry conditions).
                            scaled_timeout = min(max(base_timeout, int(tf_secs * 0.5)), 15_552_000)
                        else:
                            scaled_timeout = base_timeout
                        if time.time() - queued_at > scaled_timeout:
                            logger.warning(
                                f"Queued order {order_id} for {queued['symbol']} timed out "
                                f"after {scaled_timeout}s. Cancelling."
                            )
                            try:
                                await asyncio.to_thread(self.trader.cancel_order, order_id)
                            except Exception as e:
                                logger.error(f"Failed to cancel timed-out order {order_id}: {e}")
                            # Remove from queue regardless of cancel success
                            self.queued_orders.remove(queued)
                            self._state_dirty = True
                            # If this was an exit order, cancel its OCO pair
                            if queued.get("is_exit_order"):
                                oco_pair_id = queued.get("oco_pair")
                                if oco_pair_id:
                                    try:
                                        await asyncio.to_thread(self.trader.cancel_order, oco_pair_id)
                                        logger.info(f"Cancelled OCO pair {oco_pair_id} for timed-out exit order {order_id}")
                                    except Exception as e:
                                        logger.warning(f"Failed to cancel OCO order {oco_pair_id}: {e}")
                                    async with self._queued_orders_lock:
                                        self.queued_orders = [
                                            q for q in self.queued_orders
                                            if q.get("order_id") != oco_pair_id
                                        ]
                                # Clear exit order IDs from position
                                pos = self.positions.get(queued["symbol"])
                                if pos:
                                    pos.pop("stop_loss_order_id", None)
                                    pos.pop("take_profit_order_id", None)
                                if self.notifier:
                                    stock_name = await self._get_stock_name(queued["symbol"])
                                    display_symbol = self._format_symbol_display(queued["symbol"], stock_name, queued.get("timeframe"))
                                    await self.notifier.send_notification(
                                        f"🔗 OCO pair {oco_pair_id} cancelled for {display_symbol} (main order timed out).",
                                        summary={"symbol": queued["symbol"], "action": "CANCEL", "reason": "OCO pair cancelled due to timeout"}
                                    )
                            if self.notifier:
                                stock_name = await self._get_stock_name(queued['symbol'])
                                tf = queued.get('timeframe')
                                display = self._format_symbol_display(queued['symbol'], stock_name, tf)
                                await self.notifier.send_notification(
                                    f"⏰ Queued {queued['side']} order for {display} timed out and was cancelled.",
                                    summary={
                                        "symbol": queued['symbol'],
                                        "action": "CANCEL",
                                        "reason": "Queued order timeout",
                                    }
                                )
                            continue

                    paper_order = await asyncio.to_thread(self.trader.get_order, order_id)
                    if paper_order is None:
                        logger.warning(f"Order {order_id} not found for {queued['symbol']}, removing from queue.")
                        async with self._queued_orders_lock:
                            self.queued_orders.remove(queued)
                        self._state_dirty = True
                        continue

                    status = paper_order.status
                    if isinstance(status, str):
                        status = status.lower()

                    # --- For stop/stop_limit exit orders, cancel OCO pair as soon as stop price is reached ---
                    if (queued.get("is_exit_order")
                            and queued.get("order_type") in ("stop", "stop_limit")
                            and queued.get("side") == "sell"
                            and queued.get("oco_pair") is not None):
                        stop_price = queued.get("stop_price")
                        if stop_price is not None:
                            # Fetch current price
                            try:
                                base = queued["symbol"].split("/")[0]
                                quotes = await self._get_quotes_async([base], timeout=45.0)
                                ticker = quotes.get(base)
                            except Exception:
                                pass
                            if ticker and ticker.get("last") is not None:
                                current_price = ticker["last"]
                                if current_price <= stop_price:
                                    # Stop triggered – cancel OCO pair immediately
                                    oco_pair_id = queued["oco_pair"]
                                    try:
                                        await asyncio.to_thread(self.trader.cancel_order, oco_pair_id)
                                        logger.info(
                                            f"Stop triggered for {queued['symbol']} at {current_price:.4f}, "
                                            f"cancelled OCO pair {oco_pair_id}"
                                        )
                                    except Exception as e:
                                        logger.warning(f"Failed to cancel OCO order {oco_pair_id}: {e}")
                                    # Remove the cancelled take-profit from queued_orders
                                    self.queued_orders = [
                                        q for q in self.queued_orders
                                        if q.get("order_id") != oco_pair_id
                                    ]
                                    # Clear OCO reference so we don't try again
                                    queued["oco_pair"] = None
                                    # Clear take-profit order ID from position
                                    pos = self.positions.get(queued["symbol"])
                                    if pos:
                                        pos.pop("take_profit_order_id", None)
                                    # Notify user
                                    if self.notifier:
                                        stock_name = await self._get_stock_name(queued["symbol"])
                                        display_symbol = self._format_symbol_display(
                                            queued["symbol"], stock_name, queued.get("timeframe")
                                        )
                                        await self.notifier.send_notification(
                                            f"🛑 Stop triggered for {display_symbol} at {current_price:.4f}, "
                                            f"take‑profit order cancelled.",
                                            summary={
                                                "symbol": queued["symbol"],
                                                "action": "CANCEL",
                                                "reason": "Stop triggered, OCO pair cancelled",
                                            }
                                        )
                                    self._state_dirty = True

                    # Determine how much has been filled since the last check
                    filled_qty = float(paper_order.filled_qty) if paper_order.filled_qty else 0.0
                    filled_avg_price = float(paper_order.filled_avg_price) if paper_order.filled_avg_price else 0.0
                    last_filled_qty = queued.get('filled_qty', 0.0)
                    delta_qty = filled_qty - last_filled_qty

                    if delta_qty > 0:
                        # A new fill occurred (partial or final)
                        delta_cost = delta_qty * filled_avg_price
                        # Build a trade dict for this delta
                        # Recompute the actual fee for this fill (PaperTrader already
                        # deducted it from the balance, but does not store it in the order)
                        from src.exchanges.fees import calculate_transaction_costs
                        _quote_ccy = queued['symbol'].split("/")[1] if "/" in queued['symbol'] else self.base_currency
                        _fee_costs = calculate_transaction_costs(
                            queued['side'].upper(), filled_avg_price, delta_qty, symbol=queued['symbol']
                        )
                        trade_dict = {
                            'id': str(paper_order.id),
                            'symbol': queued['symbol'],
                            'side': queued['side'],
                            'amount': delta_qty,
                            'price': filled_avg_price,
                            'cost': delta_cost,
                            'fee': {'cost': _fee_costs["total_costs"], 'currency': _quote_ccy},
                            'status': 'closed',
                            'timestamp': int(time.time() * 1000),
                        }
                        # Update tracking fields
                        queued['filled_qty'] = filled_qty
                        queued['filled_cost'] = queued.get('filled_cost', 0.0) + delta_cost

                        if queued['side'] == 'buy':
                            # Update remaining quote amount
                            original_amount = queued.get('original_amount', queued['amount'])
                            queued['amount'] = original_amount - queued['filled_cost']
                            await self._handle_queued_buy_fill(trade_dict, queued)
                        else:
                            # Update remaining base amount
                            original_amount = queued.get('original_amount', queued['amount'])
                            queued['amount'] = original_amount - filled_qty
                            await self._handle_queued_sell_fill(trade_dict, queued, partial=True)

                        # --- OCO handling for exit orders ---
                        if queued.get("is_exit_order"):
                            oco_pair_id = queued.get("oco_pair")
                            if oco_pair_id:
                                try:
                                    await asyncio.to_thread(self.trader.cancel_order, oco_pair_id)
                                    logger.info(f"Cancelled OCO pair {oco_pair_id} for {queued['symbol']}")
                                except Exception as e:
                                    logger.warning(f"Failed to cancel OCO order {oco_pair_id}: {e}")
                                async with self._queued_orders_lock:
                                    self.queued_orders = [
                                        q for q in self.queued_orders
                                        if q.get("order_id") != oco_pair_id
                                    ]
                            
                            # If the fill was partial, cancel the remaining part of this exit order
                            # to avoid leaving a dangling order that is no longer linked to the position.
                            # The risk management loop will handle the remaining position.
                            if filled_qty < queued.get('original_amount', queued['amount']):
                                try:
                                    await asyncio.to_thread(self.trader.cancel_order, order_id)
                                    logger.info(f"Cancelled remaining part of partially filled exit order {order_id} for {queued['symbol']}")
                                except Exception as e:
                                    logger.warning(f"Failed to cancel remaining part of exit order {order_id}: {e}")
                                async with self._queued_orders_lock:
                                    self.queued_orders = [
                                        q for q in self.queued_orders
                                        if q.get("order_id") != order_id
                                    ]

                            pos = self.positions.get(queued["symbol"])
                            if pos:
                                pos.pop("stop_loss_order_id", None)
                                pos.pop("take_profit_order_id", None)
                                # Place replacement exit orders for the remaining position to avoid
                                # a protection gap until the next risk-management loop tick.
                                if pos.get("amount", 0) > 0 and pos.get("stop_loss") is not None:
                                    from src.strategies.base import Signal as _Signal
                                    _dummy_params = {
                                        "trailing_take_profit": pos.get("trailing_take_profit", False),
                                        "partial_take_profit_levels": pos.get("partial_take_profit_levels"),
                                        "partial_take_profit_pct": pos.get("partial_take_profit_pct"),
                                    }
                                    _dummy_signal = _Signal(
                                        action="BUY",
                                        confidence=1.0,
                                        reasoning="Replacing exit orders after partial exit-order fill",
                                        stop_loss_order_type=pos.get("stop_loss_order_type"),
                                        stop_loss_stop_price=pos.get("stop_loss"),
                                        stop_loss_limit_price=None,
                                        take_profit_order_type=pos.get("take_profit_order_type"),
                                        take_profit_limit_price=pos.get("take_profit"),
                                        strategy_params=_dummy_params,
                                    )
                                    _exit_prices = {
                                        "stop_loss_price": pos.get("stop_loss"),
                                        "take_profit_price": pos.get("take_profit"),
                                    }
                                    try:
                                        await self._place_exit_orders(
                                            queued["symbol"], _dummy_signal, _exit_prices, pos.get("timeframe")
                                        )
                                    except Exception as _e:
                                        logger.warning(
                                            f"Failed to place replacement exit orders after partial "
                                            f"fill for {queued['symbol']}: {_e}"
                                        )
                            # Notify user
                            if self.notifier:
                                stock_name = await self._get_stock_name(queued["symbol"])
                                display_symbol = self._format_symbol_display(queued["symbol"], stock_name, queued.get("timeframe"))
                                await self.notifier.send_notification(
                                    f"🔗 OCO pair {oco_pair_id} cancelled for {display_symbol} (other order filled).",
                                    summary={
                                        "symbol": queued["symbol"],
                                        "action": "CANCEL",
                                        "reason": "OCO pair cancelled",
                                    }
                                )

                    if status == 'filled':
                        logger.info(f"Queued limit order {order_id} for {queued['symbol']} completely filled.")
                        async with self._queued_orders_lock:
                            self.queued_orders.remove(queued)
                        self._state_dirty = True

                    elif status in ('rejected', 'canceled', 'cancelled', 'expired'):
                        logger.warning(
                            f"Queued order {order_id} for {queued['symbol']} ended as {status}, removing."
                        )
                        if self.notifier:
                            stock_name = await self._get_stock_name(queued['symbol'])
                            tf = queued.get('timeframe')
                            display = self._format_symbol_display(queued['symbol'], stock_name, tf)
                            await self.notifier.send_notification(
                                f"❌ Queued {queued['side']} order for {display} {status}.",
                                summary={
                                    "symbol": queued['symbol'],
                                    "action": "INFO",
                                    "reason": f"Order {status}",
                                }
                            )
                        async with self._queued_orders_lock:
                            self.queued_orders.remove(queued)
                        self._state_dirty = True
                        if queued.get("is_exit_order"):
                            oco_pair_id = queued.get("oco_pair")
                            if oco_pair_id:
                                try:
                                    await asyncio.to_thread(self.trader.cancel_order, oco_pair_id)
                                    logger.info(f"Cancelled OCO pair {oco_pair_id} for {status} exit order {order_id}")
                                except Exception as e:
                                    logger.warning(f"Failed to cancel OCO order {oco_pair_id}: {e}")
                                async with self._queued_orders_lock:
                                    self.queued_orders = [
                                        q for q in self.queued_orders
                                        if q.get("order_id") != oco_pair_id
                                    ]
                            pos = self.positions.get(queued["symbol"])
                            if pos:
                                pos.pop("stop_loss_order_id", None)
                                pos.pop("take_profit_order_id", None)
                            if self.notifier:
                                stock_name = await self._get_stock_name(queued["symbol"])
                                display_symbol = self._format_symbol_display(queued["symbol"], stock_name, queued.get("timeframe"))
                                await self.notifier.send_notification(
                                    f"🔗 OCO pair {oco_pair_id} cancelled for {display_symbol} (main order {status}).",
                                    summary={"symbol": queued["symbol"], "action": "CANCEL", "reason": f"OCO pair cancelled due to main order {status}"}
                                )

                    # else: still open / partially_filled / accepted – keep waiting
            except Exception as e:
                logger.error(f"Error processing queued orders: {e}", exc_info=True)
            await asyncio.sleep(15)  # check every 15 seconds for faster fill detection

    async def _handle_queued_buy_fill(self, trade_dict: Dict[str, Any], queued: Dict[str, Any]):
        """Process a queued BUY limit order that has filled in the simulator."""
        symbol = trade_dict['symbol']
        parts = symbol.split("/")
        if len(parts) != 2:
            logger.error(f"Invalid symbol format in queued buy fill: {symbol}")
            return
        base, quote = parts
        fee = trade_dict.get('fee', {})
        fee_cost = float(fee.get('cost', 0.0) or 0.0)
        fee_currency = fee.get('currency', '')
        cost_basis = trade_dict['cost'] + (fee_cost if fee_currency == quote else 0.0)
        net_base = trade_dict['amount'] - (fee_cost if fee_currency == base else 0.0)

        signal_dict = queued.get('signal', {}) or {}
        params = signal_dict.get('strategy_params', {}) or {}
        timeframe = queued.get('timeframe')
        atr = queued.get('atr')
        fill_price = trade_dict['price']

        # Determine stop-loss percentage based on method
        stop_method = params.get("stop_loss_method", "fixed")
        if stop_method == "atr_multiple" and atr is not None and atr > 0 and fill_price > 0:
            atr_mult = params.get("stop_loss_atr_multiple")
            if atr_mult is not None:
                sl_pct = (atr_mult * atr) / fill_price
            else:
                sl_pct = params.get("stop_loss_pct")
        else:
            sl_pct = params.get("stop_loss_pct")

        # Determine take-profit percentage based on method
        if "take_profit_atr_multiple" in params and atr is not None and atr > 0 and fill_price > 0:
            tp_atr_mult = params["take_profit_atr_multiple"]
            tp_pct = (tp_atr_mult * atr) / fill_price
        else:
            tp_pct = params.get("take_profit_pct")
        trailing_stop = params.get("trailing_stop", False)
        trailing_stop_distance_pct = params.get("trailing_stop_distance_pct")
        indicator_config = signal_dict.get('indicator_config')

        if symbol in self.positions:
            old_cost_basis = self.positions[symbol].get("cost_basis", self.positions[symbol]["amount"] * self.positions[symbol]["price"])
            old_net_base = self.positions[symbol].get("net_base", self.positions[symbol]["amount"])
            new_cost_basis = old_cost_basis + cost_basis
            new_net_base = old_net_base + net_base
            new_price = new_cost_basis / new_net_base if new_net_base > 0 else 0.0
            self.positions[symbol]["amount"] = new_net_base
            self.positions[symbol]["price"] = new_price
            self.positions[symbol]["cost_basis"] = new_cost_basis
            self.positions[symbol]["net_base"] = new_net_base
            # Preserve existing absolute SL/TP prices when scaling in.
            # Recalculating based on the new weighted average would shift
            # them from where the LLM originally intended.
            self.positions[symbol]["take_profit_atr_multiple"] = params.get("take_profit_atr_multiple")
            self.positions[symbol]["trailing_stop"] = trailing_stop
            self.positions[symbol]["trailing_stop_distance_pct"] = trailing_stop_distance_pct
            self.positions[symbol]["max_hold_time_seconds"] = params.get("max_hold_time_seconds")
            self.positions[symbol]["trailing_stop_activation_pct"] = params.get("trailing_stop_activation_pct")
            self.positions[symbol]["trailing_take_profit"] = params.get("trailing_take_profit", False)
            self.positions[symbol]["trailing_take_profit_distance_pct"] = params.get("trailing_take_profit_distance_pct")
            self.positions[symbol]["breakeven_activation_pct"] = params.get("breakeven_activation_pct")
            partial_levels = params.get("partial_take_profit_levels")
            if partial_levels:
                self.positions[symbol]["partial_take_profit_levels"] = partial_levels
                self.positions[symbol]["partial_tp_levels_triggered"] = []
                self.positions[symbol]["partial_tp_depth_wait_start"] = {}
                self.positions[symbol]["partial_take_profit_pct"] = None
                self.positions[symbol]["partial_take_profit_fraction"] = None
                self.positions[symbol]["partial_tp_triggered"] = None
            else:
                self.positions[symbol]["partial_take_profit_pct"] = params.get("partial_take_profit_pct")
                self.positions[symbol]["partial_take_profit_fraction"] = params.get("partial_take_profit_fraction")
                self.positions[symbol]["partial_tp_triggered"] = False
            self.positions[symbol]["cooldown_after_loss_seconds"] = params.get("cooldown_after_loss_seconds", 0)
            self.positions[symbol]["news_sentiment_exit_threshold"] = params.get("news_sentiment_exit_threshold")
            self.positions[symbol]["max_unrealized_loss_pct"] = params.get("max_unrealized_loss_pct")
            self.positions[symbol]["timeframe"] = timeframe
            self.positions[symbol]["indicator_config"] = indicator_config
            self.positions[symbol]["entry_order_type"] = queued.get('order_type', 'market')
            self.positions[symbol]["buy_confidence"] = signal_dict.get('confidence', 0.0)
            self.positions[symbol]["buy_reasoning"] = (signal_dict.get('reasoning', '') or '')[:200]
        else:
            entry_price = cost_basis / net_base if net_base > 0 else trade_dict["price"]
            self.positions[symbol] = {
                "symbol": symbol,
                "side": "buy",
                "amount": net_base,
                "price": entry_price,
                "timestamp": trade_dict["timestamp"],
                "stop_loss": entry_price * (1 - sl_pct) if sl_pct else None,
                "take_profit": entry_price * (1 + tp_pct) if tp_pct else None,
                "take_profit_atr_multiple": params.get("take_profit_atr_multiple"),
                "cost_basis": cost_basis,
                "net_base": net_base,
                "buy_confidence": signal_dict.get('confidence', 0.0),
                "buy_reasoning": (signal_dict.get('reasoning', '') or '')[:200],
                "trailing_stop": trailing_stop,
                "trailing_stop_distance_pct": trailing_stop_distance_pct,
                "max_hold_time_seconds": params.get("max_hold_time_seconds"),
                "trailing_stop_activation_pct": params.get("trailing_stop_activation_pct"),
                "trailing_take_profit": params.get("trailing_take_profit", False),
                "trailing_take_profit_distance_pct": params.get("trailing_take_profit_distance_pct"),
                "breakeven_activation_pct": params.get("breakeven_activation_pct"),
                "partial_take_profit_levels": params.get("partial_take_profit_levels"),
                "partial_tp_levels_triggered": [],
                "partial_tp_depth_wait_start": {},
                "original_amount": net_base,
                "partial_take_profit_pct": params.get("partial_take_profit_pct") if not params.get("partial_take_profit_levels") else None,
                "partial_take_profit_fraction": params.get("partial_take_profit_fraction") if not params.get("partial_take_profit_levels") else None,
                "partial_tp_triggered": False if not params.get("partial_take_profit_levels") else None,
                "cooldown_after_loss_seconds": params.get("cooldown_after_loss_seconds", 0),
                "news_sentiment_exit_threshold": params.get("news_sentiment_exit_threshold"),
                "max_unrealized_loss_pct": params.get("max_unrealized_loss_pct"),
                "timeframe": timeframe,
                "indicator_config": indicator_config,
                "entry_order_type": queued.get('order_type', 'market'),
            }

        custom_interval = params.get("strategy_interval_seconds")
        if custom_interval is not None:
            self._strategy_intervals[symbol] = custom_interval

        trade_dict['strategy_type'] = signal_dict.get('strategy_type')
        trade_dict['timeframe'] = timeframe
        trade_dict['buy_confidence'] = signal_dict.get('confidence', 0.0)
        trade_dict['buy_reasoning'] = (signal_dict.get('reasoning', '') or '')[:200]
        self._append_trade(trade_dict)
        self._balance_cache = None
        async with self._cycle_spent_lock:
            self._cycle_spent += trade_dict['cost']
        await asyncio.to_thread(insert_trade, trade_dict)
        await self._save_state(force=True)
        if self.notifier:
            stock_name = await self._get_stock_name(symbol)
            display_symbol = self._format_symbol_display(symbol, stock_name, timeframe)
            buy_msg = f"🟢 BUY {display_symbol}: {trade_dict['amount']:.6f} @ {trade_dict['price']:.4f}"
            buy_summary = {
                "symbol": symbol,
                "action": "BUY",
                "price": trade_dict["price"],
                "amount": trade_dict["amount"],
                "confidence": signal_dict.get('confidence', 0.0),
                "reason": (signal_dict.get('reasoning', '') or '')[:200],
                "strategy_type": signal_dict.get('strategy_type'),
                "indicators": {"atr": atr},
            }
            if signal_dict.get('model_type'):
                buy_summary["model_type"] = signal_dict.get('model_type')
            if signal_dict.get('llm_provider'):
                buy_summary["llm_provider"] = signal_dict.get('llm_provider')
            if signal_dict.get('llm_model'):
                buy_summary["llm_model"] = signal_dict.get('llm_model')
            await self.notifier.send_notification(buy_msg, summary=buy_summary)

        # Place native exit orders for the new/updated position
        signal_dict = queued.get('signal', {}) or {}
        if signal_dict:
            try:
                # Reconstruct a Signal from the stored dict, filtering to only
                # valid Signal fields and providing fallbacks for required fields.
                import dataclasses as _dc
                valid_keys = {f.name for f in _dc.fields(Signal)}
                filtered = {k: v for k, v in signal_dict.items() if k in valid_keys}
                # Ensure required fields have fallbacks
                if "action" not in filtered:
                    filtered["action"] = "BUY"
                if "confidence" not in filtered:
                    filtered["confidence"] = 0.0
                if "reasoning" not in filtered:
                    filtered["reasoning"] = ""
                reconstructed_signal = Signal(**filtered)
                exit_prices = self._compute_exit_order_prices(
                    entry_price=self.positions[symbol]["price"],
                    signal=reconstructed_signal,
                    atr=queued.get('atr'),
                )
                await self._place_exit_orders(symbol, reconstructed_signal, exit_prices, queued.get('timeframe'))
            except Exception as e:
                logger.error(f"Failed to place exit orders after queued buy fill for {symbol}: {e}")
                if self.notifier:
                    stock_name = await self._get_stock_name(symbol)
                    display_symbol = self._format_symbol_display(symbol, stock_name, queued.get('timeframe'))
                    await self.notifier.send_notification(
                        f"⚠️ Exit order placement failed for {display_symbol} after queued fill: {e}",
                        summary={
                            "symbol": symbol,
                            "action": "ERROR",
                            "reason": f"Exit order placement failed after queued fill: {str(e)[:200]}",
                        }
                    )

    async def _handle_queued_sell_fill(self, trade_dict: Dict[str, Any], queued: Dict[str, Any], partial: bool = False):
        """Process a queued SELL limit order that has filled in the simulator.

        When *partial* is True, only a portion of the order has filled; the
        position is prorated and updated rather than removed.
        """
        symbol = trade_dict['symbol']
        parts = symbol.split("/")
        if len(parts) != 2:
            logger.error(f"Invalid symbol format in queued sell fill: {symbol}")
            return
        base, quote = parts
        pos = self.positions.get(symbol)
        fee = trade_dict.get('fee', {})
        fee_cost = float(fee.get('cost', 0.0) or 0.0)
        fee_currency = fee.get('currency', '')
        net_quote = trade_dict['cost'] - (fee_cost if fee_currency == quote else 0.0)
        exit_reason = queued.get('exit_reason', 'limit_order')
        trade_dict['exit_reason'] = exit_reason
        signal_dict = queued.get('signal', {}) or {}
        trade_dict['strategy_type'] = signal_dict.get('strategy_type')
        trade_dict['timeframe'] = queued.get('timeframe')
        if pos:
            trade_dict['buy_confidence'] = pos.get("buy_confidence", 0.0)
            trade_dict['buy_reasoning'] = pos.get("buy_reasoning", "")
        if pos and "timestamp" in pos:
            trade_dict['hold_time_seconds'] = (trade_dict['timestamp'] - pos["timestamp"]) / 1000.0
        else:
            trade_dict['hold_time_seconds'] = None

        if partial and pos:
            # Prorated cost basis for the sold portion
            cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
            net_base = pos.get("net_base", pos["amount"])
            prorated_cost_basis = cost_basis * (trade_dict['amount'] / net_base) if net_base > 0 else 0.0
            realized_pnl = net_quote - prorated_cost_basis
            trade_dict['realized_pnl'] = realized_pnl
            trade_dict['cost_basis'] = prorated_cost_basis

            # Update position
            remaining_amount = pos["amount"] - trade_dict['amount']
            remaining_cost_basis = cost_basis - prorated_cost_basis
            remaining_net_base = net_base - trade_dict['amount']

            # Cancel old exit orders because quantity changed
            await self._cancel_exit_orders(symbol)

            if remaining_amount <= 0 or remaining_net_base <= 0:
                # Position fully closed via partial fills
                if realized_pnl < 0:
                    self.last_loss_time[symbol] = time.time()
                    self.cooldown_durations[symbol] = pos.get("cooldown_after_loss_seconds", 0)
                pos.pop("_stop_loss_triggered", None)
                pos.pop("_stop_loss_review_count", None)
                async with self._positions_lock:
                    self.positions.pop(symbol, None)
                self._strategy_intervals.pop(symbol, None)
                self._last_strategy_eval.pop(symbol, None)
                self._last_decisions.pop(symbol, None)
                self._pending_entries.pop(symbol, None)
                await self._remove_symbol_if_paused(symbol)
            else:
                async with self._positions_lock:
                    self.positions[symbol]["amount"] = remaining_amount
                    self.positions[symbol]["cost_basis"] = remaining_cost_basis
                    self.positions[symbol]["net_base"] = remaining_net_base
                    self.positions[symbol]["price"] = remaining_cost_basis / remaining_net_base if remaining_net_base > 0 else 0.0

                # Replace exit orders for the remaining amount
                from src.strategies.base import Signal
                async with self._positions_lock:
                    dummy_params = {
                        "trailing_take_profit": self.positions[symbol].get("trailing_take_profit", False),
                        "partial_take_profit_levels": self.positions[symbol].get("partial_take_profit_levels"),
                        "partial_take_profit_pct": self.positions[symbol].get("partial_take_profit_pct"),
                    }
                    dummy_signal = Signal(
                        action="BUY",
                        confidence=1.0,
                        reasoning="Replacing exit orders after partial sell fill",
                        stop_loss_order_type=self.positions[symbol].get("stop_loss_order_type"),
                        stop_loss_stop_price=self.positions[symbol].get("stop_loss"),
                        stop_loss_limit_price=None,
                        take_profit_order_type=self.positions[symbol].get("take_profit_order_type"),
                        take_profit_limit_price=self.positions[symbol].get("take_profit"),
                        strategy_params=dummy_params,
                    )
                    exit_prices = {
                        "stop_loss_price": self.positions[symbol].get("stop_loss"),
                        "take_profit_price": self.positions[symbol].get("take_profit"),
                    }
                await self._place_exit_orders(symbol, dummy_signal, exit_prices, self.positions[symbol].get("timeframe"))
        else:
            # Full fill (non-partial) – original logic
            # Cancel any remaining exit orders before removing the position
            await self._cancel_exit_orders(symbol)
            if pos:
                cost_basis = pos.get("cost_basis", pos["amount"] * pos["price"])
                realized_pnl = net_quote - cost_basis
            else:
                realized_pnl = 0.0
            trade_dict['realized_pnl'] = realized_pnl
            trade_dict['cost_basis'] = pos.get("cost_basis", 0.0) if pos else 0.0
            if realized_pnl < 0:
                self.last_loss_time[symbol] = time.time()
                self.cooldown_durations[symbol] = pos.get("cooldown_after_loss_seconds", 0) if pos else 0
            if pos:
                pos.pop("_stop_loss_triggered", None)
                pos.pop("_stop_loss_review_count", None)
            async with self._positions_lock:
                self.positions.pop(symbol, None)
            self._strategy_intervals.pop(symbol, None)
            self._last_strategy_eval.pop(symbol, None)
            self._last_decisions.pop(symbol, None)
            self._pending_entries.pop(symbol, None)
            await self._remove_symbol_if_paused(symbol)

        self._append_trade(trade_dict)
        self._balance_cache = None
        await asyncio.to_thread(insert_trade, trade_dict)
        await self._save_state(force=True)
        if self.notifier:
            reason_labels = {
                "manual_sell": "🖐️ Manual",
                "manual_sell_all": "🖐️ Manual (Sell All)",
                "stop_loss": "⛔ Stop-Loss",
                "take_profit": "✅ Take-Profit",
                "max_hold_time": "⏰ Max Hold Time",
                "news_sentiment_exit": "📰 News Sentiment",
                "force_close": "🔻 Force Close",
                "external_sell": "🔄 External Sell",
                "delisted": "🗑️ Delisted",
            }
            reason_label = reason_labels.get(exit_reason, exit_reason) if exit_reason else None
            reason_str = f" [{reason_label}]" if reason_label else ""
            stock_name = await self._get_stock_name(symbol)
            tf = queued.get('timeframe') or (pos.get("timeframe") if pos else None)
            display_symbol = self._format_symbol_display(symbol, stock_name, tf)
            partial_str = " (partial)" if partial else ""
            sell_msg = f"🔴 SELL{reason_str}{partial_str} {display_symbol}: {trade_dict['amount']:.6f} @ {trade_dict['price']:.4f}"
            if pos:
                cb = trade_dict.get('cost_basis', 0.0) or (pos.get("cost_basis", pos["amount"] * pos["price"]) if pos else 0.0)
                pnl_pct = (realized_pnl / cb * 100) if cb > 0 else 0.0
                sell_msg += f" | P&L: {realized_pnl:+.4f} ({pnl_pct:+.2f}%)"
            sell_summary = {
                "symbol": symbol,
                "action": "SELL",
                "price": trade_dict["price"],
                "amount": trade_dict["amount"],
                "confidence": signal_dict.get('confidence', 0.0),
                "reason": (signal_dict.get('reasoning', '') or '')[:200],
                "exit_reason": exit_reason,
                "realized_pnl": realized_pnl,
                "strategy_type": signal_dict.get('strategy_type'),
                "indicators": {"atr": queued.get('atr')},
            }
            if signal_dict.get('model_type'):
                sell_summary["model_type"] = signal_dict.get('model_type')
            if signal_dict.get('llm_provider'):
                sell_summary["llm_provider"] = signal_dict.get('llm_provider')
            if signal_dict.get('llm_model'):
                sell_summary["llm_model"] = signal_dict.get('llm_model')
            await self.notifier.send_notification(sell_msg, summary=sell_summary)

    async def _cleanup_orphaned_orders(self):
        """Periodically cancel any open orders that are older than 10 minutes,
        but never cancel orders that are still being tracked as queued."""
        await asyncio.sleep(120)  # initial delay
        # --- Notify mode: no orphaned order cleanup ---
        if settings.TRADING_MODE == "notify":
            return
        while self._running:
            try:
                open_orders = await asyncio.to_thread(self.trader.get_open_orders)
                now = time.time()
                # Build a set of order IDs that are currently queued (waiting for fill)
                queued_ids = {q.get('order_id') for q in self.queued_orders if q.get('order_id')}
                for order in open_orders:
                    order_id = order.get('id')
                    if order_id in queued_ids:
                        continue   # this order is being monitored by _process_queued_orders
                    created_at = order.get('timestamp', 0) / 1000.0  # ms to seconds
                    if now - created_at > 600:  # 10 minutes
                        logger.warning(
                            f"Cancelling orphaned order {order_id} for {order['symbol']} "
                            f"(open for {now - created_at:.0f}s)."
                        )
                        await asyncio.to_thread(self.trader.cancel_order, order_id)
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
        of 'ENI.MI/EUR'). This method tries multiple formats to find a match.
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
        self, symbol: str, action: str, ticker: Dict[str, Any]
    ) -> Optional[float]:
        """Compute a default aggressive limit price for extended‑hours trading."""
        if action == "BUY":
            last = ticker.get('last')
            if last and last > 0:
                limit = last * 1.002
                if last >= 1.0:
                    limit = round(limit, 2)
                else:
                    limit = round(limit, 4)
                return limit
        elif action == "SELL":
            last = ticker.get('last')
            if last and last > 0:
                limit = last * 0.998
                if last >= 1.0:
                    limit = round(limit, 2)
                else:
                    limit = round(limit, 4)
                return limit
        return None

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
        # Build params hash for dedup lookup
        source_candles = historical_ohlcv or raw_candles or []
        last_ts = source_candles[-1][0] if source_candles else 0
        candle_count = len(source_candles)
        params_hash = hashlib.md5(
            json.dumps(variant_params, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        # Check database for a recent identical backtest (dedup within 1 hour)
        try:
            recent = await asyncio.to_thread(
                get_recent_backtest_result, symbol, assigned_tf, params_hash, 3600
            )
            if recent:
                logger.debug(f"Backtest DB cache hit for {symbol} {assigned_tf} (params_hash={params_hash})")
                return recent["stats"], recent["summary"]
        except Exception:
            pass

        # Run backtest with concurrency limiting
        async with self._backtest_semaphore:
            variant_signal = Signal(
                action="BUY",
                confidence=preliminary_signal.confidence,
                reasoning=preliminary_signal.reasoning,
                strategy_params=variant_params,
            )
            bt_stats, bt_summary = await self._run_backtest_from_signal(
                symbol=symbol,
                signal=variant_signal,
                atr=atr,
                current_price=current_price,
                tf_secs=tf_secs,
                assigned_tf=assigned_tf,
                historical_ohlcv=historical_ohlcv,
                raw_candles=raw_candles,
                base_balance=base_balance,
                is_btp=is_btp,
            )

        # Persist the result to the database
        if bt_stats is not None:
            try:
                await asyncio.to_thread(
                    save_backtest_result, symbol, assigned_tf, params_hash,
                    variant_params, bt_stats, bt_summary
                )
            except Exception as e:
                logger.warning(f"Failed to persist backtest result to DB for {symbol}: {e}")

        return bt_stats, bt_summary

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
        bt_params = signal.strategy_params or {}
        bt_sl_pct = bt_params.get("stop_loss_pct", 0.02)
        bt_tp_pct = bt_params.get("take_profit_pct", 0.05)
        bt_sl_atr_mult = bt_params.get("stop_loss_atr_multiple")
        bt_tp_atr_mult = bt_params.get("take_profit_atr_multiple")
        bt_max_hold = bt_params.get("max_hold_time_seconds")
        bt_trailing = bt_params.get("trailing_stop", False)
        bt_trail_dist = bt_params.get("trailing_stop_distance_pct")
        bt_trail_act = bt_params.get("trailing_stop_activation_pct")
        bt_entry_config = bt_params.get("backtest_entry_config")

        bt_period_days = bt_params.get("backtest_period_days")
        if bt_period_days is not None:
            bt_period_days = max(30, min(int(bt_period_days), settings.OHLCV_RETENTION_DAYS))
            bt_since_ms = int(time.time() * 1000) - bt_period_days * 24 * 60 * 60 * 1000
            bt_limit = int((bt_period_days * 86400) / tf_secs) + 100
            bt_db_candles = await asyncio.to_thread(
                get_ohlcv, symbol, assigned_tf, since_ms=bt_since_ms, limit=bt_limit
            )
            if bt_db_candles:
                bt_candles = [
                    [c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
                    for c in bt_db_candles
                ]
            else:
                bt_candles = historical_ohlcv or raw_candles
        else:
            bt_candles = historical_ohlcv or raw_candles

        # Early skip: if the assigned timeframe cannot possibly have enough candles
        # given the data retention period, skip backtesting entirely instead of
        # falling back to a much shorter timeframe whose results would be misleading.
        tf_seconds_bt = self._timeframe_to_seconds(assigned_tf)
        max_possible_candles = (settings.OHLCV_RETENTION_DAYS * 86400) / tf_seconds_bt
        if max_possible_candles < 5:
            return None, (
                f"Backtesting skipped for {assigned_tf}: only ~{int(max_possible_candles)} candles possible "
                f"with {settings.OHLCV_RETENTION_DAYS} days retention (need ≥5). "
                f"Rely on LLM analysis, fundamentals, and multi-timeframe indicators instead."
            )

        # --- Fallback to shorter timeframes when the assigned timeframe has too few candles ---
        MIN_BACKTEST_CANDLES = 20
        backtest_fallback_note = ""
        if bt_candles is None or len(bt_candles) < MIN_BACKTEST_CANDLES:
            if assigned_tf in settings.OHLCV_TIMEFRAMES:
                tf_idx = settings.OHLCV_TIMEFRAMES.index(assigned_tf)
                for shorter_tf in settings.OHLCV_TIMEFRAMES[tf_idx + 1:]:
                    shorter_tf_secs = self._timeframe_to_seconds(shorter_tf)
                    try:
                        if bt_period_days is not None:
                            fb_since_ms = int(time.time() * 1000) - bt_period_days * 24 * 60 * 60 * 1000
                            fb_limit = int((bt_period_days * 86400) / shorter_tf_secs) + 100
                            fb_db_candles = await asyncio.to_thread(
                                get_ohlcv, symbol, shorter_tf, since_ms=fb_since_ms, limit=fb_limit
                            )
                        else:
                            fb_db_candles = await asyncio.to_thread(
                                get_ohlcv, symbol, shorter_tf, limit=500
                            )
                        if fb_db_candles and len(fb_db_candles) >= MIN_BACKTEST_CANDLES:
                            bt_candles = [
                                [c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]]
                                for c in fb_db_candles
                            ]
                            backtest_fallback_note = (
                                f" ⚠️ FALLBACK WARNING: Backtest was run on {shorter_tf} candles, NOT {assigned_tf}. "
                                f"The assigned {assigned_tf} timeframe had insufficient candles (< {MIN_BACKTEST_CANDLES}) "
                                f"with {settings.OHLCV_RETENTION_DAYS} days retention. "
                                f"Results from {shorter_tf} may not accurately represent {assigned_tf} behavior — treat with caution."
                            )
                            logger.info(
                                f"Backtest fallback for {symbol}: assigned_tf={assigned_tf} had insufficient candles, "
                                f"using {shorter_tf} ({len(bt_candles)} candles)."
                            )
                            break
                    except Exception as e:
                        logger.debug(f"Backtest fallback to {shorter_tf} failed for {symbol}: {e}")

        bt_position_fraction = bt_params.get("position_size_fraction", 1.0 / self.effective_max_symbols if self.effective_max_symbols > 0 else 1.0)
        bt_trade_value = base_balance * bt_position_fraction
        if bt_trade_value > 0:
            from src.exchanges.fees import calculate_transaction_costs
            buy_costs = calculate_transaction_costs("BUY", 100.0, bt_trade_value / 100.0, symbol=symbol)
            sell_costs = calculate_transaction_costs("SELL", 100.0, bt_trade_value / 100.0, symbol=symbol)
            total_fee_pct = (buy_costs["total_costs"] + sell_costs["total_costs"]) / bt_trade_value
            bt_fee_rate = total_fee_pct / 2
        else:
            bt_fee_rate = 0.006

        atr_series = None
        adx_series = None
        rsi_series = None
        macd_hist_series = None
        if bt_candles and len(bt_candles) >= 2:
            def _compute_bt_indicator_series():
                _atr_series = None
                _adx_series = None
                _rsi_series = None
                _macd_hist_series = None
                try:
                    # Compute ATR series (needed for dynamic ATR stops or trailing stops)
                    if bt_params.get("trailing_stop_atr_multiple") or bt_sl_atr_mult or bt_tp_atr_mult:
                        _atr_series = compute_atr_series(bt_candles, period=14)

                    # Compute ADX series (always needed for the backtester's trend-strength filter)
                    _adx_series = compute_adx_series(bt_candles, period=14)

                    # Compute RSI and MACD series for additional backtest filters
                    _rsi_series = compute_rsi_series(bt_candles, period=14)
                    _, _, _macd_hist_series = compute_macd_series(bt_candles)
                except Exception:
                    pass
                return _atr_series, _adx_series, _rsi_series, _macd_hist_series

            atr_series, adx_series, rsi_series, macd_hist_series = await asyncio.to_thread(_compute_bt_indicator_series)

        # Fetch LLM-configured thresholds for backtest filters
        bt_max_rsi = 70.0
        try:
            raw = await asyncio.to_thread(self.redis.get, "trading:skip_eval_rsi_overbought")
            if raw:
                bt_max_rsi = float(raw)
        except Exception:
            pass

        # Fetch portfolio caps for position sizing simulation
        bt_global_risk_mult = 1.0
        bt_max_port_exp = None
        bt_max_port_risk = None
        try:
            raw = await asyncio.to_thread(self.redis.get, "trading:global_risk_multiplier")
            if raw:
                bt_global_risk_mult = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:max_portfolio_exposure_pct")
            if raw:
                bt_max_port_exp = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:max_portfolio_stop_risk_pct")
            if raw:
                bt_max_port_risk = float(raw)
        except Exception:
            pass

        if bt_candles and len(bt_candles) >= 20:
            bt_kwargs = dict(
                stop_loss_pct=bt_sl_pct,
                take_profit_pct=bt_tp_pct,
                stop_loss_atr_multiple=bt_sl_atr_mult,
                take_profit_atr_multiple=bt_tp_atr_mult,
                max_hold_time_seconds=bt_max_hold,
                trailing_stop=bt_trailing,
                trailing_stop_distance_pct=bt_trail_dist,
                trailing_stop_activation_pct=bt_trail_act,
                partial_take_profit_levels=bt_params.get("partial_take_profit_levels"),
                breakeven_activation_pct=bt_params.get("breakeven_activation_pct"),
                trailing_take_profit=bt_params.get("trailing_take_profit", False),
                trailing_take_profit_distance_pct=bt_params.get("trailing_take_profit_distance_pct"),
                trailing_stop_atr_multiple=bt_params.get("trailing_stop_atr_multiple"),
                atr_values=atr_series,
                max_unrealized_loss_pct=bt_params.get("max_unrealized_loss_pct"),
                adx_values=adx_series,
                rsi_values=rsi_series,
                max_rsi=bt_max_rsi,
                macd_hist_values=macd_hist_series,
                fee_rate=bt_fee_rate,
                fee_model="intesa",
                trade_value=bt_trade_value,
                is_btp=is_btp,
                cooldown_after_loss_seconds=bt_params.get("cooldown_after_loss_seconds"),
                slippage_pct=0.001,
                slippage_model="dynamic",
                slippage_base_pct=0.001,
                slippage_max_pct=0.01,
                backtest_entry_config=bt_entry_config,
                direction="long",
                simulate_position_sizing=True,
                initial_balance=base_balance,
                confidence=signal.confidence,
                confidence_sizing_weight=bt_params.get("confidence_sizing_weight", 0.0),
                global_risk_multiplier=bt_global_risk_mult,
                position_size_multiplier=bt_params.get("position_size_multiplier", 1.0),
                max_risk_per_trade_pct=bt_params.get("max_risk_per_trade_pct"),
                max_portfolio_risk_pct=bt_params.get("max_portfolio_risk_pct"),
                max_portfolio_exposure_pct=bt_max_port_exp,
                max_portfolio_stop_risk_pct=bt_max_port_risk,
                position_size_fraction=bt_position_fraction,
                gap_tolerance_mult=1.5,
                on_gaps="warn",
            )
            backtest_stats = await asyncio.to_thread(
                backtest_strategy,
                candles=bt_candles,
                **bt_kwargs,
            )
            bt_entry_config_used = bt_entry_config is not None and isinstance(bt_entry_config, dict) and len(bt_entry_config) > 0
            bt_summary = format_backtest_summary(backtest_stats, entry_config_used=bt_entry_config_used)
            if backtest_fallback_note:
                bt_summary += backtest_fallback_note

            if len(bt_candles) >= 100:
                wf_stats = await asyncio.to_thread(
                    walk_forward_backtest,
                    candles=bt_candles,
                    num_windows=5,
                    **bt_kwargs,
                )
                bt_summary = bt_summary + "\n" + format_walk_forward_summary(wf_stats)

            return backtest_stats, bt_summary
        if backtest_fallback_note:
            return None, f"Insufficient data for backtest (need ≥{MIN_BACKTEST_CANDLES} candles).{backtest_fallback_note}"
        return None, f"Insufficient data for backtest for {assigned_tf} (need ≥{MIN_BACKTEST_CANDLES} candles with {settings.OHLCV_RETENTION_DAYS} days retention)."

    async def _prepare_simulation_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch all necessary data and build the strategy prompt for simulation."""
        symbol_entry = next((e for e in self.current_symbols if e["symbol"] == symbol), None)
        if not symbol_entry:
            return {"error": f"Symbol {symbol} not found in current_symbols"}
        
        assigned_tf = symbol_entry["timeframe"]

        symbol_data = await self._fetch_symbol_market_data(symbol, assigned_tf)
        if symbol_data is None:
            return {"error": "No ticker data"}
        ticker = symbol_data["ticker"]
        current_price = symbol_data["current_price"]
        fundamentals = symbol_data["fundamentals"]
        balance = symbol_data["balance"]
        base_balance = symbol_data["base_balance"]
        ohlcv_data = symbol_data["ohlcv_data"]
        is_btp = symbol_data["is_btp"]
        tf_seconds = symbol_data["tf_seconds"]
        multi_tf_indicators = symbol_data["multi_tf_indicators"]
        multi_tf_raw_candles = symbol_data["multi_tf_raw_candles"]
        atr = symbol_data["atr"]
        rsi = symbol_data["rsi"]
        macd = symbol_data["macd"]
        macd_signal = symbol_data["macd_signal"]
        macd_hist = symbol_data["macd_hist"]
        bb_upper = symbol_data["bb_upper"]
        bb_middle = symbol_data["bb_middle"]
        bb_lower = symbol_data["bb_lower"]
        ema_9 = symbol_data["ema_9"]
        ema_21 = symbol_data["ema_21"]
        stochastic_k = symbol_data["stochastic_k"]
        stochastic_d = symbol_data["stochastic_d"]
        adx = symbol_data["adx"]
        plus_di = symbol_data["plus_di"]
        minus_di = symbol_data["minus_di"]
        obv = symbol_data["obv"]
        mfi = symbol_data["mfi"]
        cci = symbol_data["cci"]
        williams_r = symbol_data["williams_r"]
        ichimoku = symbol_data["ichimoku"]
        donchian_channels = symbol_data["donchian_channels"]
        parabolic_sar = symbol_data["parabolic_sar"]
        keltner_channels = symbol_data["keltner_channels"]
        vwap = symbol_data["vwap"]
        daily_pivot_points = symbol_data["daily_pivot_points"]
        per_symbol_budget = base_balance / self.effective_max_symbols if self.effective_max_symbols > 0 else 0.0

        # Fetch indicator config from position if exists
        ind_cfg = self.positions.get(symbol, {}).get('indicator_config') if symbol in self.positions else None

        atr_multi_tf = {}
        for tf in settings.OHLCV_TIMEFRAMES:
            ind = multi_tf_indicators.get(tf, {})
            tf_atr = ind.get('atr')
            if tf_atr is not None and tf_atr > 0:
                atr_multi_tf[tf] = tf_atr

        atr_percentile = None
        if atr is not None and atr > 0:
            atr_percentile_key = f"atr_percentile:{symbol}"
            try:
                stored_atr = await asyncio.to_thread(self.redis.get, atr_percentile_key)
                if stored_atr:
                    atr_history = json.loads(stored_atr)
                else:
                    atr_history = []
                if len(atr_history) >= 5:
                    sorted_atr = sorted(atr_history)
                    rank = sum(1 for v in sorted_atr if v <= atr)
                    atr_percentile = round(rank / len(sorted_atr) * 100, 1)
            except Exception:
                pass

        market_regime = await self._classify_market_regime(
            adx=adx, plus_di=plus_di, minus_di=minus_di, ema_9=ema_9, ema_21=ema_21,
            bb_upper=bb_upper, bb_lower=bb_lower, bb_middle=bb_middle,
            atr=atr, atr_percentile=atr_percentile, current_price=current_price
        )

        raw_candles = multi_tf_raw_candles.get(assigned_tf)
        historical_ohlcv = None
        try:
            since_ms = int(time.time() * 1000) - settings.OHLCV_RETENTION_DAYS * 24 * 60 * 60 * 1000
            hist_limit = int((settings.OHLCV_RETENTION_DAYS * 86400) / tf_seconds) + 100
            db_candles = await asyncio.to_thread(get_ohlcv, symbol, assigned_tf, since_ms=since_ms, limit=hist_limit)
            if db_candles:
                historical_ohlcv = [[c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in db_candles]
        except Exception:
            pass

        perf = await asyncio.to_thread(self._compute_performance_metrics)
        trade_pattern_analysis = await asyncio.to_thread(self._compute_trade_pattern_analysis)
        symbol_event = None
        if settings.NEWS_ENABLED and detect_upcoming_events is not None:
            try:
                symbol_event = await asyncio.to_thread(detect_upcoming_events, symbol)
            except Exception:
                pass

        aggregate_sentiment = None
        if settings.NEWS_ENABLED:
            try:
                aggregate_sentiment = await self._get_cached_sentiment(symbol)
            except Exception:
                pass

        sentiment_trend_val = None
        if aggregate_sentiment:
            base_symbol = symbol.split("/")[0]
            current_compound = aggregate_sentiment.get("avg_compound")
            prev_key = f"sentiment:prev:{base_symbol}"
            prev_raw = await asyncio.to_thread(self.redis.get, prev_key)
            prev_compound = float(prev_raw) if prev_raw else None
            if current_compound is not None and prev_compound is not None:
                sentiment_trend_val = round(current_compound - prev_compound, 4)

        volume_trend_val = None
        current_volume = ticker.get('quoteVolume', 0) or 0
        if current_volume > 0:
            volume_trend_val = await self._compute_volume_trend(symbol, current_volume, timeframe=assigned_tf)

        full_market_breadth = None
        try:
            full_breadth_raw = await asyncio.to_thread(self.redis.get, "market:breadth:full")
            if full_breadth_raw:
                full_market_breadth = json.loads(full_breadth_raw)
        except Exception:
            pass
        session_info = self._get_session_info()

        now_rome = datetime.now(timezone.utc).astimezone(ZoneInfo(settings.MARKET_TIMEZONE))
        weekday = now_rome.weekday()
        if weekday < 5:
            rome_minutes = now_rome.hour * 60 + now_rome.minute
            close_minutes = settings.MARKET_CLOSE_HOUR * 60 + settings.MARKET_CLOSE_MINUTE
            minutes_to_market_close = close_minutes - rome_minutes
            if minutes_to_market_close < 0: minutes_to_market_close = 0
        else:
            minutes_to_market_close = None

        global_risk_mult = await self._get_global_risk_multiplier()

        max_port_exp = max_port_risk = None
        try:
            raw = await asyncio.to_thread(self.redis.get, "trading:max_portfolio_exposure_pct")
            if raw: max_port_exp = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:max_portfolio_stop_risk_pct")
            if raw: max_port_risk = float(raw)
        except: pass

        min_viable_amount = settings.MIN_VIABLE_TRADE_AMOUNT
        try:
            raw = await asyncio.to_thread(self.redis.get, "trading:min_viable_trade_amount")
            if raw: min_viable_amount = float(raw)
        except: pass

        sim_min_hold_time_mult = 1.0
        sim_min_stop_atr_mult = 1.0
        try:
            raw = await asyncio.to_thread(self.redis.get, "trading:min_max_hold_time_mult")
            if raw: sim_min_hold_time_mult = float(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:min_stop_loss_atr_mult")
            if raw: sim_min_stop_atr_mult = float(raw)
        except: pass

        # --- Emulate _process_symbol context ---
        open_positions = [pos for pos in self.positions.values() if pos.get("symbol") == symbol]
        position_info = self.positions.get(symbol)
        unrealized_pnl = None
        if position_info:
            unrealized_pnl = (current_price - position_info['price']) * position_info['amount']

        _portfolio = await self._compute_portfolio_exposure_summary(base_balance)
        portfolio_total_value = _portfolio["portfolio_total_value"]
        portfolio_exposure = _portfolio["portfolio_exposure"]
        portfolio_stop_risk = _portfolio["portfolio_stop_risk"]
        portfolio_exposure_pct = _portfolio["portfolio_exposure_pct"]
        portfolio_stop_risk_pct = _portfolio["portfolio_stop_risk_pct"]
        portfolio_available_capital = _portfolio["portfolio_available_capital"]

        recent_trades = [t for t in self.trade_history if t.get("side") == "sell"][-5:]
        recent_trades_summary = [
            {
                "symbol": t["symbol"],
                "realized_pnl": t.get("realized_pnl", 0.0),
                "strategy": t.get("strategy_type", "unknown"),
            }
            for t in recent_trades
        ]

        past_trades = [t for t in self.trade_history if t.get("symbol") == symbol and t.get("side") == "sell"][-10:]

        historical_backtest_results = await asyncio.to_thread(
            get_backtest_results_for_symbol, symbol, assigned_tf, 10
        )

        try:
            asset = await self._get_asset_info(symbol)
            min_order_amount = float(asset.min_order_size) if asset.min_order_size else None
        except Exception:
            min_order_amount = None
        if min_order_amount is not None and current_price:
            min_order_cost = min_order_amount * current_price
        else:
            min_order_cost = None

        max_sl_reviews = settings.MAX_STOP_LOSS_REVIEWS
        max_tp_reviews = settings.MAX_TAKE_PROFIT_REVIEWS
        max_partial_tp_reviews = settings.MAX_PARTIAL_TP_REVIEWS
        max_dust_sweep_reviews = settings.MAX_DUST_SWEEP_REVIEWS
        try:
            raw = await asyncio.to_thread(self.redis.get, "trading:max_stop_loss_reviews")
            if raw: max_sl_reviews = int(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:max_take_profit_reviews")
            if raw: max_tp_reviews = int(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:max_partial_tp_reviews")
            if raw: max_partial_tp_reviews = int(raw)
            raw = await asyncio.to_thread(self.redis.get, "trading:max_dust_sweep_reviews")
            if raw: max_dust_sweep_reviews = int(raw)
        except Exception:
            pass

        # Scale stop-loss review limit for long-term timeframes (same as _process_symbol)
        if tf_seconds >= settings.LONG_TERM_TF_SECONDS:  # >= 1 month
            max_sl_reviews = min(max_sl_reviews, settings.LONG_TERM_MAX_STOP_LOSS_REVIEWS)
        elif tf_seconds >= 604_800:  # >= 1 week
            max_sl_reviews = min(max_sl_reviews, settings.WEEKLY_MAX_STOP_LOSS_REVIEWS)

        trading_paused = False  # Force False for simulation

        max_hold_expired = False
        max_hold_expired_count = 0
        stop_loss_triggered = False
        stop_loss_review_count = 0
        take_profit_triggered = False
        take_profit_review_count = 0
        partial_tp_triggered = False
        partial_tp_review_count = 0
        partial_tp_triggered_levels = []
        dust_sweep_triggered = False
        dust_sweep_review_count = 0
        if symbol in self.positions:
            pos = self.positions[symbol]
            max_hold_expired = pos.get("_max_hold_expired", False)
            max_hold_expired_count = pos.get("_max_hold_expired_count", 0)
            stop_loss_triggered = pos.get("_stop_loss_triggered", False)
            stop_loss_review_count = pos.get("_stop_loss_review_count", 0)
            take_profit_triggered = pos.get("_take_profit_triggered", False)
            take_profit_review_count = pos.get("_take_profit_review_count", 0)
            partial_tp_triggered = pos.get("_partial_tp_triggered", False) or pos.get("_partial_tp_triggered_single", False)
            partial_tp_review_count = pos.get("_partial_tp_review_count", 0) or pos.get("_partial_tp_single_review_count", 0)
            partial_tp_triggered_levels = pos.get("_partial_tp_triggered_levels", [])
            dust_sweep_triggered = pos.get("_dust_sweep_triggered", False)
            dust_sweep_review_count = pos.get("_dust_sweep_review_count", 0)

        partial_tp_executed_levels = self.positions[symbol].get("partial_tp_levels_triggered", []) if symbol in self.positions else []

        remaining = max(0.0, base_balance - self._cycle_spent)

        prompt = await asyncio.to_thread(
            build_analysis_prompt,
            symbol=symbol, ticker=ticker, balance=balance, open_positions=open_positions,
            per_symbol_budget=per_symbol_budget, max_symbols=self.effective_max_symbols,
            base_currency=self.base_currency, performance=perf, ohlcv_data=ohlcv_data,
            assigned_timeframe=assigned_tf, atr=atr, atr_multi_tf=atr_multi_tf, rsi=rsi,
            macd=macd, macd_signal=macd_signal, macd_hist=macd_hist, bb_upper=bb_upper,
            bb_middle=bb_middle, bb_lower=bb_lower, ema_9=ema_9, ema_21=ema_21,
            stochastic_k=stochastic_k, stochastic_d=stochastic_d, adx=adx, plus_di=plus_di,
            minus_di=minus_di, obv=obv, mfi=mfi, cci=cci, williams_r=williams_r,
            unrealized_pnl=unrealized_pnl, position_info=position_info,
            drawdown_pct=perf.get("equity_curve", {}).get("drawdown_pct"),
            raw_candles=raw_candles, recent_trades=recent_trades_summary, historical_ohlcv=historical_ohlcv,
            min_order_amount=min_order_amount, min_order_cost=min_order_cost, all_symbols=self.current_symbols,
            past_trades=past_trades, cycle_spent=self._cycle_spent, remaining_balance=remaining,
            market_regime=market_regime, multi_tf_raw_candles=multi_tf_raw_candles,
            multi_tf_indicators=multi_tf_indicators, session_info=session_info,
            sentiment_trend=sentiment_trend_val, volume_trend=volume_trend_val,
            ichimoku=ichimoku, market_breadth=getattr(self, '_market_breadth', None),
            full_market_breadth=full_market_breadth, parabolic_sar=parabolic_sar,
            keltner_channels=keltner_channels, donchian_channels=donchian_channels,
            atr_percentile=atr_percentile, global_risk_multiplier=global_risk_mult,
            trading_paused=trading_paused, max_hold_expired=max_hold_expired, max_hold_expired_count=max_hold_expired_count,
            stop_loss_triggered=stop_loss_triggered, stop_loss_review_count=stop_loss_review_count,
            take_profit_triggered=take_profit_triggered, take_profit_review_count=take_profit_review_count,
            partial_tp_triggered=partial_tp_triggered, partial_tp_review_count=partial_tp_review_count,
            partial_tp_triggered_levels=partial_tp_triggered_levels if partial_tp_triggered_levels else None,
            partial_tp_executed_levels=partial_tp_executed_levels,
            dust_sweep_triggered=dust_sweep_triggered, dust_sweep_review_count=dust_sweep_review_count,
            max_stop_loss_reviews=max_sl_reviews, max_take_profit_reviews=max_tp_reviews,
            max_partial_tp_reviews=max_partial_tp_reviews, max_dust_sweep_reviews=max_dust_sweep_reviews,
            portfolio_exposure_pct=portfolio_exposure_pct, portfolio_stop_risk_pct=portfolio_stop_risk_pct,
            portfolio_total_value=portfolio_total_value, portfolio_open_count=len(self.positions),
            portfolio_available_capital=portfolio_available_capital, last_decision=self._last_decisions.get(symbol),
            minutes_to_market_close=minutes_to_market_close,
            current_strategy_interval_seconds=self._strategy_intervals.get(symbol, tf_seconds),
            max_portfolio_exposure_pct=max_port_exp, max_portfolio_stop_risk_pct=max_port_risk,
            trade_pattern_analysis=trade_pattern_analysis, symbol_event=symbol_event,
            queued_orders=self.queued_orders, fundamentals=fundamentals, vwap=vwap,
            daily_pivot_points=daily_pivot_points,
            min_hold_time_mult=sim_min_hold_time_mult,
            min_stop_atr_mult=sim_min_stop_atr_mult,
            min_viable_trade_amount=min_viable_amount,
            historical_backtest_results=historical_backtest_results,
        )
        # Add quote staleness warning if the price data is outdated
        staleness_warning = self._get_quote_staleness_warning(ticker)
        if staleness_warning:
            prompt += staleness_warning

        # Compute complexity and model tier for perfect emulation
        _conflicting = False
        if rsi is not None and macd_hist is not None:
            if (rsi < 30 and macd_hist < 0) or (rsi > 70 and macd_hist > 0):
                _conflicting = True
        strategy_complexity = self._compute_prompt_complexity(
            num_candidates=len(self.current_symbols),
            volatility_percentile=atr_percentile,
            rsi=rsi, macd=macd, macd_signal=macd_signal, macd_hist=macd_hist,
            bb_upper=bb_upper, bb_middle=bb_middle, bb_lower=bb_lower,
            ema_9=ema_9, ema_21=ema_21, stochastic_k=stochastic_k,
            adx=adx, plus_di=plus_di, minus_di=minus_di,
            mfi=mfi, cci=cci, williams_r=williams_r, ichimoku=ichimoku,
            market_breadth=getattr(self, '_market_breadth', None),
            full_market_breadth=full_market_breadth,
            sentiment_trend_magnitude=abs(sentiment_trend_val) if sentiment_trend_val is not None else None,
            volume_trend=volume_trend_val, market_regime=market_regime,
            unrealized_pnl=unrealized_pnl,
            drawdown_pct=perf.get("equity_curve", {}).get("drawdown_pct"),
            portfolio_exposure_pct=portfolio_exposure_pct,
            portfolio_stop_risk_pct=portfolio_stop_risk_pct,
            is_critical=(max_hold_expired or stop_loss_triggered or take_profit_triggered or partial_tp_triggered or dust_sweep_triggered),
            trading_paused=trading_paused, symbol_event=symbol_event, fundamentals=fundamentals,
            consecutive_losses=perf.get("equity_curve", {}).get("consecutive_losses", 0),
            current_price=current_price, conflicting_signals=_conflicting,
        )
        strategy_model_type = self._choose_model_tier(
            atr=atr, atr_percentile=atr_percentile, rsi=rsi, macd=macd, macd_signal=macd_signal, macd_hist=macd_hist,
            bb_upper=bb_upper, bb_middle=bb_middle, bb_lower=bb_lower, ema_9=ema_9, ema_21=ema_21,
            stochastic_k=stochastic_k, adx=adx, plus_di=plus_di, minus_di=minus_di,
            mfi=mfi, cci=cci, williams_r=williams_r, ichimoku=ichimoku,
            market_regime=market_regime, market_breadth=getattr(self, '_market_breadth', None),
            full_market_breadth=full_market_breadth, sentiment_trend_val=sentiment_trend_val,
            volume_trend=volume_trend_val, unrealized_pnl=unrealized_pnl,
            drawdown_pct=perf.get("equity_curve", {}).get("drawdown_pct"),
            portfolio_exposure_pct=portfolio_exposure_pct,
            portfolio_stop_risk_pct=portfolio_stop_risk_pct,
            is_critical=(max_hold_expired or stop_loss_triggered or take_profit_triggered or partial_tp_triggered or dust_sweep_triggered),
            trading_paused=trading_paused, symbol_event=symbol_event, fundamentals=fundamentals,
            consecutive_losses=perf.get("equity_curve", {}).get("consecutive_losses", 0),
            current_price=current_price,
        )
        effective_temp = self._get_effective_temperature(strategy_model_type, strategy_complexity)

        # --- Build market snapshot for caching (identical to _process_symbol) ---
        market_snapshot = {
            "symbol": symbol,
            "ticker": ticker,
            "balance": balance,
            "open_positions": open_positions,
            "per_symbol_budget": per_symbol_budget,
            "max_symbols": self.effective_max_symbols,
            "performance": perf,
            "ohlcv_data": ohlcv_data,
            "assigned_timeframe": assigned_tf,
            "atr": atr,
            "atr_multi_tf": atr_multi_tf,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "ema_9": ema_9,
            "ema_21": ema_21,
            "stochastic_k": stochastic_k,
            "stochastic_d": stochastic_d,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "obv": obv,
            "mfi": mfi,
            "cci": cci,
            "williams_r": williams_r,
            "ichimoku": ichimoku,
            "donchian_channels": donchian_channels,
            "drawdown_pct": perf.get("equity_curve", {}).get("drawdown_pct"),
            "raw_candles": raw_candles,
            "recent_trades": recent_trades_summary,
            "historical_ohlcv": historical_ohlcv,
            "min_order_amount": min_order_amount,
            "min_order_cost": min_order_cost,
            "all_symbols": self.current_symbols,
            "past_trades": past_trades,
            "aggregate_sentiment": aggregate_sentiment,
            "cycle_spent": self._cycle_spent,
            "remaining_balance": remaining,
            "market_regime": market_regime,
            "multi_tf_raw_candles": multi_tf_raw_candles,
            "multi_tf_indicators": multi_tf_indicators,
            "session_info": session_info,
            "sentiment_trend": sentiment_trend_val,
            "volume_trend": volume_trend_val,
            "market_breadth": getattr(self, '_market_breadth', None),
            "full_market_breadth": full_market_breadth,
            "parabolic_sar": parabolic_sar,
            "keltner_channels": keltner_channels,
            "atr_percentile": atr_percentile,
            "global_risk_multiplier": global_risk_mult,
            "trading_paused": trading_paused,  # False for simulation
            "last_decision": self._last_decisions.get(symbol),
        }
        market_hash = compute_market_hash(market_snapshot)

        return {
            "ticker": ticker, "analysis_prompt": prompt, "atr": atr, "assigned_tf": assigned_tf,
            "tf_seconds": tf_seconds, "historical_ohlcv": historical_ohlcv,
            "raw_candles": raw_candles, "current_price": current_price,
            "base_balance": base_balance, "is_btp": is_btp,
            "model_type": strategy_model_type, "temperature": effective_temp,
            "market_hash": market_hash,
            "per_symbol_budget": per_symbol_budget,
            "min_order_amount": min_order_amount,
            "min_order_cost": min_order_cost,
            "remaining_balance": remaining,
            "portfolio_total_value": portfolio_total_value,
            "portfolio_exposure_pct": portfolio_exposure_pct,
            "portfolio_stop_risk_pct": portfolio_stop_risk_pct,
            "portfolio_available_capital": portfolio_available_capital,
            "max_portfolio_exposure_pct": max_port_exp,
            "max_portfolio_stop_risk_pct": max_port_risk,
            "global_risk_multiplier": global_risk_mult,
            "min_stop_atr_mult": sim_min_stop_atr_mult,
            "min_hold_time_mult": sim_min_hold_time_mult,
            "has_position": symbol in self.positions,
            "historical_backtest_results": historical_backtest_results,
        }

    async def simulate_backtest(self, symbol: str) -> Dict[str, Any]:
        """Simulate Step 1a (analysis), Step 1b (variants), and run backtest without executing trades."""
        data = await self._prepare_simulation_data(symbol)
        if "error" in data:
            return data

        model_type = data.get("model_type", "mind")
        temperature = data.get("temperature", 0.2)
        market_hash = data.get("market_hash")

        # Step 1a: Analysis
        try:
            step1a_result = await asyncio.wait_for(
                asyncio.to_thread(
                    get_cached_llm_response,
                    compact_prompt(data["analysis_prompt"]),
                    COMPACTED_SYSTEM_PROMPT, 60,
                    market_hash=market_hash,
                    model_type=model_type,
                    temperature=temperature,
                ),
                timeout=settings.LLM_TIMEOUT
            )
            step1a_response = step1a_result["response"]
        except Exception as e:
            return {"error": f"LLM Step 1a call failed: {e}"}

        analysis = self._parse_analysis_response(step1a_response)
        if analysis is None:
            return {"error": "Failed to parse Step 1a analysis response", "raw_response": step1a_response}

        # Step 1b: Backtest variants
        variants_prompt = await asyncio.to_thread(
            build_backtest_variants_prompt,
            symbol=symbol,
            analysis=analysis,
            ticker=data["ticker"],
            current_price=data["current_price"],
            atr=data["atr"],
            assigned_timeframe=data["assigned_tf"],
            base_currency=self.base_currency,
            base_balance=data["base_balance"],
            per_symbol_budget=data["per_symbol_budget"],
            min_order_amount=data.get("min_order_amount"),
            min_order_cost=data.get("min_order_cost"),
            remaining_balance=data.get("remaining_balance"),
            portfolio_total_value=data.get("portfolio_total_value"),
            portfolio_exposure_pct=data.get("portfolio_exposure_pct"),
            portfolio_stop_risk_pct=data.get("portfolio_stop_risk_pct"),
            portfolio_available_capital=data.get("portfolio_available_capital"),
            max_portfolio_exposure_pct=data.get("max_portfolio_exposure_pct"),
            max_portfolio_stop_risk_pct=data.get("max_portfolio_stop_risk_pct"),
            global_risk_multiplier=data.get("global_risk_multiplier"),
            min_stop_atr_mult=data.get("min_stop_atr_mult", 1.0),
            min_hold_time_mult=data.get("min_hold_time_mult", 1.0),
            trading_paused=False,
            has_position=data.get("has_position", False),
            historical_backtest_results=data.get("historical_backtest_results"),
        )

        try:
            step1b_result = await asyncio.wait_for(
                asyncio.to_thread(
                    get_cached_llm_response,
                    compact_prompt(variants_prompt),
                    COMPACTED_SYSTEM_PROMPT, 60,
                    market_hash=compute_market_hash({"step": "1b", "analysis": analysis}),
                    model_type=model_type,
                    temperature=temperature,
                ),
                timeout=settings.LLM_TIMEOUT
            )
            step1b_response = step1b_result["response"]
        except Exception as e:
            return {"error": f"LLM Step 1b call failed: {e}"}

        try:
            preliminary_strategy = create_strategy_from_llm(step1b_response)
            preliminary_signal = preliminary_strategy.generate_signal({})
        except ValueError as e:
            return {"error": f"Failed to parse Step 1b response: {e}", "raw_response": step1b_response}

        if preliminary_signal.action in ("BUY", "HOLD"):
            # Determine which variant param sets to backtest
            variants_to_test = []
            if preliminary_signal.backtest_variants:
                variants_to_test = list(preliminary_signal.backtest_variants)
            else:
                variants_to_test.append(preliminary_signal.strategy_params or {})
            # --- Deduplicate variants with identical key risk parameters ---
            variants_to_test = self._deduplicate_variants(variants_to_test)
            # Safety cap: limit to configured max variants to prevent excessive backtest time
            if len(variants_to_test) > settings.MAX_BACKTEST_VARIANTS:
                logger.warning(
                    f"LLM returned {len(variants_to_test)} backtest variants for {symbol}, "
                    f"capping to {settings.MAX_BACKTEST_VARIANTS}"
                )
                variants_to_test = variants_to_test[:settings.MAX_BACKTEST_VARIANTS]

            # Limit number of variants based on available data length
            source_candles = data.get("historical_ohlcv") or data.get("raw_candles") or []
            if source_candles and len(source_candles) < 50:
                variants_to_test = variants_to_test[:2]
            elif source_candles and len(source_candles) < 100:
                variants_to_test = variants_to_test[:3]

            async def _sim_run_variant(vp: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    bt_stats, bt_summary = await self._run_backtest_variant(
                        symbol=symbol,
                        variant_params=vp,
                        preliminary_signal=preliminary_signal,
                        atr=data["atr"],
                        current_price=data["current_price"],
                        tf_secs=data["tf_seconds"],
                        assigned_tf=data["assigned_tf"],
                        historical_ohlcv=data["historical_ohlcv"],
                        raw_candles=data["raw_candles"],
                        base_balance=data["base_balance"],
                        is_btp=data["is_btp"],
                    )
                    if bt_stats is not None:
                        return {"variant_params": vp, "summary": bt_summary, "stats": bt_stats}
                    else:
                        return {"variant_params": vp, "summary": bt_summary or "Insufficient data for backtest.", "stats": {}}
                except Exception as e:
                    logger.warning(f"Backtest variant failed for {symbol}: {e}")
                    return {"variant_params": vp, "summary": f"Backtest error: {e}", "stats": {}}

            backtest_results = list(await asyncio.gather(*[_sim_run_variant(vp) for vp in variants_to_test]))

            combined_bt_summary = " | ".join(
                f"V{i+1}: {r['summary']}" for i, r in enumerate(backtest_results)
            ) if backtest_results else "No backtest performed"

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

        # Step 1a: Analysis
        try:
            step1a_result = await asyncio.wait_for(
                asyncio.to_thread(
                    get_cached_llm_response,
                    compact_prompt(data["analysis_prompt"]),
                    COMPACTED_SYSTEM_PROMPT, 60,
                    market_hash=market_hash,
                    model_type=model_type,
                    temperature=temperature,
                ),
                timeout=settings.LLM_TIMEOUT
            )
            step1a_response = step1a_result["response"]
        except Exception as e:
            return {"error": f"LLM Step 1a call failed: {e}"}

        analysis = self._parse_analysis_response(step1a_response)
        if analysis is None:
            return {"error": "Failed to parse Step 1a analysis response", "raw_response": step1a_response}

        # Step 1b: Backtest variants
        variants_prompt = await asyncio.to_thread(
            build_backtest_variants_prompt,
            symbol=symbol,
            analysis=analysis,
            ticker=data["ticker"],
            current_price=data["current_price"],
            atr=data["atr"],
            assigned_timeframe=data["assigned_tf"],
            base_currency=self.base_currency,
            base_balance=data["base_balance"],
            per_symbol_budget=data["per_symbol_budget"],
            min_order_amount=data.get("min_order_amount"),
            min_order_cost=data.get("min_order_cost"),
            remaining_balance=data.get("remaining_balance"),
            portfolio_total_value=data.get("portfolio_total_value"),
            portfolio_exposure_pct=data.get("portfolio_exposure_pct"),
            portfolio_stop_risk_pct=data.get("portfolio_stop_risk_pct"),
            portfolio_available_capital=data.get("portfolio_available_capital"),
            max_portfolio_exposure_pct=data.get("max_portfolio_exposure_pct"),
            max_portfolio_stop_risk_pct=data.get("max_portfolio_stop_risk_pct"),
            global_risk_multiplier=data.get("global_risk_multiplier"),
            min_stop_atr_mult=data.get("min_stop_atr_mult", 1.0),
            min_hold_time_mult=data.get("min_hold_time_mult", 1.0),
            trading_paused=False,
            has_position=data.get("has_position", False),
            historical_backtest_results=data.get("historical_backtest_results"),
        )

        try:
            step1b_result = await asyncio.wait_for(
                asyncio.to_thread(
                    get_cached_llm_response,
                    compact_prompt(variants_prompt),
                    COMPACTED_SYSTEM_PROMPT, 60,
                    market_hash=compute_market_hash({"step": "1b", "analysis": analysis}),
                    model_type=model_type,
                    temperature=temperature,
                ),
                timeout=settings.LLM_TIMEOUT
            )
            step1b_response = step1b_result["response"]
        except Exception as e:
            return {"error": f"LLM Step 1b call failed: {e}"}

        try:
            preliminary_strategy = create_strategy_from_llm(step1b_response)
            preliminary_signal = preliminary_strategy.generate_signal({})
        except ValueError as e:
            return {"error": f"Failed to parse Step 1b response: {e}", "raw_response": step1b_response}

        if preliminary_signal.action == "SELL":
            return {
                "step1_response": step1b_response,
                "step2_response": "N/A (Step 1 action is SELL)",
                "action": preliminary_signal.action,
                "backtest_summary": "No backtest performed (action is SELL)",
            }

        # Determine which variant param sets to backtest
        variants_to_test = []
        if preliminary_signal.backtest_variants:
            variants_to_test = list(preliminary_signal.backtest_variants)
        else:
            variants_to_test.append(preliminary_signal.strategy_params or {})
        # --- Deduplicate variants with identical key risk parameters ---
        variants_to_test = self._deduplicate_variants(variants_to_test)
        # Safety cap: limit to configured max variants to prevent excessive backtest time
        if len(variants_to_test) > settings.MAX_BACKTEST_VARIANTS:
            logger.warning(
                f"LLM returned {len(variants_to_test)} backtest variants for {symbol}, "
                f"capping to {settings.MAX_BACKTEST_VARIANTS}"
            )
            variants_to_test = variants_to_test[:settings.MAX_BACKTEST_VARIANTS]

        # Limit number of variants based on available data length
        source_candles = data.get("historical_ohlcv") or data.get("raw_candles") or []
        if source_candles and len(source_candles) < 50:
            variants_to_test = variants_to_test[:2]
        elif source_candles and len(source_candles) < 100:
            variants_to_test = variants_to_test[:3]

        async def _sim_run_variant(vp: Dict[str, Any]) -> Dict[str, Any]:
            try:
                bt_stats, bt_summary = await self._run_backtest_variant(
                    symbol=symbol,
                    variant_params=vp,
                    preliminary_signal=preliminary_signal,
                    atr=data["atr"],
                    current_price=data["current_price"],
                    tf_secs=data["tf_seconds"],
                    assigned_tf=data["assigned_tf"],
                    historical_ohlcv=data["historical_ohlcv"],
                    raw_candles=data["raw_candles"],
                    base_balance=data["base_balance"],
                    is_btp=data["is_btp"],
                )
                if bt_stats is not None:
                    return {"variant_params": vp, "summary": bt_summary, "stats": bt_stats}
                else:
                    return {"variant_params": vp, "summary": bt_summary or "Insufficient data for backtest.", "stats": {}}
            except Exception as e:
                logger.warning(f"Backtest variant failed for {symbol}: {e}")
                return {"variant_params": vp, "summary": f"Backtest error: {e}", "stats": {}}

        backtest_results = list(await asyncio.gather(*[_sim_run_variant(vp) for vp in variants_to_test]))

        combined_bt_summary = " | ".join(
            f"V{i+1}: {r['summary']}" for i, r in enumerate(backtest_results)
        ) if backtest_results else "No backtest performed"

        total_variants_proposed = len(preliminary_signal.backtest_variants) if preliminary_signal.backtest_variants else 1
        step2_prompt = build_final_decision_prompt(
            symbol=symbol,
            ticker=data["ticker"],
            preliminary_decision={
                "action": preliminary_signal.action,
                "confidence": preliminary_signal.confidence,
                "reasoning": preliminary_signal.reasoning,
                "strategy_params": preliminary_signal.strategy_params,
                "timeframe": data["assigned_tf"],
            },
            backtest_results=backtest_results,
            base_currency=self.base_currency,
            trading_paused=False,
            total_variants_proposed=total_variants_proposed,
            historical_backtest_results=data.get("historical_backtest_results"),
        )
        
        # Append position info if exists
        if symbol in self.positions:
            pos = self.positions[symbol]
            step2_prompt += (
                f"\n**Existing Position:** You already hold {pos['amount']:.6f} "
                f"at entry {pos['price']:.4f}. A BUY will ADD to this position (scale in).\n"
            )

        try:
            step2_result = await asyncio.wait_for(
                asyncio.to_thread(
                    get_cached_llm_response,
                    compact_prompt(step2_prompt),
                    COMPACTED_SYSTEM_PROMPT,
                    60,
                    model_type=model_type,
                    temperature=temperature,
                ),
                timeout=settings.LLM_TIMEOUT
            )
            step2_response = step2_result["response"]
        except Exception as e:
            return {
                "step1_response": step1b_response,
                "error": f"LLM Step 2 call failed: {e}",
                "action": preliminary_signal.action,
                "backtest_summary": combined_bt_summary,
            }

        # Parse Step 2 response to get the final action
        try:
            final_strategy = create_strategy_from_llm(step2_response)
        except ValueError:
            # Retry with correction prompt
            logger.warning(f"Simulation Step 2 parse failed for {symbol}. Retrying.")
            correction = (
                "Your previous response was not valid JSON. "
                "Output ONLY a single JSON object. "
                "Here is the request:\n\n" + step2_prompt
            )
            try:
                retry_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        get_cached_llm_response,
                        compact_prompt(correction),
                        COMPACTED_SYSTEM_PROMPT, 30,
                        model_type="actuator",
                        temperature=temperature,
                    ),
                    timeout=settings.LLM_TIMEOUT
                )
                step2_response = retry_result["response"]
                final_strategy = create_strategy_from_llm(step2_response)
            except Exception as e2:
                return {
                    "step1_response": step1b_response,
                    "step2_response": step2_response,
                    "error": f"Failed to parse LLM Step 2 response after retry: {e2}",
                    "action": preliminary_signal.action,
                    "backtest_summary": combined_bt_summary,
                }

        final_signal = final_strategy.generate_signal({})

        return {
            "step1_response": step1b_response,
            "step2_response": step2_response,
            "action": final_signal.action,
            "backtest_summary": combined_bt_summary,
            "backtest_results": backtest_results,
        }
