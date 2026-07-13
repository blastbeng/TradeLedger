"""Market data management component for the TradingEngine.

Handles OHLCV downloads, gap filling, and indicator computation.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

from src.config.settings import settings
from src.database import get_ohlcv, save_indicators, get_symbol_name_from_db, save_discovered_symbol, get_latest_ohlcv_timestamp, insert_ohlcv_batch
from src.exchanges.market_data import get_tradable_assets, discover_btp_bonds, discover_italian_ucits_etfs, _check_yf_circuit, _get_yf_session, get_bars_range, get_quotes, get_quotes_cached
from src.indicators import compute_all_indicators

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


class MarketDataManager:
    """Handles market data downloads and indicator computation for the TradingEngine."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.shared_state = engine.shared_state
        self.event_bus = event_bus
        self.event_bus.subscribe("get_stock_name", self.get_stock_name)
        self.event_bus.subscribe("get_tradable_assets", self.get_tradable_assets)
        self.event_bus.subscribe("get_btp_bonds", self.get_btp_bonds)
        self.event_bus.subscribe("get_etf_symbols", self.get_etf_symbols)
        self.event_bus.subscribe("get_asset_info", self.get_asset_info)
        self.event_bus.subscribe("get_quotes_async", self._get_quotes_async)
        self.event_bus.subscribe("get_quotes_batched", self._get_quotes_batched)
        self.event_bus.subscribe("get_all_position_tickers", self._get_all_position_tickers)
        self.event_bus.subscribe("get_all_position_tickers_sync", self._get_all_position_tickers_sync)
        self.event_bus.subscribe("get_tickers_for_symbols_sync", self._get_tickers_for_symbols_sync)
        self.event_bus.subscribe("backfill_new_symbol", self._backfill_new_symbol)
        self.event_bus.subscribe("get_clock", self.get_clock)
        self.event_bus.subscribe("compute_and_store_indicators", self.compute_and_store_indicators)
        self._clock_cache: Optional[Any] = None
        self._clock_cache_time: float = 0.0

    def invalidate_clock_cache(self):
        self._clock_cache = None

    async def get_clock(self, ttl: float = 30.0) -> Optional[ClockInfo]:
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
            logger.error(f"Failed to get market clock from pandas_market_calendars: {type(e).__name__}: {e}")
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

    async def get_stock_name(self, symbol: str) -> str:
        """Return the human-readable company name for a symbol, cached in Redis.

        Uses yfinance to fetch the name.
        """
        engine = self.engine
        base = symbol.split("/")[0] if "/" in symbol else symbol

        if re.match(r'^IT[A-Z0-9]{10}$', base):
            # It's a BTP bond, try to get the name from the BTP cache (includes DB-merged BTPs)
            try:
                btp_bonds = await self.get_btp_bonds()
                for b in btp_bonds:
                    if b["isin"] == base:
                        name = b.get("name") or base
                        if name and name != b["isin"]:
                            return name
            except Exception as e:
                logger.debug(f"get_stock_name: BTP cache lookup failed for {base}: {type(e).__name__}: {e}")
            # Fallback: try DB directly
            try:
                db_name = await asyncio.to_thread(get_symbol_name_from_db, base)
                if db_name:
                    return db_name
            except Exception:
                pass

            # If we got a name from the BTP cache, save it to DB for future lookups
            try:
                btp_bonds = await self.get_btp_bonds()
                for b in btp_bonds:
                    if b["isin"] == base:
                        name = b.get("name") or base
                        if name and name != b["isin"]:
                            await asyncio.to_thread(
                                save_discovered_symbol, base, base, "btp", name,
                                country="italy"
                            )
                            return name
            except Exception as e:
                logger.debug(f"get_stock_name: failed to save name to DB for {base}: {type(e).__name__}: {e}")
            return base

        # Check Redis cache first
        cache_key = f"stock_name:{base}"
        try:
            cached = await asyncio.to_thread(engine.redis.get, cache_key)
            if cached:
                return cached.decode() if isinstance(cached, bytes) else cached
        except Exception:
            pass

        # Check discovered_symbols table (works even when yf circuit is open)
        try:
            db_name = await asyncio.to_thread(get_symbol_name_from_db, base)
            if db_name:
                try:
                    await asyncio.to_thread(engine.redis.setex, cache_key, 7 * 24 * 3600, db_name)
                except Exception as e:
                    logger.debug(f"get_stock_name: failed to cache name in Redis for {base}: {type(e).__name__}: {e}")
                return db_name
        except Exception as e:
            logger.debug(f"get_stock_name: DB name lookup failed for {base}: {type(e).__name__}: {e}")

        if _check_yf_circuit():
            return base

        try:
            def _fetch_yf_name():
                import yfinance as yf
                ticker = yf.Ticker(base, session=_get_yf_session())
                info = ticker.info
                return info.get("longName") or info.get("shortName") or base
            name = await asyncio.to_thread(_fetch_yf_name)
        except Exception as e:
            logger.debug(f"get_stock_name: yfinance fetch failed for {base}: {type(e).__name__}: {e}")
            name = base

        # If yfinance returned a name, save it to the DB for future use
        if name and name != base:
            try:
                db_base = base
                suffix = settings.TICKER_SUFFIX
                if suffix and db_base.endswith(suffix):
                    db_base = db_base[:-len(suffix)]
                save_discovered_symbol(db_base, None, None, name, country=None)
            except Exception as e:
                logger.debug(f"get_stock_name: failed to save name to DB for {base}: {type(e).__name__}: {e}")

        # Cache for 7 days (names rarely change)
        try:
            await asyncio.to_thread(engine.redis.setex, cache_key, 7 * 24 * 3600, name)
        except Exception as e:
            logger.debug(f"get_stock_name: failed to cache name in Redis for {base}: {type(e).__name__}: {e}")
        return name

    async def get_tradable_assets(self) -> List[str]:
        """Return tradable assets, cached for 5 minutes to reduce API calls."""
        engine = self.engine
        now = time.time()
        if engine._tradable_assets_cache and (now - engine._tradable_assets_cache_time) < 300:
            return engine._tradable_assets_cache
        async with engine._tradable_assets_lock:
            # Double-check cache after acquiring lock (another task may have populated it)
            now = time.time()
            if engine._tradable_assets_cache and (now - engine._tradable_assets_cache_time) < 300:
                return engine._tradable_assets_cache
            assets = await asyncio.to_thread(get_tradable_assets)
            engine._tradable_assets_cache = assets
            engine._tradable_assets_cache_time = now
            return assets

    async def get_btp_bonds(self) -> List[Dict[str, Any]]:
        """Return BTP bonds, cached for 30 minutes to reduce scraping calls."""
        engine = self.engine
        now = time.time()
        if engine._btp_bonds_cache and (now - engine._btp_bonds_cache_time) < 1800:
            return engine._btp_bonds_cache
        async with engine._tradable_assets_lock:
            # Double-check after lock
            now = time.time()
            if engine._btp_bonds_cache and (now - engine._btp_bonds_cache_time) < 1800:
                return engine._btp_bonds_cache
            bonds = await asyncio.to_thread(discover_btp_bonds)
            # Merge with DB-saved BTPs so nothing is lost between runs
            try:
                from src.database import get_all_discovered_symbols
                db_symbols = await asyncio.to_thread(get_all_discovered_symbols)
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
            engine._btp_bonds_cache = bonds
            engine._btp_bonds_cache_time = now
            return bonds

    async def get_etf_symbols(self) -> List[str]:
        """Return Italian UCITS ETF symbols, cached for 1 hour."""
        engine = self.engine
        now = time.time()
        if engine._etf_symbols_cache and (now - engine._etf_symbols_cache_time) < 3600:
            return engine._etf_symbols_cache
        async with engine._tradable_assets_lock:
            # Double-check after lock
            now = time.time()
            if engine._etf_symbols_cache and (now - engine._etf_symbols_cache_time) < 3600:
                return engine._etf_symbols_cache
            symbols = await asyncio.to_thread(discover_italian_ucits_etfs)
            engine._etf_symbols_cache = symbols
            engine._etf_symbols_cache_time = now
            return symbols

    async def get_asset_info(self, symbol: str) -> Any:
        """Return asset info (min order size, name, etc.), cached for 1 hour.

        Fetches from yfinance (subject to circuit breaker) with database
        fallback for the name. Returns permissive defaults only when no
        data is available.
        """
        engine = self.engine
        base = symbol.split("/")[0] if "/" in symbol else symbol
        now = time.time()
        if base in engine._asset_cache and (now - engine._asset_cache_time.get(base, 0)) < 3600:
            return engine._asset_cache[base]

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
                logger.warning(f"yfinance asset info fetch failed for {base}: {type(e).__name__}: {e}")

        # Database fallback for name
        if name == base:
            try:
                db_name = await asyncio.to_thread(get_symbol_name_from_db, base)
                if db_name:
                    name = db_name
            except Exception:
                pass

        # Default to permissive 0.0 when no minimum was found
        if min_order_size is None:
            min_order_size = 0.0

        asset = AssetInfo(name=name, min_order_size=min_order_size, fractionable=fractionable)
        engine._asset_cache[base] = asset
        engine._asset_cache_time[base] = now
        return asset

    async def _get_quotes_async(self, symbols: List[str], timeout: float = 45.0) -> Dict[str, Dict[str, Any]]:
        """Fetch quotes using the dedicated quote thread pool with a timeout.
        This prevents slow yfinance calls from blocking the default asyncio thread pool."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self.engine._quote_executor, get_quotes, symbols),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Quote fetch timed out for {len(symbols)} symbols")
            return {}
        except Exception as e:
            logger.warning(f"Quote fetch failed for {len(symbols)} symbols: {type(e).__name__}: {e}")
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

    async def _get_all_position_tickers(self) -> Dict[str, Dict[str, Any]]:
        """Fetch tickers for all open positions, batching missing ones into a single API call."""
        engine = self.engine
        self.shared_state._portfolio_exposure_cache = None
        tickers: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        for sym in self.shared_state.positions:
            missing.append(sym.split("/")[0])
        if missing:
            try:
                raw = await self._get_quotes_batched(missing, timeout_per_chunk=45.0)
                for sym in self.shared_state.positions:
                    base = sym.split("/")[0]
                    if base in raw:
                        tickers[sym] = raw[base]
            except Exception as e:
                logger.warning(f"Batch quote fetch failed for positions: {type(e).__name__}: {e}")
        return tickers

    def _get_all_position_tickers_sync(self) -> Dict[str, Dict[str, Any]]:
        """Fetch tickers for all open positions synchronously, batching missing ones.

        Uses get_quotes_cached (Redis/DB only, no network calls) to avoid
        blocking the default asyncio thread pool with slow yfinance requests.
        """
        engine = self.engine
        self.shared_state._portfolio_exposure_cache = None
        tickers: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        for sym in self.shared_state.positions:
            missing.append(sym.split("/")[0])
        if missing:
            try:
                raw = get_quotes_cached(missing)
                for sym in self.shared_state.positions:
                    base = sym.split("/")[0]
                    if base in raw:
                        tickers[sym] = raw[base]
            except Exception as e:
                logger.warning(f"Sync batch quote fetch failed for positions: {type(e).__name__}: {e}")
        return tickers

    def _get_tickers_for_symbols_sync(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch latest quotes for a list of symbols synchronously, batching missing ones.

        Uses get_quotes_cached (Redis/DB only, no network calls) to avoid
        blocking the default asyncio thread pool with slow yfinance requests.
        """
        engine = self.engine
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
                logger.warning(f"Sync batch quote fetch failed: {type(e).__name__}: {e}")
        return tickers

    async def _backfill_ohlcv(self, symbol: str, timeframe: str, start_ms: int, end_ms: int, max_candles: int = None, ignore_existing: bool = False, force: bool = False, quiet: bool = False) -> int:
        """Fetch and store all missing OHLCV candles between start_ms and end_ms.
        Returns the number of candles inserted."""
        logger.debug(f"Backfill started for {symbol} {timeframe}: {start_ms} → {end_ms}")
        if ignore_existing:
            since = start_ms
        else:
            loop = asyncio.get_running_loop()
            latest_ts = await loop.run_in_executor(self.engine._db_executor, get_latest_ohlcv_timestamp, symbol, timeframe)
            if latest_ts is None:
                since = start_ms
            else:
                # Skip the API call entirely if the latest candle is recent enough
                # (within one candle interval of now). No new data to fetch.
                if not force:
                    interval_ms = self.engine._timeframe_to_ms(timeframe)
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
                async with self.engine._download_semaphore:
                    loop = asyncio.get_running_loop()
                    candles = await loop.run_in_executor(
                        self.engine._download_executor,
                        get_bars_range,
                        symbol.split("/")[0], timeframe, since, 10000
                    )
            except Exception as e:
                logger.warning(f"get_bars_range failed for {symbol} {timeframe} at {since}: {e}")
                break

            if not candles:
                break

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.engine._db_executor, insert_ohlcv_batch, symbol, timeframe, candles)
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

    async def compute_and_store_indicators(self, symbol: str, timeframe: str, candles: List[List]) -> Optional[Dict[str, Any]]:
        """Compute indicators for a symbol/timeframe using TA-Lib and store in DB."""
        engine = self.engine
        if not candles or len(candles) < 2:
            return None
        try:
            async with engine._indicator_semaphore:
                loop = asyncio.get_running_loop()
                ind = await loop.run_in_executor(engine._download_executor, compute_all_indicators, candles)
            if ind:
                latest_ts = candles[-1][0]
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(engine._db_executor, save_indicators, symbol, timeframe, latest_ts, ind)
                logger.debug(f"Indicators computed and stored for {symbol} {timeframe}")
                return ind
            return None
        except Exception as e:
            logger.warning(f"Failed to compute/store indicators for {symbol} {timeframe}: {type(e).__name__}: {e}")
            return None
    async def _fill_gaps(self, symbol: str, timeframe: str):
        """Detect and fill gaps in stored OHLCV data for a symbol/timeframe."""
        engine = self.engine
        interval_ms = engine._timeframe_to_ms(timeframe)
        if interval_ms <= 0:
            return

        # Get all stored timestamps
        loop = asyncio.get_running_loop()
        candles = await loop.run_in_executor(engine._db_executor, get_ohlcv, symbol, timeframe, 50000)
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
        max_gaps_per_cycle = settings.MAX_GAPS_PER_CYCLE  # Limit gap fills per cycle to avoid rate limits

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
        engine = self.engine
        loop = asyncio.get_running_loop()
        try:
            inserted = await self._backfill_ohlcv(symbol, timeframe, start_ms, end_ms, quiet=quiet, force=force)
            if inserted > 0 or force:
                await self._fill_gaps(symbol, timeframe)
                db_candles = await loop.run_in_executor(engine._db_executor, get_ohlcv, symbol, timeframe, 200)
                if db_candles:
                    raw_candles = [[c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in db_candles]
                    await self.compute_and_store_indicators(symbol, timeframe, raw_candles)
        except Exception as e:
            logger.warning(f"Download failed for {symbol} {timeframe}: {type(e).__name__}: {e}")
