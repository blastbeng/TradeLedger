import asyncio
import hashlib
import logging
import re
import threading
import time
import warnings
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import json
import requests
import yfinance as yf
import httpx
from bs4 import BeautifulSoup

from src.config.settings import settings
from src.utils.redis_client import get_redis_client
from src.utils.btp_policy import BTPPolicy
from src.database import save_quotes_batch, get_quotes_from_db, get_latest_close_prices
from src.exchanges.proxy_utils import DynamicProxyRotator, _dynamic_rotator, _get_proxies
from src.exchanges.borsa_italiana_utils import (
    _check_bi_circuit,
    _record_bi_error,
    _reset_bi_circuit,
    _get_borsa_italiana_token,
    _invalidate_borsa_token_cache,
    _get_isin_and_info_from_borsa_italiana,
    get_borsa_italiana_quote,
    get_borsa_italiana_candles,
    BORSA_TIMEFRAME_MAP,
    _fetch_btp_details,
    discover_btp_bonds,
)
from src.exchanges.yf_session import (
    _yf_download_with_timeout,
    _check_yf_circuit,
    _record_yf_error,
    _reset_yf_circuit,
    _invalidate_yf_session,
    _get_yf_session,
    _yf_rate_limiter,
    YFinanceRateLimiter,
)
from src.exchanges.candle_utils import (
    _validate_and_clean_candles,
    _aggregate_candles,
    _merge_candles,
)
from src.exchanges.alphavantage_utils import (
    _av_rate_limiter,
    get_alphavantage_quote,
    get_alphavantage_candles,
)
from src.exchanges.iex_utils import (
    _iex_rate_limiter,
    get_iex_quote,
    get_iex_candles,
)
from src.exchanges.asset_discovery import (
    _fetch_info,
    _discover_wikipedia_tickers,
    _load_static_tickers,
    _get_hardcoded_tickers,
    _discover_financedatabase_tickers,
    discover_italian_ucits_etfs,
    _save_discovered_assets_to_db,
    get_tradable_assets,
    set_notifier,
)

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


class CircuitBreaker:
    """Generic circuit breaker for market data sources."""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

    def record_success(self):
        self.failure_count = 0

    def is_open(self) -> bool:
        if self.failure_count >= self.failure_threshold:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.failure_count = 0
                return False
            return True
        return False


_av_circuit_breaker = CircuitBreaker()
_iex_circuit_breaker = CircuitBreaker()

_get_quotes_lock = threading.Lock()

def _get_isin_from_yfinance(base_symbol: str) -> Optional[str]:
    """Fetch the ISIN code for a symbol, using DB first, then yfinance as fallback."""
    from src.database import get_isin_from_db, save_discovered_symbol

    # Strip suffix for DB lookup (DB stores base symbols without suffix)
    suffix = settings.TICKER_SUFFIX
    db_symbol = base_symbol
    if suffix and db_symbol.endswith(suffix):
        db_symbol = db_symbol[:-len(suffix)]

    # Check DB first (not Redis)
    cached = get_isin_from_db(db_symbol)
    if cached:
        # If strict country filter is enabled, ignore cached ISINs that don't match the target country
        if settings.COUNTRY_FILTER_STRICT and not cached.startswith(settings.TARGET_COUNTRY):
            logger.debug(f"DB has non-target ISIN {cached} for {db_symbol}, ignoring and re-fetching.")
        else:
            return cached

    isin = None
    # If yfinance circuit is open, we can't fetch the ISIN from yfinance.
    if not _check_yf_circuit():
        yf_symbol = f"{db_symbol}{suffix}" if suffix and not db_symbol.endswith(suffix) else db_symbol
        try:
            ticker = yf.Ticker(yf_symbol, session=_get_yf_session())
            isin = ticker.isin
            if isin:
                isin = isin.strip()
                if isin == '-' or not isin:
                    isin = None
                elif settings.COUNTRY_FILTER_STRICT and not isin.startswith(settings.TARGET_COUNTRY):
                    logger.debug(f"yfinance returned non-target ISIN {isin} for {base_symbol}, discarding.")
                    isin = None
        except (RuntimeError, ValueError, KeyError, AttributeError, OSError) as e:
            logger.debug(f"Failed to fetch ISIN for {base_symbol} from yfinance: {e}")
            isin = None

    # Fallback to Borsa Italiana search if yfinance failed or circuit is open
    if not isin:
        bi_isin, _, _ = _get_isin_and_info_from_borsa_italiana(db_symbol)
        if bi_isin:
            isin = bi_isin

    if isin:
        # Save to DB with the base symbol (no suffix)
        try:
            save_country = settings.TARGET_COUNTRY if settings.COUNTRY_FILTER_STRICT else None
            save_discovered_symbol(db_symbol, isin, "stock", None, country=save_country)
        except (RuntimeError, ValueError, OSError):
            pass

    return isin


# Map our timeframe strings to yfinance interval strings
TIMEFRAME_MAP = {
    "1h": "60m",
    "1d": "1d",
    "1w": "1wk",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "3Y": "3y",
    "5Y": "5y",
}

TIMEFRAME_MS = {
    "1h": 3600_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
    "3M": 7_776_000_000,
    "6M": 15_552_000_000,
    "1Y": 31_536_000_000,
    "3Y": 94_608_000_000,
    "5Y": 157_680_000_000,
}

def _enrich_quotes_with_btp_details(result: Dict[str, Dict[str, Any]], symbols: List[str]):
    """Enrich quote results with BTP maturity, coupon, and name from discovered_symbols."""
    btp_symbols = [s for s in symbols if BTPPolicy.is_btp(s)]
    if not btp_symbols:
        return
    try:
        from src.database import get_btp_details_from_db
        details = get_btp_details_from_db(btp_symbols)
        for sym in btp_symbols:
            if sym in result and details.get(sym):
                d = details[sym]
                if d.get("maturity"):
                    result[sym]["maturity"] = d["maturity"]
                if d.get("coupon") is not None:
                    result[sym]["coupon"] = d["coupon"]
                if not result[sym].get("name") and d.get("name"):
                    result[sym]["name"] = d["name"]
    except (RuntimeError, ValueError, KeyError, OSError) as e:
        logger.debug(f"Failed to enrich BTP details in quotes: {type(e).__name__}: {e}")


def _finalize_and_persist_quotes(
    result: Dict[str, Dict[str, Any]],
    symbols: List[str],
    redis_client
) -> None:
    """Finalize quote data (change_24h, bid/ask fallback), persist to Redis/DB, and enrich BTP details."""
    # --- Final pass: compute change_24h and percentage from DB daily candles ---
    symbols_with_price = [
        sym for sym in result
        if result[sym].get("last") is not None and result[sym]["last"] > 0
    ]
    if symbols_with_price:
        try:
            db_change_data = get_latest_close_prices(symbols_with_price)
            for sym in symbols_with_price:
                if sym in db_change_data:
                    prev_close = db_change_data[sym].get("prev_close")
                    if prev_close and prev_close > 0:
                        last = result[sym]["last"]
                        if result[sym].get("change_24h") is None:
                            result[sym]["change_24h"] = last - prev_close
                        if result[sym].get("percentage") is None:
                            result[sym]["percentage"] = round((last - prev_close) / prev_close * 100, 4)
        except (RuntimeError, ValueError, KeyError, OSError) as e:
            logger.warning(f"Failed to recompute change_24h/percentage from DB candles: {type(e).__name__}: {e}")

    # Validate quotes against existing DB data to prevent bad data overwriting good data
    if symbols_with_price:
        try:
            existing_quotes = get_quotes_from_db(symbols_with_price, max_age_seconds=86400)
            for sym in symbols_with_price:
                new_last = result[sym].get("last")
                new_update = result[sym].get("last_update")
                existing_last = existing_quotes.get(sym, {}).get("last")
                existing_update = existing_quotes.get(sym, {}).get("last_update")
                
                # If existing quote is newer, don't overwrite with stale data
                if existing_update and new_update and existing_update > new_update:
                    logger.debug(f"Quote validation: existing quote for {sym} is newer than new quote. Keeping existing.")
                    result[sym] = existing_quotes[sym]
                    continue

                if existing_last and existing_last > 0 and new_last and new_last > 0:
                    deviation = abs(new_last - existing_last) / existing_last
                    if deviation > 0.5:
                        # Only revert if the existing quote is recent (within 1 hour)
                        # to avoid blocking valid large price moves over longer periods
                        if existing_update and (int(time.time() * 1000) - existing_update < 3600 * 1000):
                            logger.warning(
                                f"Quote validation failed for {sym}: new price {new_last} deviates "
                                f"by {deviation*100:.2f}% from recent existing price {existing_last}. Reverting to existing."
                            )
                            result[sym] = existing_quotes[sym]
                        else:
                            logger.info(
                                f"Quote validation: large deviation for {sym} ({deviation*100:.2f}%) "
                                f"but existing quote is old. Accepting new price."
                            )
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to validate quotes against DB: {type(e).__name__}: {e}")

    # Ensure bid/ask are never NULL when last is available — use last as fallback
    for sym in result:
        if result[sym].get("last") is not None and result[sym]["last"] > 0:
            if result[sym].get("bid") is None:
                result[sym]["bid"] = result[sym]["last"]
            if result[sym].get("ask") is None:
                result[sym]["ask"] = result[sym]["last"]

    # Persist to Redis and database
    quotes_to_save = {}
    for sym, q in result.items():
        if q.get("last") is not None:
            try:
                redis_client.set(f"quote:{sym}", json.dumps(q), ex=300)
            except (TypeError, ValueError, RuntimeError):
                pass
            quotes_to_save[sym] = q
    if quotes_to_save:
        try:
            save_quotes_batch(quotes_to_save)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to save quotes to database: {type(e).__name__}: {e}")

    # Enrich BTP quotes with maturity, coupon, and name from discovered_symbols
    _enrich_quotes_with_btp_details(result, symbols)


def get_quotes(symbols: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """Fetch latest quotes for a list of symbols using yfinance batch download.

    Returns a dict mapping symbol -> {last, bid, ask, volume, change_24h, percentage, quoteVolume}.
    Uses yf.download for efficient batch fetching.
    A global lock ensures only one batch download runs at a time to prevent rate limits.
    """
    if symbols is None:
        symbols = []
    if not symbols:
        return {}

    # If a fetch is already in progress, try to fall back to cache.
    # If the cache is empty for any requested symbol, wait for the lock
    # to avoid returning empty quotes during startup.
    if not _get_quotes_lock.acquire(blocking=False):
        cached = get_quotes_cached(symbols)
        if any(cached.get(s, {}).get("last") is None for s in symbols):
            with _get_quotes_lock:
                # Re-check cache after acquiring lock, the previous fetch might have populated it
                cached = get_quotes_cached(symbols)
                if any(cached.get(s, {}).get("last") is None for s in symbols):
                    return _get_quotes_impl(symbols)
                return cached
        return cached

    try:
        return _get_quotes_impl(symbols)
    finally:
        _get_quotes_lock.release()


def _get_quotes_impl(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    # Sanitize symbols: remove $ prefix and /currency suffix
    symbols = [s.lstrip('$').split('/')[0] for s in symbols]

    redis_client = get_redis_client()
    result = {}
    missing_symbols = []

    # Check per-symbol Redis cache first
    for sym in symbols:
        cache_key = f"quote:{sym}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                result[sym] = json.loads(cached)
            else:
                missing_symbols.append(sym)
        except (TypeError, ValueError, RuntimeError):
            missing_symbols.append(sym)

    if not missing_symbols:
        return result

    # Check database for quotes not in Redis cache (up to 24 hours old)
    try:
        db_quotes = get_quotes_from_db(missing_symbols, max_age_seconds=86400)
        for sym in list(missing_symbols):
            if sym in db_quotes:
                result[sym] = db_quotes[sym]
                missing_symbols.remove(sym)
                # Refresh Redis cache from DB data
                try:
                    redis_client.set(f"quote:{sym}", json.dumps(db_quotes[sym]), ex=300)
                except (TypeError, ValueError, RuntimeError, ConnectionError, TimeoutError, OSError):
                    pass
        if db_quotes:
            logger.debug(f"Loaded {len(db_quotes)} quotes from database (Redis miss fallback)")
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning(f"DB quote fetch failed: {type(e).__name__}: {e}", exc_info=True)

    # --- Try DB close prices first (fast, no network call) ---
    # This ensures quotes are available even when yfinance is rate-limited or blocked.
    # The OHLCV data is populated by background download tasks using borsaitaliana as primary source.
    if missing_symbols:
        try:
            db_candles = get_latest_close_prices(missing_symbols)
            for sym in list(missing_symbols):
                if sym in db_candles and db_candles[sym].get("last", 0) > 0:
                    candle_ts = db_candles[sym].get("candle_timestamp")
                    if candle_ts and (int(time.time() * 1000) - candle_ts > 48 * 3600 * 1000):
                        logger.debug(f"get_quotes: DB close price for {sym} is stale (older than 48h), skipping.")
                        continue

                    last = db_candles[sym]["last"]
                    prev_close = db_candles[sym].get("prev_close")
                    volume = db_candles[sym].get("volume")

                    abs_change = None
                    pct = None
                    if prev_close and prev_close > 0:
                        abs_change = last - prev_close
                        pct = round((abs_change / prev_close) * 100, 4)

                    result[sym] = {
                        "last": last,
                        "bid": last,
                        "ask": last,
                        "volume": volume,
                        "change_24h": abs_change,
                        "percentage": pct,
                        "quoteVolume": volume,
                        "last_update": db_candles[sym].get("candle_timestamp"),
                        "source": "db_close",
                    }
                    # Do not remove from missing_symbols so yfinance can still try to update it
        except (RuntimeError, ValueError, KeyError, OSError) as e:
            logger.warning(f"get_quotes: DB close price fallback failed: {type(e).__name__}: {e}")

    # --- Try Borsa Italiana first for symbols with known ISINs ---
    # Borsa Italiana is the primary source for Italian market quotes.
    # yfinance is only a fallback for symbols without an ISIN.
    # Attempt Borsa Italiana for ALL symbols (it resolves ISIN on-demand, BTPs use ISIN directly)
    bi_symbols = list(missing_symbols)

    if bi_symbols:
        for sym in bi_symbols:
            try:
                bi_quote = get_borsa_italiana_quote(sym)
                if bi_quote and bi_quote.get("last") is not None:
                    result.setdefault(sym, {"last": None, "bid": None, "ask": None, "volume": None, "change_24h": None, "percentage": None, "quoteVolume": None}).update(bi_quote)
                    result[sym]["last_update"] = int(time.time() * 1000)
                    result[sym]["source"] = "borsa_italiana"
                    logger.debug(f"get_quotes: Borsa Italiana provided quote for {sym}")
            except Exception as e:
                logger.warning(f"get_quotes: Borsa Italiana failed for {sym}: {type(e).__name__}: {e}")

    if not missing_symbols:
        # All symbols got prices from cache/DB — finalize, persist, and return
        _finalize_and_persist_quotes(result, symbols, redis_client)
        return result

    # Initialize result with None for all still-missing symbols that don't have a DB quote yet
    for sym in missing_symbols:
        if sym not in result or result[sym].get("last") is None:
            result[sym] = {"last": None, "bid": None, "ask": None, "volume": None, "change_24h": None, "percentage": None, "quoteVolume": None}

    # Symbols still missing a valid price after Borsa Italiana — try yfinance
    stock_symbols = [s for s in missing_symbols if not BTPPolicy.is_btp(s) and result.get(s, {}).get("last") is None]
    # Only use yfinance for symbols with a valid Italian ISIN to avoid wrong-country data
    stock_symbols = [s for s in stock_symbols if _get_isin_from_yfinance(s) is not None]

    # --- Batch fetch ALL price data using yf.download (single HTTP request) ---
    # This replaces the slow sequential fast_info calls that caused timeouts.
    # We get last price, volume, and previous close from one batch download.
    # Bid/ask are fetched on-demand by _process_symbol via get_yahoo_quote.
    if stock_symbols and not _check_yf_circuit():
        try:
            # Log proxy status for debugging
            if settings.HTTP_PROXY_ENABLED:
                if settings.HTTP_PROXIES:
                    logger.debug(f"get_quotes: HTTP_PROXY_ENABLED with {len(settings.HTTP_PROXIES)} static proxies")
                else:
                    logger.debug("get_quotes: HTTP_PROXY_ENABLED with dynamic proxy rotator")
            else:
                logger.debug("get_quotes: HTTP_PROXY not enabled")
            batch_hist = _yf_download_with_timeout(
                stock_symbols,
                period="2d",
                interval="1d",
                auto_adjust=False,
                actions=False,
                group_by="ticker",
                session=_get_yf_session(),
            )
            if batch_hist is None or batch_hist.empty:
                logger.warning(
                    f"get_quotes: yf.download returned empty data for {len(stock_symbols)} symbols. "
                    f"Yahoo Finance may be rate-limiting or blocking requests."
                )
            for sym in stock_symbols:
                try:
                    if len(stock_symbols) > 1:
                        if batch_hist is None or batch_hist.empty or sym not in batch_hist.columns.levels[0]:
                            continue
                        sym_data = batch_hist[sym]
                    else:
                        sym_data = batch_hist

                    if len(sym_data) >= 1:
                        last = sym_data["Close"].iloc[-1]
                        if last is not None and not pd.isna(last) and last > 0:
                            result[sym]["last"] = float(last)
                            result[sym]["last_update"] = int(time.time() * 1000)
                            result[sym]["source"] = "yfinance"
                        vol = sym_data["Volume"].iloc[-1] if "Volume" in sym_data.columns else None
                        if vol is not None and not pd.isna(vol):
                            result[sym]["volume"] = float(vol)
                            result[sym]["quoteVolume"] = float(vol)
                    if len(sym_data) >= 2:
                        prev_close = sym_data["Close"].iloc[-2]
                        if prev_close is not None and not pd.isna(prev_close) and prev_close > 0:
                            last_val = result[sym].get("last")
                            if last_val is not None:
                                result[sym]["change_24h"] = last_val - prev_close
                                result[sym]["percentage"] = ((last_val - prev_close) / prev_close) * 100
                except (KeyError, ValueError, AttributeError, IndexError):
                    pass
        except Exception as e:
            logger.warning(f"Batch download failed: {type(e).__name__}: {e}")
    elif stock_symbols and _check_yf_circuit():
        logger.warning(
            f"get_quotes: yfinance circuit breaker is OPEN — skipping quote fetch for {len(stock_symbols)} symbols. "
            f"Quotes will be served from Redis cache or database if available."
        )

    # --- Try Alpha Vantage for stocks still missing valid prices ---
    missing_after_yf = [
        sym for sym in missing_symbols
        if result.get(sym, {}).get("last") is None
    ]
    if missing_after_yf and not _av_circuit_breaker.is_open():
        for sym in missing_after_yf[:10]:
            try:
                av_quote = get_alphavantage_quote(sym)
                if av_quote:
                    result[sym].update(av_quote)
                    result[sym]["last_update"] = int(time.time() * 1000)
                    result[sym]["source"] = "alphavantage"
                    _av_circuit_breaker.record_success()
                else:
                    _av_circuit_breaker.record_failure()
            except Exception as e:
                logger.warning(f"get_quotes: Alpha Vantage failed for {sym}: {type(e).__name__}: {e}")
                _av_circuit_breaker.record_failure()

    # --- Try IEX Cloud for stocks still missing valid prices ---
    missing_after_av = [
        sym for sym in missing_symbols
        if result.get(sym, {}).get("last") is None
    ]
    if missing_after_av and not _iex_circuit_breaker.is_open():
        for sym in missing_after_av[:10]:
            try:
                iex_quote = get_iex_quote(sym)
                if iex_quote:
                    result[sym].update(iex_quote)
                    result[sym]["last_update"] = int(time.time() * 1000)
                    result[sym]["source"] = "iex"
                    _iex_circuit_breaker.record_success()
                else:
                    _iex_circuit_breaker.record_failure()
            except Exception as e:
                logger.warning(f"get_quotes: IEX Cloud failed for {sym}: {type(e).__name__}: {e}")
                _iex_circuit_breaker.record_failure()

    # Finalize, persist, and enrich quotes
    _finalize_and_persist_quotes(result, symbols, redis_client)

    # Summary log
    valid_count = sum(1 for sym in missing_symbols if result[sym].get("last") is not None)
    if valid_count == 0 and missing_symbols:
        if _check_yf_circuit():
            logger.debug(
                f"get_quotes: 0/{len(missing_symbols)} symbols got valid prices "
                f"(circuit breaker open)."
            )
        else:
            logger.warning(
                f"get_quotes: 0/{len(missing_symbols)} symbols got valid prices. "
                f"Check yfinance connectivity and proxy settings."
            )
    else:
        logger.debug(f"get_quotes: {valid_count}/{len(missing_symbols)} symbols got valid prices")

    return result


def get_quotes_cached(symbols: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """Fetch quotes from Redis cache and database only. No network calls.

    Used by the symbol re-evaluation loop which must never block on yfinance
    or Borsa Italiana API calls. The background quote refresh loop is
    responsible for keeping Redis and the database up to date.
    """
    if symbols is None:
        symbols = []
    if not symbols:
        return {}

    # Sanitize symbols: remove $ prefix and /currency suffix
    symbols = [s.lstrip('$').split('/')[0] for s in symbols]

    redis_client = get_redis_client()
    result = {}
    missing_symbols = []

    # Check per-symbol Redis cache first
    for sym in symbols:
        cache_key = f"quote:{sym}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                result[sym] = json.loads(cached)
            else:
                missing_symbols.append(sym)
        except (TypeError, ValueError, RuntimeError):
            missing_symbols.append(sym)

    if not missing_symbols:
        return result

    # Check database for quotes not in Redis cache (up to 24 hours old)
    try:
        db_quotes = get_quotes_from_db(missing_symbols, max_age_seconds=86400)
        for sym in list(missing_symbols):
            if sym in db_quotes:
                result[sym] = db_quotes[sym]
                missing_symbols.remove(sym)
                # Refresh Redis cache from DB data
                try:
                    redis_client.set(f"quote:{sym}", json.dumps(db_quotes[sym]), ex=300)
                except (TypeError, ValueError, RuntimeError, ConnectionError, TimeoutError, OSError):
                    pass
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning(f"get_quotes_cached: DB quote fetch failed: {type(e).__name__}: {e}", exc_info=True)

    # Try DB close prices for anything still missing
    if missing_symbols:
        try:
            db_candles = get_latest_close_prices(missing_symbols)
            for sym in list(missing_symbols):
                if sym in db_candles and db_candles[sym].get("last", 0) > 0:
                    candle_ts = db_candles[sym].get("candle_timestamp")
                    if candle_ts and (int(time.time() * 1000) - candle_ts > 48 * 3600 * 1000):
                        logger.debug(f"get_quotes_cached: DB close price for {sym} is stale (older than 48h), skipping.")
                        continue

                    last = db_candles[sym]["last"]
                    prev_close = db_candles[sym].get("prev_close")
                    volume = db_candles[sym].get("volume")

                    abs_change = None
                    pct = None
                    if prev_close and prev_close > 0:
                        abs_change = last - prev_close
                        pct = round((abs_change / prev_close) * 100, 4)

                    result[sym] = {
                        "last": last,
                        "bid": last,
                        "ask": last,
                        "volume": volume,
                        "change_24h": abs_change,
                        "percentage": pct,
                        "quoteVolume": volume,
                        "last_update": db_candles[sym].get("candle_timestamp"),
                        "source": "db_close",
                    }
                    missing_symbols.remove(sym)
        except (RuntimeError, ValueError, KeyError, OSError) as e:
            logger.warning(f"get_quotes_cached: DB close price fallback failed: {type(e).__name__}: {e}")

    # Initialize remaining missing symbols with None values
    for sym in missing_symbols:
        result[sym] = {"last": None, "bid": None, "ask": None, "volume": None,
                       "change_24h": None, "percentage": None, "quoteVolume": None}

    # Finalize, persist, and enrich quotes
    _finalize_and_persist_quotes(result, symbols, redis_client)

    return result


def get_multi_timeframe_bars(
    symbol: str = "", timeframes: List[str] = None, limit: int = 24
) -> Dict[str, List[List[float]]]:
    if timeframes is None:
        timeframes = []
    """Fetch OHLCV bars for a symbol across multiple timeframes using yfinance.

    Returns a dict mapping timeframe -> list of candles [timestamp_ms, open, high, low, close, volume].
    """
    if not timeframes:
        return {}

    # Sanitize symbol: remove $ prefix and /currency suffix
    symbol = symbol.lstrip('$').split('/')[0]

    # Format symbol for Yahoo Finance: BTP ISINs are used as-is, stocks get TICKER_SUFFIX if missing
    yf_symbol = symbol
    if not BTPPolicy.is_btp(symbol) and settings.TICKER_SUFFIX and not symbol.endswith(settings.TICKER_SUFFIX):
        yf_symbol = f"{symbol}{settings.TICKER_SUFFIX}"

    redis_client = get_redis_client()
    cache_ttl = 60 if any(tf in ("1h",) for tf in timeframes) else 300

    # Check if we have ISIN (resolve on-demand if not in DB) — once for all timeframes
    db_isin = _get_isin_from_yfinance(symbol)
    has_isin = db_isin is not None

    result = {}
    for tf in timeframes:
        interval = TIMEFRAME_MAP.get(tf)
        if not interval:
            logger.warning(f"Unsupported timeframe: {tf}")
            continue

        needs_aggregation = tf in ("6M", "1Y", "3Y", "5Y")
        fetch_interval = "1mo" if needs_aggregation else interval

        cache_key = f"ohlcv:{symbol}:{tf}:{limit}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                result[tf] = json.loads(cached)
                continue
        except (TypeError, ValueError, RuntimeError):
            pass

        # BTPs: only borsaitaliana, no yfinance
        if BTPPolicy.is_btp(symbol):
            borsa_candles = get_borsa_italiana_candles(symbol, tf, limit=limit)
            result[tf] = borsa_candles or []
            if borsa_candles:
                try:
                    redis_client.set(cache_key, json.dumps(result[tf]), ex=cache_ttl)
                except (TypeError, ValueError, RuntimeError):
                    pass
            continue

        # If we have ISIN, only use borsaitaliana (skip yfinance to avoid rate limits)
        borsa_candles = None
        if has_isin:
            borsa_candles = get_borsa_italiana_candles(symbol, tf, limit=limit)
            if borsa_candles:
                result[tf] = borsa_candles[-limit:] if limit else borsa_candles
                try:
                    redis_client.set(cache_key, json.dumps(result[tf]), ex=cache_ttl)
                except (TypeError, ValueError, RuntimeError):
                    pass
                continue
            # Borsa Italiana failed — fall back to yfinance since we have a valid IT ISIN
        else:
            # No valid Italian ISIN — do not use yfinance to avoid wrong-country data
            result[tf] = []
            continue

        # Use yfinance as fallback (only reached if has_isin is True and Borsa failed)
        yf_candles: List[List] = []
        if not _check_yf_circuit():
            yf_symbol = symbol
            if not BTPPolicy.is_btp(symbol) and settings.TICKER_SUFFIX and not symbol.endswith(settings.TICKER_SUFFIX):
                yf_symbol = f"{symbol}{settings.TICKER_SUFFIX}"

            try:
                ticker = yf.Ticker(yf_symbol, session=_get_yf_session())
                if fetch_interval in ("5m", "15m"):
                    period = "60d"
                elif fetch_interval == "60m":
                    period = "730d"
                else:
                    period = "max"
                hist = ticker.history(period=period, interval=fetch_interval, auto_adjust=False, actions=False)
                if not hist.empty:
                    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
                    hist = hist[[col for col in ohlcv_cols if col in hist.columns]]
                    candles = []
                    for idx, row in hist.iterrows():
                        ts = int(idx.timestamp() * 1000)
                        candles.append([ts, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]])
                    if needs_aggregation:
                        candles = _aggregate_candles(candles, tf)
                    yf_candles = candles
            except (RuntimeError, ValueError, KeyError, AttributeError, OSError) as e:
                logger.debug(f"yfinance fetch failed for {symbol} {tf}: {e}")

        # Try Alpha Vantage if yfinance returned nothing
        av_candles = None
        if not yf_candles and not _av_circuit_breaker.is_open():
            try:
                av_candles = get_alphavantage_candles(symbol, tf, limit=limit)
                if av_candles:
                    logger.debug(f"Alpha Vantage provided candles for {symbol} {tf}")
                    _av_circuit_breaker.record_success()
                else:
                    _av_circuit_breaker.record_failure()
            except Exception as e:
                logger.warning(f"Alpha Vantage candles failed for {symbol} {tf}: {type(e).__name__}: {e}")
                _av_circuit_breaker.record_failure()

        # Try IEX Cloud if both yfinance and Alpha Vantage returned nothing
        iex_candles = None
        if not yf_candles and not av_candles and not _iex_circuit_breaker.is_open():
            try:
                iex_candles = get_iex_candles(symbol, tf, limit=limit)
                if iex_candles:
                    logger.debug(f"IEX Cloud provided candles for {symbol} {tf}")
                    _iex_circuit_breaker.record_success()
                else:
                    _iex_circuit_breaker.record_failure()
            except Exception as e:
                logger.warning(f"IEX Cloud candles failed for {symbol} {tf}: {type(e).__name__}: {e}")
                _iex_circuit_breaker.record_failure()

        # Merge all sources (borsa > yf > av > iex precedence by timestamp)
        merged = _merge_candles(borsa_candles, yf_candles)
        if av_candles:
            merged = _merge_candles(av_candles, merged)
        if iex_candles:
            merged = _merge_candles(iex_candles, merged)
        if merged:
            merged = _validate_and_clean_candles(merged, symbol)
            result[tf] = merged[-limit:] if limit else merged
            try:
                redis_client.set(cache_key, json.dumps(result[tf]), ex=cache_ttl)
            except (TypeError, ValueError, RuntimeError):
                pass
        else:
            result[tf] = []
    return result


def get_bars_range(
    symbol: str = "", timeframe: str = "", start_ms: int = 0, limit: int = 500
) -> List[List[float]]:
    """Fetch OHLCV bars from a start timestamp (ms) up to the present using yfinance.

    Returns a list of candles [timestamp_ms, open, high, low, close, volume].
    """
    # Sanitize symbol: remove $ prefix and /currency suffix
    symbol = symbol.lstrip('$').split('/')[0]

    interval = TIMEFRAME_MAP.get(timeframe)
    if not interval:
        logger.warning(f"Unsupported timeframe: {timeframe}")
        return []

    # Yahoo Finance does not support 6mo, 1y, 3y, 5y intervals natively.
    # We fetch 1mo data and aggregate it.
    needs_aggregation = timeframe in ("6M", "1Y", "3Y", "5Y")
    fetch_interval = "1mo" if needs_aggregation else interval

    redis_client = get_redis_client()
    cache_key = f"ohlcv_range:{symbol}:{timeframe}:{start_ms}:{limit}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except (TypeError, ValueError, RuntimeError):
        pass

    # BTPs: only borsaitaliana, no yfinance
    if BTPPolicy.is_btp(symbol):
        borsa_candles = get_borsa_italiana_candles(symbol, timeframe, limit=limit, start_ms=start_ms)
        if borsa_candles:
            try:
                redis_client.set(cache_key, json.dumps(borsa_candles), ex=300)
            except (TypeError, ValueError, RuntimeError):
                pass
            return borsa_candles
        return []

    # Check if we have ISIN (resolve on-demand if not in DB)
    db_isin = _get_isin_from_yfinance(symbol)
    has_isin = db_isin is not None

    # If we have ISIN, only use borsaitaliana (skip yfinance to avoid rate limits)
    borsa_candles = None
    if has_isin:
        borsa_candles = get_borsa_italiana_candles(symbol, timeframe, limit=limit, start_ms=start_ms)
        if borsa_candles:
            if limit and len(borsa_candles) > limit:
                borsa_candles = borsa_candles[-limit:]
            try:
                redis_client.set(cache_key, json.dumps(borsa_candles), ex=300)
            except (TypeError, ValueError, RuntimeError, ConnectionError, TimeoutError, OSError):
                pass
            return borsa_candles
        # Borsa Italiana failed — fall back to yfinance since we have a valid IT ISIN
    else:
        # No valid Italian ISIN — do not use yfinance to avoid wrong-country data
        return []

    # Use yfinance as fallback (only reached if has_isin is True and Borsa failed)
    yf_candles: List[List] = []
    if not _check_yf_circuit():
        yf_symbol = symbol
        if settings.TICKER_SUFFIX and not symbol.endswith(settings.TICKER_SUFFIX):
            yf_symbol = f"{symbol}{settings.TICKER_SUFFIX}"

        start_dt = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
        end_dt = datetime.now(timezone.utc)

        if interval in ("5m", "15m", "60m"):
            earliest_allowed = datetime.now(timezone.utc) - timedelta(days=730)
            if start_dt < earliest_allowed:
                start_dt = earliest_allowed

        try:
            ticker = yf.Ticker(yf_symbol, session=_get_yf_session())
            hist = ticker.history(start=start_dt, end=end_dt, interval=fetch_interval, auto_adjust=False, actions=False)
            if not hist.empty:
                ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
                hist = hist[[col for col in ohlcv_cols if col in hist.columns]]
                candles = []
                for idx, row in hist.iterrows():
                    ts = int(idx.timestamp() * 1000)
                    candles.append([ts, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]])
                if needs_aggregation:
                    candles = _aggregate_candles(candles, timeframe)
                yf_candles = candles
        except (RuntimeError, ValueError, KeyError, AttributeError, OSError) as e:
            logger.debug(f"yfinance fetch failed for {symbol} {timeframe}: {e}")

    # Try Alpha Vantage if yfinance returned nothing
    av_candles = None
    if not yf_candles and not _av_circuit_breaker.is_open():
        try:
            av_candles = get_alphavantage_candles(symbol, timeframe, limit=limit, start_ms=start_ms)
            if av_candles:
                logger.debug(f"Alpha Vantage provided candles for {symbol} {timeframe}")
                _av_circuit_breaker.record_success()
            else:
                _av_circuit_breaker.record_failure()
        except Exception as e:
            logger.warning(f"Alpha Vantage candles failed for {symbol} {timeframe}: {type(e).__name__}: {e}")
            _av_circuit_breaker.record_failure()

    # Try IEX Cloud if both yfinance and Alpha Vantage returned nothing
    iex_candles = None
    if not yf_candles and not av_candles and not _iex_circuit_breaker.is_open():
        try:
            iex_candles = get_iex_candles(symbol, timeframe, limit=limit, start_ms=start_ms)
            if iex_candles:
                logger.debug(f"IEX Cloud provided candles for {symbol} {timeframe}")
                _iex_circuit_breaker.record_success()
            else:
                _iex_circuit_breaker.record_failure()
        except Exception as e:
            logger.warning(f"IEX Cloud candles failed for {symbol} {timeframe}: {type(e).__name__}: {e}")
            _iex_circuit_breaker.record_failure()

    # Merge all sources (borsa > yf > av > iex precedence by timestamp)
    merged = _merge_candles(borsa_candles, yf_candles)
    if av_candles:
        merged = _merge_candles(av_candles, merged)
    if iex_candles:
        merged = _merge_candles(iex_candles, merged)
    merged = _validate_and_clean_candles(merged, symbol)

    if merged:
        if limit and len(merged) > limit:
            merged = merged[-limit:]
        try:
            redis_client.set(cache_key, json.dumps(merged), ex=300)
        except (TypeError, ValueError, RuntimeError):
            pass
        return merged
    return []


