import asyncio
import collections
import hashlib
import logging
import random
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
import concurrent.futures
from bs4 import BeautifulSoup

from src.config.settings import settings
from src.utils.redis_client import get_redis_client
from src.database import save_quotes_batch, get_quotes_from_db, get_latest_close_prices

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

_get_quotes_lock = threading.Lock()

# Dedicated thread pool for yf.download with a hard timeout wrapper.
# This prevents yf.download from hanging threads indefinitely when
# curl_cffi's own timeout doesn't fire.
_yf_download_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="yf-download-timeout"
)


def _yf_download_with_timeout(symbols, **kwargs):
    """Run yf.download with a hard 30-second timeout to prevent indefinite hangs.

    If the timeout fires, the underlying thread is abandoned (it will eventually
    die on its own due to the curl_cffi HTTP timeout). Returns None on timeout.
    """
    future = _yf_download_executor.submit(yf.download, symbols, **kwargs)
    try:
        return future.result(timeout=30)
    except concurrent.futures.TimeoutError:
        count = len(symbols) if isinstance(symbols, list) else 1
        logger.warning(f"yf.download timed out after 30s for {count} symbols")
        return None


# --- yfinance Circuit Breaker ---
_yf_error_count = 0
_yf_last_error_time = 0.0
_yf_circuit_open_until = 0.0
_yf_lock = threading.Lock()

YF_MAX_ERRORS = 20
YF_CIRCUIT_COOLDOWN = 300  # 5 minutes

_yf_session_cache = None
_yf_session_lock = threading.Lock()
_yf_session_created = False  # True only after successful creation or confirmed ImportError

def _check_yf_circuit() -> bool:
    """Return True if the circuit is open (yfinance should be skipped)."""
    with _yf_lock:
        return time.time() < _yf_circuit_open_until

def _record_yf_error():
    """Record a yfinance error and potentially trip the circuit breaker."""
    global _yf_error_count, _yf_last_error_time, _yf_circuit_open_until
    with _yf_lock:
        now = time.time()
        if now - _yf_last_error_time > 300:
            _yf_error_count = 0
        _yf_error_count += 1
        _yf_last_error_time = now
        if _yf_error_count >= YF_MAX_ERRORS:
            if _yf_circuit_open_until < now:
                logger.error(f"yfinance circuit breaker tripped due to {_yf_error_count} errors. Blocking yfinance calls for {YF_CIRCUIT_COOLDOWN}s.")
            _yf_circuit_open_until = now + YF_CIRCUIT_COOLDOWN

def _reset_yf_circuit():
    """Reset the circuit breaker after a successful call."""
    global _yf_error_count
    with _yf_lock:
        _yf_error_count = 0


class YFinanceRateLimiter:
    """Sliding window rate limiter for yfinance requests."""
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def acquire(self):
        # Read live settings to allow runtime reload without restart
        if not settings.YF_RATE_LIMIT_ENABLED or settings.YF_RATE_LIMIT_MAX_REQUESTS <= 0:
            return
        with self._lock:
            now = time.time()
            window = settings.YF_RATE_LIMIT_WINDOW_SECONDS
            max_req = settings.YF_RATE_LIMIT_MAX_REQUESTS
            # Remove timestamps outside the window
            while self._timestamps and self._timestamps[0] <= now - window:
                self._timestamps.popleft()
            if len(self._timestamps) >= max_req:
                # Rate limit exceeded — fail fast instead of blocking.
                # Sleeping would hold the thread pool worker hostage and cause
                # asyncio.wait_for timeouts in the engine.  Raising lets the
                # caller fall back to the database immediately.
                raise ConnectionError("yfinance rate limit exceeded")
            self._timestamps.append(time.time())


_yf_rate_limiter = YFinanceRateLimiter(
    max_requests=settings.YF_RATE_LIMIT_MAX_REQUESTS,
    window_seconds=settings.YF_RATE_LIMIT_WINDOW_SECONDS,
)

class YFinance401Filter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "HTTP Error 401" in msg or "Unauthorized" in msg:
            _record_yf_error()
            if _check_yf_circuit():
                return False  # suppress log spam when circuit is open
        return True

# Attach filter to yfinance logger
logging.getLogger("yfinance").addFilter(YFinance401Filter())

def _get_yf_session():
    """Return a cached curl_cffi session that impersonates Chrome for yfinance requests.

    Yahoo Finance increasingly blocks requests that don't look like a real
    browser.  curl_cffi can impersonate Chrome's TLS fingerprint, which
    avoids 401/429 responses.  The session is created once and reused for
    all subsequent calls to avoid the overhead of repeated session creation.
    """
    global _yf_session_cache, _yf_session_created

    # Fast path: return cached session without acquiring lock
    if _yf_session_created:
        return _yf_session_cache

    with _yf_session_lock:
        # Double-check after acquiring lock
        if _yf_session_created:
            return _yf_session_cache

        try:
            from curl_cffi import requests as curl_requests

            proxies = None
            if settings.HTTP_PROXY_ENABLED and settings.HTTP_PROXIES:
                import random
                proxy = random.choice(settings.HTTP_PROXIES)
                proxies = {"http": proxy, "https": proxy}
                logger.debug(f"Using proxy for yfinance: {proxy}")

            class YFinanceSessionWrapper(curl_requests.Session):
                def request(self, *args, **kwargs):
                    if _check_yf_circuit():
                        raise ConnectionError("yfinance circuit breaker is open")
                    _yf_rate_limiter.acquire()
                    # Enforce a timeout to prevent indefinite hangs
                    kwargs.setdefault('timeout', 15.0)
                    response = super().request(*args, **kwargs)
                    if response.status_code == 401:
                        _record_yf_error()
                    else:
                        _reset_yf_circuit()
                    return response

            _yf_session_cache = YFinanceSessionWrapper(impersonate="chrome", proxies=proxies)
            _yf_session_created = True
            return _yf_session_cache
        except ImportError:
            logger.warning("curl_cffi not installed – yfinance requests may be blocked.")
            _yf_session_cache = None
            _yf_session_created = True
            return None
        except Exception as e:
            logger.warning(f"Failed to create curl_cffi session: {e}")
            # Don't set _yf_session_created so we retry on the next call
            return None


class DynamicProxyRotator:
    def __init__(self):
        self.proxy_source_url = "https://free-proxy-list.net"
        self.test_url = "http://httpbin.org"
        self.valid_proxies = []
        self._last_refresh = 0.0

    def fetch_raw_proxies(self):
        """Scrapes the latest free proxies from the web."""
        try:
            response = requests.get(self.proxy_source_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            proxies = []
            
            table = soup.find('div', class_='table-responsive')
            if not table:
                return []
                
            for row in table.find('tbody').find_all('tr'):
                cols = row.find_all('td')
                if cols:
                    ip = cols[0].text.strip()
                    port = cols[1].text.strip()
                    proxies.append(f"http://{ip}:{port}")
            return proxies
        except Exception as e:
            logger.warning(f"Error fetching raw proxies: {e}")
            return []

    async def _validate_single_proxy(self, client, proxy):
        """Asynchronously tests a single proxy's viability."""
        try:
            response = await client.get(self.test_url, proxy=proxy, timeout=3.0)
            if response.status_code == 200:
                self.valid_proxies.append(proxy)
        except Exception:
            pass

    async def refresh_proxy_pool(self):
        """Fetches and tests all proxies, rebuilding the valid pool."""
        raw_proxies = self.fetch_raw_proxies()
        logger.info(f"Fetched {len(raw_proxies)} raw proxies. Validating speed and uptime...")
        
        self.valid_proxies = []
        async with httpx.AsyncClient() as client:
            tasks = [self._validate_single_proxy(client, proxy) for proxy in raw_proxies]
            await asyncio.gather(*tasks)
            
        self._last_refresh = time.time()
        logger.info(f"Pool updated! {len(self.valid_proxies)} proxies are ready to use.")

    def get_proxy(self):
        """Returns a random valid proxy from the active pool."""
        if not self.valid_proxies:
            return None
        return random.choice(self.valid_proxies)

_dynamic_rotator = DynamicProxyRotator()


def _get_proxies() -> Optional[str]:
    """Return a random proxy string for httpx if enabled, else None."""
    if not settings.HTTP_PROXY_ENABLED:
        return None

    # Trigger background refresh if pool is empty or stale (every 30 mins)
    if not _dynamic_rotator.valid_proxies or (time.time() - _dynamic_rotator._last_refresh > 1800):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_dynamic_rotator.refresh_proxy_pool())
        except RuntimeError:
            pass  # No event loop running

    # Randomly decide to use YF_PROXIES or dynamic rotator
    use_dynamic = False
    if _dynamic_rotator.valid_proxies:
        if settings.HTTP_PROXIES:
            use_dynamic = random.choice([True, False])
        else:
            use_dynamic = True
    elif not settings.HTTP_PROXIES:
        return None

    if use_dynamic:
        proxy = _dynamic_rotator.get_proxy()
        if proxy:
            logger.debug(f"Using dynamic proxy: {proxy}")
            return proxy
    
    if settings.HTTP_PROXIES:
        proxy = random.choice(settings.HTTP_PROXIES)
        logger.debug(f"Using static proxy: {proxy}")
        return proxy
    
    return None


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
        return cached

    # If yfinance circuit is open, we can't fetch the ISIN from yfinance.
    if _check_yf_circuit():
        return None

    yf_symbol = f"{db_symbol}{suffix}" if suffix and not db_symbol.endswith(suffix) else db_symbol
    try:
        ticker = yf.Ticker(yf_symbol, session=_get_yf_session())
        isin = ticker.isin
        if isin:
            isin = isin.strip()
            if isin == '-' or not isin:
                return None
            # Save to DB with the base symbol (no suffix)
            try:
                save_country = settings.TARGET_COUNTRY if settings.COUNTRY_FILTER_STRICT else None
                save_discovered_symbol(db_symbol, isin, None, "", country=save_country)
            except Exception:
                pass
            return isin
    except Exception as e:
        logger.debug(f"Failed to fetch ISIN for {base_symbol}: {e}")
    return None


# Borsa Italiana timeframe conversion map (daily data → resampled via pandas)
BORSA_TIMEFRAME_MAP = {
    "1d": "1d",       # Daily (uses intraday endpoint)
    "1M": "1M",       # Monthly
    "3M": "3M",       # Quarterly
    "6M": "6M",       # Semi-annual
    "1Y": "1Y",       # Year End
    "3Y": "3Y",       # 3-Year
    "5Y": "5Y",       # 5-Year
}

# --- Borsa Italiana token cache ---
_borsa_token_cache: Dict[str, tuple] = {}  # {cache_key: (timestamp, token)}
_borsa_token_cache_lock = threading.Lock()
_BORSA_TOKEN_CACHE_TTL = 300  # 5 minutes


def _get_borsa_italiana_token(isin: str, market_code: str) -> Optional[str]:
    """Dynamically fetch the bearer token from the Borsa Italiana summary chart page, with caching."""
    cache_key = f"{isin}-{market_code}"
    now = time.time()

    # Check cache first
    with _borsa_token_cache_lock:
        cached = _borsa_token_cache.get(cache_key)
        if cached and (now - cached[0]) < _BORSA_TOKEN_CACHE_TTL:
            return cached[1]

    url = f"https://grafici.borsaitaliana.it/summary-chart/{isin}-{market_code}?lang=it"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        with httpx.Client(proxy=_get_proxies(), timeout=15.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            # Extract token from <chart-allinone ... token="..." ...>
            # Use BeautifulSoup for robust parsing, with regex as a fallback
            soup = BeautifulSoup(response.text, "html.parser")
            chart_tag = soup.find("chart-allinone")
            token = chart_tag.get("token") if chart_tag else None
            
            if not token:
                # Fallback to regex if BeautifulSoup fails to find the tag
                match = re.search(r'<chart-allinone[^>]*token="([^"]+)"', response.text)
                if match:
                    token = match.group(1)
            
            if token:
                # Cache the token
                with _borsa_token_cache_lock:
                    _borsa_token_cache[cache_key] = (now, token)
                return token
                
            logger.warning(f"Could not find Borsa Italiana token for {isin}-{market_code}")
            return None
    except Exception as e:
        logger.warning(f"Failed to fetch Borsa Italiana token for {isin}-{market_code}: {e}")
        return None


def get_borsa_italiana_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch the latest real-time quote from Borsa Italiana for an Italian stock or BTP.

    Returns a dict with keys: last, bid, ask, volume, change_24h, percentage,
    quoteVolume, last_update, source.
    Returns None if the quote cannot be fetched.
    """
    base = symbol.split("/")[0] if "/" in symbol else symbol

    if re.match(r'^IT[A-Z0-9]{10}$', base):
        isin = base
    else:
        isin = _get_isin_from_yfinance(base)
        if not isin:
            return None

    market_code = settings.MARKET_CODE
    token = _get_borsa_italiana_token(isin, market_code)
    if not token:
        return None

    headers = {
        "accept": "*/*",
        "accept-language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        "authorization": f"Bearer {token}",
        "priority": "u=1, i",
        "referer": f"https://grafici.borsaitaliana.it/summary-chart/{isin}-{market_code}?lang=it",
        "sec-ch-ua": '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
    }

    try:
        with httpx.Client(proxy=_get_proxies(), timeout=10.0, follow_redirects=True) as client:
            url = f"https://grafici.borsaitaliana.it/api/instruments/{isin},{market_code},ISIN/intraday?resolution=1MN"
            response = client.get(url, headers=headers)
            response.raise_for_status()

            try:
                data = response.json()
            except (json.JSONDecodeError, ValueError):
                match = re.search(r'<pre>(.*?)</pre>', response.text, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(1))
                    except json.JSONDecodeError:
                        return None
                else:
                    return None

            intraday_points = data.get("intradayPoint", [])
            if intraday_points:
                latest = intraday_points[-1]
                last_price = float(latest.get("endPx", 0))
                if last_price > 0:
                    vol = float(latest.get("vol", 0) or 0)
                    return {
                        "last": last_price,
                        "bid": last_price,
                        "ask": last_price,
                        "volume": vol,
                        "quoteVolume": vol,
                        "last_update": int(time.time() * 1000),
                        "source": "borsa_italiana",
                    }
    except Exception as e:
        logger.debug(f"Borsa Italiana quote fetch failed for {symbol}: {e}")
    return None


def get_borsa_italiana_candles(
    symbol: str,
    timeframe: str,
    limit: int = 500,
    start_ms: int = None,
) -> Optional[List[List]]:
    """Download OHLCV candles from borsaitaliana.it grafici API as the primary source.

    Fetches daily candles from the borsaitaliana chart API and resamples
    them to the requested timeframe using pandas.
    For 1d timeframe, uses the intraday endpoint.

    Returns list of [timestamp_ms, open, high, low, close, volume].
    Returns None if the download fails or the timeframe is not supported.
    """
    if timeframe not in BORSA_TIMEFRAME_MAP:
        return None

    base = symbol.split("/")[0] if "/" in symbol else symbol

    # For BTPs, the symbol IS the ISIN
    if re.match(r'^IT[A-Z0-9]{10}$', base):
        isin = base
    else:
        isin = _get_isin_from_yfinance(base)
        if not isin:
            logger.debug(f"Could not get ISIN for {symbol}, skipping borsaitaliana")
            return None

    # Determine market code for referer URL
    market_code = settings.MARKET_CODE

    # Dynamically fetch the bearer token
    token = _get_borsa_italiana_token(isin, market_code)
    if not token:
        logger.warning(f"Skipping Borsa Italiana download for {symbol} {timeframe}: no token found.")
        return None

    # Headers matching the browser request exactly
    headers = {
        "accept": "*/*",
        "accept-language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        "authorization": f"Bearer {token}",
        "priority": "u=1, i",
        "referer": f"https://grafici.borsaitaliana.it/summary-chart/{isin}-{market_code}?lang=it",
        "sec-ch-ua": '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
    }

    try:
        with httpx.Client(proxy=_get_proxies(), timeout=15.0, follow_redirects=True) as client:
            if timeframe == "1d":
                # For 1d, use the 1M history endpoint to get daily candles
                url = f"https://grafici.borsaitaliana.it/api/instruments/{isin},{market_code},ISIN/history/period?period=1M&adjustment=true&add-last-price=true"
                logger.debug(f"Fetching 1d data from history endpoint: {url}")
                response = client.get(url, headers=headers)
                response.raise_for_status()
            else:
                # For other timeframes, use the history endpoint with the correct period
                period = BORSA_TIMEFRAME_MAP.get(timeframe)
                url = f"https://grafici.borsaitaliana.it/api/instruments/{isin},{market_code},ISIN/history/period?period={period}&adjustment=true&add-last-price=true"
                response = client.get(url, headers=headers)
                response.raise_for_status()

        # The API returns JSON wrapped in HTML <pre> tags
        text = response.text
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            # Extract JSON from HTML <pre> tags
            match = re.search(r'<pre>(.*?)</pre>', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    logger.error(f"Could not parse JSON from borsaitaliana response for {symbol}")
                    return None
            else:
                logger.error(f"No JSON data found in borsaitaliana response for {symbol}")
                return None

        # Extract candle data — handle both "history" and "intraday" response formats
        history = data.get("history", {})
        history_dt = history.get("historyDt", [])
        intraday_points = data.get("intradayPoint", [])

        rows = []
        if intraday_points and timeframe != "1d":
            # Intraday response format (1d endpoint)
            # Fields: time (YYYYMMDD-HH:MM:SS), beginPx, highPx, lowPx, endPx, vol
            for item in intraday_points:
                time_str = item.get("time", "")
                if not time_str:
                    continue
                try:
                    # Format: "YYYYMMDD-HH:MM:SS"
                    dt = datetime.strptime(time_str, "%Y%m%d-%H:%M:%S")
                    ts = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
                    rows.append([
                        ts,
                        float(item["beginPx"]),
                        float(item["highPx"]),
                        float(item["lowPx"]),
                        float(item["endPx"]),
                        float(item.get("vol", 0) or 0),
                    ])
                except (ValueError, KeyError) as e:
                    logger.error(f"Failed to parse borsaitaliana intraday candle for {symbol}: {e}")
                    continue
        elif history_dt:
            # History response format (1M, 3M, 1Y, etc. endpoints)
            # Fields: dt (YYYYMMDD), openPx, highPx, lowPx, closePx, qty
            for item in history_dt:
                dt_str = item.get("dt", "")
                if not dt_str:
                    continue
                try:
                    if len(dt_str) == 8:
                        # YYYYMMDD (daily)
                        dt = datetime.strptime(dt_str, "%Y%m%d")
                    elif len(dt_str) == 14:
                        # YYYYMMDDHHMMSS (intraday)
                        dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
                    elif len(dt_str) == 12:
                        # YYMMDDHHMMSS (intraday, 2-digit year)
                        dt = datetime.strptime(dt_str, "%y%m%d%H%M%S")
                    else:
                        continue
                    ts = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
                    close_px = item.get("closePx") or item.get("lastPx") or item.get("setPx") or 0
                    rows.append([
                        ts,
                        float(item.get("openPx") or close_px),
                        float(item.get("highPx") or close_px),
                        float(item.get("lowPx") or close_px),
                        float(close_px),
                        float(item.get("qty", 0) or 0),
                    ])
                except (ValueError, KeyError) as e:
                    logger.error(f"Failed to parse borsaitaliana history candle for {symbol}: {e}")
                    continue
        else:
            logger.warning(f"Empty history from borsaitaliana for {symbol} {timeframe}")
            return None

        if not rows:
            return None

        # Sort by timestamp
        rows.sort(key=lambda c: c[0])

        # Filter by start_ms if provided
        if start_ms is not None:
            rows = [c for c in rows if c[0] >= start_ms]

        # For 1d intraday data, aggregate 1-minute candles into daily candles
        if timeframe == "1d" and len(rows) > 1:
            # Check if we have intraday granularity (timestamps within same day)
            first_ts = rows[0][0]
            second_ts = rows[1][0]
            if (second_ts - first_ts) < 86400000:  # less than 1 day apart = intraday
                logger.debug(f"Aggregating {len(rows)} intraday candles into daily candles for {symbol}")
                df = pd.DataFrame(rows, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
                df.set_index('Date', inplace=True)
                ohlcv_rules = {
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }
                df = df.resample('1D').agg(ohlcv_rules)
                df.dropna(subset=['Open'], inplace=True)
                df.reset_index(inplace=True)
                rows = []
                for _, row in df.iterrows():
                    ts = int(row['Date'].timestamp() * 1000)
                    vol = float(row['Volume']) if pd.notna(row['Volume']) else 0.0
                    rows.append([ts, float(row['Open']), float(row['High']), float(row['Low']), float(row['Close']), vol])
                rows.sort(key=lambda c: c[0])

        # Sort by timestamp
        rows.sort(key=lambda c: c[0])

        # Apply limit
        if limit and len(rows) > limit:
            rows = rows[-limit:]

        if rows:
            logger.info(f"Downloaded {len(rows)} candles from borsaitaliana for {symbol} {timeframe}")
            return rows

        return None

    except Exception as e:
        logger.debug(f"Borsaitaliana candle download failed for {symbol} {timeframe}: {e}")
        return None

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

def _aggregate_candles(candles: List[List], target_tf: str) -> List[List]:
    """Aggregate monthly candles into larger timeframes (6M, 1Y, 3Y, 5Y)."""
    if not candles or target_tf not in ("6M", "1Y", "3Y", "5Y"):
        return candles

    grouped = {}
    for c in candles:
        ts = c[0]
        dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        year = dt.year
        month = dt.month

        if target_tf == "6M":
            period_key = (year, (month - 1) // 6)
        elif target_tf == "1Y":
            period_key = year
        elif target_tf == "3Y":
            period_key = year // 3
        elif target_tf == "5Y":
            period_key = year // 5

        if period_key not in grouped:
            grouped[period_key] = {
                'timestamp': ts,
                'open': c[1],
                'high': c[2],
                'low': c[3],
                'close': c[4],
                'volume': c[5]
            }
        else:
            grouped[period_key]['high'] = max(grouped[period_key]['high'], c[2])
            grouped[period_key]['low'] = min(grouped[period_key]['low'], c[3])
            grouped[period_key]['close'] = c[4]
            grouped[period_key]['volume'] += c[5]

    result = []
    for key in sorted(grouped.keys()):
        g = grouped[key]
        result.append([
            g['timestamp'],
            g['open'],
            g['high'],
            g['low'],
            g['close'],
            g['volume']
        ])
    return result


def _merge_candles(borsa_candles: Optional[List[List]], yf_candles: Optional[List[List]]) -> List[List]:
    """Merge two candle lists, deduplicating by timestamp (borsaitaliana takes precedence)."""
    if not borsa_candles and not yf_candles:
        return []
    if not borsa_candles:
        return yf_candles
    if not yf_candles:
        return borsa_candles
    merged = {}
    for c in yf_candles:
        merged[c[0]] = c
    for c in borsa_candles:  # borsaitaliana overrides yfinance for same timestamp
        merged[c[0]] = c
    return sorted(merged.values(), key=lambda c: c[0])


def _fetch_info(symbol: str, max_retries: int = 2) -> tuple[Optional[str], Optional[str]]:
    """Fetch the country and name from yfinance info for a symbol, with retries.

    Returns a tuple (country, name) on success, or (None, None) if yfinance
    could not provide the information after all retries.
    """
    if _check_yf_circuit():
        return None, None
    import time as _time
    for attempt in range(max_retries + 1):
        try:
            ticker = yf.Ticker(symbol, session=_get_yf_session())
            info = ticker.info
            country = info.get("country")
            name = info.get("longName") or info.get("shortName")
            if country or name:
                return country, name
            # country is None or empty – retry if attempts remain
            if attempt < max_retries:
                _time.sleep(0.5 * (2 ** attempt))
                continue
        except Exception as e:
            logger.debug(f"Failed to fetch info for {symbol} (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                _time.sleep(0.5 * (2 ** attempt))
    return None, None


def _discover_wikipedia_tickers(urls: List[str], index_name: str) -> List[str]:
    """Scrape a Wikipedia constituent list from one or more URLs.

    Returns base symbols (suffix stripped). Tries each URL in order; returns
    the first non‑empty result. Also extracts and caches ISIN codes in Redis.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    redis_client = get_redis_client()
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tables = pd.read_html(response.text)
        except Exception as e:
            logger.debug(f"Failed to scrape {url}: {str(e)[:200]}")
            continue

        for table in tables:
            # Flatten multi‑level column names
            if isinstance(table.columns, pd.MultiIndex):
                table.columns = [' '.join(col).strip() for col in table.columns.values]
            
            # Try to find ticker and ISIN columns by name
            ticker_col = None
            isin_col = None
            for col in table.columns:
                col_str = str(col).lower()
                if any(kw in col_str for kw in ("ticker", "symbol", "code", "simbolo", "codice", "yahoo", "borsa")):
                    ticker_col = col
                if "isin" in col_str:
                    isin_col = col
            
            if ticker_col is None:
                # Last resort: look for a column whose values look like tickers
                for col in table.columns:
                    sample = table[col].dropna().astype(str).head(20).tolist()
                    if not sample:
                        continue
                    
                    ticker_like = 0
                    non_empty = 0
                    for s in sample:
                        s_clean = s.strip().upper()
                        if not s_clean:
                            continue
                        non_empty += 1
                        # Match typical ticker patterns like ENI, ENI.MI, etc. (avoid ISINs)
                        if re.match(r'^[A-Z0-9]{1,10}(\.[A-Z]{2})?$', s_clean):
                            ticker_like += 1
                    
                    # If at least 60% of non-empty values look like tickers, use this column
                    if non_empty > 0 and ticker_like >= non_empty * 0.6:
                        ticker_col = col
                        break

            # If no explicit ISIN column, look for a column with ISIN-like values
            if isin_col is None:
                for col in table.columns:
                    if col == ticker_col:
                        continue
                    sample = table[col].dropna().astype(str).head(20).tolist()
                    isin_like = 0
                    non_empty = 0
                    for s in sample:
                        s_clean = s.strip().upper()
                        if not s_clean:
                            continue
                        non_empty += 1
                        if re.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", s_clean):
                            isin_like += 1
                    if non_empty > 0 and isin_like >= non_empty * 0.8:
                        isin_col = col
                        break
            
            if ticker_col is not None:
                tickers = table[ticker_col].dropna().astype(str).tolist()
                isins = table[isin_col].dropna().astype(str).tolist() if isin_col is not None else []
                base_symbols = []
                for i, t in enumerate(tickers):
                    t = t.strip().upper()
                    # Skip ISINs (e.g., IT0001233417)
                    if re.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", t):
                        continue
                    # Remove any text after a space or parenthesis (e.g., "ENI.MI (ENI)")
                    t = re.split(r'[\s(]', t)[0]
                    base = t.split(".")[0] if "." in t else t
                    if re.match(r"^[A-Z0-9]+$", base):
                        base_symbols.append(base)
                        # Cache ISIN if available
                        if i < len(isins):
                            isin = isins[i].strip().upper()
                            if re.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", isin):
                                try:
                                    from src.database import save_discovered_symbol
                                    save_discovered_symbol(base, isin, "stock", "", country="italy")
                                except Exception:
                                    pass
                
                if base_symbols:
                    logger.info(f"Discovered {len(base_symbols)} {index_name} tickers from {url}")
                    return base_symbols

    logger.warning(f"No ticker column found in any Wikipedia page for {index_name}.")
    return []


def _load_static_tickers() -> List[str]:
    """Load base symbols from a static CSV file if present."""
    import os
    path = os.path.join(settings.DATA_DIR, "italian_tickers.csv")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return [line.strip().upper() for line in f if line.strip() and re.match(r"^[A-Z0-9]+$", line.strip())]
    except Exception as e:
        logger.warning(f"Failed to load static tickers file: {e}")
        return []


def _get_hardcoded_tickers() -> List[str]:
    """Return a hardcoded list of major Italian tickers as a last resort."""
    return [
        "ENI", "ENEL", "ISP", "UCG", "STLA", "TIT", "RACE", "AZM", "BAMI", "MB",
        "LDO", "TEN", "PRY", "SPM", "BPE", "EXO", "NEXI", "A2A", "RNST", "SRG",
        "INW", "DHER", "PST", "BZU", "CPR", "TRN", "BMO", "AQUA", "BRS", "TGY",
        "IWM", "MOL", "HER", "BIA", "CNH", "ST", "UNI", "VBT", "AMP", "BKB"
    ]


def _discover_financedatabase_tickers() -> List[str]:
    """Discover base tickers using the FinanceDatabase library based on TARGET_COUNTRY."""
    try:
        import financedatabase as fd
    except ImportError:
        logger.warning("financedatabase not installed. Skipping FinanceDatabase ticker discovery.")
        return []

    country = settings.TARGET_COUNTRY.capitalize()
    suffix = settings.TICKER_SUFFIX
    try:
        equities = fd.Equities()
        df = equities.select(country=country)
        if df is None or df.empty:
            logger.warning(f"No tickers found in FinanceDatabase for country: {country}")
            return []

        base_symbols = []
        for symbol in df.index:
            # We only want symbols that match our configured suffix (e.g., .MI)
            if suffix and symbol.endswith(suffix):
                base = symbol[:-len(suffix)]
                if re.match(r"^[A-Z0-9]+$", base):
                    base_symbols.append(base)
            elif not suffix:
                # If no suffix is configured, just take the base symbol
                base = symbol.split(".")[0] if "." in symbol else symbol
                if re.match(r"^[A-Z0-9]+$", base):
                    base_symbols.append(base)

        logger.info(f"Discovered {len(base_symbols)} tickers from FinanceDatabase for {country}")
        return base_symbols
    except Exception as e:
        logger.warning(f"FinanceDatabase ticker discovery failed: {e}")
        return []


def discover_italian_ucits_etfs() -> List[str]:
    """Discover Italian-focused UCITS ETFs using FinanceDatabase."""
    redis_client = get_redis_client()
    cache_key = "italian_ucits_etfs"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    try:
        import financedatabase as fd
    except ImportError:
        logger.warning("financedatabase not installed. Skipping ETF discovery.")
        return []

    keywords = [k.strip().lower() for k in settings.ETF_ITALY_KEYWORDS.split(",") if k.strip()]
    if not keywords:
        return []

    try:
        etfs = fd.ETFs()
        # Try to filter by country first, fallback to all if unsupported
        try:
            df = etfs.select(country="Italy")
        except Exception:
            df = etfs.select()

        if df is None or df.empty:
            return []

        base_symbols = []
        for symbol, row in df.iterrows():
            name = str(row.get('name', '')).lower()
            # MANDATORY SAFETY FILTER: Must contain "UCITS"
            if 'ucits' not in name and 'ucits' not in symbol.lower():
                continue

            # Apply keyword filter to ensure Italian economy focus
            if not any(kw in name for kw in keywords):
                continue

            # Extract short alphanumeric symbol (strip exchange suffix)
            base = symbol.split(".")[0] if "." in symbol else symbol
            if re.match(r"^[A-Z0-9]+$", base):
                base_symbols.append(base)

        logger.info(f"Discovered {len(base_symbols)} Italian UCITS ETFs matching keywords.")
        # Cache for 24 hours
        try:
            redis_client.set(cache_key, json.dumps(base_symbols), ex=86400)
        except Exception:
            pass
        # Save ETF symbols to DB
        try:
            from src.database import save_discovered_symbols_batch
            symbols_to_save = [
                {"symbol": sym, "isin": None, "asset_type": "etf", "name": "", "country": "italy"}
                for sym in base_symbols
            ]
            if symbols_to_save:
                save_discovered_symbols_batch(symbols_to_save)
        except Exception as e:
            logger.warning(f"Failed to save ETF symbols to DB: {e}")
        return base_symbols
    except Exception as e:
        logger.warning(f"Failed to discover Italian UCITS ETFs: {e}")
        return []


def _save_discovered_assets_to_db(base_symbols: List[str], etf_symbols: List[str] = None):
    """Save discovered stock/ETF base symbols to the database. ISINs are fetched on demand."""
    from src.database import save_discovered_symbols_batch, get_isin_map_from_db
    etf_set = set(etf_symbols or [])
    symbols_to_save = []
    for sym in base_symbols:
        base = sym.split(".")[0] if "." in sym else sym
        if re.match(r'^IT[A-Z0-9]{10}$', base):
            continue  # BTPs are saved separately
        asset_type = "etf" if base in etf_set else "stock"
        symbols_to_save.append({
            "symbol": base,
            "isin": None,
            "asset_type": asset_type,
            "name": "",
            "country": None,
        })
    if not symbols_to_save:
        return
    # Batch check which symbols already have ISINs in DB — single query instead of N
    try:
        sym_list = [s["symbol"] for s in symbols_to_save]
        isin_map = get_isin_map_from_db(sym_list)
        # Filter out symbols that already have an ISIN
        symbols_to_save = [s for s in symbols_to_save if not isin_map.get(s["symbol"])]
    except Exception:
        pass  # If batch lookup fails, save all (upsert handles it)
    if symbols_to_save:
        try:
            save_discovered_symbols_batch(symbols_to_save)
            logger.info(f"Saved {len(symbols_to_save)} discovered symbols to database")
        except Exception as e:
            logger.warning(f"Failed to save discovered symbols to database: {e}")


def get_tradable_assets() -> List[str]:
    """Return a list of tradable Italian equity symbols, filtered by country.

    Discovers base symbols dynamically from the FTSE MIB Wikipedia page and
    from news RSS feeds, then appends the configured ticker suffix and
    verifies via yfinance that each symbol's country matches the configured
    TARGET_COUNTRY. Results are cached in Redis for 24 hours.
    """
    # Discover tickers from Wikipedia (FTSE MIB constituents)
    base_symbols = _discover_wikipedia_tickers(
        ["https://it.wikipedia.org/wiki/FTSE_MIB", "https://en.wikipedia.org/wiki/FTSE_MIB"],
        "FTSE MIB"
    )

    # --- FTSE Italia All-Share constituents ---
    all_share = _discover_wikipedia_tickers(
        ["https://it.wikipedia.org/wiki/FTSE_Italia_All-Share", "https://en.wikipedia.org/wiki/FTSE_Italia_All-Share"],
        "FTSE Italia All-Share"
    )
    if all_share:
        existing = set(base_symbols)
        for t in all_share:
            if t not in existing:
                base_symbols.append(t)
                existing.add(t)

    # --- User-configured additional tickers ---
    extra = settings.ADDITIONAL_TICKERS
    if extra:
        existing = set(base_symbols)
        for t in extra:
            t_clean = t.strip().upper()
            if t_clean and t_clean not in existing:
                base_symbols.append(t_clean)
                existing.add(t_clean)

    # --- FinanceDatabase ticker discovery ---
    if settings.FINANCEDATABASE_TICKER_DISCOVERY_ENABLED:
        fd_tickers = _discover_financedatabase_tickers()
        if fd_tickers:
            existing = set(base_symbols)
            for t in fd_tickers:
                if t not in existing:
                    base_symbols.append(t)
                    existing.add(t)

    # --- Italian UCITS ETF discovery ---
    etf_symbols = discover_italian_ucits_etfs()
    if etf_symbols:
        existing = set(base_symbols)
        for etf in etf_symbols:
            if etf not in existing:
                base_symbols.append(etf)
                existing.add(etf)

    # Discover additional tickers from news RSS feeds
    try:
        from src.news.fetcher import discover_tickers_from_news
        news_tickers = discover_tickers_from_news()
        if news_tickers:
            logger.info(f"Discovered {len(news_tickers)} tickers from news feeds")
            # Merge, ensuring uniqueness
            existing = set(base_symbols)
            for t in news_tickers:
                if t not in existing:
                    base_symbols.append(t)
                    existing.add(t)
    except Exception as e:
        logger.warning(f"News ticker discovery failed: {e}")

    # --- Fallback: try static CSV file, then hardcoded list ---
    if not base_symbols:
        static = _load_static_tickers()
        if static:
            logger.info(f"Loaded {len(static)} tickers from static file.")
            base_symbols = static

    if not base_symbols:
        hardcoded = _get_hardcoded_tickers()
        if hardcoded:
            logger.info(f"Loaded {len(hardcoded)} tickers from hardcoded fallback list.")
            base_symbols = hardcoded

    suffix = settings.TICKER_SUFFIX
    target_country = settings.TARGET_COUNTRY.lower()
    if not base_symbols:
        logger.warning("No tickers discovered from Wikipedia, Euronext, or news feeds. Checking Redis cache and DB for previously discovered symbols...")
        # Try Redis cache first (may have symbols from a previous successful run)
        redis_client = get_redis_client()
        cache_key = f"tradable_assets:{settings.TARGET_COUNTRY}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                import json
                cached_list = json.loads(cached)
                # Merge with DB-saved symbols
                try:
                    from src.database import get_all_discovered_symbols
                    db_symbols = get_all_discovered_symbols()
                    existing_set = set(cached_list)
                    for db_entry in db_symbols:
                        db_sym = db_entry["symbol"]
                        db_country = db_entry.get("country")
                        if db_country is not None and db_country.lower() != target_country:
                            continue
                        if re.match(r'^IT[A-Z0-9]{10}$', db_sym):
                            if db_sym not in existing_set:
                                cached_list.append(db_sym)
                                existing_set.add(db_sym)
                        else:
                            candidate = f"{db_sym}{suffix}" if suffix and not db_sym.endswith(suffix) else db_sym
                            if candidate not in existing_set:
                                cached_list.append(candidate)
                                existing_set.add(candidate)
                    logger.info(f"Discovery failed but recovered {len(cached_list)} symbols from Redis cache + DB")
                except Exception as e:
                    logger.warning(f"Failed to merge DB symbols with cached list: {e}")
                if cached_list:
                    return cached_list
        except Exception:
            pass
        # No Redis cache — try DB only
        try:
            from src.database import get_all_discovered_symbols
            db_symbols = get_all_discovered_symbols()
            if db_symbols:
                db_only_list = []
                for db_entry in db_symbols:
                    db_sym = db_entry["symbol"]
                    db_country = db_entry.get("country")
                    # Skip symbols confirmed to be non-Italian
                    if db_country is not None and db_country.lower() != target_country:
                        continue
                    if settings.COUNTRY_FILTER_STRICT and db_country is None and not re.match(r'^IT[A-Z0-9]{10}$', db_sym):
                        continue
                    if re.match(r'^IT[A-Z0-9]{10}$', db_sym):
                        db_only_list.append(db_sym)
                    else:
                        candidate = f"{db_sym}{suffix}" if suffix and not db_sym.endswith(suffix) else db_sym
                        db_only_list.append(candidate)
                if db_only_list:
                    logger.info(f"Discovery failed but recovered {len(db_only_list)} symbols from DB only")
                    return db_only_list
        except Exception as e:
            logger.warning(f"Failed to recover symbols from DB: {e}")
        return []

    # Save discovered symbols to DB.
    # In strict mode, defer saving until AFTER country filtering to avoid
    # polluting the DB with non-Italian symbols.
    if not settings.COUNTRY_FILTER_STRICT:
        try:
            etf_set = set(etf_symbols)
            _save_discovered_assets_to_db(base_symbols, list(etf_set))
        except Exception as e:
            logger.warning(f"Failed to save discovered assets to DB: {e}")

    suffix = settings.TICKER_SUFFIX
    candidates = []
    for sym in base_symbols:
        if re.match(r'^IT[A-Z0-9]{10}$', sym):
            candidates.append(sym)          # BTP ISIN – no suffix
        else:
            candidates.append(f"{sym}{suffix}")

    # Check Redis cache
    redis_client = get_redis_client()
    cache_key = f"tradable_assets:{settings.TARGET_COUNTRY}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            import json
            cached_list = json.loads(cached)
            # Merge with DB-saved symbols even on cache hit
            try:
                from src.database import get_all_discovered_symbols
                db_symbols = get_all_discovered_symbols()
                existing_set = set(cached_list)
                for db_entry in db_symbols:
                    db_sym = db_entry["symbol"]
                    db_country = db_entry.get("country")
                    # Skip symbols confirmed to be non-Italian
                    if db_country is not None and db_country.lower() != target_country:
                        continue
                    if settings.COUNTRY_FILTER_STRICT and db_country is None and not re.match(r'^IT[A-Z0-9]{10}$', db_sym):
                        continue
                    if re.match(r'^IT[A-Z0-9]{10}$', db_sym):
                        if db_sym not in existing_set:
                            cached_list.append(db_sym)
                            existing_set.add(db_sym)
                    else:
                        candidate = f"{db_sym}{suffix}" if suffix and not db_sym.endswith(suffix) else db_sym
                        if candidate not in existing_set:
                            cached_list.append(candidate)
                            existing_set.add(candidate)
            except Exception as e:
                logger.warning(f"Failed to merge DB symbols with cached list: {e}")
            return cached_list
    except Exception:
        pass

    # Filter candidates by country using yfinance
    strict = settings.COUNTRY_FILTER_STRICT
    filtered = []
    for symbol in candidates:
        # BTP ISINs start with IT and are Italian bonds, skip yfinance country check
        if re.match(r'^IT[A-Z0-9]{10}$', symbol):
            if target_country == "italy":
                filtered.append(symbol)
            continue

        country, name = _fetch_info(symbol)
        # Save the fetched country and name to the database for future filtering.
        # In strict mode, only save Italian symbols to DB.
        if country is not None and (not settings.COUNTRY_FILTER_STRICT or country.lower() == target_country):
            try:
                from src.database import save_discovered_symbol
                db_base = symbol
                if suffix and db_base.endswith(suffix):
                    db_base = db_base[:-len(suffix)]
                save_discovered_symbol(db_base, None, None, name or "", country=country)
            except Exception:
                pass
        elif name and not settings.COUNTRY_FILTER_STRICT:
            # Country is None but name is available — save the name for display
            try:
                from src.database import save_discovered_symbol
                db_base = symbol
                if suffix and db_base.endswith(suffix):
                    db_base = db_base[:-len(suffix)]
                save_discovered_symbol(db_base, None, None, name or "", country=None)
            except Exception:
                pass
        if country is None:
            # yfinance failed to return country info.
            # In lenient mode (default), keep the symbol because it was
            # discovered from Italian sources (Wikipedia FTSE MIB,
            # FinanceDatabase country=Italy, news feeds, etc.).
            # In strict mode, drop it.
            if strict:
                logger.debug(f"Symbol {symbol} skipped (country unknown, strict mode)")
            else:
                filtered.append(symbol)
                logger.debug(f"Symbol {symbol} kept (country unknown, assumed from Italian source)")
        elif country.lower() == target_country:
            filtered.append(symbol)
        else:
            logger.debug(f"Symbol {symbol} skipped (country={country}, target={target_country})")

    # In strict mode, save only confirmed-Italian symbols to DB
    if settings.COUNTRY_FILTER_STRICT:
        try:
            from src.database import save_discovered_symbols_batch
            etf_set = set(etf_symbols)
            symbols_to_save = []
            for symbol in filtered:
                if re.match(r'^IT[A-Z0-9]{10}$', symbol):
                    continue  # BTPs are saved separately
                base = symbol
                if suffix and base.endswith(suffix):
                    base = base[:-len(suffix)]
                asset_type = "etf" if base in etf_set else "stock"
                symbols_to_save.append({
                    "symbol": base,
                    "isin": None,
                    "asset_type": asset_type,
                    "name": "",
                    "country": target_country,
                })
            if symbols_to_save:
                save_discovered_symbols_batch(symbols_to_save)
                logger.info(f"Saved {len(symbols_to_save)} confirmed-Italian symbols to DB (strict mode)")
        except Exception as e:
            logger.warning(f"Failed to save filtered assets to DB: {e}")

    # Cache the filtered list for 24 hours
    try:
        import json
        redis_client.set(cache_key, json.dumps(filtered), ex=86400)
    except Exception as e:
        logger.warning(f"Failed to cache tradable assets: {e}")

    # Merge with previously discovered symbols from DB so nothing is lost
    try:
        from src.database import get_all_discovered_symbols
        db_symbols = get_all_discovered_symbols()
        existing_set = set(filtered)
        for db_entry in db_symbols:
            db_sym = db_entry["symbol"]
            db_country = db_entry.get("country")
            # Skip symbols confirmed to be non-Italian
            if db_country is not None and db_country.lower() != target_country:
                continue
            if re.match(r'^IT[A-Z0-9]{10}$', db_sym):
                # BTP ISIN — add as-is (no suffix)
                if db_sym not in existing_set:
                    filtered.append(db_sym)
                    existing_set.add(db_sym)
            else:
                # Stock/ETF — add with suffix
                candidate = f"{db_sym}{suffix}" if suffix and not db_sym.endswith(suffix) else db_sym
                if candidate not in existing_set:
                    filtered.append(candidate)
                    existing_set.add(candidate)
        logger.info(f"Merged {len(db_symbols)} symbols from DB, total: {len(filtered)}")
    except Exception as e:
        logger.warning(f"Failed to merge discovered symbols from DB: {e}")

    logger.info(f"Tradable assets for {settings.TARGET_COUNTRY}: {len(filtered)} symbols")
    return filtered


def _enrich_quotes_with_btp_details(result: Dict[str, Dict[str, Any]], symbols: List[str]):
    """Enrich quote results with BTP maturity, coupon, and name from discovered_symbols."""
    btp_symbols = [s for s in symbols if re.match(r'^IT[A-Z0-9]{10}$', s)]
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
    except Exception as e:
        logger.debug(f"Failed to enrich BTP details in quotes: {e}")


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

    # Try to acquire the lock with a timeout to prevent indefinite blocking
    # when a previous yf.download call hangs and never releases the lock.
    if not _get_quotes_lock.acquire(timeout=30):
        logger.warning(
            "get_quotes: could not acquire lock within 30s (previous call may be hung). "
            "Falling back to cached/DB quotes only."
        )
        return get_quotes_cached(symbols)
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
        except Exception:
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
                except Exception:
                    pass
        if db_quotes:
            logger.debug(f"Loaded {len(db_quotes)} quotes from database (Redis miss fallback)")
    except Exception as e:
        logger.warning(f"DB quote fetch failed: {e}", exc_info=True)

    # --- Try DB close prices first (fast, no network call) ---
    # This ensures quotes are available even when yfinance is rate-limited or blocked.
    # The OHLCV data is populated by background download tasks using borsaitaliana as primary source.
    if missing_symbols:
        try:
            db_candles = get_latest_close_prices(missing_symbols)
            for sym in list(missing_symbols):
                if sym in db_candles and db_candles[sym].get("last", 0) > 0:
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
        except Exception as e:
            logger.warning(f"get_quotes: DB close price fallback failed: {e}")

    if not missing_symbols:
        # All symbols got prices from cache/DB — cache and return
        quotes_to_save = {}
        for sym, q in result.items():
            if q.get("last") is not None:
                try:
                    redis_client.set(f"quote:{sym}", json.dumps(q), ex=300)
                except Exception:
                    pass
                quotes_to_save[sym] = q
        if quotes_to_save:
            try:
                save_quotes_batch(quotes_to_save)
            except Exception as e:
                logger.warning(f"Failed to save quotes to database: {e}")
        return result

    # Initialize result with None for all still-missing symbols that don't have a DB quote yet
    for sym in missing_symbols:
        if sym not in result or result[sym].get("last") is None:
            result[sym] = {"last": None, "bid": None, "ask": None, "volume": None, "change_24h": None, "percentage": None, "quoteVolume": None}

    # Filter out BTP ISINs as they are not supported by yfinance and should be served from DB
    stock_symbols = [s for s in missing_symbols if not re.match(r'^IT[A-Z0-9]{10}$', s)]

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
                        if batch_hist is None or batch_hist.empty or sym not in batch_hist.columns.levels[1]:
                            continue
                        sym_data = batch_hist[sym]
                    else:
                        sym_data = batch_hist

                    if len(sym_data) >= 1:
                        last = sym_data["Close"].iloc[-1]
                        if last is not None and not pd.isna(last) and last > 0:
                            result[sym]["last"] = float(last)
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
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Batch download failed: {e}")
    elif stock_symbols and _check_yf_circuit():
        logger.warning(
            f"get_quotes: yfinance circuit breaker is OPEN — skipping quote fetch for {len(stock_symbols)} symbols. "
            f"Quotes will be served from Redis cache or database if available."
        )

    # --- Try Borsa Italiana for Italian stocks AND BTPs still missing valid prices ---
    # This is a per-symbol HTTP call (token + intraday), so limit to a small batch.
    missing_after_yf = [
        sym for sym in missing_symbols
        if result.get(sym, {}).get("last") is None
    ]
    if missing_after_yf:
        # Limit to 5 symbols to avoid excessive API calls
        for sym in missing_after_yf[:5]:
            bi_quote = get_borsa_italiana_quote(sym)
            if bi_quote:
                result[sym].update(bi_quote)
                logger.debug(f"get_quotes: Borsa Italiana provided quote for {sym}")

    # --- Final pass: compute change_24h and percentage from DB daily candles ---
    # For ALL symbols with a valid last price, recompute change_24h and percentage
    # from the latest 2 daily candles in the database. This ensures consistency
    # and eliminates NULL values when daily candle data is available.
    # DB candles are the primary source; yfinance is only a fallback for the last price.
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
                        result[sym]["change_24h"] = last - prev_close
                        result[sym]["percentage"] = round((last - prev_close) / prev_close * 100, 4)
        except Exception as e:
            logger.warning(f"Failed to recompute change_24h/percentage from DB candles: {e}")

    # Cache the result per-symbol in Redis (5 minutes) and save to database
    quotes_to_save = {}
    for sym in result:
        if result[sym].get("last") is not None:
            try:
                redis_client.set(f"quote:{sym}", json.dumps(result[sym]), ex=300)
            except Exception:
                pass
            quotes_to_save[sym] = result[sym]

    # Save to database for persistence (survives Redis flushes and yfinance outages)
    if quotes_to_save:
        try:
            save_quotes_batch(quotes_to_save)
        except Exception as e:
            logger.warning(f"Failed to save quotes to database: {e}")

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

    # Enrich BTP quotes with maturity, coupon, and name from discovered_symbols
    _enrich_quotes_with_btp_details(result, symbols)

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
        except Exception:
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
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"get_quotes_cached: DB quote fetch failed: {e}", exc_info=True)

    # Try DB close prices for anything still missing
    if missing_symbols:
        try:
            db_candles = get_latest_close_prices(missing_symbols)
            for sym in list(missing_symbols):
                if sym in db_candles and db_candles[sym].get("last", 0) > 0:
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
        except Exception as e:
            logger.warning(f"get_quotes_cached: DB close price fallback failed: {e}")

    # Initialize remaining missing symbols with None values
    for sym in missing_symbols:
        result[sym] = {"last": None, "bid": None, "ask": None, "volume": None,
                       "change_24h": None, "percentage": None, "quoteVolume": None}

    # --- Final pass: compute change_24h and percentage from DB daily candles ---
    # For ALL symbols with a valid last price, recompute change_24h and percentage
    # from the latest 2 daily candles in the database. This ensures consistency
    # and eliminates NULL values when daily candle data is available.
    # DB candles are the primary source; yfinance is only a fallback for the last price.
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
                        result[sym]["change_24h"] = last - prev_close
                        result[sym]["percentage"] = round((last - prev_close) / prev_close * 100, 4)
        except Exception as e:
            logger.warning(f"get_quotes_cached: Failed to recompute change_24h/percentage from DB candles: {e}")

    # Persist DB close prices to Redis and the quotes table so that other
    # consumers (web dashboard, re-evaluation, etc.) can access them even
    # when yfinance is unavailable.  This is a fast batch DB write, not a
    # network call, so it respects the "no network calls" contract of this
    # function.
    quotes_to_save = {}
    for sym, q in result.items():
        if q.get("last") is not None:
            try:
                redis_client.set(f"quote:{sym}", json.dumps(q), ex=300)
            except Exception:
                pass
            quotes_to_save[sym] = q
    if quotes_to_save:
        try:
            save_quotes_batch(quotes_to_save)
        except Exception as e:
            logger.warning(f"get_quotes_cached: Failed to save DB close prices to quotes table: {e}")

    # Enrich BTP quotes with maturity, coupon, and name from discovered_symbols
    _enrich_quotes_with_btp_details(result, symbols)

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
    if not re.match(r'^IT[A-Z0-9]{10}$', symbol) and settings.TICKER_SUFFIX and not symbol.endswith(settings.TICKER_SUFFIX):
        yf_symbol = f"{symbol}{settings.TICKER_SUFFIX}"

    redis_client = get_redis_client()
    cache_ttl = 60 if any(tf in ("1h",) for tf in timeframes) else 300

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
        except Exception:
            pass

        # --- 1. Always try borsaitaliana first ---
        borsa_candles = get_borsa_italiana_candles(symbol, tf, limit=limit)

        # BTPs: only borsaitaliana, no yfinance
        if re.match(r'^IT[A-Z0-9]{10}$', symbol):
            result[tf] = borsa_candles or []
            if borsa_candles:
                try:
                    redis_client.set(cache_key, json.dumps(result[tf]), ex=cache_ttl)
                except Exception:
                    pass
            continue

        # Check if we have ISIN in DB (strip suffix — DB stores base symbols)
        from src.database import get_isin_from_db
        db_lookup_symbol = symbol
        if settings.TICKER_SUFFIX and db_lookup_symbol.endswith(settings.TICKER_SUFFIX):
            db_lookup_symbol = db_lookup_symbol[:-len(settings.TICKER_SUFFIX)]
        db_isin = get_isin_from_db(db_lookup_symbol)
        has_isin = db_isin is not None

        # If we have ISIN, only use borsaitaliana (skip yfinance to avoid rate limits)
        if has_isin:
            if borsa_candles:
                result[tf] = borsa_candles[-limit:] if limit else borsa_candles
                try:
                    redis_client.set(cache_key, json.dumps(result[tf]), ex=cache_ttl)
                except Exception:
                    pass
            else:
                result[tf] = []
            continue

        # No ISIN — use yfinance as fallback
        yf_candles: List[List] = []
        if not _check_yf_circuit():
            yf_symbol = symbol
            if not re.match(r'^IT[A-Z0-9]{10}$', symbol) and settings.TICKER_SUFFIX and not symbol.endswith(settings.TICKER_SUFFIX):
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
            except Exception as e:
                logger.debug(f"yfinance fetch failed for {symbol} {tf}: {e}")

        # Merge both sources
        merged = _merge_candles(borsa_candles, yf_candles)
        if merged:
            result[tf] = merged[-limit:] if limit else merged
            try:
                redis_client.set(cache_key, json.dumps(result[tf]), ex=cache_ttl)
            except Exception:
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
    except Exception:
        pass

    # --- 1. Always try borsaitaliana first ---
    borsa_candles = get_borsa_italiana_candles(symbol, timeframe, limit=limit, start_ms=start_ms)

    # BTPs: only borsaitaliana, no yfinance
    if re.match(r'^IT[A-Z0-9]{10}$', symbol):
        if borsa_candles:
            try:
                redis_client.set(cache_key, json.dumps(borsa_candles), ex=300)
            except Exception:
                pass
            return borsa_candles
        return []

    # Check if we have ISIN in DB (strip suffix — DB stores base symbols)
    from src.database import get_isin_from_db
    db_lookup_symbol = symbol
    if settings.TICKER_SUFFIX and db_lookup_symbol.endswith(settings.TICKER_SUFFIX):
        db_lookup_symbol = db_lookup_symbol[:-len(settings.TICKER_SUFFIX)]
    db_isin = get_isin_from_db(db_lookup_symbol)
    has_isin = db_isin is not None

    # If we have ISIN, only use borsaitaliana (skip yfinance to avoid rate limits)
    if has_isin:
        if borsa_candles:
            if limit and len(borsa_candles) > limit:
                borsa_candles = borsa_candles[-limit:]
            try:
                redis_client.set(cache_key, json.dumps(borsa_candles), ex=300)
            except Exception:
                pass
            return borsa_candles
        return []

    # No ISIN — use yfinance as fallback
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
        except Exception as e:
            logger.debug(f"yfinance fetch failed for {symbol} {timeframe}: {e}")

    # Merge both sources
    merged = _merge_candles(borsa_candles, yf_candles)

    if merged:
        if limit and len(merged) > limit:
            merged = merged[-limit:]
        try:
            redis_client.set(cache_key, json.dumps(merged), ex=300)
        except Exception:
            pass
        return merged
    return []


def _fetch_btp_details(isin: str) -> Dict[str, Optional[Any]]:
    """Fetch BTP details (maturity, coupon, name) from the individual BTP page.

    Scrapes the 'Info Strumento' section of the Borsa Italiana BTP page
    to extract the Scadenza (maturity), coupon rate, and denomination.
    Results are cached in Redis for 24 hours.
    """
    # Check Redis cache first
    redis_client = get_redis_client()
    cache_key = f"btp_details:{isin}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    url = f"https://www.borsaitaliana.it/borsa/obbligazioni/mot/obbligazioni-in-euro/scheda/{isin}-MOTX.html?lang=it"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        with httpx.Client(proxy=_get_proxies(), timeout=15.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            details: Dict[str, Optional[Any]] = {}

            # Robust approach: find ALL tables on the page and look for the
            # relevant keys in any of them. The "Info Strumento" section is
            # the only place with "Scadenza" and "Tasso Cedola" fields.
            for table in soup.find_all("table"):
                for row in table.find_all("tr"):
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        key = cells[0].get_text(strip=True)
                        val = cells[1].get_text(strip=True)
                        if "Scadenza" in key:
                            details["maturity"] = val
                        elif "Tasso Cedola su base Annua" in key:
                            if val:
                                val_cleaned = val.replace(",", ".")
                                try:
                                    details["coupon"] = float(val_cleaned)
                                except ValueError:
                                    details["coupon"] = val
                            # If empty, leave coupon unset (zero-coupon bond)
                        elif "Denominazione" in key:
                            details["name"] = val

            # Cache the result (24h for populated details, 1h for empty to allow retry)
            cache_ttl = 86400 if details else 3600
            try:
                redis_client.set(cache_key, json.dumps(details), ex=cache_ttl)
            except Exception:
                pass

            return details
    except Exception as e:
        logger.debug(f"Failed to fetch BTP details for {isin}: {e}")
        return {}


def discover_btp_bonds() -> List[Dict[str, Any]]:
    """Discover and parse BTP bonds from Borsa Italiana."""
    redis_client = get_redis_client()
    cache_key = "btp_bonds_list"
    import json

    bonds = None
    try:
        cached = redis_client.get(cache_key)
        if cached:
            bonds = json.loads(cached)
    except Exception:
        pass

    _btp_list_was_cached = bonds is not None

    if bonds is None:
        url = settings.BTP_URL
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        bonds = []

        for page in range(1, 11):
            page_url = f"{url}?&page={page}"
            try:
                with httpx.Client(proxy=_get_proxies(), timeout=15.0, follow_redirects=True) as client:
                    response = client.get(page_url, headers=headers)
                if response.status_code != 200:
                    break

                soup = BeautifulSoup(response.text, "html.parser")
                table = soup.find("table")
                if not table:
                    break

                rows = table.find_all("tr")
                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) < 3:
                        continue

                    isin_text = cols[0].get_text(separator=" ", strip=True)
                    isin_match = re.search(r'IT[A-Z0-9]{10}', isin_text)
                    if not isin_match:
                        continue
                    isin = isin_match.group(0)

                    name = cols[1].get_text(strip=True)

                    last_price_str = cols[2].get_text(strip=True).replace(",", ".")
                    try:
                        last_price = float(last_price_str) if last_price_str else None
                    except ValueError:
                        last_price = None

                    coupon_str = cols[3].get_text(strip=True).replace(",", ".") if len(cols) > 3 else ""
                    try:
                        coupon = float(coupon_str) if coupon_str else None
                    except ValueError:
                        coupon = None

                    maturity = cols[4].get_text(strip=True) if len(cols) > 4 else None
                    change_pct = 0.0

                    if last_price is not None:
                        bonds.append({
                            "isin": isin,
                            "name": name,
                            "last_price": last_price,
                            "change_pct": change_pct,
                            "coupon": coupon,
                            "maturity": maturity,
                        })
            except Exception as e:
                logger.warning(f"Failed to fetch BTP page {page}: {e}")
                break

        # Only cache non-empty results so failed scrapes retry on next call
        if bonds:
            try:
                redis_client.set(cache_key, json.dumps(bonds), ex=1800)
            except Exception as e:
                logger.warning(f"Failed to cache BTP bonds: {e}")

    # Only fetch individual BTP details when the list was freshly scraped
    # (not from Redis cache). Bonds that already have maturity/coupon from
    # the list page are skipped. Individual page fetches are cached in Redis
    # for 24 hours by _fetch_btp_details itself.
    if not _btp_list_was_cached:
        for bond in bonds:
            isin = bond["isin"]
            # Skip if maturity and coupon are already present from the list page
            if bond.get("maturity") and bond.get("coupon") is not None:
                continue
            details = _fetch_btp_details(isin)
            if details:
                if details.get("maturity"):
                    bond["maturity"] = details["maturity"]
                if details.get("coupon") is not None:
                    bond["coupon"] = details["coupon"]
                if details.get("name"):
                    bond["name"] = details["name"]
            time.sleep(0.2)  # small delay to avoid rate limiting

    # Always save BTP bonds to DB (idempotent upsert — ensures DB stays populated)
    if bonds:
        try:
            from src.database import save_discovered_symbols_batch
            symbols_to_save = [
                {"symbol": b["isin"], "isin": b["isin"], "asset_type": "btp", "name": b.get("name", ""),
                 "maturity": b.get("maturity"), "coupon": b.get("coupon"), "country": "italy"}
                for b in bonds
            ]
            if symbols_to_save:
                save_discovered_symbols_batch(symbols_to_save)
        except Exception as e:
            logger.warning(f"Failed to save BTP bonds to DB: {e}")

    return bonds
