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
from bs4 import BeautifulSoup

from src.config.settings import settings
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)


# --- yfinance Circuit Breaker ---
_yf_error_count = 0
_yf_last_error_time = 0.0
_yf_circuit_open_until = 0.0
_yf_lock = threading.Lock()

YF_MAX_ERRORS = 10
YF_CIRCUIT_COOLDOWN = 3600  # 1 hour

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
        if not settings.YF_RATE_LIMIT_ENABLED or self.max_requests <= 0:
            return
        with self._lock:
            now = time.time()
            # Remove timestamps outside the window
            while self._timestamps and self._timestamps[0] <= now - self.window_seconds:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.max_requests:
                # Calculate sleep time until the oldest request exits the window
                sleep_time = self.window_seconds - (now - self._timestamps[0])
                if sleep_time > 0:
                    logger.debug(f"yfinance rate limit reached, sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                # Clean up again after sleeping
                now = time.time()
                while self._timestamps and self._timestamps[0] <= now - self.window_seconds:
                    self._timestamps.popleft()
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
    """Return a curl_cffi session that impersonates Chrome for yfinance requests.

    Yahoo Finance increasingly blocks requests that don't look like a real
    browser.  curl_cffi can impersonate Chrome's TLS fingerprint, which
    avoids 401/429 responses.
    """
    try:
        from curl_cffi import requests as curl_requests

        proxies = None
        if settings.YF_PROXY_ENABLED and settings.YF_PROXIES:
            import random
            proxy = random.choice(settings.YF_PROXIES)
            proxies = {"http": proxy, "https": proxy}
            logger.debug(f"Using proxy for yfinance: {proxy}")

        class YFinanceSessionWrapper(curl_requests.Session):
            def request(self, *args, **kwargs):
                if _check_yf_circuit():
                    raise ConnectionError("yfinance circuit breaker is open")
                _yf_rate_limiter.acquire()
                # Enforce a timeout to prevent indefinite hangs
                kwargs.setdefault('timeout', 7.0)
                response = super().request(*args, **kwargs)
                if response.status_code == 401:
                    _record_yf_error()
                else:
                    _reset_yf_circuit()
                return response

        return YFinanceSessionWrapper(impersonate="chrome", proxies=proxies)
    except ImportError:
        logger.warning("curl_cffi not installed – yfinance requests may be blocked.")
        return None
    except Exception as e:
        logger.warning(f"Failed to create curl_cffi session: {e}")
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
    if not settings.YF_PROXY_ENABLED:
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
        if settings.YF_PROXIES:
            use_dynamic = random.choice([True, False])
        else:
            use_dynamic = True
    elif not settings.YF_PROXIES:
        return None

    if use_dynamic:
        proxy = _dynamic_rotator.get_proxy()
        if proxy:
            logger.debug(f"Using dynamic proxy: {proxy}")
            return proxy
    
    if settings.YF_PROXIES:
        proxy = random.choice(settings.YF_PROXIES)
        logger.debug(f"Using static proxy: {proxy}")
        return proxy
    
    return None


def _get_isin_from_yfinance(base_symbol: str) -> Optional[str]:
    """Fetch the ISIN code for a symbol using yfinance, cached in Redis for 7 days."""
    if _check_yf_circuit():
        return None
    redis_client = get_redis_client()
    cache_key = f"isin:{base_symbol}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return cached
    except Exception:
        pass

    suffix = settings.TICKER_SUFFIX
    yf_symbol = f"{base_symbol}{suffix}" if suffix and not base_symbol.endswith(suffix) else base_symbol
    try:
        ticker = yf.Ticker(yf_symbol, session=_get_yf_session())
        isin = ticker.isin
        if isin:
            try:
                redis_client.set(cache_key, isin, ex=7 * 24 * 3600)
            except Exception:
                pass
            return isin
    except Exception as e:
        logger.debug(f"Failed to fetch ISIN for {base_symbol}: {e}")
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

    Returns list of [timestamp_ms, open, high, low, close, volume].
    Returns None if the download fails or the timeframe is not supported
    (e.g., 1h intraday is not available from borsaitaliana).
    """
    # 1h and other intraday timeframes are not available from borsaitaliana
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

    # The API always returns daily candles for the requested period.
    # We always fetch 5Y to get maximum available data, then resample.
    api_period = "5Y"

    # The exchange code is always XMIL for the grafici API
    url = f"https://grafici.borsaitaliana.it/api/instruments/{isin},XMIL,ISIN/history/period?period={api_period}&adjustment=true&add-last-price=true"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
    }

    try:
        with httpx.Client(proxy=_get_proxies(), timeout=8.0, follow_redirects=True) as client:
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

        # Extract the history data
        history = data.get("history", {})
        history_dt = history.get("historyDt", [])

        if not history_dt:
            logger.warning(f"Empty history from borsaitaliana for {symbol} {timeframe}")
            return None

        # Build candle list from the API response
        # Date format is "YYYYMMDD" string
        rows = []
        for item in history_dt:
            dt_str = item.get("dt", "")
            if not dt_str or len(dt_str) != 8:
                continue
            try:
                dt = datetime.strptime(dt_str, "%Y%m%d")
                ts = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
                rows.append([
                    ts,
                    float(item["openPx"]),
                    float(item["highPx"]),
                    float(item["lowPx"]),
                    float(item["closePx"]),
                    float(item.get("qty", 0) or 0),
                ])
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to parse borsaitaliana candle for {symbol}: {e}")
                continue

        if not rows:
            return None

        # Sort by timestamp
        rows.sort(key=lambda c: c[0])

        # Filter by start_ms if provided
        if start_ms is not None:
            rows = [c for c in rows if c[0] >= start_ms]

        # Resample to requested timeframe if needed (not 1d)
        pandas_freq = BORSA_TIMEFRAME_MAP.get(timeframe)
        if pandas_freq is not None:
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
            df = df.resample(pandas_freq).agg(ohlcv_rules)
            df.dropna(subset=['Open'], inplace=True)
            df.reset_index(inplace=True)
            rows = []
            for _, row in df.iterrows():
                ts = int(row['Date'].timestamp() * 1000)
                vol = float(row['Volume']) if pd.notna(row['Volume']) else 0.0
                rows.append([ts, float(row['Open']), float(row['High']), float(row['Low']), float(row['Close']), vol])

        # Sort again after resampling
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

# Borsa Italiana timeframe conversion map (daily data → resampled via pandas)
BORSA_TIMEFRAME_MAP = {
    "1d": None,       # Daily native (no conversion needed)
    "1w": "W",         # Weekly
    "1M": "ME",        # Month End
    "3M": "3ME",       # Quarterly
    "6M": "6ME",       # Semi-annual
    "1Y": "YE",        # Year End
    "3Y": "3YE",       # 3-Year
    "5Y": "5YE",       # 5-Year
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


def _fetch_country(symbol: str, max_retries: int = 2) -> Optional[str]:
    """Fetch the country property from yfinance info for a symbol, with retries.

    Returns the country string on success, or None if yfinance could not
    provide the information after all retries.
    """
    if _check_yf_circuit():
        return None
    import time as _time
    for attempt in range(max_retries + 1):
        try:
            ticker = yf.Ticker(symbol, session=_get_yf_session())
            info = ticker.info
            country = info.get("country")
            if country:
                return country
            # country is None or empty – retry if attempts remain
            if attempt < max_retries:
                _time.sleep(0.5 * (2 ** attempt))
                continue
        except Exception as e:
            logger.debug(f"Failed to fetch country for {symbol} (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                _time.sleep(0.5 * (2 ** attempt))
    return None


def _discover_wikipedia_tickers(urls: List[str], index_name: str) -> List[str]:
    """Scrape a Wikipedia constituent list from one or more URLs.

    Returns base symbols (suffix stripped). Tries each URL in order; returns
    the first non‑empty result.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
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
            
            # Try to find a ticker column by name
            ticker_col = None
            for col in table.columns:
                col_str = str(col).lower()
                if any(kw in col_str for kw in ("ticker", "symbol", "code", "isin", "simbolo", "codice", "yahoo", "borsa")):
                    ticker_col = col
                    break
            
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
            
            if ticker_col is not None:
                tickers = table[ticker_col].dropna().astype(str).tolist()
                base_symbols = []
                for t in tickers:
                    t = t.strip().upper()
                    # Skip ISINs (e.g., IT0001233417)
                    if re.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", t):
                        continue
                    # Remove any text after a space or parenthesis (e.g., "ENI.MI (ENI)")
                    t = re.split(r'[\s(]', t)[0]
                    base = t.split(".")[0] if "." in t else t
                    if re.match(r"^[A-Z0-9]+$", base):
                        base_symbols.append(base)
                
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
        return base_symbols
    except Exception as e:
        logger.warning(f"Failed to discover Italian UCITS ETFs: {e}")
        return []


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

    if not base_symbols:
        logger.warning("No tickers discovered from Wikipedia, Euronext, or news feeds.")
        return []

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
            return json.loads(cached)
    except Exception:
        pass

    # Filter candidates by country using yfinance
    target_country = settings.TARGET_COUNTRY.lower()
    strict = settings.COUNTRY_FILTER_STRICT
    filtered = []
    for symbol in candidates:
        # BTP ISINs start with IT and are Italian bonds, skip yfinance country check
        if re.match(r'^IT[A-Z0-9]{10}$', symbol):
            if target_country == "italy":
                filtered.append(symbol)
            continue

        country = _fetch_country(symbol)
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

    # Cache the filtered list for 24 hours
    try:
        import json
        redis_client.set(cache_key, json.dumps(filtered), ex=86400)
    except Exception as e:
        logger.warning(f"Failed to cache tradable assets: {e}")

    logger.info(f"Tradable assets for {settings.TARGET_COUNTRY}: {len(filtered)} symbols")
    return filtered


def get_quotes(symbols: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """Fetch latest quotes for a list of symbols using yfinance batch download.

    Returns a dict mapping symbol -> {last, bid, ask, volume, change_24h, percentage, quoteVolume}.
    Uses yf.download for efficient batch fetching.
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

    # Initialize result with None for all missing symbols
    for sym in missing_symbols:
        result[sym] = {"last": None, "bid": None, "ask": None, "volume": None, "change_24h": None, "percentage": None, "quoteVolume": None}

    btp_symbols = [s for s in missing_symbols if re.match(r'^IT[A-Z0-9]{10}$', s)]
    stock_symbols = [s for s in missing_symbols if s not in btp_symbols]

    # Fetch BTP quotes from Borsa Italiana cache
    if btp_symbols:
        try:
            btp_bonds = discover_btp_bonds()
            btp_map = {b["isin"]: b for b in btp_bonds}
            for sym in btp_symbols:
                if sym in btp_map:
                    b = btp_map[sym]
                    result[sym]["last"] = b["last_price"]
                    result[sym]["bid"] = b["last_price"]
                    result[sym]["ask"] = b["last_price"]
                    result[sym]["change_24h"] = b["change_pct"]
                    result[sym]["percentage"] = b["change_pct"]
                    result[sym]["name"] = b.get("name")
                    result[sym]["coupon"] = b.get("coupon")
                    result[sym]["maturity"] = b.get("maturity")
        except Exception as e:
            logger.warning(f"Failed to fetch BTP quotes: {e}")

    # --- Batch fetch previous close for all stock symbols ---
    prev_closes = {}
    if stock_symbols and not _check_yf_circuit():
        try:
            batch_hist = yf.download(
                stock_symbols,
                period="2d",
                interval="1d",
                auto_adjust=False,
                actions=False,
                group_by="ticker",
                session=_get_yf_session(),
            )
            for sym in stock_symbols:
                if sym in batch_hist.columns.levels[1]:
                    sym_data = batch_hist[sym]
                    if len(sym_data) >= 2:
                        prev_close = sym_data["Close"].iloc[-2]
                        if prev_close and prev_close > 0:
                            prev_closes[sym] = prev_close
        except Exception as e:
            logger.warning(f"Batch daily history failed: {e}")

    for sym in stock_symbols:
        if _check_yf_circuit():
            break
        try:
            ticker = yf.Ticker(sym, session=_get_yf_session())
            info = ticker.fast_info
            last = info.get("lastPrice")
            if last is not None:
                result[sym]["last"] = float(last)
            bid = info.get("bid")
            ask = info.get("ask")
            if bid is not None:
                result[sym]["bid"] = bid
            if ask is not None:
                result[sym]["ask"] = ask
            vol = info.get("lastVolume")
            if vol is not None:
                result[sym]["volume"] = vol
                result[sym]["quoteVolume"] = vol

            prev_close = prev_closes.get(sym)
            if last is not None and prev_close is not None:
                change = ((last - prev_close) / prev_close) * 100
                result[sym]["change_24h"] = change
                result[sym]["percentage"] = change
        except Exception:
            pass

    # Cache the result per-symbol for 60 seconds
    for sym in missing_symbols:
        if result[sym].get("last") is not None:
            try:
                redis_client.set(f"quote:{sym}", json.dumps(result[sym]), ex=60)
            except Exception:
                pass

    return result


def _get_btp_name(isin: str) -> str:
    """Get the BTP name from the cached BTP bonds list."""
    try:
        btp_bonds = discover_btp_bonds()
        for b in btp_bonds:
            if b["isin"] == isin:
                return b["name"]
    except Exception:
        pass
    return isin

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

        # --- 2. Also fetch from yfinance (not just fallback — always merge) ---
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

        # --- 3. Merge both sources ---
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

    # --- 2. Also fetch from yfinance (not just fallback — always merge) ---
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

    # --- 3. Merge both sources ---
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


def discover_btp_bonds() -> List[Dict[str, Any]]:
    """Discover and parse BTP bonds from Borsa Italiana."""
    redis_client = get_redis_client()
    cache_key = "btp_bonds_list"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            import json
            return json.loads(cached)
    except Exception:
        pass

    url = settings.BORSA_ITALIANA_BTP_URL
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    bonds = []
    import json

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

                # Extract Coupon (Cedola) and Maturity (Scadenza)
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

    try:
        redis_client.set(cache_key, json.dumps(bonds), ex=300)  # Cache for 5 minutes
    except Exception as e:
        logger.warning(f"Failed to cache BTP bonds: {e}")

    return bonds
