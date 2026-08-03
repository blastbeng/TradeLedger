import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import pandas as pd
import httpx
from bs4 import BeautifulSoup

from src.config.settings import settings
from src.utils.btp_policy import BTPPolicy
from src.exchanges.proxy_utils import _get_proxies
from src.exchanges.candle_utils import _validate_and_clean_candles
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

# --- Borsa Italiana Circuit Breaker ---
_bi_error_count = 0
_bi_last_error_time = 0.0
_bi_circuit_open_until = 0.0
_bi_lock = threading.Lock()

_bi_http_client: Optional[httpx.Client] = None
_bi_http_client_lock = threading.Lock()

def _get_bi_client(timeout: float = 15.0) -> httpx.Client:
    """Return a shared httpx.Client with connection pooling for Borsa Italiana requests."""
    global _bi_http_client
    with _bi_http_client_lock:
        if _bi_http_client is None or _bi_http_client.is_closed:
            _bi_http_client = httpx.Client(
                proxy=_get_proxies(),
                timeout=timeout,
                follow_redirects=True,
            )
        return _bi_http_client

BI_MAX_ERRORS = 20
BI_CIRCUIT_COOLDOWN = 300  # 5 minutes

def _check_bi_circuit() -> bool:
    """Return True if the Borsa Italiana circuit is open (calls should be skipped)."""
    with _bi_lock:
        return time.time() < _bi_circuit_open_until

def _record_bi_error(exc: Optional[Exception] = None):
    """Record a Borsa Italiana error and potentially trip the circuit breaker."""
    global _bi_error_count, _bi_last_error_time, _bi_circuit_open_until
    with _bi_lock:
        now = time.time()
        if now - _bi_last_error_time > 300:
            _bi_error_count = 0
        _bi_error_count += 1
        _bi_last_error_time = now
        if _bi_error_count >= BI_MAX_ERRORS:
            if _bi_circuit_open_until < now:
                exc_msg = f" Last error: {exc}" if exc else ""
                logger.error(f"Borsa Italiana circuit breaker tripped due to {_bi_error_count} errors. Blocking BI calls for {BI_CIRCUIT_COOLDOWN}s.{exc_msg}")
            _bi_circuit_open_until = now + BI_CIRCUIT_COOLDOWN

def _reset_bi_circuit():
    """Reset the Borsa Italiana circuit breaker after a successful call."""
    global _bi_error_count
    with _bi_lock:
        _bi_error_count = 0


# --- Borsa Italiana token cache ---
_borsa_token_cache: Dict[str, tuple] = {}  # {cache_key: (timestamp, token)}
_borsa_token_cache_lock = threading.Lock()
_BORSA_TOKEN_CACHE_TTL = settings.BORSA_TOKEN_CACHE_TTL


def _get_borsa_italiana_token(isin: str, market_code: str) -> Optional[str]:
    """Dynamically fetch the bearer token from the Borsa Italiana summary chart page, with caching."""
    if _check_bi_circuit():
        return None

    cache_key = f"market-{market_code}"
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
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = _get_bi_client(timeout=15.0)
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
                _reset_bi_circuit()
                return token

            logger.warning(f"Could not find Borsa Italiana token for {isin}-{market_code}")
            return None
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError, AttributeError, OSError) as e:
            _record_bi_error(e)
            logger.warning(f"Failed to fetch Borsa Italiana token for {isin}-{market_code} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None


def _invalidate_borsa_token_cache(isin: str, market_code: str) -> None:
    """Remove a cached Borsa Italiana token so it is re-fetched on next use."""
    cache_key = f"market-{market_code}"
    with _borsa_token_cache_lock:
        _borsa_token_cache.pop(cache_key, None)


def _get_isin_and_info_from_borsa_italiana(base_symbol: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Fetch ISIN, country, and name from Borsa Italiana search page.

    Returns (isin, country, name). Country is always 'Italy' if found.
    """
    if _check_bi_circuit():
        return None, None, None

    url = f"https://www.borsaitaliana.it/borsa/searchengine/search.html?lang=it&q={base_symbol}&Cerca=Search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        client = _get_bi_client(timeout=10.0)
        response = client.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "/borsa/search/scheda.html?code=" in href:
                match = re.search(r'code=([^&]+)', href)
                if match:
                    isin = match.group(1)
                    if re.match(r'^[A-Z]{2}[A-Z0-9]{9}\d$', isin):
                        name = a_tag.get_text(strip=True)
                        if name.lower() == base_symbol.lower():
                            _reset_bi_circuit()
                            return isin, "Italy", name
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, OSError) as e:
        _record_bi_error(e)
        logger.error(f"Borsa Italiana search failed for {base_symbol}: {type(e).__name__}: {e}")
    return None, None, None


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

def get_borsa_italiana_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch the latest real-time quote from Borsa Italiana for an Italian stock or BTP.

    Returns a dict with keys: last, bid, ask, volume, change_24h, percentage,
    quoteVolume, last_update, source.
    Returns None if the quote cannot be fetched.
    """
    if _check_bi_circuit():
        return None

    base = symbol.split("/")[0] if "/" in symbol else symbol

    if BTPPolicy.is_btp(base):
        isin = base
    else:
        from src.exchanges.market_data import _get_isin_from_yfinance
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
        client = _get_bi_client(timeout=10.0)
        url = f"https://grafici.borsaitaliana.it/api/instruments/{isin},{market_code},ISIN/intraday?resolution=1MN"
        for attempt in range(2):
            try:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403) and attempt == 0:
                    logger.debug(f"Borsa Italiana token expired for {symbol}, refreshing...")
                    _invalidate_borsa_token_cache(isin, market_code)
                    token = _get_borsa_italiana_token(isin, market_code)
                    if not token:
                        return None
                    headers["authorization"] = f"Bearer {token}"
                    continue
                raise

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
                _reset_bi_circuit()
                return {
                    "last": last_price,
                    "bid": last_price,
                    "ask": last_price,
                    "volume": vol,
                    "quoteVolume": vol,
                    "last_update": int(time.time() * 1000),
                    "source": "borsa_italiana",
                }
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, OSError) as e:
        _record_bi_error(e)
        logger.warning(f"Borsa Italiana quote fetch failed for {symbol}: {type(e).__name__}: {e}")
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
    if _check_bi_circuit():
        return None

    if timeframe not in BORSA_TIMEFRAME_MAP:
        return None

    base = symbol.split("/")[0] if "/" in symbol else symbol

    # For BTPs, the symbol IS the ISIN
    if BTPPolicy.is_btp(base):
        isin = base
    else:
        from src.exchanges.market_data import _get_isin_from_yfinance
        isin = _get_isin_from_yfinance(base)
        if not isin:
            logger.debug(f"No ISIN found for {symbol}, skipping borsaitaliana")
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
        client = _get_bi_client(timeout=15.0)
        if timeframe == "1d":
            url = f"https://grafici.borsaitaliana.it/api/instruments/{isin},{market_code},ISIN/history/period?period=1M&adjustment=true&add-last-price=true"
            logger.debug(f"Fetching 1d data from history endpoint: {url}")
        else:
            period = BORSA_TIMEFRAME_MAP.get(timeframe)
            url = f"https://grafici.borsaitaliana.it/api/instruments/{isin},{market_code},ISIN/history/period?period={period}&adjustment=true&add-last-price=true"

        for attempt in range(2):
            try:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403) and attempt == 0:
                    logger.debug(f"Borsa Italiana token expired for {symbol}, refreshing...")
                    _invalidate_borsa_token_cache(isin, market_code)
                    token = _get_borsa_italiana_token(isin, market_code)
                    if not token:
                        logger.warning(f"Skipping Borsa Italiana download for {symbol} {timeframe}: no token found after refresh.")
                        return None
                    headers["authorization"] = f"Bearer {token}"
                    continue
                raise

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
            _reset_bi_circuit()
            return _validate_and_clean_candles(rows, symbol)

        return None

    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, OSError) as e:
        _record_bi_error(e)
        logger.warning(f"Borsaitaliana candle download failed for {symbol} {timeframe}: {type(e).__name__}: {e}")
        return None


def _fetch_btp_details(isin: str) -> Dict[str, Optional[Any]]:
    """Fetch BTP details (maturity, coupon, name) from the individual BTP page.

    Scrapes the 'Info Strumento' section of the Borsa Italiana BTP page
    to extract the Scadenza (maturity), coupon rate, and denomination.
    Results are cached in Redis for 24 hours.
    """
    if _check_bi_circuit():
        return {}

    # Check Redis cache first
    redis_client = get_redis_client()
    cache_key = f"btp_details:{isin}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except (TypeError, ValueError, RuntimeError):
        pass

    url = f"https://www.borsaitaliana.it/borsa/obbligazioni/mot/obbligazioni-in-euro/scheda/{isin}-MOTX.html?lang=it"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        client = _get_bi_client(timeout=15.0)
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
                        # Validate maturity date format (expected dd/mm/yyyy)
                        try:
                            datetime.strptime(val, "%d/%m/%Y")
                            details["maturity"] = val
                        except ValueError:
                            logger.warning(f"Invalid maturity date format for {isin}: {val}")
                    elif "Tasso Cedola su base Annua" in key:
                        if val:
                            val_cleaned = val.replace(",", ".")
                            try:
                                coupon_val = float(val_cleaned)
                                # Validate coupon rate range (0% to 15%)
                                if 0.0 <= coupon_val <= 15.0:
                                    details["coupon"] = coupon_val
                                else:
                                    logger.warning(f"Coupon rate out of expected range for {isin}: {coupon_val}")
                            except ValueError:
                                logger.warning(f"Invalid coupon rate format for {isin}: {val}")
                        # If empty, leave coupon unset (zero-coupon bond)
                    elif "Denominazione" in key:
                        details["name"] = val

        # Cache the result (24h for populated details, 1h for empty to allow retry)
        cache_ttl = 86400 if details else 3600
        try:
            redis_client.set(cache_key, json.dumps(details), ex=cache_ttl)
        except (TypeError, ValueError, RuntimeError):
            pass

        _reset_bi_circuit()
        return details
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, AttributeError, OSError) as e:
        _record_bi_error(e)
        logger.warning(f"Failed to fetch BTP details for {isin}: {type(e).__name__}: {e}")
        return {}


def discover_btp_bonds() -> List[Dict[str, Any]]:
    """Discover and parse BTP bonds from Borsa Italiana."""
    if _check_bi_circuit():
        return []

    redis_client = get_redis_client()
    cache_key = "btp_bonds_list"
    import json

    bonds = None
    try:
        cached = redis_client.get(cache_key)
        if cached:
            bonds = json.loads(cached)
    except (TypeError, ValueError, RuntimeError):
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
                client = _get_bi_client(timeout=15.0)
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
                        coupon_val = float(coupon_str) if coupon_str else None
                        if coupon_val is not None and not (0.0 <= coupon_val <= 15.0):
                            logger.warning(f"Coupon rate out of expected range for {isin}: {coupon_val}")
                            coupon_val = None
                        coupon = coupon_val
                    except ValueError:
                        logger.warning(f"Invalid coupon rate format for {isin}: {coupon_str}")
                        coupon = None

                    maturity_str = cols[4].get_text(strip=True) if len(cols) > 4 else None
                    maturity = None
                    if maturity_str:
                        try:
                            datetime.strptime(maturity_str, "%d/%m/%Y")
                            maturity = maturity_str
                        except ValueError:
                            logger.warning(f"Invalid maturity date format for {isin}: {maturity_str}")
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
            except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, OSError) as e:
                _record_bi_error(e)
                logger.warning(f"Failed to fetch BTP page {page}: {type(e).__name__}: {e}")
                break

        # Only cache non-empty results so failed scrapes retry on next call
        if bonds:
            try:
                redis_client.set(cache_key, json.dumps(bonds), ex=1800)
            except (TypeError, ValueError, RuntimeError) as e:
                logger.warning(f"Failed to cache BTP bonds: {e}")

    if bonds:
        _reset_bi_circuit()

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
                {"symbol": b["isin"], "isin": b["isin"], "asset_type": "btp", "name": b.get("name") or None,
                 "maturity": b.get("maturity"), "coupon": b.get("coupon"), "country": "italy"}
                for b in bonds
            ]
            if symbols_to_save:
                save_discovered_symbols_batch(symbols_to_save)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to save BTP bonds to DB: {e}")

    return bonds
