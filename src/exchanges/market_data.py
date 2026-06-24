import hashlib
import logging
import re
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


def _get_yf_session():
    """Return a curl_cffi session that impersonates Chrome for yfinance requests.

    Yahoo Finance increasingly blocks requests that don't look like a real
    browser.  curl_cffi can impersonate Chrome's TLS fingerprint, which
    avoids 401/429 responses.
    """
    try:
        from curl_cffi import requests as curl_requests
        return curl_requests.Session(impersonate="chrome")
    except ImportError:
        logger.warning("curl_cffi not installed – yfinance requests may be blocked.")
        return None
    except Exception as e:
        logger.warning(f"Failed to create curl_cffi session: {e}")
        return None


def _get_isin_from_yfinance(symbol: str) -> Optional[str]:
    """Fetch the ISIN for a stock/ETF symbol from yfinance, cached in Redis for 7 days."""
    redis_client = get_redis_client()
    cache_key = f"isin:{symbol}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return cached
    except Exception:
        pass

    try:
        yf_symbol = symbol
        if not re.match(r'^IT[A-Z0-9]{10}$', symbol) and settings.TICKER_SUFFIX and not symbol.endswith(settings.TICKER_SUFFIX):
            yf_symbol = f"{symbol}{settings.TICKER_SUFFIX}"
        ticker = yf.Ticker(yf_symbol, session=_get_yf_session())
        isin = ticker.isin
        if isin and len(isin) > 0:
            try:
                redis_client.setex(cache_key, 7 * 24 * 3600, isin)
            except Exception:
                pass
            return isin
    except Exception as e:
        logger.debug(f"Failed to fetch ISIN for {symbol}: {e}")
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

INVESTINY_TIMEFRAME_MAP = {
    "1h": "1H",
    "1d": "D",
    "1w": "W",
    "1M": "M",
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


def _get_from_date_for_timeframe(tf: str, now: datetime) -> datetime:
    """Determine a reasonable from_date for Borsa Italiana fallback based on timeframe."""
    if tf == "1h":
        return now - timedelta(days=60)
    elif tf == "1d":
        return now - timedelta(days=365 * 2)
    elif tf == "1w":
        return now - timedelta(days=365 * 10)
    elif tf == "1M":
        return now - timedelta(days=365 * 20)
    elif tf == "3M":
        return now - timedelta(days=365 * 20)
    elif tf == "6M":
        return now - timedelta(days=365 * 30)
    elif tf == "1Y":
        return now - timedelta(days=365 * 30)
    elif tf == "3Y":
        return now - timedelta(days=365 * 30)
    elif tf == "5Y":
        return now - timedelta(days=365 * 30)
    else:
        return now - timedelta(days=365 * 10)


# Mapping of common BTP ISINs to their Investing.com numerical pairId.
# These are used as a fast cache; if an ISIN is not found here, the dynamic
# search API is used.  The engineer must fill in correct IDs.
BTP_ID_MAP: Dict[str, int] = {
    "IT0001086567": 172,   # BTP 10Y (generic yield)
    # Add more entries as needed
}

def _fetch_country(symbol: str, max_retries: int = 2) -> Optional[str]:
    """Fetch the country property from yfinance info for a symbol, with retries.

    Returns the country string on success, or None if yfinance could not
    provide the information after all retries.
    """
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

    # Check Redis cache for all symbols
    redis_client = get_redis_client()
    cache_key = f"quotes:{hashlib.md5(json.dumps(symbols).encode()).hexdigest()}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    result = {}
    btp_symbols = [s for s in symbols if re.match(r'^IT[A-Z0-9]{10}$', s)]
    stock_symbols = [s for s in symbols if s not in btp_symbols]

    # Initialize result with None for all symbols
    for sym in symbols:
        result[sym] = {"last": None, "bid": None, "ask": None, "volume": None, "change_24h": None, "percentage": None, "quoteVolume": None}

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
    if stock_symbols:
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
        try:
            ticker = yf.Ticker(sym, session=_get_yf_session())
            # fast_info gives last price, bid, ask, volume
            info = ticker.fast_info
            last = info.get("lastPrice")
            if last is not None:
                result[sym]["last"] = float(last)
            result[sym]["bid"] = info.get("bid")
            result[sym]["ask"] = info.get("ask")
            result[sym]["volume"] = info.get("lastVolume")
            result[sym]["quoteVolume"] = info.get("lastVolume")

            # Use the pre-fetched previous close
            prev_close = prev_closes.get(sym)
            if last is not None and prev_close is not None:
                change = ((last - prev_close) / prev_close) * 100
                result[sym]["change_24h"] = change
                result[sym]["percentage"] = change
        except Exception:
            pass

    # Cache the result for 60 seconds
    try:
        redis_client.set(cache_key, json.dumps(result), ex=60)
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

def _get_btp_investing_id(isin: str, name: str) -> Optional[int]:
    """Search and cache the Investing.com ID for a BTP using a direct HTTP call."""
    # 1. Check static map
    if isin in BTP_ID_MAP:
        return BTP_ID_MAP[isin]

    redis_client = get_redis_client()
    cache_key = f"investing_id:{isin}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return int(cached)
    except Exception:
        pass

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.investing.com/",
        "Origin": "https://www.investing.com",
    }

    # --- Search API ---
    try:
        url = "https://tvc6.investing.com/search"
        logger.info(f"Searching Investing.com ID for BTP {isin} ({name})")

        for query in [isin, name]:
            params = {"query": query, "limit": 1, "type": ""}
            response = httpx.get(url, params=params, timeout=10.0, headers=headers)
            if response.status_code == 200:
                results = response.json()
                logger.debug(f"Search response for '{query}': {results}")
                if results:
                    # The API may return 'ticker' or 'pairId'
                    investing_id = results[0].get("ticker") or results[0].get("pairId")
                    if investing_id is not None:
                        investing_id = int(investing_id)
                        redis_client.set(cache_key, str(investing_id), ex=86400)
                        return investing_id
    except Exception as e:
        logger.warning(f"Search API failed for BTP {isin} ({name}): {e}")

    # --- Fallback: scrape the bond page for data-pair-id ---
    try:
        search_url = f"https://www.investing.com/search/?q={isin}"
        resp = httpx.get(search_url, headers=headers, timeout=15.0, follow_redirects=True)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Look for a link that contains the ISIN and has a data-pair-id attribute
            for a in soup.find_all("a", href=True):
                if isin in a.get_text() or (name and name.lower() in a.get_text().lower()):
                    pair_id = a.get("data-pair-id")
                    if pair_id:
                        redis_client.set(cache_key, str(pair_id), ex=86400)
                        return int(pair_id)
            # Alternative: look for any element with data-pair-id on the page
            for tag in soup.find_all(attrs={"data-pair-id": True}):
                pair_id = tag["data-pair-id"]
                redis_client.set(cache_key, str(pair_id), ex=86400)
                return int(pair_id)
    except Exception as e:
        logger.warning(f"Fallback scrape failed for BTP {isin}: {e}")

    return None

def get_btp_candles(
    investing_id: int,
    from_date: str,
    to_date: str,
    interval: str = "D",
) -> pd.DataFrame:
    """
    Fetch BTP OHLCV candles from Investing.com's internal charting API.

    Args:
        investing_id: Numerical pairId on Investing.com.
        from_date: Start date as 'DD/MM/YYYY' or 'YYYY-MM-DD'.
        to_date: End date as 'DD/MM/YYYY' or 'YYYY-MM-DD'.
        interval: Resolution (default 'D' for daily). Use keys from
                  INVESTINY_TIMEFRAME_MAP (e.g., '1H', 'D', 'W', 'M').

    Returns:
        pd.DataFrame with DatetimeIndex named 'Date' and columns
        Open, High, Low, Close, Volume.  Empty DataFrame if no data.
    """
    # --- Parse dates -------------------------------------------------
    def _parse_date(s: str) -> datetime:
        for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Unsupported date format: {s}")

    try:
        start_dt = _parse_date(from_date)
        end_dt = _parse_date(to_date)
    except ValueError as e:
        logger.warning(f"Date parsing error in get_btp_candles: {e}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # --- Map interval to Investing.com resolution --------------------
    resolution = INVESTINY_TIMEFRAME_MAP.get(interval)
    if not resolution:
        logger.warning(f"Unsupported interval: {interval}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # --- Call the API ------------------------------------------------
    url = "https://tvc6.investing.com/history"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    params = {
        "symbol": investing_id,
        "resolution": resolution,
        "from": int(start_dt.timestamp()),
        "to": int(end_dt.timestamp()),
    }

    try:
        response = httpx.get(url, params=params, timeout=15.0, headers=headers)
        if response.status_code != 200:
            logger.warning(f"Investing.com API returned {response.status_code} for id {investing_id}")
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        data = response.json()
        if data.get("s") != "ok":
            logger.warning(
                f"Investing.com history API returned status '{data.get('s')}' for id {investing_id}. "
                f"Response: {data}"
            )
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    except Exception as e:
        logger.warning(f"Failed to fetch candles for id {investing_id}: {e}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # --- Build DataFrame --------------------------------------------
    timestamps = data.get("t", [])
    opens = data.get("o", [])
    highs = data.get("h", [])
    lows = data.get("l", [])
    closes = data.get("c", [])
    volumes = data.get("v", [])

    if not timestamps:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # Convert timestamps (seconds) to datetime
    dates = pd.to_datetime(timestamps, unit="s", utc=True)

    df = pd.DataFrame({
        "Open":   [float(o) if o is not None else None for o in opens],
        "High":   [float(h) if h is not None else None for h in highs],
        "Low":    [float(l) if l is not None else None for l in lows],
        "Close":  [float(c) if c is not None else None for c in closes],
        "Volume": [int(v)   if v is not None else 0     for v in volumes],
    }, index=dates)

    df.index.name = "Date"
    # Drop rows where all OHLC are NaN (optional, but keeps data clean)
    df.dropna(subset=["Open", "High", "Low", "Close"], how="all", inplace=True)
    return df

def _fetch_btp_candles_from_borsaitaliana(
    isin: str,
    timeframe: str,
    from_date: datetime,
    to_date: datetime,
    limit: int
) -> Optional[List[List]]:
    """
    Fetch BTP OHLCV candles from Borsa Italiana hidden JSON endpoint.
    Returns list of [timestamp_ms, open, high, low, close, volume]
    or None on failure.
    """
    url = f"https://www.borsaitaliana.it/borsa/obbligazioni/mot/btp/scheda/{isin}-MOTX"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": f"https://www.borsaitaliana.it/borsa/obbligazioni/mot/btp/scheda/{isin}-MOTX.html?lang=it"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            logger.warning(f"Borsa Italiana API returned {response.status_code} for BTP {isin} {timeframe}")
            return None
        
        data = response.json()
        if not data or not isinstance(data, list):
            return None

        candles = []
        from_ts = from_date.timestamp() * 1000
        to_ts = to_date.timestamp() * 1000

        for row in data:
            if not isinstance(row, list) or len(row) < 6:
                continue
            ts = row[0]
            if ts is None:
                continue
            if ts < from_ts or ts > to_ts:
                continue
            candles.append([
                int(ts),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            ])

        if not candles:
            return None

        # Sort by time ascending
        candles.sort(key=lambda x: x[0])
        # Apply limit (most recent)
        if limit and len(candles) > limit:
            candles = candles[-limit:]

        return candles

    except Exception as e:
        logger.warning(f"Failed to fetch BTP candles from Borsa Italiana for {isin}: {e}")
        return None


def _fetch_stock_candles_from_borsaitaliana(
    symbol: str,
    timeframe: str,
    from_date: datetime,
    to_date: datetime,
    limit: int,
    isin: Optional[str] = None,
) -> Optional[List[List]]:
    """
    Fetch stock/ETF OHLCV candles from Borsa Italiana charting API.
    Returns list of [timestamp_ms, open, high, low, close, volume]
    or None on failure.
    """
    TIMEFRAME_MAP_BI = {
        "1h": "1h",
        "1d": "1d",
        "1w": "1w",
        "1M": "1M",
        "3M": "3M",
        "6M": "6M",
        "1Y": "1Y",
        "3Y": "3Y",
        "5Y": "5Y",
    }
    sample_time = TIMEFRAME_MAP_BI.get(timeframe)
    if not sample_time:
        logger.warning(f"Unsupported timeframe for Borsa Italiana Stock: {timeframe}")
        return None

    # Strip any known suffix to get the base ticker (e.g., "MTS.MI" -> "MTS")
    base_symbol = symbol
    for suffix in [".MI", ".mi"]:
        if base_symbol.endswith(suffix):
            base_symbol = base_symbol[:-len(suffix)]
            break

    # Dynamically determine the TimeFrame zoom level based on the requested date range
    days = (to_date - from_date).days
    if days <= 30:
        tf_str = "1m"
    elif days <= 90:
        tf_str = "3m"
    elif days <= 180:
        tf_str = "6m"
    elif days <= 365:
        tf_str = "1y"
    elif days <= 365 * 3:
        tf_str = "3y"
    elif days <= 365 * 5:
        tf_str = "5y"
    elif days <= 365 * 10:
        tf_str = "10y"
    else:
        tf_str = "max"

    url = "https://charts.borsaitaliana.it/charts/services/ChartWService.asmx/GetPrices"
    headers = {
        "Host": "charts.borsaitaliana.it",
        "Origin": "https://www.borsaitaliana.it",
        "Referer": "https://www.borsaitaliana.it/",
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    # Build list of keys to try: ISIN-based first, then symbol-based fallback
    keys_to_try = []
    if isin:
        keys_to_try.append(f"{isin}.MTAA")
        keys_to_try.append(f"{isin}.ETFP")
    keys_to_try.append(f"{base_symbol}.MTA")
    keys_to_try.append(base_symbol)

    # Remove duplicates while preserving order
    seen = set()
    unique_keys = []
    for k in keys_to_try:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)

    for key in unique_keys:
        payload = {
            "request": {
                "SampleTime": sample_time,
                "TimeFrame": tf_str,
                "RequestedDataSetType": "ohlc",
                "ChartPriceType": "price",
                "Key": key,
                "OffSet": 0,
                "FromDate": None,
                "ToDate": None,
                "UseDelay": False,
                "KeyType": "Topic",
                "KeyType2": "Topic",
                "Language": "it-IT",
            }
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            if response.status_code != 200:
                logger.warning(f"Borsa Italiana API returned {response.status_code} for {key} {timeframe}")
                continue
            raw = response.json()
            prices = raw.get("d", {}).get("Prices")
            if not prices:
                logger.warning(f"Borsa Italiana returned no prices for {key} {timeframe}: {raw}")
                continue

            candles = []
            from_ts = from_date.timestamp() * 1000
            to_ts = to_date.timestamp() * 1000

            for p in prices:
                ts = p.get("Time")
                if ts is None:
                    continue
                if ts < from_ts or ts > to_ts:
                    continue
                candles.append([
                    ts,
                    float(p.get("Open", 0)),
                    float(p.get("High", 0)),
                    float(p.get("Low", 0)),
                    float(p.get("Close", 0)),
                    float(p.get("Volume", 0)),
                ])

            if not candles:
                continue

            # Sort by time ascending
            candles.sort(key=lambda x: x[0])
            # Apply limit (most recent)
            if limit and len(candles) > limit:
                candles = candles[-limit:]

            return candles

        except Exception:
            continue

    return None


def _fetch_btp_candles(
    isin: str, name: str, timeframe: str,
    from_date: datetime, to_date: datetime, limit: int
) -> List[List[float]]:
    """Fetch BTP candles using Borsa Italiana first, falling back to Investing.com."""
    # Try Borsa Italiana first
    candles = _fetch_btp_candles_from_borsaitaliana(isin, timeframe, from_date, to_date, limit)
    if candles:
        return candles

    # Fallback to Investing.com
    investing_id = _get_btp_investing_id(isin, name)
    if not investing_id:
        return []

    # Convert datetimes to string format expected by get_btp_candles
    from_str = from_date.strftime("%Y-%m-%d")
    to_str   = to_date.strftime("%Y-%m-%d")

    # Map our timeframe to investiny interval
    interval = INVESTINY_TIMEFRAME_MAP.get(timeframe)
    if not interval:
        logger.warning(f"Unsupported timeframe for BTP: {timeframe}")
        return []

    df = get_btp_candles(investing_id, from_str, to_str, interval=interval)
    if df.empty:
        return []

    # Convert DataFrame to list of lists
    candles = []
    for idx, row in df.iterrows():
        ts_ms = int(idx.timestamp() * 1000)
        candles.append([
            ts_ms,
            row["Open"],
            row["High"],
            row["Low"],
            row["Close"],
            int(row["Volume"]),
        ])

    return candles[-limit:]



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

        # Use investiny for BTPs (ISINs)
        if re.match(r'^IT[A-Z0-9]{10}$', symbol):
            name = _get_btp_name(symbol)
            logger.info(f"Fetching BTP candles for {symbol} ({name}) timeframe {tf}")
            now = datetime.now(timezone.utc)
            inv_interval = INVESTINY_TIMEFRAME_MAP.get(tf)
            if inv_interval == "1H":
                from_date = now - timedelta(days=60)
            elif inv_interval == "D":
                from_date = now - timedelta(days=365)
            elif inv_interval == "W":
                from_date = now - timedelta(days=365*5)
            elif inv_interval == "M":
                from_date = now - timedelta(days=365*10)
            else:
                from_date = now - timedelta(days=365*10)
            
            candles = _fetch_btp_candles(symbol, name, tf, from_date, now, limit)
            result[tf] = candles
            if candles:
                try:
                    redis_client.set(cache_key, json.dumps(candles), ex=cache_ttl)
                except Exception:
                    pass
            continue
        try:
            ticker = yf.Ticker(yf_symbol, session=_get_yf_session())
            # yfinance intraday data limits: 60 days for 5m/15m, 730 days for 60m
            if fetch_interval in ("5m", "15m"):
                period = "60d"
            elif fetch_interval == "60m":
                period = "730d"
            else:
                period = "max"
            hist = ticker.history(period=period, interval=fetch_interval, auto_adjust=False, actions=False)
            yf_candles = None
            if not hist.empty:
                # Filter to essential OHLCV columns only (drop Dividends, Stock Splits, etc.)
                ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
                hist = hist[[col for col in ohlcv_cols if col in hist.columns]]
                candles = []
                for idx, row in hist.iterrows():
                    ts = int(idx.timestamp() * 1000)
                    candles.append([ts, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]])
                
                if needs_aggregation:
                    candles = _aggregate_candles(candles, tf)
                
                yf_candles = candles[-limit:]
            
            # Borsa Italiana fallback if yfinance data is insufficient
            if yf_candles is None or len(yf_candles) < 3:
                isin = _get_isin_from_yfinance(symbol)
                now = datetime.now(timezone.utc)
                from_date = _get_from_date_for_timeframe(tf, now)
                bi_candles = _fetch_stock_candles_from_borsaitaliana(symbol, tf, from_date, now, limit, isin=isin)
                
                if bi_candles is not None and len(bi_candles) > 0:
                    if yf_candles is None or len(bi_candles) > len(yf_candles):
                        result[tf] = bi_candles
                        try:
                            redis_client.set(cache_key, json.dumps(bi_candles), ex=cache_ttl)
                        except Exception:
                            pass
                        continue
            
            if yf_candles is not None and len(yf_candles) > 0:
                result[tf] = yf_candles
                try:
                    redis_client.set(cache_key, json.dumps(yf_candles), ex=cache_ttl)
                except Exception:
                    pass
            else:
                result[tf] = []
        except Exception as e:
            logger.warning(f"Failed to fetch bars for {symbol} {tf}: {e}")
            # Try Borsa Italiana fallback even on exception
            try:
                isin = _get_isin_from_yfinance(symbol)
                now = datetime.now(timezone.utc)
                from_date = _get_from_date_for_timeframe(tf, now)
                fallback = _fetch_stock_candles_from_borsaitaliana(symbol, tf, from_date, now, limit, isin=isin)
                result[tf] = fallback if fallback is not None else []
                if fallback:
                    try:
                        redis_client.set(cache_key, json.dumps(fallback), ex=cache_ttl)
                    except Exception:
                        pass
            except Exception:
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

    # Use investiny for BTPs (ISINs)
    if re.match(r'^IT[A-Z0-9]{10}$', symbol):
        name = _get_btp_name(symbol)
        from_date = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
        to_date = datetime.now(timezone.utc)
        candles = _fetch_btp_candles(symbol, name, timeframe, from_date, to_date, limit)
        if candles:
            try:
                redis_client.set(cache_key, json.dumps(candles), ex=300)
            except Exception:
                pass
        return candles

    # Format symbol for Yahoo Finance: BTP ISINs are used as-is, stocks get TICKER_SUFFIX if missing
    yf_symbol = symbol
    if not re.match(r'^IT[A-Z0-9]{10}$', symbol) and settings.TICKER_SUFFIX and not symbol.endswith(settings.TICKER_SUFFIX):
        yf_symbol = f"{symbol}{settings.TICKER_SUFFIX}"

    start_dt = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
    end_dt = datetime.now(timezone.utc)

    # Yahoo Finance only provides intraday data for the last 730 days.
    # Clamp the start date to avoid "requested range must be within the last 730 days" errors.
    if interval in ("5m", "15m", "60m"):
        earliest_allowed = datetime.now(timezone.utc) - timedelta(days=730)
        if start_dt < earliest_allowed:
            logger.warning(
                f"Clamping start date for {symbol} {timeframe} from {start_dt} to {earliest_allowed} "
                f"(Yahoo intraday limit 730 days)"
            )
            start_dt = earliest_allowed

    try:
        ticker = yf.Ticker(yf_symbol, session=_get_yf_session())
        hist = ticker.history(start=start_dt, end=end_dt, interval=fetch_interval, auto_adjust=False, actions=False)
        yf_candles = None
        if not hist.empty:
            # Filter to essential OHLCV columns only (drop Dividends, Stock Splits, etc.)
            ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
            hist = hist[[col for col in ohlcv_cols if col in hist.columns]]
            candles = []
            for idx, row in hist.iterrows():
                ts = int(idx.timestamp() * 1000)
                candles.append([ts, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]])
            if needs_aggregation:
                candles = _aggregate_candles(candles, timeframe)
            yf_candles = candles[-limit:]
        
        # Borsa Italiana fallback if yfinance data is insufficient
        if yf_candles is None or len(yf_candles) < 3:
            isin = _get_isin_from_yfinance(symbol)
            bi_candles = _fetch_stock_candles_from_borsaitaliana(symbol, timeframe, start_dt, end_dt, limit, isin=isin)
            if bi_candles is not None and len(bi_candles) > 0:
                if yf_candles is None or len(bi_candles) > len(yf_candles):
                    result = bi_candles
                    try:
                        redis_client.set(cache_key, json.dumps(result), ex=300)
                    except Exception:
                        pass
                    return result
        
        if yf_candles is not None and len(yf_candles) > 0:
            result = yf_candles
            try:
                redis_client.set(cache_key, json.dumps(result), ex=300)
            except Exception:
                pass
            return result
        return []
    except Exception as e:
        logger.warning(f"Failed to fetch bars range for {symbol} {timeframe}: {e}")
        # Try Borsa Italiana fallback even on exception
        try:
            isin = _get_isin_from_yfinance(symbol)
            fallback = _fetch_stock_candles_from_borsaitaliana(symbol, timeframe, start_dt, end_dt, limit, isin=isin)
            if fallback is not None:
                if fallback:
                    try:
                        redis_client.set(cache_key, json.dumps(fallback), ex=300)
                    except Exception:
                        pass
                return fallback
        except Exception:
            pass
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
            response = httpx.get(page_url, headers=headers, timeout=15.0, follow_redirects=True)
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
