import logging
import re
import warnings
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
import yfinance as yf
import httpx
from bs4 import BeautifulSoup

from src.config.settings import settings
from src.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

# Map our timeframe strings to yfinance interval strings
TIMEFRAME_MAP = {
    "1h": "60m",
    "1d": "1d",
}

def _fetch_country(symbol: str) -> Optional[str]:
    """Fetch the country property from yfinance info for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get("country")
    except Exception as e:
        logger.debug(f"Failed to fetch country for {symbol}: {e}")
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
            # Try to find a ticker column
            ticker_col = None
            for col in table.columns:
                col_str = str(col).lower()
                if any(kw in col_str for kw in ("ticker", "symbol", "code", "isin", "simbolo", "codice")):
                    ticker_col = col
                    break
            if ticker_col is None:
                # Last resort: look for a column whose values look like tickers
                for col in table.columns:
                    sample = table[col].dropna().astype(str).head(10).tolist()
                    # Match typical ticker patterns like ENI, ENI.MI, etc. (avoid ISINs)
                    if all(re.match(r'^[A-Z0-9]{1,6}(\.[A-Z]{2})?$', s) for s in sample):
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
    candidates = [f"{sym}{suffix}" for sym in base_symbols]

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
    filtered = []
    for symbol in candidates:
        country = _fetch_country(symbol)
        if country is not None and country.lower() == target_country:
            filtered.append(symbol)
        else:
            logger.debug(f"Symbol {symbol} skipped (country={country}, target={target_country})")

    # Cache the filtered list for 24 hours
    try:
        import json
        redis_client.setex(cache_key, 86400, json.dumps(filtered))
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
        except Exception as e:
            logger.warning(f"Failed to fetch BTP quotes: {e}")

    # Batch in chunks of 10 to avoid yfinance rate limits and timeouts
    chunk_size = 10
    for i in range(0, len(stock_symbols), chunk_size):
        chunk = stock_symbols[i:i+chunk_size]
        try:
            # Fetch intraday data for latest price and volume
            intraday = yf.download(chunk, period="1d", interval="5m", progress=False, group_by='column')
            if not intraday.empty:
                last_row = intraday.iloc[-1]
                for sym in chunk:
                    try:
                        if len(chunk) > 1:
                            last = last_row[("Close", sym)]
                            volume = last_row[("Volume", sym)]
                        else:
                            last = last_row["Close"]
                            volume = last_row["Volume"]
                        if not pd.isna(last):
                            result[sym]["last"] = float(last)
                        if not pd.isna(volume):
                            result[sym]["volume"] = float(volume)
                            result[sym]["quoteVolume"] = float(volume)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Batch intraday download failed for chunk {i//chunk_size}: {e}")

        try:
            # Fetch daily data for previous close to calculate change_24h
            daily = yf.download(chunk, period="2d", interval="1d", progress=False, group_by='column')
            if not daily.empty:
                for sym in chunk:
                    try:
                        if len(chunk) > 1:
                            prev_close = daily[("Close", sym)].iloc[-2]
                        else:
                            prev_close = daily["Close"].iloc[-2]
                        if not pd.isna(prev_close) and result[sym]["last"]:
                            last = result[sym]["last"]
                            change_24h = ((last - prev_close) / prev_close * 100) if prev_close > 0 else None
                            result[sym]["change_24h"] = change_24h
                            result[sym]["percentage"] = change_24h
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Batch daily download failed for chunk {i//chunk_size}: {e}")

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
    result = {}
    for tf in timeframes:
        interval = TIMEFRAME_MAP.get(tf)
        if not interval:
            logger.warning(f"Unsupported timeframe: {tf}")
            continue
        try:
            ticker = yf.Ticker(symbol)
            # yfinance intraday data is limited to 60 days
            period = "60d" if interval in ("5m", "15m", "60m") else "1y"
            hist = ticker.history(period=period, interval=interval)
            if not hist.empty:
                candles = []
                for idx, row in hist.iterrows():
                    ts = int(idx.timestamp() * 1000)
                    candles.append([ts, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]])
                # Take the last `limit` candles
                result[tf] = candles[-limit:]
            else:
                result[tf] = []
        except Exception as e:
            logger.warning(f"Failed to fetch bars for {symbol} {tf}: {e}")
            result[tf] = []
    return result


def get_bars_range(
    symbol: str = "", timeframe: str = "", start_ms: int = 0, limit: int = 500
) -> List[List[float]]:
    """Fetch OHLCV bars from a start timestamp (ms) up to the present using yfinance.

    Returns a list of candles [timestamp_ms, open, high, low, close, volume].
    """
    interval = TIMEFRAME_MAP.get(timeframe)
    if not interval:
        logger.warning(f"Unsupported timeframe: {timeframe}")
        return []
    start_dt = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
    end_dt = datetime.now(timezone.utc)
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_dt, end=end_dt, interval=interval)
        if not hist.empty:
            candles = []
            for idx, row in hist.iterrows():
                ts = int(idx.timestamp() * 1000)
                candles.append([ts, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]])
            return candles[-limit:]
        return []
    except Exception as e:
        logger.warning(f"Failed to fetch bars range for {symbol} {timeframe}: {e}")
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

                change_pct = 0.0

                if last_price is not None:
                    bonds.append({
                        "isin": isin,
                        "name": name,
                        "last_price": last_price,
                        "change_pct": change_pct
                    })
        except Exception as e:
            logger.warning(f"Failed to fetch BTP page {page}: {e}")
            break

    try:
        redis_client.setex(cache_key, 300, json.dumps(bonds))  # Cache for 5 minutes
    except Exception as e:
        logger.warning(f"Failed to cache BTP bonds: {e}")

    return bonds
