import logging
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
import yfinance as yf

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
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            tables = pd.read_html(response.text)
        except Exception as e:
            logger.debug(f"Failed to scrape {url}: {e}")
            continue

        for table in tables:
            # Flatten multi‑level column names
            if isinstance(table.columns, pd.MultiIndex):
                table.columns = [' '.join(col).strip() for col in table.columns.values]
            # Try to find a ticker column
            ticker_col = None
            for col in table.columns:
                col_str = str(col).lower()
                if any(kw in col_str for kw in ("ticker", "symbol", "code", "isin", "ticker symbol", "simbolo")):
                    ticker_col = col
                    break
            if ticker_col is None:
                # Last resort: look for a column whose values look like tickers
                for col in table.columns:
                    sample = table[col].dropna().astype(str).head(5).tolist()
                    if all(re.match(r'^[A-Z0-9\.]+$', s) for s in sample):
                        ticker_col = col
                        break
            if ticker_col is not None:
                tickers = table[ticker_col].dropna().astype(str).tolist()
                base_symbols = []
                for t in tickers:
                    t = t.strip().upper()
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


def _discover_euronext_milan_tickers() -> List[str]:
    """Download the Euronext ISIN directory CSV and extract all Milan-listed tickers.

    Returns a list of base symbols (suffix stripped). Returns an empty list
    if the download or parsing fails.
    """
    import io
    urls_to_try = [
        "https://live.euronext.com/en/isin-directory/download?format=csv",
        "https://live.euronext.com/en/isin-directory/download?format=csv&market=XMIL",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for url in urls_to_try:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            # Auto-detect delimiter (comma or semicolon)
            df = pd.read_csv(io.StringIO(response.text), sep=None, engine='python', dtype=str)
            break
        except Exception as e:
            logger.debug(f"Euronext CSV attempt failed for {url}: {e}")
            continue
    else:
        logger.warning("All Euronext CSV URLs failed.")
        return []

    # --- Find the market column ---
    market_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ("market", "exchange", "trading venue", "mic")):
            market_col = col
            break
    if market_col is None:
        logger.warning("Euronext CSV has no recognisable market column. Columns: %s", list(df.columns))
        return []

    # Filter for Milan
    milan_mask = df[market_col].str.lower().str.contains("milan", na=False)
    milan_df = df[milan_mask]
    if milan_df.empty:
        logger.warning("No Milan rows found in Euronext CSV (market column: %s).", market_col)
        return []

    # --- Find the ticker column ---
    ticker_col = None
    for col in milan_df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ("ticker", "symbol", "code", "isin", "name")):
            # Prefer shorter columns (ticker is usually short)
            if ticker_col is None or len(str(milan_df[col].iloc[0])) < len(str(milan_df[ticker_col].iloc[0])):
                ticker_col = col
    if ticker_col is None:
        logger.warning("Euronext CSV has no recognisable ticker column. Columns: %s", list(milan_df.columns))
        return []

    tickers = milan_df[ticker_col].dropna().astype(str).tolist()
    base_symbols = []
    for t in tickers:
        t = t.strip().upper()
        # Remove any exchange suffix (e.g., ".MI", ".MIL")
        base = t.split(".")[0] if "." in t else t
        if re.match(r"^[A-Z0-9]+$", base):
            base_symbols.append(base)

    if base_symbols:
        logger.info(f"Discovered {len(base_symbols)} Milan tickers from Euronext ISIN directory")
    return base_symbols


def _discover_euronext_milan_from_html() -> List[str]:
    """Scrape the Euronext Milan equities list page for ticker symbols."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        url = "https://live.euronext.com/en/markets/milan/equities/list"
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(response.text)
    except Exception as e:
        logger.warning(f"Failed to scrape Euronext Milan HTML page: {e}")
        return []

    for table in tables:
        # Flatten multi‑level columns
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = [' '.join(col).strip() for col in table.columns.values]
        ticker_col = None
        for col in table.columns:
            col_str = str(col).lower()
            if any(kw in col_str for kw in ("ticker", "symbol", "code", "isin", "name")):
                # Prefer shorter columns
                if ticker_col is None or len(str(table[col].iloc[0])) < len(str(table[ticker_col].iloc[0])):
                    ticker_col = col
        if ticker_col is None:
            # Fallback: any column with short uppercase strings
            for col in table.columns:
                sample = table[col].dropna().astype(str).head(5).tolist()
                if all(re.match(r'^[A-Z0-9\.]+$', s) for s in sample):
                    ticker_col = col
                    break
        if ticker_col is not None:
            tickers = table[ticker_col].dropna().astype(str).tolist()
            base_symbols = []
            for t in tickers:
                t = t.strip().upper()
                base = t.split(".")[0] if "." in t else t
                if re.match(r"^[A-Z0-9]+$", base):
                    base_symbols.append(base)
            if base_symbols:
                logger.info(f"Discovered {len(base_symbols)} Milan tickers from Euronext HTML page")
                return base_symbols

    logger.warning("No ticker column found on Euronext Milan HTML page.")
    return []


def _discover_euronext_milan_json() -> List[str]:
    """Fetch the Euronext Milan instrument list via the JSON API.

    Returns base symbols (suffix stripped). Returns an empty list on failure.
    """
    import json
    url = "https://live.euronext.com/en/ajax/getDirectoryDownload?format=json&market=XMIL"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.warning(f"Euronext JSON API failed: {e}")
        return []

    # The JSON structure is a list of objects; each has "isin", "name", "symbol", "market", etc.
    if not isinstance(data, list):
        logger.warning("Euronext JSON response is not a list.")
        return []

    base_symbols = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # The market is already filtered by the API parameter, but double-check
        market = item.get("market", "")
        if "milan" not in market.lower():
            continue
        symbol = item.get("symbol") or item.get("ticker") or item.get("code")
        if not symbol:
            continue
        symbol = symbol.strip().upper()
        base = symbol.split(".")[0] if "." in symbol else symbol
        if re.match(r"^[A-Z0-9]+$", base):
            base_symbols.append(base)

    if base_symbols:
        logger.info(f"Discovered {len(base_symbols)} Milan tickers from Euronext JSON API")
    return base_symbols


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

    # --- Euronext Milan ISIN directory (official list) ---
    if settings.EURONEXT_TICKER_DISCOVERY_ENABLED:
        euronext_tickers = _discover_euronext_milan_tickers()
        if not euronext_tickers:
            euronext_tickers = _discover_euronext_milan_from_html()
        if not euronext_tickers:
            euronext_tickers = _discover_euronext_milan_json()
        if euronext_tickers:
            existing = set(base_symbols)
            for t in euronext_tickers:
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
    # Initialize result with None for all symbols
    for sym in symbols:
        result[sym] = {"last": None, "bid": None, "ask": None, "volume": None, "change_24h": None, "percentage": None, "quoteVolume": None}

    try:
        # Fetch intraday data for latest price and volume
        intraday = yf.download(symbols, period="1d", interval="5m", progress=False, group_by='column')
        if not intraday.empty:
            last_row = intraday.iloc[-1]
            for sym in symbols:
                try:
                    if len(symbols) > 1:
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
        logger.warning(f"Batch intraday download failed: {e}")

    try:
        # Fetch daily data for previous close to calculate change_24h
        daily = yf.download(symbols, period="2d", interval="1d", progress=False, group_by='column')
        if not daily.empty:
            for sym in symbols:
                try:
                    if len(symbols) > 1:
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
        logger.warning(f"Batch daily download failed: {e}")

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
