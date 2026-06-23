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
    "5m": "5m",
    "15m": "15m",
    "1h": "60m",
    "4h": "60m",  # yfinance doesn't have 4h, using 60m as fallback
    "1d": "1d",
}

# Fallback list of FTSE MIB constituents in case Wikipedia scraping fails
FTSE_MIB_FALLBACK_TICKERS = [
    "ENI", "ENEL", "ISP", "UCG", "STLAM", "G", "RACE", "ASSM", "BNP", "TIT",
    "LDO", "BAMI", "MONC", "AZM", "STG", "EXO", "PRY", "REC", "BZU", "FBK",
    "A2A", "HER", "TEN", "INW", "NEXI", "AMP", "PST", "BPSO", "IG", "SPM",
    "CNX", "DAN", "RWAY", "BRE", "UNI", "MGP", "PLT", "BIO", "ALSO", "Ei",
    "IP", "WDA", "ARL", "SRG", "BGN", "MOL", "CEM", "DLG", "TIP", "CLF",
    "KRN", "BCC", "FCT", "ALB", "BEC", "CIR", "DOW", "Ei", "FERR", "GAM",
    "IOL", "LUX", "MARR", "NOS", "OVS", "PAN", "QDM", "RCS", "SAY", "TRV",
    "VBT", "WAM", "ZV"
]


def _fetch_country(symbol: str) -> Optional[str]:
    """Fetch the country property from yfinance info for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get("country")
    except Exception as e:
        logger.debug(f"Failed to fetch country for {symbol}: {e}")
        return None


def _discover_ftse_mib_tickers() -> List[str]:
    """Scrape the FTSE MIB constituent list from Wikipedia.

    Returns a list of base symbols (suffix stripped). Returns a fallback list
    if scraping fails.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(
            "https://en.wikipedia.org/wiki/FTSE_MIB",
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        tables = pd.read_html(response.text)
    except Exception as e:
        logger.warning(f"Failed to scrape FTSE MIB table from Wikipedia: {e}. Using fallback list.")
        return FTSE_MIB_FALLBACK_TICKERS

    for table in tables:
        # Look for a column named "Ticker" or "Symbol" (case-insensitive)
        ticker_col = None
        for col in table.columns:
            col_str = str(col).lower()
            if "ticker" in col_str or "symbol" in col_str:
                ticker_col = col
                break
        if ticker_col is not None:
            tickers = table[ticker_col].dropna().astype(str).tolist()
            # Clean tickers: remove any existing suffix (split by '.' and take first part)
            base_symbols = []
            for t in tickers:
                t = t.strip().upper()
                # Remove any exchange suffix (e.g., ".MI", ".L")
                base = t.split(".")[0] if "." in t else t
                # Keep only alphanumeric base symbols
                if re.match(r"^[A-Z0-9]+$", base):
                    base_symbols.append(base)
            if base_symbols:
                logger.info(f"Discovered {len(base_symbols)} FTSE MIB tickers from Wikipedia")
                return base_symbols

    logger.warning("No ticker column found in Wikipedia FTSE MIB tables. Using fallback list.")
    return FTSE_MIB_FALLBACK_TICKERS


def get_tradable_assets() -> List[str]:
    """Return a list of tradable Italian equity symbols, filtered by country.

    Discovers base symbols dynamically from the FTSE MIB Wikipedia page and
    from news RSS feeds, then appends the configured ticker suffix and
    verifies via yfinance that each symbol's country matches the configured
    TARGET_COUNTRY. Results are cached in Redis for 24 hours.
    """
    # Discover tickers from Wikipedia (FTSE MIB constituents)
    base_symbols = _discover_ftse_mib_tickers()

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

    if not base_symbols:
        logger.warning("No tickers discovered from Wikipedia or news feeds.")
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
