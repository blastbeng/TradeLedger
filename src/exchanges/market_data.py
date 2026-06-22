import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

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


def _fetch_country(symbol: str) -> Optional[str]:
    """Fetch the country property from yfinance info for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get("country")
    except Exception as e:
        logger.debug(f"Failed to fetch country for {symbol}: {e}")
        return None


def get_tradable_assets(trading_client=None) -> List[str]:
    """Return a list of tradable Italian equity symbols, filtered by country.

    Builds candidate symbols by appending the configured ticker suffix to each
    base symbol, then verifies via yfinance that each symbol's country matches
    the configured TARGET_COUNTRY. Results are cached in Redis for 24 hours.
    """
    # Comprehensive list of major Italian stocks (FTSE MIB and mid-cap constituents)
    base_symbols = [
        "ENI", "ENEL", "ISP", "UCG", "STM", "TIT", "FERRARI", "MONC", "AZM",
        "RACE", "BAMI", "MB", "TEN", "PRY", "BPE", "EXO", "INW", "NEXI",
        "REC", "SPM", "BZU", "DIA", "HER", "IPG", "LDO", "STL", "WBG",
        "A2A", "BMO", "CNF", "ERG", "GAM", "ITM", "KOS", "NHF", "PST",
        "SAL", "SRG", "TOD", "UNI", "USC", "VLT", "ZUC",
    ]
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


def get_quotes(data_client=None, symbols: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """Fetch latest quotes for a list of symbols using yfinance fast_info.

    Returns a dict mapping symbol -> {last, bid, ask, volume, change_24h, percentage, quoteVolume}.
    """
    if symbols is None:
        symbols = []
    if not symbols:
        return {}
    result = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.fast_info
            last = info.get("lastPrice")
            bid = info.get("bid")
            ask = info.get("ask")
            volume = info.get("volume")
            prev_close = info.get("previousClose")
            change_24h = ((last - prev_close) / prev_close * 100) if last and prev_close and prev_close > 0 else None
            result[sym] = {
                "last": last,
                "bid": bid,
                "ask": ask,
                "volume": volume,
                "change_24h": change_24h,
                "percentage": change_24h,
                "quoteVolume": volume,
            }
        except Exception as e:
            logger.debug(f"fast_info failed for {sym}: {e}")
            result[sym] = {"last": None, "bid": None, "ask": None, "volume": None, "change_24h": None, "percentage": None, "quoteVolume": None}
    return result


def get_multi_timeframe_bars(
    data_client=None, symbol: str = "", timeframes: List[str] = None, limit: int = 24
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
    data_client=None, symbol: str = "", timeframe: str = "", start_ms: int = 0, limit: int = 500
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
