import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import yfinance as yf

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Map our timeframe strings to yfinance interval strings
TIMEFRAME_MAP = {
    "5m": "5m",
    "15m": "15m",
    "1h": "60m",
    "4h": "60m",  # yfinance doesn't have 4h, using 60m as fallback
    "1d": "1d",
}


def get_tradable_assets(trading_client=None) -> List[str]:
    """Return a list of tradable Italian equity symbols."""
    # Hardcoded list of major Italian stocks (FTSE MIB constituents)
    base_symbols = [
        "ENI", "ENEL", "ISP", "UCG", "STM", "TIT", "FERRARI", "MONC", "AZM",
        "RACE", "BAMI", "MB", "TEN", "PRY", "BPE", "EXO", "INW", "NEXI",
        "REC", "SPM", "BZU", "DIA", "HER", "IPG", "LDO", "STL", "WBG",
        "A2A", "BMO", "CNF", "ERG", "GAM", "ITM", "KOS", "NHF", "PST",
        "SAL", "SRG", "TOD", "UNI", "USC", "VLT", "ZUC"
    ]
    suffix = settings.TICKER_SUFFIX
    return [f"{sym}{suffix}" for sym in base_symbols]


def get_quotes(data_client=None, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch latest quotes for a list of symbols using yfinance.

    Returns a dict mapping symbol -> {last, bid, ask, volume, change_24h}.
    """
    if not symbols:
        return {}
    result = {}
    try:
        tickers = yf.Tickers(" ".join(symbols))
        hist = tickers.history(period="1d", interval="1d")
        for sym in symbols:
            if sym in hist:
                close = hist[sym]["Close"].iloc[-1] if not hist[sym].empty else None
                open_price = hist[sym]["Open"].iloc[-1] if not hist[sym].empty else None
                volume = hist[sym]["Volume"].iloc[-1] if not hist[sym].empty else None
                change_24h = ((close - open_price) / open_price * 100) if close and open_price and open_price > 0 else None
                result[sym] = {
                    "last": close,
                    "bid": None,
                    "ask": None,
                    "volume": volume,
                    "change_24h": change_24h,
                    "percentage": change_24h,
                    "quoteVolume": volume,
                }
            else:
                result[sym] = {"last": None, "bid": None, "ask": None, "volume": None, "change_24h": None, "percentage": None, "quoteVolume": None}
    except Exception as e:
        logger.warning(f"yfinance quote fetch failed: {e}")
        for sym in symbols:
            result.setdefault(sym, {"last": None, "bid": None, "ask": None, "volume": None, "change_24h": None, "percentage": None, "quoteVolume": None})
    return result


def get_multi_timeframe_bars(
    data_client=None, symbol: str, timeframes: List[str], limit: int = 24
) -> Dict[str, List[List[float]]]:
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
    data_client=None, symbol: str, timeframe: str, start_ms: int, limit: int = 500
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
