import logging
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import httpx

from src.config.settings import settings
from src.utils.symbol_utils import is_btp_isin
from src.exchanges.proxy_utils import _get_proxies
from src.exchanges.yf_session import YFinanceRateLimiter
from src.exchanges.candle_utils import _validate_and_clean_candles

logger = logging.getLogger(__name__)

# --- Alpha Vantage Rate Limiter ---
_av_rate_limiter = YFinanceRateLimiter(
    max_requests=settings.ALPHAVANTAGE_RATE_LIMIT_PER_MIN,
    window_seconds=60,
    use_yf_settings=False,
)


def get_alphavantage_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch a single quote from Alpha Vantage. Returns None on failure."""
    if not settings.ALPHAVANTAGE_ENABLED or not settings.ALPHAVANTAGE_API_KEY:
        return None

    base = symbol.split("/")[0] if "/" in symbol else symbol
    if is_btp_isin(base):
        return None  # Alpha Vantage does not support BTPs

    # Strip suffix for Alpha Vantage (it uses plain US-style tickers)
    suffix = settings.TICKER_SUFFIX
    if suffix and base.endswith(suffix):
        base = base[:-len(suffix)]

    try:
        _av_rate_limiter.acquire()
    except ConnectionError:
        return None

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={base}&apikey={settings.ALPHAVANTAGE_API_KEY}"
    try:
        with httpx.Client(proxy=_get_proxies(), timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            quote = data.get("Global Quote")
            if not quote:
                return None
            last = float(quote.get("05. price", 0))
            if last <= 0:
                return None
            change = float(quote.get("09. change", 0) or 0)
            pct = float(quote.get("10. change percent", "0").rstrip("%") or 0)
            vol = float(quote.get("06. volume", 0) or 0)
            return {
                "last": last,
                "bid": last,
                "ask": last,
                "volume": vol,
                "change_24h": change,
                "percentage": pct,
                "quoteVolume": vol,
                "last_update": int(time.time() * 1000),
                "source": "alphavantage",
            }
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, OSError) as e:
        logger.warning(f"Alpha Vantage quote failed for {symbol}: {type(e).__name__}: {e}")
        return None


def get_alphavantage_candles(
    symbol: str, timeframe: str, limit: int = 500, start_ms: int = None
) -> Optional[List[List]]:
    """Fetch OHLCV candles from Alpha Vantage. Returns None on failure."""
    if not settings.ALPHAVANTAGE_ENABLED or not settings.ALPHAVANTAGE_API_KEY:
        return None

    base = symbol.split("/")[0] if "/" in symbol else symbol
    if is_btp_isin(base):
        return None

    suffix = settings.TICKER_SUFFIX
    if suffix and base.endswith(suffix):
        base = base[:-len(suffix)]

    # Map our timeframes to Alpha Vantage functions
    av_function = None
    av_interval = None
    if timeframe == "1h":
        av_function = "TIME_SERIES_INTRADAY"
        av_interval = "60min"
    elif timeframe == "1d":
        av_function = "TIME_SERIES_DAILY"
    elif timeframe == "1w":
        av_function = "TIME_SERIES_WEEKLY"
    elif timeframe == "1M":
        av_function = "TIME_SERIES_MONTHLY"
    else:
        return None  # Unsupported timeframe for Alpha Vantage

    try:
        _av_rate_limiter.acquire()
    except ConnectionError:
        return None

    params = {
        "function": av_function,
        "symbol": base,
        "apikey": settings.ALPHAVANTAGE_API_KEY,
        "outputsize": "full" if (start_ms is not None or limit > 100) else "compact",
    }
    if av_interval:
        params["interval"] = av_interval

    try:
        with httpx.Client(proxy=_get_proxies(), timeout=15.0) as client:
            response = client.get("https://www.alphavantage.co/query", params=params)
            response.raise_for_status()
            data = response.json()

            # Find the time series key (varies by function)
            ts_key = None
            for key in data:
                if "Time Series" in key:
                    ts_key = key
                    break
            if not ts_key:
                return None

            ts = data[ts_key]
            rows = []
            for dt_str, values in ts.items():
                try:
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    ts_ms = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
                except ValueError:
                    try:
                        dt = datetime.strptime(dt_str, "%Y-%m-%d")
                        ts_ms = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
                    except ValueError:
                        continue
                if start_ms is not None and ts_ms < start_ms:
                    continue
                rows.append([
                    ts_ms,
                    float(values.get("1. open", 0)),
                    float(values.get("2. high", 0)),
                    float(values.get("3. low", 0)),
                    float(values.get("4. close", 0)),
                    float(values.get("5. volume", 0) or 0),
                ])

            if not rows:
                return None
            rows.sort(key=lambda c: c[0])
            if limit and len(rows) > limit:
                rows = rows[-limit:]
            return _validate_and_clean_candles(rows, symbol)
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, OSError) as e:
        logger.warning(f"Alpha Vantage candles failed for {symbol} {timeframe}: {type(e).__name__}: {e}")
        return None
