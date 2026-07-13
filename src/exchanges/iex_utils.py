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

# --- IEX Cloud ---
_iex_rate_limiter = YFinanceRateLimiter(
    max_requests=100,  # IEX free tier: 100 req/min
    window_seconds=60,
    use_yf_settings=False,
)


def get_iex_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch a single quote from IEX Cloud. Returns None on failure."""
    if not settings.IEX_ENABLED or not settings.IEX_API_KEY:
        return None

    base = symbol.split("/")[0] if "/" in symbol else symbol
    if is_btp_isin(base):
        return None

    suffix = settings.TICKER_SUFFIX
    if suffix and base.endswith(suffix):
        base = base[:-len(suffix)]

    try:
        _iex_rate_limiter.acquire()
    except ConnectionError:
        return None

    url = f"https://cloud.iexapis.com/stable/stock/{base}/quote?token={settings.IEX_API_KEY}"
    try:
        with httpx.Client(proxy=_get_proxies(), timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            last = float(data.get("latestPrice", 0) or 0)
            if last <= 0:
                return None
            vol = float(data.get("latestVolume", 0) or 0)
            change = float(data.get("change", 0) or 0)
            pct = float(data.get("changePercent", 0) or 0)
            return {
                "last": last,
                "bid": float(data.get("iexBidPrice", 0) or last),
                "ask": float(data.get("iexAskPrice", 0) or last),
                "volume": vol,
                "change_24h": change,
                "percentage": pct,
                "quoteVolume": vol,
                "last_update": int(time.time() * 1000),
                "source": "iex",
            }
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, OSError) as e:
        logger.warning(f"IEX quote failed for {symbol}: {type(e).__name__}: {e}")
        return None


def get_iex_candles(
    symbol: str, timeframe: str, limit: int = 500, start_ms: int = None
) -> Optional[List[List]]:
    """Fetch OHLCV candles from IEX Cloud. Returns None on failure."""
    if not settings.IEX_ENABLED or not settings.IEX_API_KEY:
        return None

    base = symbol.split("/")[0] if "/" in symbol else symbol
    if is_btp_isin(base):
        return None

    suffix = settings.TICKER_SUFFIX
    if suffix and base.endswith(suffix):
        base = base[:-len(suffix)]

    # Map timeframes to IEX chart ranges
    iex_range = None
    if timeframe == "1h":
        iex_range = "1d"
    elif timeframe == "1d":
        iex_range = "1m"  # 1 month of daily data
    elif timeframe == "1w":
        iex_range = "3m"
    elif timeframe == "1M":
        iex_range = "1y"
    elif timeframe in ("3M", "6M"):
        iex_range = "1y"
    elif timeframe in ("1Y", "3Y", "5Y"):
        iex_range = "5y"
    else:
        return None

    try:
        _iex_rate_limiter.acquire()
    except ConnectionError:
        return None

    url = f"https://cloud.iexapis.com/stable/stock/{base}/chart/{iex_range}?token={settings.IEX_API_KEY}"
    try:
        with httpx.Client(proxy=_get_proxies(), timeout=15.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            if not data:
                return None

            rows = []
            for item in data:
                dt_str = item.get("date", "")
                minute = item.get("minute", "")
                try:
                    if minute:
                        dt = datetime.strptime(f"{dt_str} {minute}", "%Y-%m-%d %H:%M")
                    else:
                        dt = datetime.strptime(dt_str, "%Y-%m-%d")
                    ts_ms = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
                except ValueError:
                    continue
                if start_ms is not None and ts_ms < start_ms:
                    continue
                o = float(item.get("open", 0) or 0)
                h = float(item.get("high", 0) or 0)
                l = float(item.get("low", 0) or 0)
                c = float(item.get("close", 0) or 0)
                v = float(item.get("volume", 0) or 0)
                if o <= 0 or c <= 0:
                    continue
                rows.append([ts_ms, o, h, l, c, v])

            if not rows:
                return None
            rows.sort(key=lambda c: c[0])
            if limit and len(rows) > limit:
                rows = rows[-limit:]
            return _validate_and_clean_candles(rows, symbol)
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, OSError) as e:
        logger.warning(f"IEX candles failed for {symbol} {timeframe}: {type(e).__name__}: {e}")
        return None
